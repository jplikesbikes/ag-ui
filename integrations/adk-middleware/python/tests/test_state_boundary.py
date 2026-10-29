"""Backend bookkeeping in session state never reaches the client.

The middleware keeps its own records in session state (pending tool calls, LRO
id remaps, the thread/app/user stamps, the processed-message ledger). Clients
echo back the state they are shown on every run, so a record the client sees
comes back stale: an answered call returns to ``pending_tool_calls`` and stays
pending for the life of the thread. The run's context is the one exception: it
is the client's own input, echoed back on purpose (#964).
"""

import pytest

from ag_ui.core import EventType, Tool as AGUITool, ToolMessage, UserMessage
from ag_ui_adk import ADKAgent, AGUIToolset
from ag_ui_adk.endpoint import create_adk_app
from ag_ui_adk.event_translator import EventTranslator
from ag_ui_adk.session_manager import (
    CONTEXT_STATE_KEY,
    INTERNAL_STATE_KEYS,
    SessionManager,
)
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps import App, ResumabilityConfig
from google.adk.sessions import InMemorySessionService
from httpx import ASGITransport, AsyncClient

from tests.hitl_helpers import ScriptedLlm, collect, run_input, tool_call

APP = "state_boundary_app"
USER_ID = "test_user"
QUESTION = UserMessage(id="u-1", role="user", content="Chart my usage")
CHART_TOOLS = [
    AGUITool(
        name="render_chart",
        description="Render a chart",
        parameters={"type": "object", "properties": {}},
    )
]
HIDDEN_KEYS = INTERNAL_STATE_KEYS - {CONTEXT_STATE_KEY}


@pytest.fixture(autouse=True)
def reset_session_manager():
    SessionManager.reset_instance()
    yield
    SessionManager.reset_instance()


def _chart_agent(store):
    llm = ScriptedLlm(model="scripted", first_call={"name": "render_chart"})
    return ADKAgent.from_app(
        App(
            name=APP,
            root_agent=LlmAgent(name="chart_agent", model=llm, tools=[AGUIToolset()]),
            resumability_config=ResumabilityConfig(is_resumable=True),
        ),
        user_id=USER_ID,
        session_service=store,
        use_thread_id_as_session_id=True,
    )


async def _answered_chart(store, thread, *, state=None):
    """Ask, then answer the chart call, echoing ``state`` with the answer."""
    agent = _chart_agent(store)
    asked = await collect(agent, run_input(thread, "run-1", [QUESTION], tools=CHART_TOOLS))
    call_id, _ = tool_call(asked, "render_chart")
    answer = ToolMessage(id="t-1", role="tool", tool_call_id=call_id, content="{}")
    answered = await collect(
        agent,
        run_input(
            thread, "run-2", [QUESTION, answer], tools=CHART_TOOLS,
            **({"state": state} if state is not None else {}),
        ),
    )
    return agent, call_id, asked + answered


def _paths(delta) -> list:
    return [(op.model_dump() if hasattr(op, "model_dump") else op)["path"] for op in delta]


def _shown_keys(events) -> set:
    shown = set()
    for event in events:
        if event.type == EventType.STATE_SNAPSHOT:
            shown |= set(event.snapshot)
        elif event.type == EventType.STATE_DELTA:
            shown |= {path.lstrip("/").split("/")[0] for path in _paths(event.delta)}
    return shown


@pytest.mark.parametrize("key", sorted(HIDDEN_KEYS))
def test_translated_state_hides_backend_keys(key):
    translator = EventTranslator()
    state = {key: "backend", CONTEXT_STATE_KEY: [], "visible": 42}

    assert translator._create_state_snapshot_event(state).snapshot == {
        CONTEXT_STATE_KEY: [],
        "visible": 42,
    }
    delta = translator._create_state_delta_event(state, "thread", "run").delta
    assert sorted(_paths(delta)) == [f"/{CONTEXT_STATE_KEY}", "/visible"]


@pytest.mark.asyncio
async def test_a_tool_round_trip_shows_the_client_only_its_state():
    store = InMemorySessionService()
    _, _, events = await _answered_chart(store, "t-shown", state={"app_key": 1})

    shown = _shown_keys(events)
    assert "app_key" in shown
    assert not shown & HIDDEN_KEYS


@pytest.mark.asyncio
async def test_echoed_state_does_not_restore_an_answered_call():
    """A client holding an earlier snapshot echoes ``pending_tool_calls`` back.

    Accepting a tool result removes its call from the pending list; the echo,
    written after that, must not put it back.
    """
    store = InMemorySessionService()
    agent = _chart_agent(store)
    asked = await collect(agent, run_input("t-echo", "run-1", [QUESTION], tools=CHART_TOOLS))
    call_id, _ = tool_call(asked, "render_chart")
    session = await store.get_session(app_name=APP, user_id=USER_ID, session_id="t-echo")
    echoed = dict(session.state)
    assert call_id in echoed["pending_tool_calls"]

    answer = ToolMessage(id="t-1", role="tool", tool_call_id=call_id, content="{}")
    await collect(
        agent,
        run_input("t-echo", "run-2", [QUESTION, answer], tools=CHART_TOOLS, state=echoed),
    )

    session = await store.get_session(app_name=APP, user_id=USER_ID, session_id="t-echo")
    assert call_id not in (session.state.get("pending_tool_calls") or [])


@pytest.mark.asyncio
async def test_agents_state_shows_the_client_only_its_state():
    store = InMemorySessionService()
    agent, _, _ = await _answered_chart(store, "t-endpoint", state={"app_key": 1})

    async with AsyncClient(
        transport=ASGITransport(app=create_adk_app(agent)),
        base_url="http://test",
    ) as client:
        response = await client.post("/agents/state", json={"threadId": "t-endpoint"})

    assert response.status_code == 200
    state = response.json()["state"]
    assert state["app_key"] == 1
    assert not set(state) & HIDDEN_KEYS
