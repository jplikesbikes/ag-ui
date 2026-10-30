"""How a run's tool results and new messages reach ADK.

ADK re-runs a confirmed tool only while the confirmation's FunctionResponse is
the last user event (#1839), and leaves any event holding a confirmation out of
the model's context. So a confirmation answer runs on its own, the turn's other
results go in an event of their own, and the run's new messages follow, unless
the answer paused the run again; then they wait for the run that answers it.

The model here rejects the one request shape Gemini rejects (a function call
turn not answered by the next turn), so an invalid request fails as it would in
production.
"""

from typing import Any, List, Optional

import pytest
from pydantic import Field

from ag_ui.core import (
    AssistantMessage,
    EventType,
    ResumeEntry,
    SystemMessage,
    Tool as AGUITool,
    ToolMessage,
    UserMessage,
)
from ag_ui_adk import ADKAgent, AGUIToolset
from ag_ui_adk.session_manager import SessionManager
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps import App, ResumabilityConfig
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.genai import types

from tests.hitl_helpers import collect, run_input, tool_call

RC = "adk_request_confirmation"
APP = "dispatch_app"
USER_ID = "test_user"
ASK = UserMessage(id="u-1", role="user", content="Run the dangerous action")
TYPED = UserMessage(id="u-2", role="user", content="Actually, never mind")
MORE = UserMessage(id="u-3", role="user", content="And one more thing")
PICK = AGUITool(name="pick_color", description="client tool", parameters={"type": "object", "properties": {}})


@pytest.fixture(autouse=True)
def reset_session_manager():
    SessionManager.reset_instance()
    yield
    SessionManager.reset_instance()


class StrictLlm(BaseLlm):
    """Turn ``i`` calls the tools in ``script[i]`` (a list of names), later turns reply text.

    Rejects a request whose function call turn is not answered by the turn after it.
    """

    script: List[List[str]] = Field(default_factory=list)
    turn_count: int = 0
    requests: List[List[Any]] = Field(default_factory=list)

    async def generate_content_async(self, llm_request, stream: bool = False):
        contents = list(llm_request.contents or [])
        for i, content in enumerate(contents[:-1]):
            calls = [p.function_call.name for p in content.parts or [] if p.function_call]
            if content.role != "model" or not calls:
                continue
            answer = contents[i + 1]
            answered = [p.function_response.name for p in answer.parts or [] if p.function_response]
            if answer.role != "user" or sorted(answered) != sorted(calls):
                raise ValueError(
                    "400 INVALID_ARGUMENT: function call turn not answered by the next turn: "
                    + " | ".join(
                        f"{c.role}:" + ",".join(
                            f"call:{p.function_call.name}" if p.function_call
                            else f"resp:{p.function_response.name}" if p.function_response
                            else f"text:{(p.text or '')[:20]}"
                            for p in c.parts or []
                        )
                        for c in contents
                    )
                )
        self.requests.append(contents)
        self.turn_count += 1
        names = self.script[self.turn_count - 1] if self.turn_count <= len(self.script) else []
        parts = [
            types.Part(function_call=types.FunctionCall(
                name=name, args={"target": "foo"} if name == "dangerous_action" else {}
            ))
            for name in names
        ] or [types.Part(text="Done.")]
        yield LlmResponse(content=types.Content(role="model", parts=parts), partial=False, turn_complete=True)

    def saw(self, text: str) -> bool:
        return any(p.text == text for request in self.requests for c in request for p in c.parts or [])


class GatedTool:
    def __init__(self) -> None:
        self.decisions: List[bool] = []

        def dangerous_action(target: str, tool_context) -> dict:
            """A backend tool gated by HITL confirmation."""
            if tool_context.tool_confirmation is None:
                tool_context.request_confirmation(hint=f"Run on {target}?")
                return {"status": "awaiting_confirmation"}
            self.decisions.append(tool_context.tool_confirmation.confirmed)
            return {"status": "executed" if tool_context.tool_confirmation.confirmed else "rejected"}

        self.fn = dangerous_action


def build(script, **kwargs):
    tool, llm = GatedTool(), StrictLlm(model="strict", script=script)
    service = kwargs.pop("session_service", None) or InMemorySessionService()
    agent = ADKAgent.from_app(
        App(
            name=APP,
            root_agent=LlmAgent(name="agent", model=llm, tools=[tool.fn, AGUIToolset()]),
            resumability_config=ResumabilityConfig(is_resumable=True),
        ),
        user_id=USER_ID,
        session_service=service,
        **kwargs,
    )
    return agent, llm, tool, service


def errors(events):
    return [(e.code, e.message) for e in events if e.type == EventType.RUN_ERROR]


def finished(events):
    runs = [e for e in events if e.type == EventType.RUN_FINISHED]
    assert len(runs) == 1 and events[-1] is runs[0], [e.type for e in events]
    outcome = getattr(runs[0], "outcome", None)
    return [i.id for i in outcome.interrupts] if outcome is not None and hasattr(outcome, "interrupts") else []


def calls_message(id_, calls):
    return AssistantMessage(id=id_, role="assistant", tool_calls=[
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}
        for call_id, name, args in calls
    ])


def resume(call_id, status="resolved", confirmed=True):
    return ResumeEntry(
        interrupt_id=call_id,
        status=status,
        **({"payload": {"confirmed": confirmed}} if status == "resolved" else {}),
    )


async def session_of(agent, service, thread):
    session_id = agent._get_backend_session_id(thread, USER_ID, app_name=APP)
    return await service.get_session(app_name=APP, user_id=USER_ID, session_id=session_id)


async def deliveries(agent, service, thread, text) -> int:
    """How many user events in the session carry this text."""
    session = await session_of(agent, service, thread)
    return sum(
        1 for e in session.events if e.author == "user" and e.content
        for p in e.content.parts or [] if p.text == text
    )


async def pause(thread, script, tools=(), **kwargs):
    agent, llm, tool, service = build(script, **kwargs)
    first = await collect(agent, run_input(thread, "run-1", [ASK], tools=list(tools)))
    assert not errors(first)
    starts = [(e.tool_call_id, e.tool_call_name) for e in first if e.type == EventType.TOOL_CALL_START]
    history = [ASK, calls_message("a-1", [(i, n, tool_call(first, n)[1]) for i, n in starts])]
    return agent, llm, tool, service, dict((n, i) for i, n in starts), history




@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["resolved", "cancelled"])
async def test_answer_and_message_rerun_the_tool_then_answer_the_message(status):
    agent, llm, tool, service, ids, history = await pause(
        "t-answer", [["dangerous_action"]], emit_interrupt_outcome=True
    )
    events = await collect(agent, run_input(
        "t-answer", "run-2", [*history, TYPED], resume=[resume(ids[RC], status)]
    ))
    assert not errors(events) and finished(events) == []
    assert tool.decisions == [status == "resolved"]
    assert llm.saw(TYPED.content)
    assert await deliveries(agent, service, "t-answer", TYPED.content) == 1


@pytest.mark.asyncio
async def test_a_tool_message_answer_and_message_rerun_the_tool_then_answer_the_message():
    """The same with a plain ToolMessage answer (no interrupt outcome, no resume)."""
    agent, llm, tool, service, ids, history = await pause("t-toolmsg", [["dangerous_action"]])
    events = await collect(agent, run_input("t-toolmsg", "run-2", [
        *history,
        ToolMessage(id="tm-rc", role="tool", tool_call_id=ids[RC], content='{"confirmed": false}'),
        TYPED,
    ]))
    assert not errors(events)
    assert tool.decisions == [False]
    assert llm.saw(TYPED.content)
    assert await deliveries(agent, service, "t-toolmsg", TYPED.content) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("store", ["in_memory", "database"])
async def test_every_new_user_message_reaches_the_model(store, tmp_path):
    service = (
        InMemorySessionService() if store == "in_memory"
        else DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{tmp_path}/sessions.db")
    )
    agent, llm, _, service = build([], session_service=service)
    await collect(agent, run_input("t-two", "run-1", [ASK]))
    events = await collect(agent, run_input("t-two", "run-2", [
        ASK, AssistantMessage(id="a-1", role="assistant", content="Done."), TYPED, MORE
    ]))
    assert not errors(events)
    assert llm.saw(TYPED.content) and llm.saw(MORE.content)


@pytest.mark.asyncio
async def test_a_new_pause_is_reported_and_the_message_follows_its_answer():
    agent, llm, tool, service, ids, history = await pause(
        "t-repause", [["dangerous_action"], ["dangerous_action"]], emit_interrupt_outcome=True
    )
    paused = await collect(agent, run_input(
        "t-repause", "run-2", [*history, TYPED], resume=[resume(ids[RC])]
    ))
    assert not errors(paused)
    [second] = finished(paused)
    second_args = tool_call(paused, RC)[1]

    answered = await collect(agent, run_input("t-repause", "run-3", [
        *history, TYPED, calls_message("a-2", [(second, RC, second_args)])
    ], resume=[resume(second)]))
    assert not errors(answered) and finished(answered) == []
    assert tool.decisions == [True, True]
    assert llm.saw(TYPED.content)
    assert await deliveries(agent, service, "t-repause", TYPED.content) == 1


@pytest.mark.asyncio
async def test_typing_over_a_new_pause_delivers_both_messages():
    """The app's flow: typing instead of answering sends a cancelled resume."""
    agent, llm, tool, service, ids, history = await pause(
        "t-typing", [["dangerous_action"], ["dangerous_action"]], emit_interrupt_outcome=True
    )
    paused = await collect(agent, run_input(
        "t-typing", "run-2", [*history, TYPED], resume=[resume(ids[RC])]
    ))
    [second] = finished(paused)
    events = await collect(agent, run_input("t-typing", "run-3", [
        *history, TYPED, calls_message("a-2", [(second, RC, tool_call(paused, RC)[1])]), MORE
    ], resume=[resume(second, "cancelled")]))
    assert not errors(events)
    assert llm.saw(TYPED.content) and llm.saw(MORE.content)


@pytest.mark.asyncio
async def test_a_message_waits_for_a_frontend_call_the_rerun_made():
    agent, llm, tool, service, ids, history = await pause(
        "t-rerun-ft", [["dangerous_action"], ["pick_color"]], tools=[PICK], emit_interrupt_outcome=True
    )
    events = await collect(agent, run_input(
        "t-rerun-ft", "run-2", [*history, TYPED], tools=[PICK], resume=[resume(ids[RC])]
    ))
    assert not errors(events)
    pick_id, pick_args = tool_call(events, "pick_color")
    assert pick_id

    events = await collect(agent, run_input("t-rerun-ft", "run-3", [
        *history, TYPED, calls_message("a-2", [(pick_id, "pick_color", pick_args)]),
        ToolMessage(id="tm-pick", role="tool", tool_call_id=pick_id, content='"red"'),
    ], tools=[PICK]))
    assert not errors(events)
    assert llm.saw(TYPED.content)
    assert await deliveries(agent, service, "t-rerun-ft", TYPED.content) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [True, False])
async def test_a_partial_answer_with_a_message_is_refused_then_retried(flag):
    agent, llm, tool, service, ids, history = await pause(
        f"t-partial-{flag}", [["dangerous_action", "pick_color"]], tools=[PICK], emit_interrupt_outcome=flag
    )
    rc_answer = ToolMessage(id="tm-rc", role="tool", tool_call_id=ids[RC], content='{"confirmed": true}')
    first = [*history, TYPED] if flag else [*history, rc_answer, TYPED]
    refused = await collect(agent, run_input(
        f"t-partial-{flag}", "run-2", first, tools=[PICK], **({"resume": [resume(ids[RC])]} if flag else {})
    ))
    assert [code for code, _ in errors(refused)] == ["PENDING_TOOL_CALLS"]
    assert not llm.saw(TYPED.content)

    pick = ToolMessage(id="tm-pick", role="tool", tool_call_id=ids["pick_color"], content='"red"')
    retried = await collect(agent, run_input(
        f"t-partial-{flag}", "run-3", [*first, pick], tools=[PICK],
        **({"resume": [resume(ids[RC])]} if flag else {}),
    ))
    assert not errors(retried)
    assert tool.decisions == [True]
    assert await deliveries(agent, service, f"t-partial-{flag}", TYPED.content) == 1


@pytest.mark.asyncio
async def test_a_trailing_system_message_adds_no_model_turn():
    agent, llm, tool, service, ids, history = await pause(
        "t-system", [["dangerous_action"]], emit_interrupt_outcome=True
    )
    before = llm.turn_count
    events = await collect(agent, run_input("t-system", "run-2", [
        *history, SystemMessage(id="s-1", role="system", content="context")
    ], resume=[resume(ids[RC])]))
    assert not errors(events)
    assert tool.decisions == [True]
    assert llm.turn_count - before == 1


@pytest.mark.asyncio
async def test_the_interrupt_boundary_snapshot_keeps_a_waiting_message():
    """Spec: a MessagesSnapshot precedes the interrupting RunFinished, and the
    client drops every message it leaves out."""
    agent, llm, tool, service, ids, history = await pause(
        "t-snap", [["dangerous_action"], ["dangerous_action"]],
        emit_interrupt_outcome=True, emit_messages_snapshot=True,
    )
    paused = await collect(agent, run_input(
        "t-snap", "run-2", [*history, TYPED], resume=[resume(ids[RC])]
    ))
    assert finished(paused)
    snapshots = [e for e in paused if e.type == EventType.MESSAGES_SNAPSHOT]
    assert snapshots
    last = snapshots[-1]
    assert paused.index(last) < len(paused) - 1
    texts = [m.content for m in last.messages if m.role == "user"]
    assert TYPED.content in texts




@pytest.mark.asyncio
async def test_a_frontend_result_and_a_message_share_one_model_turn():
    """Only results that must end their invocation run on their own."""
    agent, llm, tool, service, ids, history = await pause("t-shared", [["pick_color"]], tools=[PICK])
    before = llm.turn_count
    events = await collect(agent, run_input("t-shared", "run-2", [
        *history, ToolMessage(id="tm-pick", role="tool", tool_call_id=ids["pick_color"], content='"red"'), TYPED,
    ], tools=[PICK]))
    assert not errors(events)
    assert llm.turn_count - before == 1
    assert llm.saw(TYPED.content)


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [True, False])
async def test_a_confirmation_answered_with_another_result_keeps_both(flag):
    """ADK leaves any event holding a confirmation out of the model's context."""
    agent, llm, tool, service, ids, history = await pause(
        f"t-mixed-{flag}", [["dangerous_action", "pick_color"]], tools=[PICK], emit_interrupt_outcome=flag
    )
    pick = ToolMessage(id="tm-pick", role="tool", tool_call_id=ids["pick_color"], content='"red"')
    rc = ToolMessage(id="tm-rc", role="tool", tool_call_id=ids[RC], content='{"confirmed": true}')
    events = await collect(agent, run_input(
        f"t-mixed-{flag}", "run-2", [*history, pick] if flag else [*history, rc, pick], tools=[PICK],
        **({"resume": [resume(ids[RC])]} if flag else {}),
    ))
    assert not errors(events)
    assert tool.decisions == [True]
