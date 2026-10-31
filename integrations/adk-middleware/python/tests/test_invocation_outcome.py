"""How a session's latest invocation ended, read from the session alone.

ADK persists what is needed to tell a finished turn from a failed one: an
``end_of_agent`` marker when an agent completes (resumable apps), the model's
error responses (``error_code``), on ADK 2.x the exception a tool or model
raised, and an ``INVOCATION_ABORTED`` seal. A turn whose process died leaves
none of them, and that absence is the record. ``invocation_outcome`` reads
these into A2A task states, and ``/agents/state`` returns them as
``runStatus`` so a client restoring a thread can show a failed turn as failed.

The session shapes below are the ones ADK 1.x and 2.x write for each case;
the second half drives the real runner to confirm the installed ADK writes
them.
"""

from typing import Any, List, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ag_ui.core import RunAgentInput, UserMessage
from ag_ui_adk import ADKAgent, add_adk_fastapi_endpoint, invocation_outcome
from ag_ui_adk.event_translator import (
    EMPTY_REPLY,
    INTERRUPTED,
    INTERRUPTED_IN_TOOL,
    INVOCATION_ABORTED,
    InvocationOutcome,
    latest_invocation_id,
)
from ag_ui_adk.session_manager import PENDING_CONFIRM_CHANGES_STATE_KEY, SessionManager
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.apps import App, ResumabilityConfig
from google.adk.events import Event, EventActions
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.genai import types


# ---------------------------------------------------------------------------
# Session shapes
# ---------------------------------------------------------------------------


def ev(
    author: str,
    *,
    inv: str = "e-1",
    text: Optional[str] = None,
    call: Optional[str] = None,
    response: Optional[str] = None,
    error: Optional[str] = None,
    end: bool = False,
    agent_state: Optional[dict] = None,
    transfer: Optional[str] = None,
    lro: bool = False,
    partial: bool = False,
    empty_parts: bool = False,
) -> Event:
    """One persisted event. ``call`` / ``response`` name the call id; the tool
    is named after it."""
    parts = []
    if text is not None:
        parts.append(types.Part(text=text))
    if call is not None:
        parts.append(types.Part(function_call=types.FunctionCall(id=call, name=call, args={})))
    if response is not None:
        parts.append(
            types.Part(
                function_response=types.FunctionResponse(id=response, name=response, response={})
            )
        )
    content = None
    if parts or empty_parts:
        content = types.Content(role="user" if author == "user" else "model", parts=parts)
    return Event(
        invocation_id=inv,
        author=author,
        content=content,
        error_code=error,
        error_message=f"detail of {error}" if error else None,
        partial=partial or None,
        long_running_tool_ids={call} if lro and call else None,
        actions=EventActions(
            end_of_agent=end or None,
            agent_state=agent_state,
            transfer_to_agent=transfer,
        ),
    )


def session(*events: Event, state: Optional[dict] = None) -> Session:
    return Session(id="s", app_name="app", user_id="u", state=state or {}, events=list(events))


USER = ev("user", text="hi")


def outcome(*events: Event, **kwargs: Any) -> Optional[InvocationOutcome]:
    return invocation_outcome(session(*events), **kwargs)


def state_and_code(result: Optional[InvocationOutcome]):
    return (result.state, result.code) if result else None


SHAPES = {
    # -- completed ----------------------------------------------------------
    "direct reply": (
        [USER, ev("root", text="answer"), ev("root", end=True)],
        ("completed", None),
    ),
    "tool then reply": (
        [USER, ev("root", call="lookup"), ev("root", response="lookup"),
         ev("root", text="answer"), ev("root", end=True)],
        ("completed", None),
    ),
    # On a hand-off the transferring agent closes before the target runs.
    "transfer, target replied": (
        [USER, ev("root", call="transfer_to_agent"),
         ev("root", response="transfer_to_agent", transfer="helper"),
         ev("root", end=True), ev("helper", text="answer"), ev("helper", end=True)],
        ("completed", None),
    ),
    "sequential agent, every step done": (
        [USER, ev("seq", agent_state={"current_sub_agent": "a"}), ev("a", text="A"),
         ev("a", end=True), ev("seq", agent_state={"current_sub_agent": "b"}),
         ev("b", text="B"), ev("b", end=True), ev("seq", end=True)],
        ("completed", None),
    ),
    "an error the invocation recovered from": (
        [USER, ev("root", error="RuntimeError"), ev("root", text="answer"), ev("root", end=True)],
        ("completed", None),
    ),
    "resumed after an abort seal": (
        [USER, ev("root", call="flaky"), ev("root", error="RuntimeError"),
         ev("root", response="flaky", error=INVOCATION_ABORTED),
         ev("root", text="answer"), ev("root", end=True)],
        ("completed", None),
    ),
    # ADK copies a non-STOP finish_reason onto a response that has content.
    "reply cut at MAX_TOKENS": (
        [USER, ev("root", text="a long answer, cut", error="MAX_TOKENS"), ev("root", end=True)],
        ("completed", None),
    ),
    # ADK ends an empty reply normally, so A2A would report it completed; the
    # code keeps the reason visible.
    "empty reply, no event": (
        [USER, ev("root", end=True)],
        ("completed", EMPTY_REPLY),
    ),
    "empty reply, an event without parts": (
        [USER, ev("root", empty_parts=True), ev("root", end=True)],
        ("completed", EMPTY_REPLY),
    ),
    # -- input-required -------------------------------------------------------
    "paused on a tool confirmation": (
        [USER, ev("root", call="guarded"), ev("root", response="guarded"),
         ev("root", call="adk_request_confirmation", lro=True)],
        ("input-required", None),
    ),
    "paused on a client tool": (
        [USER, ev("root", call="client_tool", lro=True)],
        ("input-required", None),
    ),
    # -- canceled -------------------------------------------------------------
    "aborted and sealed by ADK": (
        [USER, ev("root", call="slow"), ev("root", response="slow", error=INVOCATION_ABORTED)],
        ("canceled", INVOCATION_ABORTED),
    ),
    # -- failed ---------------------------------------------------------------
    "model error response": (
        [USER, ev("root", error="SAFETY"), ev("root", end=True)],
        ("failed", "SAFETY"),
    ),
    "malformed function call": (
        [USER, ev("root", error="MALFORMED_FUNCTION_CALL"), ev("root", end=True)],
        ("failed", "MALFORMED_FUNCTION_CALL"),
    ),
    "model error response after a tool": (
        [USER, ev("root", call="lookup"), ev("root", response="lookup"),
         ev("root", error="SAFETY"), ev("root", end=True)],
        ("failed", "SAFETY"),
    ),
    "model error response in a sub-agent": (
        [USER, ev("root", call="transfer_to_agent"),
         ev("root", response="transfer_to_agent", transfer="helper"),
         ev("root", end=True), ev("helper", error="SAFETY"), ev("helper", end=True)],
        ("failed", "SAFETY"),
    ),
    # ADK 2.x records an exception as an error event.
    "tool raised (ADK 2.x)": (
        [USER, ev("root", call="flaky"), ev("root", error="RuntimeError")],
        ("failed", "RuntimeError"),
    ),
    "tool raised, then resumed without closing the call (ADK 2.x)": (
        [USER, ev("root", call="flaky"), ev("root", error="RuntimeError"), ev("root", end=True)],
        ("failed", "RuntimeError"),
    ),
    "model raised after a tool (ADK 2.x)": (
        [USER, ev("root", call="lookup"), ev("root", response="lookup"),
         ev("root", error="RuntimeError")],
        ("failed", "RuntimeError"),
    ),
    "tool raised in a sub-agent (ADK 2.x)": (
        [USER, ev("root", call="transfer_to_agent"),
         ev("root", response="transfer_to_agent", transfer="helper"),
         ev("root", end=True), ev("helper", call="flaky"), ev("helper", error="RuntimeError")],
        ("failed", "RuntimeError"),
    ),
    # A killed process (and on ADK 1.x, a raised tool) leaves the call open.
    "killed mid-tool": (
        [USER, ev("root", call="slow")],
        ("failed", INTERRUPTED_IN_TOOL),
    ),
    "killed before the model answered": (
        [USER],
        ("failed", INTERRUPTED),
    ),
    "model raised after a tool (ADK 1.x)": (
        [USER, ev("root", call="lookup"), ev("root", response="lookup")],
        ("failed", INTERRUPTED),
    ),
    "killed in the sub-agent after a hand-off": (
        [USER, ev("root", call="transfer_to_agent"),
         ev("root", response="transfer_to_agent", transfer="helper"), ev("root", end=True)],
        ("failed", INTERRUPTED),
    ),
    "killed between two steps of a sequential agent": (
        [USER, ev("seq", agent_state={"current_sub_agent": "a"}), ev("a", text="A"),
         ev("a", end=True), ev("seq", agent_state={"current_sub_agent": "b"})],
        ("failed", INTERRUPTED),
    ),
    "killed mid-stream": (
        [USER, ev("root", text="partial ans", partial=True)],
        ("failed", INTERRUPTED),
    ),
}


@pytest.mark.parametrize("name", list(SHAPES))
def test_session_shape(name):
    events, expected = SHAPES[name]

    assert state_and_code(outcome(*events)) == expected


def test_the_error_message_is_never_carried():
    result = outcome(USER, ev("root", error="SAFETY"), ev("root", end=True))

    assert result.to_run_status() == {"state": "failed", "code": "SAFETY"}


def test_a_completed_run_has_no_code():
    result = outcome(USER, ev("root", text="answer"), ev("root", end=True))

    assert result.to_run_status() == {"state": "completed"}


def test_a_finish_reason_enum_becomes_its_name():
    blocked = ev("root")
    blocked.error_code = types.FinishReason.SAFETY

    assert state_and_code(outcome(USER, blocked, ev("root", end=True))) == ("failed", "SAFETY")


def test_live_is_working():
    assert state_and_code(outcome(USER, ev("root", call="slow"), live=True)) == ("working", None)


def test_a_session_without_a_turn_has_no_outcome():
    assert outcome() is None


def test_the_outcome_names_its_invocation():
    result = outcome(USER, ev("root", text="answer"), ev("root", end=True))

    assert result.invocation_id == "e-1"


def test_only_the_latest_invocation_counts():
    failed_then_done = outcome(
        USER, ev("root", call="slow"),
        ev("user", inv="e-2", text="again"), ev("root", inv="e-2", text="ok"),
        ev("root", inv="e-2", end=True),
    )
    done_then_failed = outcome(
        USER, ev("root", text="ok"), ev("root", end=True),
        ev("user", inv="e-2", text="again"), ev("root", inv="e-2", error="SAFETY"),
        ev("root", inv="e-2", end=True),
    )

    assert state_and_code(failed_then_done) == ("completed", None)
    assert state_and_code(done_then_failed) == ("failed", "SAFETY")


@pytest.mark.parametrize(
    "write",
    [
        # This package's state updates (between runs, or recording a pending call).
        ev("user", inv="state_update_1793397803"),
        # An application's own state write, appended between runs.
        Event(
            invocation_id="app-write-1", author="system",
            actions=EventActions(state_delta={"title": "Renamed"}),
        ),
    ],
    ids=["ag-ui-adk state update", "application state write"],
)
def test_state_writes_after_a_run_are_not_its_turn(write):
    failed = session(USER, ev("root", call="slow"), write)
    completed = session(USER, ev("root", text="ok"), ev("root", end=True), write)

    assert state_and_code(invocation_outcome(failed)) == ("failed", INTERRUPTED_IN_TOOL)
    assert state_and_code(invocation_outcome(completed)) == ("completed", None)
    assert latest_invocation_id(failed) == "e-1"


def test_seeded_history_is_not_a_turn():
    seeded = session(
        ev("user", inv="agui-history-1", text="old"),
        ev("assistant", inv="agui-history-1", text="old answer"),
    )

    assert invocation_outcome(seeded) is None


def test_a_pending_predictive_state_review_is_input_required():
    reviewed = session(
        USER, ev("root", text="proposed"), ev("root", end=True),
        state={PENDING_CONFIRM_CHANGES_STATE_KEY: ["review-1"]},
    )

    assert state_and_code(invocation_outcome(reviewed)) == ("input-required", None)


def test_an_answered_client_tool_is_no_longer_input_required():
    answered = outcome(
        USER, ev("root", call="client_tool", lro=True),
        ev("user", response="client_tool"), ev("root", text="thanks"), ev("root", end=True),
    )

    assert state_and_code(answered) == ("completed", None)


@pytest.mark.parametrize(
    "events, expected",
    [
        ([USER, ev("root", text="answer")], ("completed", None)),
        ([USER, ev("root", call="lookup"), ev("root", response="lookup"),
          ev("root", text="answer")], ("completed", None)),
        ([USER, ev("root", call="lookup"), ev("root", response="lookup")], ("failed", INTERRUPTED)),
        ([USER, ev("root", error="SAFETY")], ("failed", "SAFETY")),
        # Without end_of_agent markers an empty reply can't be told from a kill.
        ([USER], ("failed", INTERRUPTED)),
    ],
    ids=["reply", "tool then reply", "killed after a tool", "model error", "nothing said"],
)
def test_without_resumability_a_final_response_completes(events, expected):
    assert state_and_code(outcome(*events, resumable=False)) == expected


# ---------------------------------------------------------------------------
# The installed ADK writes these shapes
# ---------------------------------------------------------------------------


class Scripted(BaseLlm):
    """Each model call pops the next step: a response, or an exception to raise."""

    model: str = "scripted"
    steps: List[Any] = []

    async def generate_content_async(self, llm_request, stream=False):
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        yield step


def say(text: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


def calls(name: str, **args: Any) -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))],
        )
    )


def lookup(q: str) -> dict:
    """Look something up."""
    return {"answer": q}


def flaky(q: str) -> dict:
    """Always fails."""
    raise RuntimeError("tool failed")


def client_tool(q: str) -> None:
    """Answered by the client."""
    return None


def resumable_app(root) -> App:
    return App(
        name="outcome", root_agent=root,
        resumability_config=ResumabilityConfig(is_resumable=True),
    )


def llm_root(steps: List[Any], helper_steps: Optional[List[Any]] = None) -> LlmAgent:
    helper = LlmAgent(
        name="helper", model=Scripted(steps=helper_steps or []), instruction="h",
        description="A helper.",
    )
    return LlmAgent(
        name="root", model=Scripted(steps=steps), instruction="r",
        tools=[lookup, flaky, LongRunningFunctionTool(client_tool)], sub_agents=[helper],
    )


async def run_once(root, *, resumable: bool = True) -> Session:
    service = InMemorySessionService()
    if resumable:
        runner = Runner(app=resumable_app(root), session_service=service)
    else:
        runner = Runner(app_name="outcome", agent=root, session_service=service)
    created = await service.create_session(app_name="outcome", user_id="u")
    try:
        async for _ in runner.run_async(
            user_id="u", session_id=created.id,
            new_message=types.Content(role="user", parts=[types.Part(text="go")]),
        ):
            pass
    except RuntimeError:
        pass
    return await service.get_session(app_name="outcome", user_id="u", session_id=created.id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "root, expected",
    [
        (lambda: llm_root([say("answer")]), ("completed", None)),
        (lambda: llm_root([calls("lookup", q="x"), say("answer")]), ("completed", None)),
        (lambda: llm_root([calls("transfer_to_agent", agent_name="helper")], [say("answer")]),
         ("completed", None)),
        (lambda: SequentialAgent(name="seq", sub_agents=[
            LlmAgent(name="a", model=Scripted(steps=[say("A")])),
            LlmAgent(name="b", model=Scripted(steps=[say("B")])),
        ]), ("completed", None)),
        (lambda: llm_root([LlmResponse(content=None)]), ("completed", EMPTY_REPLY)),
        (lambda: llm_root([calls("client_tool", q="x")]), ("input-required", None)),
        (lambda: llm_root([LlmResponse(error_code="SAFETY", error_message="blocked")]),
         ("failed", "SAFETY")),
        (lambda: llm_root(
            [calls("transfer_to_agent", agent_name="helper")],
            [LlmResponse(error_code="MALFORMED_FUNCTION_CALL", error_message="bad")],
        ), ("failed", "MALFORMED_FUNCTION_CALL")),
    ],
    ids=["reply", "tool then reply", "transfer", "sequential", "empty reply",
         "client tool pause", "safety", "malformed call in a sub-agent"],
)
async def test_runner_session(root, expected):
    assert state_and_code(invocation_outcome(await run_once(root()))) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "steps",
    [[calls("flaky", q="x")], [calls("lookup", q="x"), RuntimeError("model failed")]],
    ids=["tool raised", "model raised after a tool"],
)
async def test_runner_session_after_an_exception_is_failed(steps):
    """ADK 2.x records the exception class as the code; ADK 1.x records nothing,
    which reads as an interrupted invocation."""
    result = invocation_outcome(await run_once(llm_root(steps)))

    assert result.state == "failed"
    assert result.code in {"RuntimeError", INTERRUPTED_IN_TOOL, INTERRUPTED}


@pytest.mark.asyncio
async def test_runner_session_without_resumability():
    done = await run_once(llm_root([calls("lookup", q="x"), say("answer")]), resumable=False)

    assert state_and_code(invocation_outcome(done, resumable=False)) == ("completed", None)


# ---------------------------------------------------------------------------
# /agents/state
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=False)
def fresh_session_manager():
    SessionManager.reset_instance()
    yield
    SessionManager.reset_instance()


def agent_for(root) -> ADKAgent:
    return ADKAgent.from_app(
        resumable_app(root), user_id="u", session_service=InMemorySessionService()
    )


def state_of(agent: ADKAgent, thread: str) -> dict:
    app = FastAPI()
    add_adk_fastapi_endpoint(app, agent)
    with TestClient(app) as client:
        response = client.post("/agents/state", json={"threadId": thread})
    assert response.status_code == 200
    return response.json()


async def run_thread(agent: ADKAgent, thread: str) -> list:
    return [
        event
        async for event in agent.run(
            RunAgentInput(
                thread_id=thread, run_id="r-1", state={}, tools=[], context=[],
                forwarded_props={},
                messages=[UserMessage(id="u-1", role="user", content="go")],
            )
        )
    ]


@pytest.mark.asyncio
async def test_agents_state_reports_a_completed_run(fresh_session_manager):
    agent = agent_for(llm_root([say("answer")]))
    await run_thread(agent, "t-done")

    assert state_of(agent, "t-done")["runStatus"] == {"state": "completed"}


@pytest.mark.asyncio
async def test_agents_state_reports_a_failed_run_by_code_only(fresh_session_manager):
    agent = agent_for(llm_root([LlmResponse(error_code="SAFETY", error_message="secret detail")]))
    await run_thread(agent, "t-failed")

    body = state_of(agent, "t-failed")

    assert body["runStatus"] == {"state": "failed", "code": "SAFETY"}
    assert "secret detail" not in str(body)


@pytest.mark.asyncio
async def test_agents_state_reports_a_live_run_as_working(fresh_session_manager):
    agent = agent_for(llm_root([say("answer")]))
    await run_thread(agent, "t-live")

    class Running:
        is_complete = False
        task = None

    agent._active_executions[("t-live", "u", "outcome")] = Running()

    assert state_of(agent, "t-live")["runStatus"] == {"state": "working"}


def test_agents_state_has_no_run_status_for_an_unknown_thread(fresh_session_manager):
    agent = agent_for(llm_root([]))

    body = state_of(agent, "t-none")

    assert body["threadExists"] is False
    assert "runStatus" not in body
