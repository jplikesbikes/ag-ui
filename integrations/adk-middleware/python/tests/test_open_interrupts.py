"""A session's open interrupts, read from the session alone.

An ``adk_request_confirmation`` call is open until a FunctionResponse answers
it; open ``confirm_changes`` reviews are kept in session state. The read, the
gate that refuses a run leaving an interrupt unanswered, and the resume path
all use that record, so they agree on every instance, including one that has
just started: its first access to a session clears ``pending_tool_calls``
when it has no execution to fulfil them, and a paused question has none.
"""

import pytest

from ag_ui.core import EventType, Interrupt, UserMessage
from ag_ui_adk.event_translator import open_interrupts
from ag_ui_adk.session_manager import PENDING_CONFIRM_CHANGES_STATE_KEY

from tests.hitl_helpers import collect, run_input, tool_call
from tests.test_confirmation_dispatch import (
    APP,
    ASK,
    RC,
    USER_ID,
    build,
    calls_message,
    errors,
    reset_session_manager,  # noqa: F401  (autouse fixture)
    resume,
    session_of,
)

TYPED = UserMessage(id="u-2", role="user", content="Something else")


async def _paused(thread, **kwargs):
    agent, llm, tool, service = build([["dangerous_action"]], emit_interrupt_outcome=True, **kwargs)
    events = await collect(agent, run_input(thread, "run-1", [ASK]))
    finished = [e for e in events if e.type == EventType.RUN_FINISHED][-1]
    call_id, call_args = tool_call(events, RC)
    history = [ASK, calls_message("a-1", [(call_id, RC, call_args)])]
    return agent, service, finished.outcome.interrupts, history


def _dumped(interrupts):
    return [i.model_dump(mode="json", exclude_none=True) for i in interrupts]


@pytest.mark.asyncio
async def test_the_read_is_what_run_finished_reported():
    agent, service, reported, _ = await _paused("t-read")

    session = await session_of(agent, service, "t-read")

    assert _dumped(open_interrupts(session)) == _dumped(reported)


@pytest.mark.asyncio
async def test_an_answered_confirmation_is_no_longer_open():
    agent, service, reported, history = await _paused("t-answered")

    events = await collect(agent, run_input(
        "t-answered", "run-2", history, resume=[resume(reported[0].id)]
    ))

    assert not errors(events)
    assert open_interrupts(await session_of(agent, service, "t-answered")) == []


@pytest.mark.asyncio
async def test_a_fresh_instance_agrees_after_its_first_access_clears_pending_calls():
    paused_on, service, reported, history = await _paused("t-fresh")
    fresh, _, tool, _ = build([], emit_interrupt_outcome=True, session_service=service)
    # Any run that reaches the session does this first on a new instance.
    await fresh._ensure_session_exists(APP, USER_ID, "t-fresh", {})
    session = await session_of(fresh, service, "t-fresh")
    assert not session.state.get("pending_tool_calls")

    # The read still reports the question...
    assert _dumped(open_interrupts(session)) == _dumped(reported)

    # ...the gate still refuses a run that leaves it unanswered...
    refused = await collect(fresh, run_input("t-fresh", "run-2", [*history, TYPED]))
    assert [code for code, _ in errors(refused)] == ["INTERRUPT_RESUME_REQUIRED"]

    # ...and answering it still hands the tool its decision.
    answered = await collect(fresh, run_input(
        "t-fresh", "run-3", history, resume=[resume(reported[0].id)]
    ))
    assert not errors(answered)
    assert tool.decisions == [True]
    assert open_interrupts(await session_of(fresh, service, "t-fresh")) == []


@pytest.mark.asyncio
async def test_a_question_paused_on_one_instance_is_answered_on_another():
    paused_on, service, reported, history = await _paused("t-other")
    other, _, tool, _ = build([], emit_interrupt_outcome=True, session_service=service)

    answered = await collect(other, run_input(
        "t-other", "run-2", history, resume=[resume(reported[0].id, confirmed=False)]
    ))

    assert not errors(answered)
    assert tool.decisions == [False]
    assert open_interrupts(await session_of(other, service, "t-other")) == []


class _Session:
    def __init__(self, state):
        self.events = []
        self.state = state


def test_a_confirm_changes_review_is_read_as_it_was_reported():
    review = Interrupt(
        id="cc-1",
        reason="confirm_changes",
        tool_call_id="cc-1",
        metadata={"predict_state": [{"state_key": "doc", "tool": "write"}]},
    )
    session = _Session({PENDING_CONFIRM_CHANGES_STATE_KEY: [review.model_dump(mode="json")]})

    assert _dumped(open_interrupts(session)) == _dumped([review])


def test_a_review_recorded_by_id_alone_is_still_open():
    session = _Session({PENDING_CONFIRM_CHANGES_STATE_KEY: ["cc-legacy"]})

    assert _dumped(open_interrupts(session)) == _dumped(
        [Interrupt(id="cc-legacy", reason="confirm_changes", tool_call_id="cc-legacy")]
    )
