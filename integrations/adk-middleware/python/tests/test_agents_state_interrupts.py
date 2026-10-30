"""/agents/state hands a restoring client the thread's open interrupts.

A client that reloads an interrupted thread has its messages and state, but
not the interrupts the closing RUN_FINISHED reported, so it cannot answer them
with ``resume``. With ``emit_interrupt_outcome`` on, the response carries them,
read from the session as the interrupt gate reads them.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ag_ui.core import EventType
from ag_ui_adk.endpoint import add_adk_fastapi_endpoint

from tests.hitl_helpers import collect, run_input, tool_call
from tests.test_confirmation_dispatch import (
    ASK,
    RC,
    build,
    calls_message,
    errors,
    reset_session_manager,  # noqa: F401  (autouse fixture)
    resume,
)


def _state(agent, thread):
    app = FastAPI()
    add_adk_fastapi_endpoint(app, agent)
    with TestClient(app) as client:
        response = client.post("/agents/state", json={"threadId": thread})
    assert response.status_code == 200
    return response.json()


async def _paused(thread, **kwargs):
    agent, _, _, _ = build([["dangerous_action"]], **kwargs)
    events = await collect(agent, run_input(thread, "run-1", [ASK]))
    return agent, events


@pytest.mark.asyncio
async def test_an_interrupted_thread_comes_back_with_its_interrupts():
    agent, events = await _paused("t-open", emit_interrupt_outcome=True)
    reported = [e for e in events if e.type == EventType.RUN_FINISHED][-1].outcome.interrupts

    body = _state(agent, "t-open")

    assert body["interrupts"] == [
        i.model_dump(mode="json", by_alias=True, exclude_none=True) for i in reported
    ]
    assert body["interrupts"][0]["toolCallId"] == reported[0].id


@pytest.mark.asyncio
async def test_an_answered_thread_has_none():
    agent, events = await _paused("t-done", emit_interrupt_outcome=True)
    call_id, call_args = tool_call(events, RC)
    answered = await collect(agent, run_input(
        "t-done", "run-2", [ASK, calls_message("a-1", [(call_id, RC, call_args)])],
        resume=[resume(call_id)],
    ))
    assert not errors(answered)

    assert _state(agent, "t-done")["interrupts"] == []


@pytest.mark.asyncio
async def test_an_agent_that_does_not_report_interrupts_adds_nothing():
    agent, _ = await _paused("t-off")

    assert "interrupts" not in _state(agent, "t-off")
