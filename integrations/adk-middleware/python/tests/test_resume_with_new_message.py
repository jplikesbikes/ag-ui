"""A run that answers a tool confirmation AND carries a new user message.

ADK re-runs a confirmed tool only when the confirmation's FunctionResponse is
the last user event in the session (``_RequestConfirmationLlmRequestProcessor``
reverse-scans for it). Sending the response and the new message in one
invocation makes the message the last user event, so the tool never sees its
confirmation. The middleware therefore dispatches the confirmation on its own
first, then the new message — both inside the one AG-UI run.
"""

import pytest

from ag_ui.core import EventType, ResumeEntry, ToolMessage
from ag_ui_adk.session_manager import SessionManager

from tests.hitl_helpers import (
    RC_TOOL_NAME,
    ConfirmationTool,
    build_confirmation_agent,
    collect,
    content_text,
    run_finished,
    run_input,
    tool_call,
)
from tests.test_interrupt_enforcement import FOLLOW_UP, USER, _history


@pytest.fixture(autouse=True)
def reset_session_manager():
    SessionManager.reset_instance()
    yield
    SessionManager.reset_instance()


async def _pause(thread: str, **agent_kwargs):
    tool = ConfirmationTool()
    agent, llm = build_confirmation_agent(tool, **agent_kwargs)
    turn1 = await collect(agent, run_input(thread, "run-1", [USER]))
    rc_id, rc_args = tool_call(turn1, RC_TOOL_NAME)
    assert rc_id
    return agent, llm, tool, [(rc_id, rc_args)]


def _texts(llm) -> list[str]:
    return [content_text(content) for content in llm.last_contents]


def _assert_one_run(events):
    assert not [e for e in events if e.type == EventType.RUN_ERROR], [
        (e.type, getattr(e, "message", None)) for e in events
    ]
    assert [e.type for e in events].count(EventType.RUN_STARTED) == 1
    run_finished(events)


class TestResumeWithNewMessage:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("confirmed", [True, False])
    async def test_resume_entry_reruns_tool_then_answers_message(self, confirmed):
        agent, llm, tool, calls = await _pause(
            f"t-resume-msg-{confirmed}", emit_interrupt_outcome=True
        )

        events = await collect(
            agent,
            run_input(
                f"t-resume-msg-{confirmed}",
                "run-2",
                [*_history(calls), FOLLOW_UP],
                resume=[
                    ResumeEntry(
                        interrupt_id=calls[0][0],
                        status="resolved",
                        payload={"confirmed": confirmed},
                    )
                ],
            ),
        )

        _assert_one_run(events)
        # The tool re-ran with the decision...
        assert [c.confirmed for c in tool.confirmations] == [confirmed]
        assert tool.executed == (1 if confirmed else 0)
        # ...and the new message still reached the model, after it.
        assert _texts(llm)[-1] == FOLLOW_UP.content

    @pytest.mark.asyncio
    async def test_tool_message_reruns_tool_then_answers_message(self):
        # The same with a plain ToolMessage answer (flag off, no resume[]).
        agent, llm, tool, calls = await _pause("t-toolmsg-msg")

        events = await collect(
            agent,
            run_input(
                "t-toolmsg-msg",
                "run-2",
                [
                    *_history(calls),
                    ToolMessage(
                        id="tm-1",
                        role="tool",
                        tool_call_id=calls[0][0],
                        content='{"confirmed": false}',
                    ),
                    FOLLOW_UP,
                ],
            ),
        )

        _assert_one_run(events)
        assert [c.confirmed for c in tool.confirmations] == [False]
        assert _texts(llm)[-1] == FOLLOW_UP.content

    @pytest.mark.asyncio
    async def test_resume_without_message_is_unchanged(self):
        agent, llm, tool, calls = await _pause("t-resume-only", emit_interrupt_outcome=True)

        events = await collect(
            agent,
            run_input(
                "t-resume-only",
                "run-2",
                _history(calls),
                resume=[
                    ResumeEntry(
                        interrupt_id=calls[0][0],
                        status="resolved",
                        payload={"confirmed": True},
                    )
                ],
            ),
        )

        _assert_one_run(events)
        assert tool.executed == 1
        assert FOLLOW_UP.content not in _texts(llm)
