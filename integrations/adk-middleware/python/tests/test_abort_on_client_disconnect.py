"""``abort_on_client_disconnect``: a run whose client stops reading aborts its
ADK invocation.

By default a run's ADK invocation keeps running in its background task after
the client stops reading the stream (it aborted the run, or the stream ended
on ``EXECUTION_TIMEOUT``). With ``abort_on_client_disconnect=True`` the agent
sets the ``abort_signal`` it passed to ``Runner.run_async`` (google-adk 2.x):
ADK stops the running tool, calls nothing further and seals the open call
with ``error_code`` ``INVOCATION_ABORTED``. On google-adk 1.x, which has no
``abort_signal``, the option has no effect.
"""

import asyncio
import logging
from typing import Any, List

import pytest

from ag_ui.core import EventType, RunAgentInput, UserMessage
from ag_ui_adk import ADKAgent
from ag_ui_adk.adk_agent import _ADK_SUPPORTS_ABORT_SIGNAL
from ag_ui_adk.session_manager import SessionManager
from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.function_tool import FunctionTool
from google.genai import types

needs_abort_signal = pytest.mark.skipif(
    not _ADK_SUPPORTS_ABORT_SIGNAL, reason="Runner.run_async(abort_signal=...) is google-adk 2.x"
)
without_abort_signal = pytest.mark.skipif(
    _ADK_SUPPORTS_ABORT_SIGNAL, reason="google-adk supports abort_signal"
)


class Scripted(BaseLlm):
    """Each model call takes the next response."""

    model: str = "scripted"
    steps: List[Any] = []

    async def generate_content_async(self, llm_request, stream=False):
        yield self.steps.pop(0)


def say(text: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


def calls(name: str, **args: Any) -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))],
        )
    )


@pytest.fixture(autouse=True)
def fresh_session_manager():
    SessionManager.reset_instance()
    yield
    SessionManager.reset_instance()


def broken(q: str) -> dict:
    """Always fails."""
    raise RuntimeError("tool failed")


class SlowToClose(BaseToolset):
    """A toolset that takes a moment to close, as one holding a connection does: the
    runner closes it after the invocation ends, so the run's task outlives its
    terminal event."""

    async def get_tools(self, readonly_context=None):
        return [FunctionTool(broken)]

    async def close(self) -> None:
        await asyncio.sleep(0.2)


class Harness:
    def __init__(self, steps, **options):
        self.gate = asyncio.Event()
        self.tool = {"started": 0, "returned": 0}

        async def slow(q: str) -> dict:
            """Blocks until the gate opens."""
            self.tool["started"] += 1
            await self.gate.wait()
            self.tool["returned"] += 1
            return {"answer": q}

        self.model = Scripted(steps=list(steps))
        self.agent = ADKAgent(
            adk_agent=LlmAgent(
                name="root", model=self.model, instruction="r",
                tools=[slow, SlowToClose()],
            ),
            app_name="abort",
            user_id="u",
            session_service=InMemorySessionService(),
            **options,
        )

    def run(self, run_id: str = "r-1") -> "asyncio.Task[list]":
        run_input = RunAgentInput(
            thread_id="t-1", run_id=run_id, state={}, tools=[], context=[],
            forwarded_props={},
            messages=[UserMessage(id="u-1", role="user", content="go")],
        )

        async def consume():
            return [event async for event in self.agent.run(run_input)]

        return asyncio.create_task(consume())

    async def tool_started(self) -> None:
        for _ in range(300):
            if self.tool["started"]:
                return
            await asyncio.sleep(0.01)
        raise AssertionError("the tool never started")

    def background_tasks(self) -> List[asyncio.Task]:
        return [execution.task for execution in self.agent._active_executions.values()]

    async def session_events(self) -> list:
        service = self.agent._session_manager._session_service
        (listed,) = (await service.list_sessions(app_name="abort", user_id="u")).sessions
        session = await service.get_session(
            app_name="abort", user_id="u", session_id=listed.id
        )
        return session.events


async def cancel(consumer: asyncio.Task) -> None:
    """The client stops reading the run's stream."""
    consumer.cancel()
    await asyncio.gather(consumer, return_exceptions=True)


def abort_seals(events) -> list:
    return [
        part.function_response
        for event in events
        if event.error_code == "INVOCATION_ABORTED"
        for part in (event.content.parts if event.content and event.content.parts else [])
        if part.function_response
    ]


@needs_abort_signal
@pytest.mark.asyncio
async def test_a_client_that_stops_reading_aborts_the_invocation():
    harness = Harness([calls("slow", q="x"), say("never said")], abort_on_client_disconnect=True)
    consumer = harness.run()
    await harness.tool_started()
    [task] = harness.background_tasks()

    await cancel(consumer)
    await asyncio.wait_for(asyncio.wait({task}), timeout=2)

    assert harness.tool == {"started": 1, "returned": 0}
    assert harness.model.steps == [say("never said")]
    events = await harness.session_events()
    [seal] = abort_seals(events)
    assert seal.name == "slow"
    assert events[-1].error_code == "INVOCATION_ABORTED"


@needs_abort_signal
@pytest.mark.asyncio
async def test_a_stream_that_ends_on_execution_timeout_aborts_the_invocation():
    harness = Harness([calls("slow", q="x"), say("never said")], abort_on_client_disconnect=True)
    harness.agent._execution_timeout = 0.3
    consumer = harness.run()
    await harness.tool_started()
    [task] = harness.background_tasks()

    events = await consumer
    await asyncio.wait_for(asyncio.wait({task}), timeout=2)

    assert events[-1].type == EventType.RUN_ERROR
    assert events[-1].code == "EXECUTION_TIMEOUT"
    assert harness.tool == {"started": 1, "returned": 0}
    assert [seal.name for seal in abort_seals(await harness.session_events())] == ["slow"]


@needs_abort_signal
@pytest.mark.asyncio
async def test_a_run_that_ends_normally_is_not_aborted():
    harness = Harness([calls("slow", q="x"), say("done")], abort_on_client_disconnect=True)
    harness.gate.set()

    events = await harness.run()

    assert events[-1].type == EventType.RUN_FINISHED
    session_events = await harness.session_events()
    assert abort_seals(session_events) == []
    assert not any(event.error_code for event in session_events)
    assert harness.model.steps == []


@needs_abort_signal
@pytest.mark.asyncio
async def test_a_run_that_ends_in_run_error_is_not_aborted(caplog):
    """The run's own RUN_ERROR reaches the client while ADK is still unwinding;
    the client read the run to its end, so nothing is aborted."""
    harness = Harness([calls("broken", q="x")], abort_on_client_disconnect=True)

    with caplog.at_level(logging.INFO, logger="ag_ui_adk"):
        events = await harness.run()

    assert events[-1].type == EventType.RUN_ERROR
    assert not [r for r in caplog.records if "Aborting the ADK invocation" in r.getMessage()]
    assert abort_seals(await harness.session_events()) == []


@pytest.mark.asyncio
async def test_by_default_the_invocation_runs_on_after_the_client_leaves():
    harness = Harness([calls("slow", q="x"), say("said anyway")])
    consumer = harness.run()
    await harness.tool_started()
    [task] = harness.background_tasks()

    await cancel(consumer)
    await asyncio.sleep(0.1)
    assert not task.done()

    harness.gate.set()
    await asyncio.wait_for(asyncio.wait({task}), timeout=2)

    assert harness.tool == {"started": 1, "returned": 1}
    assert harness.model.steps == []
    events = await harness.session_events()
    assert abort_seals(events) == []
    assert events[-1].content.parts[0].text == "said anyway"


@without_abort_signal
@pytest.mark.asyncio
async def test_without_abort_signal_the_option_warns_and_changes_nothing(caplog):
    with caplog.at_level(logging.WARNING, logger="ag_ui_adk.adk_agent"):
        harness = Harness([calls("slow", q="x"), say("said anyway")], abort_on_client_disconnect=True)
    assert "abort_on_client_disconnect=True needs" in caplog.text

    consumer = harness.run()
    await harness.tool_started()
    [task] = harness.background_tasks()
    await cancel(consumer)
    await asyncio.sleep(0.1)
    assert not task.done()

    harness.gate.set()
    await asyncio.wait_for(asyncio.wait({task}), timeout=2)
    assert harness.tool == {"started": 1, "returned": 1}
    assert harness.model.steps == []
