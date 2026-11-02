"""An execution lasts as long as its background task, not its event stream.

A run's ADK invocation runs in a background task; the client reads its events
from a queue. When the client stops reading (it aborts the run, or the stream
ends on ``EXECUTION_TIMEOUT``), the task keeps running the invocation. The
execution therefore stays in ``_active_executions`` until the task finishes,
so the next run on the thread waits for it instead of starting a second
invocation on the same session. Admission (the check for a running execution
and the registration of the new one) is one step, so runs that arrive
together start one at a time.
"""

import asyncio
from typing import Any, List

import pytest

from ag_ui.core import EventType, RunAgentInput, UserMessage
from ag_ui_adk import ADKAgent
from ag_ui_adk.session_manager import SessionManager
from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService
from google.genai import types

KEY = ("t-1", "u", "lifetime")


class Scripted(BaseLlm):
    """Each model call takes the next response; records how many overlap."""

    model: str = "scripted"
    steps: List[Any] = []
    active: int = 0
    max_active: int = 0

    async def generate_content_async(self, llm_request, stream=False):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            yield self.steps.pop(0)
        finally:
            self.active -= 1


class YieldingSessionService(InMemorySessionService):
    """An in-memory service whose calls yield to the event loop, as a database
    service's do, so concurrent runs interleave at every session access."""

    async def create_session(self, **kwargs):
        await asyncio.sleep(0)
        return await super().create_session(**kwargs)

    async def get_session(self, **kwargs):
        await asyncio.sleep(0)
        return await super().get_session(**kwargs)

    async def list_sessions(self, **kwargs):
        await asyncio.sleep(0)
        return await super().list_sessions(**kwargs)

    async def append_event(self, session, event):
        await asyncio.sleep(0)
        return await super().append_event(session, event)


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


def build(steps):
    gate, tool_runs = asyncio.Event(), []

    async def slow(q: str) -> dict:
        """Blocks until the gate opens."""
        tool_runs.append(q)
        await gate.wait()
        return {"answer": q}

    model = Scripted(steps=list(steps))
    agent = ADKAgent(
        adk_agent=LlmAgent(name="root", model=model, instruction="r", tools=[slow]),
        app_name="lifetime",
        user_id="u",
        session_service=YieldingSessionService(),
    )
    return agent, model, gate, tool_runs


def message(n: int) -> UserMessage:
    return UserMessage(id=f"u-{n}", role="user", content=f"message {n}")


def run_input(run_id: str, *messages: UserMessage) -> RunAgentInput:
    return RunAgentInput(
        thread_id="t-1", run_id=run_id, state={}, tools=[], context=[],
        forwarded_props={}, messages=list(messages),
    )


async def consume(agent: ADKAgent, run_input: RunAgentInput) -> list:
    return [event async for event in agent.run(run_input)]


async def wait_until(condition) -> None:
    for _ in range(300):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never held")


async def history(agent: ADKAgent) -> List[str]:
    """The session's texts, in order."""
    service = agent._session_manager._session_service
    (listed,) = (await service.list_sessions(app_name="lifetime", user_id="u")).sessions
    session = await service.get_session(
        app_name="lifetime", user_id="u", session_id=listed.id
    )
    return [
        part.text
        for event in session.events
        for part in (event.content.parts if event.content and event.content.parts else [])
        if part.text
    ]


@pytest.mark.asyncio
async def test_a_run_whose_client_went_away_stays_active_until_its_task_ends():
    agent, _, gate, tool_runs = build([calls("slow", q="x"), say("first answer")])
    consumer = asyncio.create_task(consume(agent, run_input("r-1", message(1))))
    await wait_until(lambda: tool_runs == ["x"])
    execution = agent._active_executions[KEY]

    consumer.cancel()  # the client stops reading the stream
    await asyncio.gather(consumer, return_exceptions=True)

    assert agent._active_executions.get(KEY) is execution
    assert not execution.is_complete

    gate.set()
    await asyncio.wait({execution.task})

    assert execution.is_complete
    assert KEY not in agent._active_executions
    assert await history(agent) == ["message 1", "first answer"]


@pytest.mark.asyncio
async def test_a_new_message_waits_for_a_run_whose_client_went_away():
    """Without the wait, the new message's invocation runs next to the one
    still in its tool, on the same session."""
    agent, model, gate, tool_runs = build(
        [calls("slow", q="x"), say("first answer"), say("second answer")]
    )
    consumer = asyncio.create_task(consume(agent, run_input("r-1", message(1))))
    await wait_until(lambda: tool_runs == ["x"])
    consumer.cancel()
    await asyncio.gather(consumer, return_exceptions=True)

    follow_up = asyncio.create_task(
        consume(agent, run_input("r-2", message(1), message(2)))
    )
    await asyncio.sleep(0.1)
    assert not follow_up.done(), "the new run must wait for the running invocation"
    assert model.steps == [say("first answer"), say("second answer")]

    gate.set()
    events = await follow_up

    assert events[-1].type == EventType.RUN_FINISHED
    assert tool_runs == ["x"]
    assert model.max_active == 1
    assert await history(agent) == [
        "message 1", "first answer", "message 2", "second answer"
    ]


@pytest.mark.asyncio
async def test_runs_waiting_on_the_same_thread_start_one_at_a_time():
    """Two runs that arrive while the thread is busy both wait. When it frees,
    the first to be admitted registers before the other checks again, so the
    other waits for it rather than starting next to it."""
    agent, model, gate, tool_runs = build(
        [calls("slow", q="x"), say("first answer"), say("answer A"), say("answer B")]
    )
    first = asyncio.create_task(consume(agent, run_input("r-1", message(1))))
    await wait_until(lambda: tool_runs == ["x"])

    waiting = [
        asyncio.create_task(consume(agent, run_input("r-2", message(1), message(2)))),
        asyncio.create_task(consume(agent, run_input("r-3", message(1), message(3)))),
    ]
    await asyncio.sleep(0.1)
    assert not any(task.done() for task in waiting)

    gate.set()
    await first
    results = await asyncio.gather(*waiting)

    assert [events[-1].type for events in results] == [EventType.RUN_FINISHED] * 2
    assert model.max_active == 1
    texts = await history(agent)
    assert texts[:2] == ["message 1", "first answer"]
    # Each message is followed by its own answer: the turns did not interleave.
    assert sorted([texts[2], texts[4]]) == ["message 2", "message 3"]
    assert texts[3] == "answer A" and texts[5] == "answer B"


@pytest.mark.asyncio
async def test_close_cancels_a_task_whose_client_went_away():
    agent, _, _, tool_runs = build([calls("slow", q="x"), say("never")])
    consumer = asyncio.create_task(consume(agent, run_input("r-1", message(1))))
    await wait_until(lambda: tool_runs == ["x"])
    execution = agent._active_executions[KEY]
    consumer.cancel()
    await asyncio.gather(consumer, return_exceptions=True)

    await agent.close()

    assert execution.task.cancelled()
    assert agent._active_executions == {}


@pytest.mark.asyncio
async def test_a_new_message_cancels_a_run_past_the_execution_timeout():
    """The client's stream ends on EXECUTION_TIMEOUT while the tool is still
    blocked. A new message does not wait on that execution forever: past the
    timeout it is cancelled, as the stale-execution cleanup does."""
    agent, _, _, tool_runs = build([calls("slow", q="x"), say("never"), say("second answer")])
    agent._execution_timeout = 0.3
    timed_out = await consume(agent, run_input("r-1", message(1)))
    assert timed_out[-1].type == EventType.RUN_ERROR
    assert timed_out[-1].code == "EXECUTION_TIMEOUT"
    execution = agent._active_executions[KEY]
    assert not execution.is_complete

    agent._execution_timeout = 60
    execution.start_time -= 60
    events = await consume(agent, run_input("r-2", message(1), message(2)))

    assert execution.task.cancelled()
    assert events[-1].type == EventType.RUN_FINISHED
    assert tool_runs == ["x"]
