"""``resume_failed_invocations``: a retry with nothing new resumes a failed turn.

A client that retries a failed run re-sends the history it already sent, so
there is no unseen message and the run used to end in an empty
``RUN_STARTED`` / ``RUN_FINISHED``: the retry did nothing. With
``resume_failed_invocations=True``, if the thread's latest invocation failed
(a tool or the model raised, the model returned an error response, the
process died mid-turn), ADK continues that invocation with
``run_async(invocation_id=..., new_message=None)``. Nothing else is resumed:
resuming a completed invocation would repeat its turn.
"""

import asyncio
from typing import Any, List

import google.adk
import pytest

from ag_ui.core import EventType, RunAgentInput, UserMessage
from ag_ui_adk import ADKAgent, invocation_outcome
from ag_ui_adk.session_manager import SessionManager
from google.adk.agents import LlmAgent
from google.adk.apps import App, ResumabilityConfig
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.genai import types

ADK_MAJOR = int(google.adk.__version__.split(".")[0])
GATE = {"open": asyncio.Event()}


class Scripted(BaseLlm):
    """Each model call takes the next step: a response, an exception to raise,
    or ``WAIT`` (block until the gate opens, as a stand-in for a dying process)."""

    model: str = "scripted"
    steps: List[Any] = []
    calls: int = 0

    async def generate_content_async(self, llm_request, stream=False):
        self.calls += 1
        step = self.steps.pop(0)
        if step == "WAIT":
            await GATE["open"].wait()
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


TOOL_RUNS: List[str] = []


def flaky_once(q: str) -> dict:
    """Fails on its first run."""
    TOOL_RUNS.append("flaky_once")
    if TOOL_RUNS.count("flaky_once") == 1:
        raise RuntimeError("tool failed")
    return {"answer": q}


def lookup(q: str) -> dict:
    """Look something up."""
    TOOL_RUNS.append("lookup")
    return {"answer": q}


async def slow(q: str) -> dict:
    """Blocks until the gate opens."""
    TOOL_RUNS.append("slow")
    await GATE["open"].wait()
    return {"answer": q}


def ask_user(q: str) -> None:
    """Answered by the user."""
    return None


@pytest.fixture(autouse=True)
def fresh_state():
    SessionManager.reset_instance()
    TOOL_RUNS.clear()
    GATE["open"] = asyncio.Event()
    yield
    SessionManager.reset_instance()


def build(steps, helper_steps=(), *, resume=True, resumable=True):
    model = Scripted(steps=list(steps))
    helper = LlmAgent(
        name="helper", model=Scripted(steps=list(helper_steps)), description="A helper."
    )
    root = LlmAgent(
        name="root", model=model, sub_agents=[helper],
        tools=[flaky_once, lookup, slow, LongRunningFunctionTool(ask_user)],
    )
    if not resumable:
        return ADKAgent(
            adk_agent=root, app_name="retry", user_id="u",
            session_service=InMemorySessionService(),
            resume_failed_invocations=resume,
        ), model
    return ADKAgent.from_app(
        App(
            name="retry", root_agent=root,
            resumability_config=ResumabilityConfig(is_resumable=True),
        ),
        user_id="u",
        session_service=InMemorySessionService(),
        resume_failed_invocations=resume,
    ), model


ASK = UserMessage(id="u-1", role="user", content="go")


def run_input(run_id: str) -> RunAgentInput:
    return RunAgentInput(
        thread_id="t-1", run_id=run_id, state={}, tools=[], context=[],
        forwarded_props={}, messages=[ASK],
    )


async def run(agent: ADKAgent, run_id: str) -> list:
    return [event async for event in agent.run(run_input(run_id))]


async def die_mid_run(agent: ADKAgent, run_id: str, blocked) -> None:
    """Start a run and stop it the way a dying process would: once ``blocked()``
    says it is stuck, its execution is cancelled, and nothing records an
    outcome."""
    consumer = asyncio.create_task(run(agent, run_id))
    for _ in range(200):
        await asyncio.sleep(0.01)
        if agent._active_executions and blocked():
            break
    else:
        raise AssertionError("the run never blocked")
    for execution in agent._active_executions.values():
        execution.task.cancel()
    consumer.cancel()
    await asyncio.gather(consumer, return_exceptions=True)
    agent._active_executions.clear()


async def session_of(agent: ADKAgent):
    service = agent._session_manager._session_service
    (listed,) = (await service.list_sessions(app_name="retry", user_id="u")).sessions
    return await service.get_session(app_name="retry", user_id="u", session_id=listed.id)


def kinds(events: list) -> list:
    return [event.type for event in events]


def text(events: list) -> str:
    return "".join(e.delta for e in events if e.type == EventType.TEXT_MESSAGE_CONTENT)


EMPTY = [EventType.RUN_STARTED, EventType.RUN_FINISHED]


async def assert_completed_once(agent: ADKAgent) -> None:
    session = await session_of(agent)
    assert invocation_outcome(session).state == "completed"
    user_texts = [
        e for e in session.events
        if e.author == "user" and e.content and any(p.text for p in e.content.parts or [])
    ]
    assert len(user_texts) == 1, "the retry must not add a user event"


@pytest.mark.asyncio
async def test_a_raised_tool_resumes_to_the_model():
    agent, _ = build([calls("flaky_once", q="x"), say("recovered")])
    assert kinds(await run(agent, "r-1"))[-1] == EventType.RUN_ERROR

    retried = await run(agent, "r-2")

    assert retried[-1].type == EventType.RUN_FINISHED
    assert text(retried) == "recovered"
    await assert_completed_once(agent)
    # ADK 2.x records the raise, so the call is closed as aborted and the tool
    # is not re-run; ADK 1.x records nothing, so the open call is executed again.
    assert TOOL_RUNS.count("flaky_once") in (1, 2)


@pytest.mark.asyncio
async def test_a_raised_model_resumes_to_the_model():
    agent, model = build([calls("lookup", q="x"), RuntimeError("model 503"), say("recovered")])
    assert kinds(await run(agent, "r-1"))[-1] == EventType.RUN_ERROR

    retried = await run(agent, "r-2")

    assert text(retried) == "recovered"
    assert TOOL_RUNS == ["lookup"]
    await assert_completed_once(agent)


@pytest.mark.asyncio
async def test_a_model_error_response_asks_the_model_again():
    """ADK closes the agent after an error response. ADK 2.x resumes a closed
    agent by calling the model again; ADK 1.x treats it as done, so there the
    retry runs nothing and the invocation stays failed."""
    agent, model = build([LlmResponse(error_code="SAFETY", error_message="blocked"), say("answer")])
    await run(agent, "r-1")
    assert invocation_outcome(await session_of(agent)).state == "failed"

    retried = await run(agent, "r-2")

    assert retried[-1].type == EventType.RUN_FINISHED
    if ADK_MAJOR >= 2:
        assert text(retried) == "answer"
        assert invocation_outcome(await session_of(agent)).state == "completed"
    else:
        assert model.calls == 1
        assert invocation_outcome(await session_of(agent)).state == "failed"


@pytest.mark.asyncio
async def test_a_tool_left_running_by_a_dead_process_runs_again():
    agent, _ = build([calls("slow", q="x"), say("answer after the tool")])
    await die_mid_run(agent, "r-1", lambda: TOOL_RUNS == ["slow"])
    assert invocation_outcome(await session_of(agent)).code == "INTERRUPTED_IN_TOOL"
    GATE["open"].set()

    retried = await run(agent, "r-2")

    assert EventType.TOOL_CALL_RESULT in kinds(retried)
    assert text(retried) == "answer after the tool"
    assert TOOL_RUNS == ["slow", "slow"]  # at least once
    await assert_completed_once(agent)


@pytest.mark.asyncio
async def test_a_hand_off_whose_target_died_resumes_in_the_target():
    agent, _ = build(
        [calls("transfer_to_agent", agent_name="helper")], ["WAIT", say("helper answers")]
    )
    helper_model = agent._adk_agent.sub_agents[0].model
    await die_mid_run(agent, "r-1", lambda: helper_model.calls == 1)
    assert invocation_outcome(await session_of(agent)).code == "INTERRUPTED"
    GATE["open"].set()

    retried = await run(agent, "r-2")

    assert text(retried) == "helper answers"
    await assert_completed_once(agent)


@pytest.mark.asyncio
async def test_a_second_retry_after_recovery_is_empty():
    agent, model = build([calls("lookup", q="x"), RuntimeError("model 503"), say("recovered")])
    await run(agent, "r-1")
    await run(agent, "r-2")
    model_calls = model.calls

    again = await run(agent, "r-3")

    assert kinds(again) == EMPTY
    assert model.calls == model_calls


@pytest.mark.asyncio
async def test_a_completed_run_is_never_resumed():
    agent, model = build([say("answer")])
    await run(agent, "r-1")

    again = await run(agent, "r-2")

    assert kinds(again) == EMPTY
    assert model.calls == 1


@pytest.mark.asyncio
async def test_a_paused_run_is_not_resumed():
    agent, model = build([calls("ask_user", q="x"), say("never")])
    await run(agent, "r-1")
    assert invocation_outcome(await session_of(agent)).state == "input-required"

    await run(agent, "r-2")

    assert model.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options", [{"resume": False}, {"resumable": False}], ids=["off", "not resumable"]
)
async def test_without_the_option_or_resumability_a_retry_stays_empty(options):
    agent, model = build([calls("lookup", q="x"), RuntimeError("model 503"), say("never")], **options)
    await run(agent, "r-1")

    again = await run(agent, "r-2")

    assert kinds(again) == EMPTY
    assert model.calls == 2
