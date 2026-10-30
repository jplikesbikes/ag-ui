"""A new message after a failed run first closes the run's unanswered calls.

A tool that raised, or a process that died mid-tool, leaves the function call
without a response. The next message then reached the model right after a call
with no answer: model APIs reject that history (ADK 1.x sends it as is), and
ADK 2.x drops the orphaned call, so the model never learns it failed. Before a
new message runs, the failed invocation's unanswered calls are answered in
ADK's own abort-seal shape. A call that waits for the user is never touched.
"""

from typing import Any, List

import pytest

from ag_ui.core import EventType, RunAgentInput, UserMessage
from ag_ui_adk import ADKAgent
from ag_ui_adk.event_translator import (
    INVOCATION_ABORTED,
    INVOCATION_ABORTED_MESSAGE,
    unanswered_call_seals,
)
from ag_ui_adk.session_manager import SessionManager
from google.adk.agents import LlmAgent
from google.adk.apps import App, ResumabilityConfig
from google.adk.events import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.genai import types


class Scripted(BaseLlm):
    """Each model call returns the next scripted response and records the
    request's history as (role, kind, tool name or text) entries. (ADK strips
    the ids it generated for calls from the request, so calls are named.)"""

    model: str = "scripted"
    steps: List[Any] = []
    requests: List[Any] = []

    async def generate_content_async(self, llm_request, stream=False):
        history = []
        for content in llm_request.contents or []:
            for part in content.parts or []:
                if part.function_call:
                    history.append((content.role, "call", part.function_call.name))
                elif part.function_response:
                    history.append((content.role, "response", part.function_response.name))
                elif part.text:
                    history.append((content.role, "text", part.text))
        self.requests.append(history)
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


def flaky(q: str) -> dict:
    """Always fails."""
    raise RuntimeError("tool failed")


def lookup(q: str) -> dict:
    """Look something up."""
    return {"answer": q}


def ask_user(q: str) -> None:
    """Answered later by the user."""
    return None


@pytest.fixture(autouse=True)
def fresh_session_manager():
    SessionManager.reset_instance()
    yield
    SessionManager.reset_instance()


def build(steps: List[Any], helper_steps: List[Any] = ()):
    model = Scripted(steps=steps)
    helper = LlmAgent(
        name="helper", model=Scripted(steps=list(helper_steps)), tools=[flaky],
        description="A helper.",
    )
    root = LlmAgent(
        name="root", model=model,
        tools=[flaky, lookup, LongRunningFunctionTool(ask_user)], sub_agents=[helper],
    )
    agent = ADKAgent.from_app(
        App(
            name="seal", root_agent=root,
            resumability_config=ResumabilityConfig(is_resumable=True),
        ),
        user_id="u",
        session_service=InMemorySessionService(),
    )
    return agent, model, helper.model


async def run(agent: ADKAgent, run_id: str, messages: list) -> list:
    return [
        event
        async for event in agent.run(
            RunAgentInput(
                thread_id="t-1", run_id=run_id, state={}, tools=[], context=[],
                forwarded_props={}, messages=messages,
            )
        )
    ]


async def session_of(agent: ADKAgent):
    service = agent._session_manager._session_service
    (listed,) = (await service.list_sessions(app_name="seal", user_id="u")).sessions
    return await service.get_session(app_name="seal", user_id="u", session_id=listed.id)


def seals_in(session) -> List[Event]:
    return [event for event in session.events if event.error_code == INVOCATION_ABORTED]


FIRST = UserMessage(id="u-1", role="user", content="first")
SECOND = UserMessage(id="u-2", role="user", content="second")


@pytest.mark.asyncio
async def test_a_raised_tool_call_is_answered_before_the_next_message():
    agent, model, _ = build([calls("flaky", q="x"), say("second answer")])
    first = await run(agent, "r-1", [FIRST])
    assert first[-1].type == EventType.RUN_ERROR

    second = await run(agent, "r-2", [FIRST, SECOND])

    assert second[-1].type == EventType.RUN_FINISHED
    # The model sees the call answered, then the new message.
    assert model.requests[-1] == [
        ("user", "text", "first"),
        ("model", "call", "flaky"),
        ("user", "response", "flaky"),
        ("user", "text", "second"),
    ]


@pytest.mark.asyncio
async def test_the_seal_has_adks_abort_shape_and_the_original_invocation():
    agent, _, _ = build([calls("flaky", q="x"), say("second answer")])
    await run(agent, "r-1", [FIRST])
    before = await session_of(agent)
    (call_event,) = [e for e in before.events if e.get_function_calls()]

    await run(agent, "r-2", [FIRST, SECOND])

    (seal,) = seals_in(await session_of(agent))
    assert seal.invocation_id == call_event.invocation_id
    assert seal.author == "root"
    assert seal.error_message == INVOCATION_ABORTED_MESSAGE
    (response,) = seal.get_function_responses()
    (call,) = call_event.get_function_calls()
    assert (response.id, response.name) == (call.id, call.name)
    assert response.response == {"error": INVOCATION_ABORTED_MESSAGE}


@pytest.mark.asyncio
async def test_a_call_in_a_sub_agent_is_sealed_by_that_agent():
    agent, _, helper_model = build(
        [calls("transfer_to_agent", agent_name="helper")],
        [calls("flaky", q="x"), say("second answer")],
    )
    await run(agent, "r-1", [FIRST])

    await run(agent, "r-2", [FIRST, SECOND])

    (seal,) = seals_in(await session_of(agent))
    assert seal.author == "helper"
    history = helper_model.requests[-1]
    assert history.index(("model", "call", "flaky")) + 1 == history.index(
        ("user", "response", "flaky")
    )


@pytest.mark.asyncio
async def test_a_call_whose_process_died_is_sealed():
    agent, model, _ = build([say("first answer"), say("second answer")])
    await run(agent, "r-1", [FIRST])
    session = await session_of(agent)
    # What a process that died mid-tool leaves behind.
    service = agent._session_manager._session_service
    await service.append_event(session, Event(
        invocation_id="e-killed", author="user",
        content=types.Content(role="user", parts=[types.Part(text="look it up")]),
    ))
    await service.append_event(session, Event(
        invocation_id="e-killed", author="root",
        content=types.Content(role="model", parts=[types.Part(
            function_call=types.FunctionCall(id="call-killed", name="lookup", args={"q": "x"})
        )]),
    ))
    agent._session_manager.invalidate_session(session.id, "seal", "u")

    await run(agent, "r-2", [FIRST, SECOND])

    (seal,) = seals_in(await session_of(agent))
    assert seal.invocation_id == "e-killed"
    history = model.requests[-1]
    assert history.index(("model", "call", "lookup")) + 1 == history.index(
        ("user", "response", "lookup")
    )


@pytest.mark.asyncio
async def test_a_completed_run_is_not_sealed():
    agent, _, _ = build([calls("lookup", q="x"), say("first answer"), say("second answer")])
    await run(agent, "r-1", [FIRST])

    await run(agent, "r-2", [FIRST, SECOND])

    assert seals_in(await session_of(agent)) == []


@pytest.mark.asyncio
async def test_a_call_waiting_for_the_user_is_left_open():
    agent, _, _ = build([calls("ask_user", q="x"), say("second answer")])
    await run(agent, "r-1", [FIRST])

    await run(agent, "r-2", [FIRST, SECOND])

    session = await session_of(agent)
    assert seals_in(session) == []
    (asked,) = [c for e in session.events for c in e.get_function_calls()]
    assert not [r for e in session.events for r in e.get_function_responses() if r.id == asked.id]


@pytest.mark.asyncio
async def test_an_empty_retry_seals_nothing():
    """Sealing is for a new message; a run with nothing new leaves the session alone."""
    agent, _, _ = build([calls("flaky", q="x")])
    await run(agent, "r-1", [FIRST])

    await run(agent, "r-2", [FIRST])

    assert seals_in(await session_of(agent)) == []


def test_the_builder_skips_long_running_and_answered_calls():
    def call_event(*ids, lro=()):
        return Event(
            invocation_id="e-1", author="root", long_running_tool_ids=set(lro) or None,
            content=types.Content(role="model", parts=[
                types.Part(function_call=types.FunctionCall(id=i, name="tool", args={}))
                for i in ids
            ]),
        )

    answered = Event(
        invocation_id="e-1", author="root",
        content=types.Content(role="user", parts=[types.Part(
            function_response=types.FunctionResponse(id="done", name="tool", response={})
        )]),
    )
    events = [call_event("done", "open"), answered, call_event("waiting", lro=["waiting"])]

    (seal,) = unanswered_call_seals(events, "e-1")

    assert [r.id for r in seal.get_function_responses()] == ["open"]
    assert unanswered_call_seals(events, "e-other") == []
