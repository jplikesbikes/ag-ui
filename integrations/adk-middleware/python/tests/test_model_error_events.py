"""A run that ends on an ADK model error response ends in RUN_ERROR.

When a model returns an error response instead of content (a safety block,
``MALFORMED_FUNCTION_CALL``, a provider error surfaced as ``error_code``), ADK
persists the response with its ``error_code`` and closes the agent normally.
The translator has nothing to render for it, so the run used to end in an
empty ``RUN_FINISHED``: a silent "success" with no output. The run now ends in
``RUN_ERROR`` whose ``code`` is ADK's ``error_code`` and whose ``message`` is
fixed text (ADK's ``error_message`` can carry provider detail).
"""

from typing import Any, List

import pytest

from ag_ui.core import EventType, RunAgentInput, UserMessage
from ag_ui_adk import ADKAgent
from ag_ui_adk.session_manager import SessionManager
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.apps import App, ResumabilityConfig
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService
from google.genai import types


class Scripted(BaseLlm):
    """Each model call returns the next scripted response."""

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


def error(code: str) -> LlmResponse:
    return LlmResponse(error_code=code, error_message=f"provider detail for {code}")


def lookup(q: str) -> dict:
    """Look something up."""
    return {"answer": q}


def llm_root(steps: List[Any], helper_steps: List[Any] = ()) -> LlmAgent:
    helper = LlmAgent(
        name="helper", model=Scripted(steps=list(helper_steps)), description="A helper."
    )
    return LlmAgent(
        name="root", model=Scripted(steps=steps), tools=[lookup], sub_agents=[helper]
    )


@pytest.fixture(autouse=True)
def fresh_session_manager():
    SessionManager.reset_instance()
    yield
    SessionManager.reset_instance()


def agent_for(root, *, resumable: bool) -> ADKAgent:
    if resumable:
        return ADKAgent.from_app(
            App(
                name="errors", root_agent=root,
                resumability_config=ResumabilityConfig(is_resumable=True),
            ),
            user_id="u",
            session_service=InMemorySessionService(),
        )
    return ADKAgent(
        adk_agent=root, app_name="errors", user_id="u",
        session_service=InMemorySessionService(),
    )


async def run(agent: ADKAgent) -> list:
    return [
        event
        async for event in agent.run(
            RunAgentInput(
                thread_id="t-1", run_id="r-1", state={}, tools=[], context=[],
                forwarded_props={},
                messages=[UserMessage(id="u-1", role="user", content="go")],
            )
        )
    ]


def terminal(events: list):
    terminals = [
        event for event in events
        if event.type in (EventType.RUN_FINISHED, EventType.RUN_ERROR)
    ]
    assert len(terminals) == 1, [event.type for event in events]
    assert events[-1] is terminals[0]
    return terminals[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("resumable", [True, False], ids=["resumable", "not resumable"])
@pytest.mark.parametrize(
    "root, code",
    [
        (lambda: llm_root([error("SAFETY")]), "SAFETY"),
        (lambda: llm_root([error("MALFORMED_FUNCTION_CALL")]), "MALFORMED_FUNCTION_CALL"),
        (lambda: llm_root([calls("lookup", q="x"), error("SAFETY")]), "SAFETY"),
        (lambda: llm_root(
            [calls("transfer_to_agent", agent_name="helper")], [error("RECITATION")]
        ), "RECITATION"),
    ],
    ids=["safety", "malformed call", "after a tool", "in a sub-agent"],
)
async def test_an_error_response_ends_the_run_in_run_error(root, code, resumable):
    events = await run(agent_for(root(), resumable=resumable))

    end = terminal(events)
    assert end.type == EventType.RUN_ERROR
    assert end.code == code
    assert end.message == "The model returned an error instead of a response."
    assert "provider detail" not in end.message


@pytest.mark.asyncio
async def test_what_the_run_did_before_the_error_still_streams():
    events = await run(agent_for(llm_root([calls("lookup", q="x"), error("SAFETY")]), resumable=True))

    types_ = [event.type for event in events]
    assert types_[0] == EventType.RUN_STARTED
    assert EventType.TOOL_CALL_RESULT in types_
    assert types_.index(EventType.TOOL_CALL_RESULT) < types_.index(EventType.RUN_ERROR)


@pytest.mark.asyncio
async def test_a_run_that_continues_after_an_error_finishes():
    root = SequentialAgent(name="seq", sub_agents=[
        LlmAgent(name="a", model=Scripted(steps=[error("SAFETY")])),
        LlmAgent(name="b", model=Scripted(steps=[say("recovered")])),
    ])

    events = await run(agent_for(root, resumable=True))

    assert terminal(events).type == EventType.RUN_FINISHED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "steps",
    [[say("answer")], [LlmResponse(content=None)]],
    ids=["reply", "empty reply"],
)
async def test_a_run_without_an_error_response_still_finishes(steps):
    """An empty reply carries no error_code: ADK ends it as a normal turn."""
    events = await run(agent_for(llm_root(steps), resumable=True))

    assert terminal(events).type == EventType.RUN_FINISHED


@pytest.mark.asyncio
async def test_the_error_response_is_still_in_the_session():
    agent = agent_for(llm_root([error("SAFETY")]), resumable=True)
    await run(agent)

    service = agent._session_manager._session_service
    (listed,) = (await service.list_sessions(app_name="errors", user_id="u")).sessions
    session = await service.get_session(app_name="errors", user_id="u", session_id=listed.id)
    assert [event.error_code for event in session.events if event.error_code] == ["SAFETY"]


@pytest.mark.asyncio
async def test_a_finish_reason_on_a_response_with_content_is_not_an_error():
    """ADK copies a non-STOP finish_reason onto ``error_code`` even when the
    response has content (a reply cut at MAX_TOKENS). The content is the answer."""
    truncated = say("a long answer, cut")
    truncated.error_code = "MAX_TOKENS"

    events = await run(agent_for(llm_root([truncated]), resumable=True))

    assert terminal(events).type == EventType.RUN_FINISHED


@pytest.mark.asyncio
async def test_a_finish_reason_enum_becomes_its_name():
    events = await run(agent_for(
        llm_root([LlmResponse(error_code=types.FinishReason.SAFETY)]), resumable=True
    ))

    assert terminal(events).code == "SAFETY"
