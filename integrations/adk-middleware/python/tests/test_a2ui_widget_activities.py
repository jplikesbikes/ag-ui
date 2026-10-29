"""A2UI widgets rendered with ADK's ``tool_context.render_ui_widget``.

ADK puts each widget on the tool's response event (``actions.render_ui_widgets``)
and the session keeps it. A widget with provider ``"a2ui"`` carries its surface's
A2UI messages under ``payload["a2ui_operations"]`` and becomes an ``a2ui-surface``
activity — the shape the A2UI middleware emits — live as an ACTIVITY_SNAPSHOT
and in restored history as an ActivityMessage, under the same id.
"""

import pytest

from ag_ui.core import (
    ActivityMessage,
    AssistantMessage,
    EventType,
    FunctionCall,
    ResumeEntry,
    ToolCall,
    UserMessage,
)
from ag_ui_adk import ADKAgent, EventTranslator, adk_events_to_messages
from ag_ui_adk.event_translator import a2ui_widget_activities, messages_snapshot_event
from ag_ui_adk.session_manager import SessionManager
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps import App, ResumabilityConfig
from google.adk.events import Event as ADKEvent
from google.adk.events.event_actions import EventActions
from google.adk.events.ui_widget import UiWidget
from google.adk.sessions import InMemorySessionService
from google.genai import types

from tests.hitl_helpers import ScriptedLlm, collect, run_finished, run_input, tool_call


@pytest.fixture(autouse=True)
def reset_session_manager():
    SessionManager.reset_instance()
    yield
    SessionManager.reset_instance()


def _ops(surface_id: str, status: str) -> list:
    return [
        {"version": "v0.9", "createSurface": {"surfaceId": surface_id, "catalogId": "c"}},
        {
            "version": "v0.9",
            "updateDataModel": {"surfaceId": surface_id, "path": "/", "value": {"status": status}},
        },
    ]


def _widget(widget_id: str, status: str = "open", provider: str = "a2ui") -> UiWidget:
    return UiWidget(
        id=widget_id,
        provider=provider,
        payload={"a2ui_operations": _ops(widget_id, status)},
    )


def _response_event(event_id: str, call_id: str, *widgets: UiWidget) -> ADKEvent:
    return ADKEvent(
        id=event_id,
        author="agent",
        content=types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=call_id, name="show", response={"status": "ok"}
                    )
                )
            ],
        ),
        actions=EventActions(render_ui_widgets=list(widgets)),
    )


def _call_event(event_id: str, call_id: str) -> ADKEvent:
    return ADKEvent(
        id=event_id,
        author="agent",
        content=types.Content(
            role="model",
            parts=[types.Part(function_call=types.FunctionCall(id=call_id, name="show", args={}))],
        ),
    )


def _text_event(event_id: str, author: str, text: str) -> ADKEvent:
    return ADKEvent(
        id=event_id,
        author=author,
        content=types.Content(role="model" if author != "user" else "user", parts=[types.Part(text=text)]),
    )


async def _translate(event: ADKEvent) -> list:
    return [e async for e in EventTranslator().translate(event, "t", "r")]


class TestLiveActivity:
    @pytest.mark.asyncio
    async def test_a2ui_widget_becomes_activity_snapshot(self):
        events = await _translate(_response_event("e1", "call-1", _widget("chart-1")))

        activities = [e for e in events if e.type == EventType.ACTIVITY_SNAPSHOT]
        assert len(activities) == 1
        activity = activities[0]
        assert activity.message_id == "a2ui-surface-chart-1"
        assert activity.activity_type == "a2ui-surface"
        assert activity.content == {"a2ui_operations": _ops("chart-1", "open")}
        assert activity.replace is True

    @pytest.mark.asyncio
    async def test_other_providers_and_payloads_are_ignored(self):
        mcp = _widget("app-1", provider="mcp")
        no_ops = UiWidget(id="w-2", provider="a2ui", payload={"other": 1})
        events = await _translate(_response_event("e1", "call-1", mcp, no_ops))

        assert not [e for e in events if e.type == EventType.ACTIVITY_SNAPSHOT]


class TestHistory:
    def test_activity_follows_the_messages_of_its_event(self):
        events = [
            _text_event("u1", "user", "Compare"),
            _call_event("e1", "call-1"),
            _response_event("e2", "call-1", _widget("chart-1")),
            _text_event("e3", "agent", "Here it is."),
        ]

        messages = adk_events_to_messages(events)

        assert [m.role for m in messages] == ["user", "assistant", "tool", "activity", "assistant"]
        activity = messages[3]
        assert isinstance(activity, ActivityMessage)
        assert activity.id == "a2ui-surface-chart-1"
        assert activity.activity_type == "a2ui-surface"
        assert activity.content == {"a2ui_operations": _ops("chart-1", "open")}

    def test_a_rerender_keeps_the_first_position_with_the_latest_content(self):
        events = [
            _call_event("e1", "call-1"),
            _response_event("e2", "call-1", _widget("question-1", "open")),
            _text_event("u2", "user", "Winback"),
            _response_event("e3", "call-1b", _widget("question-1", "answered")),
        ]

        messages = adk_events_to_messages(events)

        activities = [m for m in messages if m.role == "activity"]
        assert len(activities) == 1
        assert messages.index(activities[0]) == 2
        assert activities[0].content == {"a2ui_operations": _ops("question-1", "answered")}

    def test_a_widget_on_the_last_event_is_kept(self):
        messages = adk_events_to_messages(
            [_call_event("e1", "call-1"), _response_event("e2", "call-1", _widget("chart-1"))]
        )
        assert messages[-1].id == "a2ui-surface-chart-1"


class TestThroughADK:
    @pytest.mark.asyncio
    async def test_tool_rendered_widget_streams_and_is_restored(self):
        def show_chart(tool_context) -> dict:
            """Show a chart."""
            tool_context.render_ui_widget(_widget("chart-call"))
            return {"status": "Rendered to the user."}

        sessions = InMemorySessionService()
        agent = ADKAgent(
            adk_agent=LlmAgent(
                name="widget_agent",
                model=ScriptedLlm(model="scripted", first_call={"name": "show_chart", "args": {}}),
                tools=[show_chart],
            ),
            app_name="widget_app",
            user_id="test_user",
            session_service=sessions,
        )

        events = await collect(
            agent,
            run_input("t-widget", "run-1", [UserMessage(id="u-1", role="user", content="Chart it")]),
        )

        live = [e for e in events if e.type == EventType.ACTIVITY_SNAPSHOT]
        assert [e.message_id for e in live] == ["a2ui-surface-chart-call"]
        # The surface follows the result of the call that rendered it.
        before = events[events.index(live[0]) - 1]
        assert before.type == EventType.TOOL_CALL_RESULT

        listed = await sessions.list_sessions(app_name="widget_app", user_id="test_user")
        session = await sessions.get_session(
            app_name="widget_app", user_id="test_user", session_id=listed.sessions[0].id
        )
        history = adk_events_to_messages(session.events)
        restored = [m for m in history if m.role == "activity"]
        assert [(m.id, m.content) for m in restored] == [
            (live[0].message_id, live[0].content)
        ]
        assert history[history.index(restored[0]) - 1].role == "tool"

    @pytest.mark.asyncio
    async def test_confirmation_gated_widget_rerenders_after_resume(self):
        # A question: rendered open while the tool pauses on a confirmation,
        # re-rendered answered when the resume re-runs it.
        def ask(tool_context) -> dict:
            """Ask the user."""
            confirmation = tool_context.tool_confirmation
            if confirmation is None:
                tool_context.render_ui_widget(_widget("question-call"))
                tool_context.request_confirmation(hint="Pick one", payload={"surfaceId": "question-call"})
                return {"status": "awaiting"}
            tool_context.render_ui_widget(_widget("question-call", "answered"))
            return {"result": confirmation.payload["selected"]}

        sessions = InMemorySessionService()
        agent = ADKAgent.from_app(
            App(
                name="question_app",
                root_agent=LlmAgent(
                    name="question_agent",
                    model=ScriptedLlm(model="scripted", first_call={"name": "ask", "args": {}}),
                    tools=[ask],
                ),
                resumability_config=ResumabilityConfig(is_resumable=True),
            ),
            user_id="test_user",
            session_service=sessions,
            emit_interrupt_outcome=True,
        )
        user = UserMessage(id="u-1", role="user", content="Help me pick")

        turn1 = await collect(agent, run_input("t-q", "run-1", [user]))
        opened = [e for e in turn1 if e.type == EventType.ACTIVITY_SNAPSHOT]
        assert [e.content for e in opened] == [{"a2ui_operations": _ops("question-call", "open")}]
        interrupt = run_finished(turn1).outcome.interrupts[0]
        rc_id, rc_args = tool_call(turn1, "adk_request_confirmation")

        turn2 = await collect(
            agent,
            run_input(
                "t-q",
                "run-2",
                [
                    user,
                    AssistantMessage(
                        id="a-1",
                        role="assistant",
                        tool_calls=[
                            ToolCall(
                                id=rc_id,
                                function=FunctionCall(name="adk_request_confirmation", arguments=rc_args),
                            )
                        ],
                    ),
                ],
                resume=[
                    ResumeEntry(
                        interrupt_id=interrupt.id,
                        status="resolved",
                        payload={"confirmed": True, "payload": {"selected": ["Winback"]}},
                    )
                ],
            ),
        )
        answered = [e for e in turn2 if e.type == EventType.ACTIVITY_SNAPSHOT]
        assert [(e.message_id, e.content) for e in answered] == [
            ("a2ui-surface-question-call", {"a2ui_operations": _ops("question-call", "answered")})
        ]

        listed = await sessions.list_sessions(app_name="question_app", user_id="test_user")
        session = await sessions.get_session(
            app_name="question_app", user_id="test_user", session_id=listed.sessions[0].id
        )
        restored = [m for m in adk_events_to_messages(session.events) if m.role == "activity"]
        assert [(m.id, m.content) for m in restored] == [
            ("a2ui-surface-question-call", {"a2ui_operations": _ops("question-call", "answered")})
        ]


class TestMessagesSnapshot:
    def test_a_snapshot_with_surfaces_deletes_none_of_the_clients_activities(self):
        event = messages_snapshot_event(adk_events_to_messages(
            [_call_event("e1", "call-1"), _response_event("e2", "call-1", _widget("chart-1"))]
        ))
        assert [m.id for m in event.messages if m.role == "activity"] == ["a2ui-surface-chart-1"]
        assert event.metadata == {"@ag-ui/client": {"authoritativeActivityTypes": []}}

    def test_a_snapshot_without_activities_is_unchanged(self):
        event = messages_snapshot_event(
            [UserMessage(id="u-1", role="user", content="Hi")]
        )
        assert event.metadata is None

    def test_a_widget_without_a_dict_payload_is_ignored(self):
        widget = _widget("chart-1").model_copy(update={"payload": None})
        assert a2ui_widget_activities(
            _response_event("e2", "call-1", widget)
        ) == []

