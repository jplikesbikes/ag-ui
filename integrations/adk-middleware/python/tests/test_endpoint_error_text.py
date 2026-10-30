"""A failing endpoint tells the client what failed, not the exception's text.

Exception text can name internal resources, configuration or other users'
sessions. The run endpoint and a session backend read already answer with a
stable message; /agents/state and /capabilities now do the same for every
failure, and log the exception.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ag_ui_adk.adk_agent import ADKAgent
from ag_ui_adk.endpoint import add_adk_fastapi_endpoint

SECRET = "postgres://admin:hunter2@db.internal/sessions"


def test_capabilities_failure_hides_the_exception_text(caplog):
    agent = MagicMock(spec=ADKAgent)
    agent.get_capabilities.side_effect = RuntimeError(SECRET)
    app = FastAPI()
    add_adk_fastapi_endpoint(app, agent)

    response = TestClient(app).get("/capabilities")

    assert response.status_code == 500
    assert response.json() == {"error": "Failed to retrieve capabilities."}
    assert SECRET in caplog.text


def test_agents_state_failure_hides_the_exception_text(caplog):
    agent = ADKAgent(
        adk_agent=MagicMock(name="adk_agent"),
        app_name="error_text_app",
        user_id="user",
        use_in_memory_services=True,
    )
    app = FastAPI()
    add_adk_fastapi_endpoint(app, agent)

    session = MagicMock(id="s-1", events=[MagicMock()])
    with patch.object(
        agent._session_manager, "resolve_existing_session", AsyncMock(return_value=session)
    ), patch.object(
        agent._session_manager, "get_session_state", AsyncMock(return_value={})
    ), patch(
        "ag_ui_adk.endpoint.adk_events_to_messages", side_effect=RuntimeError(SECRET)
    ):
        response = TestClient(app).post("/agents/state", json={"threadId": "t-1"})

    assert response.status_code == 500
    assert response.json()["error"] == "Failed to read the state of this thread."
    assert SECRET not in response.text
    assert SECRET in caplog.text
