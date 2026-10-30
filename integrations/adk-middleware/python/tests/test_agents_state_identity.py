"""/agents/state reads the thread under the identity a run would use.

A run resolves its app and user through the agent (``_get_app_name`` /
``_get_user_id``: a subclass override, a static value, an extractor, or the
defaults). The state endpoint used its own chain instead, ending in the request
body's ``appName`` / ``userId``, so a client could name another app's or user's
thread and read it, and an app choosing its tenant by overriding
``_get_app_name`` was never consulted.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.adk.agents.llm_agent import LlmAgent
from google.adk.sessions import InMemorySessionService

from ag_ui.core import UserMessage
from ag_ui_adk import ADKAgent
from ag_ui_adk.endpoint import add_adk_fastapi_endpoint
from ag_ui_adk.session_manager import SessionManager

from tests.hitl_helpers import ScriptedLlm, collect, run_input

QUESTION = UserMessage(id="u-1", role="user", content="Hello")


@pytest.fixture(autouse=True)
def reset_session_manager():
    SessionManager.reset_instance()
    yield
    SessionManager.reset_instance()


class TenantAgent(ADKAgent):
    """Chooses the app per tenant, from what the request's extractor verified."""

    def _get_app_name(self, input):
        return f"app-{input.state['tenant']}"


async def _tenant_from_header(request, _input):
    return {"tenant": request.headers["x-tenant"]}


def _tenant_agent(service):
    return TenantAgent(
        adk_agent=LlmAgent(name="agent", model=ScriptedLlm(model="scripted")),
        user_id="user",
        session_service=service,
    )


@pytest.mark.asyncio
async def test_a_tenant_cannot_read_another_tenants_thread():
    service = InMemorySessionService()
    agent = _tenant_agent(service)
    await collect(agent, run_input("t-1", "run-1", [QUESTION], state={"tenant": "a"}))

    app = FastAPI()
    add_adk_fastapi_endpoint(app, agent, extract_state_from_request=_tenant_from_header)
    with TestClient(app) as client:
        own = client.post("/agents/state", headers={"x-tenant": "a"}, json={"threadId": "t-1"})
        with pytest.warns(DeprecationWarning, match="#1646"):
            other = client.post(
                "/agents/state",
                headers={"x-tenant": "b"},
                json={"threadId": "t-1", "appName": "app-a", "userId": "user"},
            )

    assert own.json()["threadExists"] is True
    assert [m["content"] for m in own.json()["messages"] if m["role"] == "user"] == ["Hello"]
    assert other.status_code == 200
    assert other.json()["threadExists"] is False
    assert other.json()["messages"] == []


@pytest.mark.asyncio
async def test_without_configured_identity_the_state_read_finds_what_the_run_wrote():
    service = InMemorySessionService()
    agent = ADKAgent(
        adk_agent=LlmAgent(name="agent", model=ScriptedLlm(model="scripted")),
        session_service=service,
    )
    await collect(agent, run_input("t-2", "run-1", [QUESTION]))

    app = FastAPI()
    add_adk_fastapi_endpoint(app, agent)
    with TestClient(app) as client:
        found = client.post("/agents/state", json={"threadId": "t-2"})
        with pytest.warns(DeprecationWarning, match="#1646"):
            steered = client.post(
                "/agents/state",
                json={"threadId": "t-2", "appName": "other-app", "userId": "other-user"},
            )

    assert found.json()["threadExists"] is True
    assert steered.json() == found.json()
