"""Unit tests for the LightOn SDK. All HTTP calls are mocked."""

import pytest
import respx
from httpx import Response

from sdk import (
    LightOn,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ValidationError,
    LightOnError,
)

BASE = "https://fake.lighton.ai/api/v3"


# Response fixtures

WORKSPACE_JSON = {
    "id": 1,
    "name": "test-workspace",
    "workspace_type": "custom",
    "document_upload_method": "manual",
    "description": "a workspace",
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-02T00:00:00Z",
    "user_role": "owner",
    "used_storage": 1024.0,
    "files_count": 5,
    "summaries": [{"language": "en", "summary": "test summary"}],
}

FILE_JSON = {
    "id": 42,
    "filename": "doc.pdf",
    "workspace": {"id": 1, "name": "test-workspace", "workspace_type": "custom"},
    "title": "My Doc",
    "extension": "pdf",
    "status": "embedded",
    "status_vision": None,
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-02T00:00:00Z",
    "total_pages": 3,
    "tags": [{"id": 1, "name": "important", "auto_assigned": False}],
    "created_by": {"id": 10, "first_name": "Jane", "last_name": "Doe", "username": "jdoe"},
    "upload_session_uuid": "abc-123",
    "external_metadata": None,
    "summaries": [],
    "message": "",
}

USER_JSON = {
    "id": 10,
    "first_name": "Jane",
    "last_name": "Doe",
    "username": "jdoe",
    "email": "jane@nexatech.com",
}

RETRIEVE_JSON = {
    "results": [
        {
            "chunk": {
                "id": 100,
                "uuid": "chunk-uuid-1",
                "content_id": "cid-1",
                "text": "Revenue grew 23% in Q1",
                "chunk_type": "text",
                "metadata": {},
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            },
            "scoring": {
                "score": 0.95,
                "distance": 0.12,
                "lexical_score": 0.8,
                "certainty": 0.9,
            },
            "workspace": {"id": 1, "name": "test-workspace"},
            "document": {"id": 42, "name": "report.pdf"},
        }
    ]
}


# Client lifecycle

class TestClientLifecycle:
    def test_context_manager(self):
        with LightOn(api_key="fake") as client:
            assert client is not None

    def test_close(self):
        client = LightOn(api_key="fake")
        client.close()


# Error handling

class TestErrorHandling:
    @respx.mock
    def test_401_raises_auth_error(self, mock_client):
        respx.get(f"{BASE}/users/me").mock(
            return_value=Response(401, json={"detail": "invalid token"})
        )
        with pytest.raises(AuthenticationError) as exc:
            mock_client.me()
        assert exc.value.status_code == 401

    @respx.mock
    def test_403_raises_auth_error(self, mock_client):
        respx.get(f"{BASE}/workspaces/999").mock(
            return_value=Response(403, json={"detail": "forbidden"})
        )
        with pytest.raises(AuthenticationError) as exc:
            mock_client.get_workspace(999)
        assert exc.value.status_code == 403

    @respx.mock
    def test_404_raises_not_found(self, mock_client):
        respx.get(f"{BASE}/files/999").mock(
            return_value=Response(404, json={"detail": "not found"})
        )
        with pytest.raises(NotFoundError):
            mock_client.get_file(999)

    @respx.mock
    def test_400_raises_validation_error(self, mock_client):
        respx.post(f"{BASE}/workspaces").mock(
            return_value=Response(400, json={"detail": "name required"})
        )
        with pytest.raises(ValidationError):
            mock_client.create_workspace("")

    @respx.mock
    def test_429_raises_rate_limit(self, mock_client):
        respx.get(f"{BASE}/files").mock(
            return_value=Response(429, json={"detail": "rate limited"})
        )
        with pytest.raises(RateLimitError):
            mock_client.list_files()

    @respx.mock
    def test_500_raises_generic_error(self, mock_client):
        respx.get(f"{BASE}/users/me").mock(
            return_value=Response(500, json={"detail": "internal error"})
        )
        with pytest.raises(LightOnError) as exc:
            mock_client.me()
        assert exc.value.status_code == 500


# User

class TestUser:
    @respx.mock
    def test_me(self, mock_client):
        respx.get(f"{BASE}/users/me").mock(
            return_value=Response(200, json={"profile": USER_JSON})
        )
        user = mock_client.me()
        assert user.id == 10
        assert user.username == "jdoe"
        assert user.email == "jane@nexatech.com"


# Workspaces

class TestWorkspaces:
    @respx.mock
    def test_list_workspaces(self, mock_client):
        respx.get(f"{BASE}/workspaces").mock(
            return_value=Response(200, json={
                "count": 1, "next": None, "previous": None,
                "results": [WORKSPACE_JSON],
            })
        )
        result = mock_client.list_workspaces()
        assert result.count == 1
        assert len(result.results) == 1
        assert result.results[0].name == "test-workspace"
        assert result.results[0].summaries[0].language == "en"

    @respx.mock
    def test_list_workspaces_with_filters(self, mock_client):
        route = respx.get(f"{BASE}/workspaces").mock(
            return_value=Response(200, json={
                "count": 0, "next": None, "previous": None, "results": [],
            })
        )
        mock_client.list_workspaces(name="test", workspace_type="custom", page=2, page_size=10)
        assert route.called
        request = route.calls.last.request
        assert "name=test" in str(request.url)
        assert "workspace_type=custom" in str(request.url)
        assert "page=2" in str(request.url)

    @respx.mock
    def test_get_workspace(self, mock_client):
        respx.get(f"{BASE}/workspaces/1").mock(
            return_value=Response(200, json=WORKSPACE_JSON)
        )
        ws = mock_client.get_workspace(1)
        assert ws.id == 1
        assert ws.files_count == 5
        assert ws.used_storage == 1024.0

    @respx.mock
    def test_create_workspace(self, mock_client):
        route = respx.post(f"{BASE}/workspaces").mock(
            return_value=Response(201, json=WORKSPACE_JSON)
        )
        ws = mock_client.create_workspace("test-workspace", description="a workspace")
        assert ws.name == "test-workspace"
        body = route.calls.last.request.content
        assert b"test-workspace" in body


# Files

class TestFiles:
    @respx.mock
    def test_list_files(self, mock_client):
        respx.get(f"{BASE}/files").mock(
            return_value=Response(200, json={
                "count": 1, "next": None, "previous": None,
                "results": [FILE_JSON],
            })
        )
        result = mock_client.list_files()
        assert result.count == 1
        assert result.results[0].filename == "doc.pdf"
        assert result.results[0].tags[0].name == "important"
        assert result.results[0].created_by.username == "jdoe"

    @respx.mock
    def test_list_files_with_filters(self, mock_client):
        route = respx.get(f"{BASE}/files").mock(
            return_value=Response(200, json={
                "count": 0, "next": None, "previous": None, "results": [],
            })
        )
        mock_client.list_files(workspace_id=1, status="embedded", search="report", extension="pdf")
        request = route.calls.last.request
        assert "workspace_id=1" in str(request.url)
        assert "status=embedded" in str(request.url)
        assert "search=report" in str(request.url)

    @respx.mock
    def test_get_file(self, mock_client):
        respx.get(f"{BASE}/files/42").mock(
            return_value=Response(200, json=FILE_JSON)
        )
        f = mock_client.get_file(42)
        assert f.id == 42
        assert f.status == "embedded"
        assert f.total_pages == 3

    @respx.mock
    def test_get_file_with_content(self, mock_client):
        route = respx.get(f"{BASE}/files/42").mock(
            return_value=Response(200, json={**FILE_JSON, "content": "hello world"})
        )
        f = mock_client.get_file(42, include_content=True)
        assert f.content == "hello world"
        assert "include_content=true" in str(route.calls.last.request.url).lower()

    @respx.mock
    def test_delete_file(self, mock_client):
        route = respx.delete(f"{BASE}/files/42").mock(
            return_value=Response(204)
        )
        mock_client.delete_file(42)
        assert route.called

    @respx.mock
    def test_upload_file(self, mock_client, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        route = respx.post(f"{BASE}/files").mock(
            return_value=Response(201, json=FILE_JSON)
        )
        f = mock_client.upload_file(test_file, workspace_id=1, title="My Doc")
        assert f.id == 42
        assert route.called
        request = route.calls.last.request
        assert b"workspace_id" in request.content
        assert b"test.txt" in request.content

    @respx.mock
    def test_update_file(self, mock_client):
        route = respx.patch(f"{BASE}/files/42").mock(
            return_value=Response(200, json={**FILE_JSON, "title": "New Title"})
        )
        f = mock_client.update_file(42, title="New Title")
        assert f.title == "New Title"
        assert route.called


# Retrieve

class TestRetrieve:
    @respx.mock
    def test_retrieve_basic(self, mock_client):
        respx.post(f"{BASE}/retrieve").mock(
            return_value=Response(200, json=RETRIEVE_JSON)
        )
        resp = mock_client.retrieve("revenue growth")
        assert len(resp.results) == 1
        assert resp.results[0].chunk.text == "Revenue grew 23% in Q1"
        assert resp.results[0].scoring.score == 0.95
        assert resp.results[0].document["name"] == "report.pdf"

    @respx.mock
    def test_retrieve_with_filters(self, mock_client):
        route = respx.post(f"{BASE}/retrieve").mock(
            return_value=Response(200, json={"results": []})
        )
        mock_client.retrieve(
            "test",
            workspace_ids=[1, 2],
            file_ids=[42],
            tag_ids=[5],
            top_k=50,
            top_n=20,
            mode="vision",
            skip_rerank=True,
        )
        import json
        body = json.loads(route.calls.last.request.content)
        assert body["query"] == "test"
        assert body["workspace_id"] == [1, 2]
        assert body["file_id"] == [42]
        assert body["tag_id"] == [5]
        assert body["top_k"] == 50
        assert body["top_n"] == 20
        assert body["mode"] == "vision"
        assert body["skip_rerank"] is True

    @respx.mock
    def test_retrieve_empty(self, mock_client):
        respx.post(f"{BASE}/retrieve").mock(
            return_value=Response(200, json={"results": []})
        )
        resp = mock_client.retrieve("nonexistent")
        assert len(resp.results) == 0


# Model parsing edge cases

class TestModelParsing:
    @respx.mock
    def test_workspace_with_no_summaries(self, mock_client):
        data = {**WORKSPACE_JSON, "summaries": []}
        respx.get(f"{BASE}/workspaces/1").mock(
            return_value=Response(200, json=data)
        )
        ws = mock_client.get_workspace(1)
        assert ws.summaries == []

    @respx.mock
    def test_file_with_no_tags_no_creator(self, mock_client):
        data = {**FILE_JSON, "tags": [], "created_by": None}
        respx.get(f"{BASE}/files/42").mock(
            return_value=Response(200, json=data)
        )
        f = mock_client.get_file(42)
        assert f.tags == []
        assert f.created_by is None

    @respx.mock
    def test_unknown_fields_ignored(self, mock_client):
        data = {**USER_JSON, "some_future_field": "value"}
        respx.get(f"{BASE}/users/me").mock(
            return_value=Response(200, json={"profile": data})
        )
        user = mock_client.me()
        assert user.id == 10
