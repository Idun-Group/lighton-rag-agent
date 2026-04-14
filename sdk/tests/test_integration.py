"""Integration tests against the live LightOn API.

These require LIGHTON_API_KEY in .env and will create/delete
real resources in the personal workspace.
"""

import time
from pathlib import Path

import pytest
from fpdf import FPDF

from sdk import LightOn


pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def personal_workspace_id(live_client):
    ws_list = live_client.list_workspaces(workspace_type="personal")
    assert ws_list.count > 0, "No personal workspace found"
    return ws_list.results[0].id


class TestUserIntegration:
    def test_me_returns_user(self, live_client):
        user = live_client.me()
        assert user.id > 0


class TestWorkspacesIntegration:
    def test_list_workspaces(self, live_client):
        result = live_client.list_workspaces()
        assert result.count > 0
        assert len(result.results) > 0

    def test_get_workspace(self, live_client, personal_workspace_id):
        ws = live_client.get_workspace(personal_workspace_id)
        assert ws.id == personal_workspace_id
        assert ws.workspace_type == "personal"

    def test_list_workspaces_pagination(self, live_client):
        page1 = live_client.list_workspaces(page=1, page_size=1)
        assert len(page1.results) <= 1


class TestFilesIntegration:
    def test_list_files(self, live_client):
        result = live_client.list_files(page_size=5)
        assert result.count >= 0

    def test_get_file(self, live_client):
        files = live_client.list_files(page_size=1)
        if files.count == 0:
            pytest.skip("No files to test with")
        f = live_client.get_file(files.results[0].id)
        assert f.id == files.results[0].id
        assert f.status in ("embedded", "pending", "parsing", "embedding")


class TestFileLifecycle:
    """Upload, poll, update, and delete a file."""

    def test_full_lifecycle(self, live_client, personal_workspace_id, tmp_path):
        test_file = tmp_path / "lifecycle_test.txt"
        test_file.write_text("Integration test content for lifecycle verification.")

        uploaded = live_client.upload_file(test_file, workspace_id=personal_workspace_id)
        assert uploaded.id > 0
        assert uploaded.status == "pending"
        file_id = uploaded.id

        try:
            for _ in range(20):
                time.sleep(2)
                f = live_client.get_file(file_id)
                if f.status in ("embedded", "fail", "embedding_failed", "parsing_failed"):
                    break
            assert f.status == "embedded", f"File ended in status: {f.status}"

            updated = live_client.update_file(file_id, title="Updated Title")
            assert updated.title == "Updated Title"
        finally:
            live_client.delete_file(file_id)

        files = live_client.list_files(search="lifecycle_test")
        found = [f for f in files.results if f.id == file_id]
        assert len(found) == 0


class TestPDFUploadAndRetrieve:
    """Upload a PDF, wait for indexing, query it, then clean up."""

    def test_pdf_round_trip(self, live_client, personal_workspace_id, tmp_path):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, (
            "The Zephyr project launched in March 2026 with a budget of EUR 500K. "
            "Lead engineer is Dr. Akira Tanaka. The project aims to reduce inference "
            "latency by 40% through speculative decoding and KV cache optimization."
        ))
        pdf_path = tmp_path / "zephyr_test.pdf"
        pdf.output(str(pdf_path))

        uploaded = live_client.upload_file(pdf_path, workspace_id=personal_workspace_id)
        file_id = uploaded.id

        try:
            for _ in range(20):
                time.sleep(2)
                f = live_client.get_file(file_id)
                if f.status in ("embedded", "fail", "embedding_failed", "parsing_failed"):
                    break
            assert f.status == "embedded"

            resp = live_client.retrieve(
                "Zephyr project budget",
                workspace_ids=[personal_workspace_id],
                top_n=3,
            )
            assert len(resp.results) > 0
            top_result = resp.results[0]
            assert "zephyr" in top_result.chunk.text.lower() or "500" in top_result.chunk.text
            assert top_result.scoring.score > 0
        finally:
            live_client.delete_file(file_id)


class TestRetrieveIntegration:
    def test_retrieve_empty_results(self, live_client, personal_workspace_id):
        resp = live_client.retrieve(
            "xyznonexistentquerythatwontmatch123",
            workspace_ids=[personal_workspace_id],
        )
        assert len(resp.results) == 0

    def test_retrieve_with_top_params(self, live_client, personal_workspace_id):
        resp = live_client.retrieve(
            "test",
            workspace_ids=[personal_workspace_id],
            top_k=5,
            top_n=2,
        )
        assert len(resp.results) <= 2
