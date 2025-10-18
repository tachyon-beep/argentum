"""CLI tests for document ingestion and retrieval."""

import json
from pathlib import Path

from click.testing import CliRunner

from argentum.cli import cli
from argentum.workspace import WorkspaceManager, DocumentStore


def _create_workspace(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("ARGENTUM_WORKSPACES_DIR", str(tmp_path / "projects"))
    manager = WorkspaceManager()
    workspace = manager.create_project("rag", title="RAG Workspace")
    return workspace.root


def test_docs_ingest_list_search(tmp_path, monkeypatch) -> None:
    workspace_root = _create_workspace(tmp_path, monkeypatch)
    doc_path = tmp_path / "policy.txt"
    doc_path.write_text("Recycling incentives and public outreach improve policy adoption.", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["project", "docs", "ingest", "rag", str(doc_path), "--tag", "policy"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert "Documents Ingested" in result.output

    result = runner.invoke(cli, ["project", "docs", "list", "rag"], catch_exceptions=False)
    assert "policy" in result.output

    result = runner.invoke(
        cli,
        [
            "project",
            "docs",
            "search",
            "rag",
            "--query",
            "recycling incentives",
            "--min-score",
            "0.0",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert "Document search" in result.output

    store = DocumentStore(Path(workspace_root) / "knowledge" / "docs")
    doc_entry = store.list_documents()[0]
    doc_id = doc_entry["doc_id"]
    chunk_id = f"{doc_id}::chunk-1"

    result = runner.invoke(
        cli,
        [
            "project",
            "docs",
            "feedback",
            "rag",
            "--doc-id",
            doc_id,
            "--chunk-id",
            chunk_id,
            "--session",
            "session-1",
            "--rating",
            "useful",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    feedback_path = Path(workspace_root) / "knowledge" / "docs" / "feedback.jsonl"
    assert feedback_path.exists()


def test_session_retrieves_documents(tmp_path, monkeypatch) -> None:
    workspace_root = _create_workspace(tmp_path, monkeypatch)
    doc_path = tmp_path / "market.txt"
    doc_path.write_text("Market analysis shows strong demand for AI knitting products.", encoding="utf-8")

    runner = CliRunner()
    runner.invoke(
        cli,
        ["project", "docs", "ingest", "rag", str(doc_path)],
        catch_exceptions=False,
    )

    # Run a debate session referencing the document
    result = runner.invoke(
        cli,
        [
            "debate",
            "AI knitting market",
            "--project",
            "rag",
            "--ministers",
            "finance",
            "--rounds",
            "1",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    session_dir = next((Path(workspace_root) / "sessions").iterdir())
    highlights_path = session_dir / "highlights.json"
    highlights = json.loads(highlights_path.read_text(encoding="utf-8"))
    assert highlights.get("retrieved_docs")
    quote_metadata = highlights.get("quotes", [{}])[0].get("metadata") or {}
    assert quote_metadata.get("citations")
    assert highlights.get("citations")
    assert "Sources:" in (highlights.get("summary") or "")
    assert highlights.get("retrieval_history")


def test_docs_purge_and_rebuild(tmp_path, monkeypatch) -> None:
    workspace_root = _create_workspace(tmp_path, monkeypatch)
    doc_path = tmp_path / "strategy.txt"
    doc_path.write_text("Adoption roadmap focusing on reliability and governance.", encoding="utf-8")

    runner = CliRunner()
    ingest_result = runner.invoke(
        cli,
        ["project", "docs", "ingest", "rag", str(doc_path)],
        catch_exceptions=False,
    )
    assert ingest_result.exit_code == 0

    store = DocumentStore(Path(workspace_root) / "knowledge" / "docs")
    doc_entry = store.list_documents()[0]
    doc_id = doc_entry["doc_id"]

    rebuild_result = runner.invoke(cli, ["project", "docs", "rebuild", "rag"], catch_exceptions=False)
    assert rebuild_result.exit_code == 0
    assert "Document Index Rebuilt" in rebuild_result.output

    # Remove source file to trigger missing warning on rebuild
    doc_path.unlink()
    missing_result = runner.invoke(cli, ["project", "docs", "rebuild", "rag"], catch_exceptions=False)
    assert missing_result.exit_code == 0
    assert "Missing sources" in missing_result.output

    purge_result = runner.invoke(cli, ["project", "docs", "purge", "rag", doc_id], catch_exceptions=False)
    assert purge_result.exit_code == 0
    assert "Document Purged" in purge_result.output

    list_result = runner.invoke(cli, ["project", "docs", "list", "rag"], catch_exceptions=False)
    assert "No documents ingested yet" in list_result.output

    search_result = runner.invoke(
        cli,
        ["project", "docs", "search", "rag", "--query", "reliability"],
        catch_exceptions=False,
    )
    assert search_result.exit_code == 0
    assert "No matching document chunks found" in search_result.output
