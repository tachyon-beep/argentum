"""Tests for the lightweight document store."""

from pathlib import Path

from argentum.knowledge.document_store import DocumentStore


def test_document_store_ingest_and_query(tmp_path) -> None:
    store = DocumentStore(tmp_path / "docs")
    text_file = tmp_path / "policy.txt"
    text_file.write_text("Recycling policy should include incentives and public education.", encoding="utf-8")

    chunks = store.ingest_file(text_file, doc_id="doc-policy", metadata={"title": "Policy"}, chunk_size=50)
    assert chunks == 1

    documents = store.list_documents()
    assert documents and documents[0]["doc_id"] == "doc-policy"

    results = store.query("recycling incentives", limit=3)
    assert results
    assert results[0]["doc_id"] == "doc-policy"
    assert results[0]["score"] > 0


def test_document_store_feedback(tmp_path) -> None:
    store = DocumentStore(tmp_path / "docs")
    feedback_file = store.base_path / "feedback.jsonl"
    assert not feedback_file.exists()

    store.record_feedback(
        doc_id="doc1",
        chunk_id="doc1::chunk-1",
        session_id="session-1",
        rating="useful",
        notes="Helped summarise policy",
    )

    assert feedback_file.exists()
    contents = feedback_file.read_text(encoding="utf-8")
    assert "doc1" in contents and "useful" in contents
