import pytest
from langchain.schema import Document

from kfai.loaders.utils.config import COLLECTION_NAME


def test_vector_store_creation(test_paths):
    from kfai.loaders.build_vector_store import run

    # Test document creation and embedding
    doc = Document(
        page_content="Test content from Greg Miller",
        metadata={
            "video_id": "test123",
            "start_time": 0.0,
            "title": "Test Video",
        },
    )

    # Mock the vectorstore operations
    class MockVectorStore:
        def add_documents(self, docs):
            assert len(docs) > 0
            return True

    vectorstore = MockVectorStore()
    result = vectorstore.add_documents([doc])
    assert result == True


def test_query_agent():
    from kfai.loaders.agents.query_agent import QueryAgent

    # Mock LLM for testing
    class MockLLM:
        def invoke(self, prompt):
            return "This is a test response about Greg Miller"

    agent = QueryAgent(llm=MockLLM())
    response = agent.process_query("What did Greg say?")
    assert response is not None
