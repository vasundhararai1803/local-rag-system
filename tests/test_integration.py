import pytest
import os
from unittest.mock import MagicMock, patch
from src.engine import LocalRAGEngine
from langchain_core.messages import AIMessage, AIMessageChunk

from langchain_core.language_models import FakeListChatModel

@patch('src.engine.QdrantVectorStore')
@patch('src.engine.ChatOllama', return_value=FakeListChatModel(responses=['{"is_on_topic": true, "reason": "Valid request."}', "Mocked answer chunk."]))
@patch('src.engine.HuggingFaceEmbeddings')
@patch('src.engine.HuggingFaceCrossEncoder')
@patch('src.engine.QdrantClient')
def test_end_to_end_rag_pipeline_with_mock_llm(mock_qdrant_client, mock_cross_encoder, mock_embeddings, mock_chat, mock_qdrant_store, monkeypatch):
    """Verifies the complete execution path through the engine using a mocked LLM response."""
    # 1. Mock the structured Pydantic output router decision
    # FakeListChatModel automatically returns the configured responses in order!

    # 2. Mock the Vector Store Retriever to return fake grounded chunks
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    
    mock_store_instance = MagicMock()
    mock_store_instance.as_retriever.return_value = mock_retriever
    mock_qdrant_store.return_value = mock_store_instance
    
    # Mock QdrantClient collection info
    mock_collection_info = MagicMock()
    mock_collection_info.points_count = 1
    mock_qdrant_client.return_value.get_collection.return_value = mock_collection_info

    # 3. Instantiate the engine and execute a stream or invoke path
    # We patch glob to return an empty list so it doesn't try to load real files
    with patch('src.engine.glob.glob', return_value=[]):
        engine = LocalRAGEngine(session_id="test_integration_session")
        
        # Override the vectorstore with our mock so it doesn't say "NO_DOCS"
        engine.vectorstore = mock_store_instance
        engine.retriever = mock_retriever
        engine._setup_lcel_graph()
        
        # Verify that the pipeline executes without raising exceptions or throwing validation errors
        try:
            result = list(engine.stream({"input": "Analyze the financial trend in this document."}))
            assert isinstance(result, list)
        except Exception as e:
            pytest.fail(f"Pipeline crashed under integration test: {e}")

@patch('src.engine.QdrantVectorStore')
@patch('src.engine.QdrantClient')
@patch('src.engine.HuggingFaceEmbeddings')
@patch('src.engine.HuggingFaceCrossEncoder')
@patch('src.engine.glob.glob')
@patch('src.engine.os.path.isfile')
def test_ingestion_idempotency_blocks_duplicate_vectors(mock_isfile, mock_glob, mock_cross_encoder, mock_embeddings, mock_qdrant_client, mock_qdrant_store, mocker):
    """Ensures the engine queries the vector database payload and drops matching hashes."""
    # Simulate finding an existing file in the directory
    mock_glob.return_value = ["dummy_path.pdf"]
    mock_isfile.return_value = True
    
    # Mock file hashing to return a dummy hash
    mocker.patch('src.engine.LocalRAGEngine._compute_file_hash', return_value="dummy_hash")
    
    # Simulate finding an existing file hash inside the Qdrant database payload scroll
    mock_client_instance = mock_qdrant_client.return_value
    mock_collection_info = MagicMock()
    mock_collection_info.points_count = 1
    mock_client_instance.get_collection.return_value = mock_collection_info
    mock_client_instance.scroll.return_value = ([mocker.MagicMock()], None) # Found match

    # If the hash is found, the engine should bypass the chunking and upload sequence entirely
    with patch('src.engine.SemanticChunker') as mock_splitter:
        engine = LocalRAGEngine(session_id="test_idempotency_session")
        # Assert that the text splitter was never even called because the database match triggered a short-circuit
        mock_splitter.assert_not_called()
