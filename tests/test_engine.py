import os
import tempfile
from src.engine import LocalRAGEngine
from langchain_core.documents import Document

def test_compute_file_hash():
    """Test that computing a SHA-256 hash is deterministic and reproducible."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"mock file content")
        tmp_path = tmp.name
        
    try:
        hash1 = LocalRAGEngine._compute_file_hash(tmp_path)
        hash2 = LocalRAGEngine._compute_file_hash(tmp_path)
        
        # Must be identical across identical files
        assert hash1 == hash2
        
        # A known payload must hash to its known digest. 
        assert hash1 == "5633d479dfae75ba7a78914ee380fa202bd6126e7c6b7c22e3ebc9e1a6ddc871"
    finally:
        os.remove(tmp_path)

def test_truncate_context_to_budget():
    """Test that the Token-Budget guard safely truncates documents exceeding the character max bounds."""
    # We will simulate 3 documents.
    # We set max_tokens=10. This equates to 10 * 4 = 40 characters limit.
    doc1 = Document(page_content="12345678901234567890") # 20 chars
    doc2 = Document(page_content="12345678901234567890") # 20 chars
    doc3 = Document(page_content="This should be sliced.") # 22 chars
    
    docs = [doc1, doc2, doc3]
    
    truncated = LocalRAGEngine._truncate_context_to_budget(docs, max_tokens=10)
    
    # doc1 (20) + doc2 (20) = 40 chars <= 40 chars max limit. 
    # doc3 would make it 62 chars, so it should be truncated.
    assert len(truncated) == 2
    assert truncated[0].page_content == doc1.page_content
    assert truncated[1].page_content == doc2.page_content
    assert doc3 not in truncated
