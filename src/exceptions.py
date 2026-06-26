class RAGException(Exception):
    """Base exception for all Local RAG System errors."""
    pass

class OffTopicException(RAGException):
    """Raised when a user query is rejected by the unified router guardrail."""
    pass

class DocumentIngestionError(RAGException):
    """Raised when the document loading or embedding pipeline fails."""
    pass
