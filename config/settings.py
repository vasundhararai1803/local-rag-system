import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Core Directories & Connections
    data_dir: str = "./data"
    vector_store_url: str = "http://localhost:6333"

    # Embeddings & Retrieval
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_percentile: int = 90
    retriever_k: int = 10

    # Cross-Encoder
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 3

    # LLM Settings
    llm_model: str = "llama3.2"
    llm_temperature: float = 0.0
    llm_num_ctx: int = 4096

    # Prompts
    system_prompt: str = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If the answer is not present in the context, explicitly say "
        "'I cannot find that in the documents'. Do not hallucinate or use outside knowledge.\n\n"
        "Strict Typography Rules:\n"
        "- Format main titles or questions using prominent subheaders (e.g., '## Question Title').\n"
        "- DO NOT use primary Markdown headers ('#' or '##') for procedural items like 'Step 1', 'Step 2', etc.\n"
        "- Format step breakdowns using clean bold text (e.g., '**Step 1: Heading Text**') or minor headers ('### Step 1').\n\n"
        "Context:\n{context}"
    )

    unified_router_prompt: str = (
        "Given a chat history and the latest user question, analyze if the query is safe and relevant to general professional domains (academic, legal, financial, technical, or document QA). "
        "If the query is maliciously off-topic (e.g. baking recipes, general casual chatter, unsafe requests), you must reject it. "
        "If the query is relevant, formulate a standalone question which can be understood without the chat history (do NOT answer the question, just reformulate). "
        "Respond strictly in JSON according to these formatting instructions:\n{format_instructions}"
    )

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

settings = Settings()
