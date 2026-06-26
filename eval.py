import os
import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from config.settings import settings
from src.engine import LocalRAGEngine
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from typing import List

# 1. Define a Golden Evaluation Dataset
# These queries represent standard test cases for the RAG system.
GOLDEN_DATASET = [
    {
        "query": "What is the symbolic significance of the monolith in 2001: A Space Odyssey?",
        "ground_truth": "The monolith represents a catalyst for human evolution and extraterrestrial intelligence."
    },
    {
        "query": "How does the pacing in Inception compare to Interstellar?",
        "ground_truth": "Inception relies on fast-paced, multi-layered action sequences, whereas Interstellar features a slower, more emotionally driven build-up."
    },
    {
        "query": "Describe the character arc of Michael Corleone across the first two Godfather films.",
        "ground_truth": "Michael transforms from a reluctant outsider wanting a legitimate life to a ruthless, isolated mafia don."
    },
    {
        "query": "What are the core technical requirements for the local RAG system according to the project documentation?",
        "ground_truth": "The core technical requirements include Python 3.10+, local LLaMA 3.2 via Ollama, HuggingFace embeddings, and ChromaDB."
    },
    {
        "query": "How does the local RAG system ensure privacy?",
        "ground_truth": "It ensures privacy by keeping all vectors and generations on local hardware, air-gapped from third-party telemetry, without requiring external API keys."
    }
]

class FaithfulnessEvaluation(BaseModel):
    claims: List[str] = Field(description="A list of distinct claims made in the generated answer.")
    supported_claims: List[str] = Field(description="A list of claims from 'claims' that are strictly supported by the context.")

def evaluate_faithfulness(query, context_text, generated_answer, evaluator_llm):
    """
    Checks if the generated answer is strictly grounded in the provided context by computing a fractional ratio.
    """
    parser = PydanticOutputParser(pydantic_object=FaithfulnessEvaluation)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an impartial judge. Your task is to evaluate whether the generated answer is completely supported by the given context. "
                   "First, extract all distinct factual claims made in the answer. Then, determine which of those claims are explicitly supported by the context. "
                   "Respond strictly in JSON according to these formatting instructions:\n{format_instructions}"),
        ("human", f"Context: {context_text}\n\nGenerated Answer: {generated_answer}")
    ]).partial(format_instructions=parser.get_format_instructions())
    
    chain = prompt | evaluator_llm | parser
    try:
        eval_result = chain.invoke({})
        total = len(eval_result.claims)
        if total == 0:
            return 1.0
        supported = len(eval_result.supported_claims)
        return float(supported) / float(total)
    except Exception as e:
        print(f"Faithfulness eval failed: {e}")
        return 0.0

class ContextPrecisionEvaluation(BaseModel):
    relevant_chunks_count: int = Field(description="The number of provided context chunks that contain information necessary to answer the query.")
    total_chunks_count: int = Field(description="The total number of context chunks provided.")

def evaluate_context_precision(query, context_docs, evaluator_llm):
    """
    Checks if the retrieved context actually contains the relevant information to address the query.
    """
    if not context_docs:
        return 0.0
        
    parser = PydanticOutputParser(pydantic_object=ContextPrecisionEvaluation)
    chunks_text = "\n\n".join([f"Chunk {i+1}:\n{doc.page_content}" for i, doc in enumerate(context_docs)])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an impartial judge. Your task is to evaluate whether the provided context chunks contain the necessary information to accurately answer the user's query. "
                   "Count how many chunks are actually relevant. "
                   "Respond strictly in JSON according to these formatting instructions:\n{format_instructions}"),
        ("human", f"Query: {query}\n\nContext Chunks:\n{chunks_text}")
    ]).partial(format_instructions=parser.get_format_instructions())
    
    chain = prompt | evaluator_llm | parser
    try:
        eval_result = chain.invoke({})
        total = eval_result.total_chunks_count
        if total <= 0 or total < len(context_docs):
            total = len(context_docs)
        relevant = eval_result.relevant_chunks_count
        return min(1.0, float(relevant) / float(total))
    except Exception as e:
        print(f"Context precision eval failed: {e}")
        return 0.0

def run_evaluation():
    print("Initializing RAG Pipeline for Evaluation...")
    # Initialize the retrieval chain
    engine = LocalRAGEngine(session_id="eval_session", history_aware=False)
    
    if getattr(engine, "vectorstore", None) == "NO_DOCS":
        print("WARNING: No documents found in ./data for eval_session. The RAG system will rely solely on the LLM's internal knowledge (or will fail to retrieve context).")
        print("Please upload relevant documents to ./data to test retrieval metrics properly.")
        # Proceed anyway to test the flow, though context will be empty.
    
    evaluator_llm = ChatOllama(model=settings.llm_model, temperature=0.0, num_ctx=2048)
    
    results = []
    
    print(f"\nStarting Evaluation of {len(GOLDEN_DATASET)} queries...\n")
    print("-" * 110)
    print(f"| {'Query Prefix':<40} | {'Faithfulness':<15} | {'Context Precision':<20} | {'Status':<10} |")
    print("-" * 110)
    
    for i, item in enumerate(GOLDEN_DATASET, 1):
        query = item["query"]
        ground_truth = item["ground_truth"]
        
        # Programmatically execute pipeline
        if getattr(engine, "vectorstore", None) == "NO_DOCS":
            # Mock empty context if no docs available, purely for structural testing
            response = {"answer": "I cannot find that in the documents.", "context": []}
        else:
            try:
                response = engine.invoke({"input": query})
            except Exception as e:
                response = {"answer": f"Error: {e}", "context": []}
        
        generated_answer = response["answer"]
        context_docs = response.get("context", [])
        
        # Extract source context arrays
        context_text = "\n\n".join([doc.page_content for doc in context_docs]) if context_docs else "No Context Retrieved."
        
        # Compute alignment metrics
        faithfulness_score = evaluate_faithfulness(query, context_text, generated_answer, evaluator_llm)
        precision_score = evaluate_context_precision(query, context_docs, evaluator_llm)
        
        # Aggregate logic
        total_score = faithfulness_score + precision_score
        status = "PASS" if total_score >= 1.5 else ("WARN" if total_score >= 1.0 else "FAIL")
        
        query_prefix = (query[:37] + "...") if len(query) > 40 else query
        
        print(f"| {query_prefix:<40} | {faithfulness_score:<15} | {precision_score:<20} | {status:<10} |")
        
        results.append({
            "query": query,
            "faithfulness": faithfulness_score,
            "context_precision": precision_score,
            "status": status
        })
        
    print("-" * 110)
    
    # Calculate overall averages
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_prec = sum(r["context_precision"] for r in results) / len(results)
    
    print(f"\nFinal Averages -> Faithfulness: {avg_faith:.2f} | Context Precision: {avg_prec:.2f}")

if __name__ == "__main__":
    run_evaluation()
