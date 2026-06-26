import os
import glob
import hashlib
import json
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from src.config import *

def compute_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def load_documents(directory_path, hashes_registry):
    documents = []
    new_hashes = {}
    for file_path in glob.glob(f"{directory_path}/**/*", recursive=True):
        if os.path.isfile(file_path):
            if file_path.endswith('ingested_hashes.json'):
                continue
            
            file_hash = compute_file_hash(file_path)
            if file_path in hashes_registry and hashes_registry[file_path] == file_hash:
                continue # Skip already processed duplicate
                
            try:
                if file_path.endswith('.txt'):
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
                elif file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                
                new_hashes[file_path] = file_hash
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    return documents, new_hashes

def build_rag_chain(session_id="langchain", history_aware=False):
    user_data_dir = os.path.join(DATA_DIR, session_id) if session_id != "langchain" else DATA_DIR
    os.makedirs(user_data_dir, exist_ok=True)
    
    registry_path = os.path.join(user_data_dir, "ingested_hashes.json")
    hashes_registry = {}
    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r') as f:
                hashes_registry = json.load(f)
        except Exception:
            hashes_registry = {}
            
    documents, new_hashes = load_documents(user_data_dir, hashes_registry)
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection_exists = False
    try:
        client.get_collection(session_id)
        collection_exists = True
    except Exception:
        pass

    if not documents and not collection_exists and not hashes_registry:
        return "NO_DOCS"
        
    if documents:
        text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=CHUNK_PERCENTILE)
        splits = text_splitter.split_documents(documents)
        
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory=CHROMA_DB_DIR,
            collection_name=session_id
        )
        
        # Save updated hashes back to disk
        hashes_registry.update(new_hashes)
        with open(registry_path, 'w') as f:
            json.dump(hashes_registry, f, indent=4)
    else:
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings,
            collection_name=session_id
        )
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE, num_ctx=LLM_NUM_CTX)
    
    # ---------------------------------------------------------
    # Modern LCEL Pipeline Setup
    # ---------------------------------------------------------
    
    # 1. Query Contextualization & Routing
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", UNIFIED_ROUTER_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
    ])
    router_chain = router_prompt | llm | JsonOutputParser()
    
    def resolve_query(inputs):
        # We always route to verify safety and relevance
        result = router_chain.invoke(inputs)
        if result.get("status") == "REJECT":
            raise ValueError("OFF_TOPIC")
        return result.get("query") or inputs["input"]

    # 2. Retrieval & Local Cross-Encoder Reranking
    def retrieve_and_rerank(query):
        docs = retriever.invoke(query)
        if not docs:
            return []
        # Score docs with HuggingFace Cross-Encoder
        scores = cross_encoder.score([(query, doc.page_content) for doc in docs])
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        # Return top N documents
        return [doc for doc, score in scored_docs[:RERANKER_TOP_N]]

    retrieve_chain = RunnableLambda(resolve_query) | RunnableLambda(retrieve_and_rerank)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 3. QA Prompt Factory
    def get_qa_prompt(inputs):
        if history_aware:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            return prompt_template.invoke({
                "context": format_docs(inputs["context"]),
                "input": inputs["input"],
                "chat_history": inputs.get("chat_history", [])
            })
        else:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", "{input}"),
            ])
            return prompt_template.invoke({
                "context": format_docs(inputs["context"]),
                "input": inputs["input"]
            })

    # 4. Streamable LCEL-style chain Wrapper
    class StreamableLCELChain:
        def invoke(self, inputs):
            context = retrieve_chain.invoke(inputs)
            inputs["context"] = context
            prompt_val = get_qa_prompt(inputs)
            answer = llm.invoke(prompt_val).content
            return {"answer": answer, "context": context}
            
        def stream(self, inputs):
            # Resolve context first
            context = retrieve_chain.invoke(inputs)
            inputs["context"] = context
            # Yield context to match ui.py schema
            yield {"context": context}
            
            # Stream the generated answer chunks
            prompt_val = get_qa_prompt(inputs)
            for chunk in llm.stream(prompt_val):
                yield {"answer": chunk.content}
                
    return StreamableLCELChain()
