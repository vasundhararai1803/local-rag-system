import os
import glob
import uuid
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama

from config.settings import settings
from src.engine import LocalRAGEngine
from src.exceptions import OffTopicException

# Load env variables
load_dotenv()

os.makedirs(settings.data_dir, exist_ok=True)

st.set_page_config(page_title="Local RAG System", page_icon="🤖")

if "user_session_id" not in st.session_state:
    st.session_state.user_session_id = f"user_{uuid.uuid4().hex}"

@st.cache_resource(show_spinner=False)
def get_rag_chain(session_id):
    return LocalRAGEngine(session_id, history_aware=True)

def process_uploaded_files(uploaded_files, session_id):
    if not uploaded_files:
        return
    with st.spinner("Processing documents..."):
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=settings.vector_store_url)
            try:
                client.delete_collection(session_id)
            except Exception:
                pass
            # Clear previous files in data directory to prevent re-ingestion
            user_data_dir = os.path.join(settings.data_dir, session_id)
            if os.path.exists(user_data_dir):
                for old_file in glob.glob(f"{user_data_dir}/*"):
                    if os.path.isfile(old_file):
                        os.remove(old_file)
        except Exception as e:
            st.warning(f"Could not cleanly reset previous data: {e}")

        user_data_dir = os.path.join(settings.data_dir, session_id)
        os.makedirs(user_data_dir, exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = os.path.join(user_data_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Clear only the isolated cached chain instance so it reloads with new docs
        get_rag_chain.clear(session_id)
        st.toast("Documents processed successfully!", icon="✅")

# --- Main App ---
st.title("🤖 Local RAG System")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Sources Cited"):
                for src in message["sources"]:
                    if isinstance(src, dict):
                        st.markdown(f"**{src['source']}**")
                        preview = src['content'][:300] + "..." if len(src['content']) > 300 else src['content']
                        st.info(preview)
                    else:
                        # Fallback for old history format
                        st.write(f"- {src}")

# React to user input
if prompt_data := st.chat_input("Ask a question about your documents...", accept_file="multiple"):
    
    # Extract files and text safely
    if hasattr(prompt_data, "files"):
        files = prompt_data.files
        prompt_text = prompt_data.text
    elif isinstance(prompt_data, dict):
        files = prompt_data.get("files", [])
        prompt_text = prompt_data.get("text", "")
    else:
        files = []
        prompt_text = prompt_data

    # Auto-process attached files
    if files:
        process_uploaded_files(files, st.session_state.user_session_id)
        
    if prompt_text:
        rag_chain = get_rag_chain(st.session_state.user_session_id)
        
        if getattr(rag_chain, "vectorstore", None) == "NO_DOCS":
            st.warning("⚠️ Please attach a PDF or text document first using the '+' button before asking questions!")
        else:
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt_text)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt_text})
    
            # Get response
            unique_sources = []
            
            with st.chat_message("assistant"):
                try:
                    # Construct chat history for the chain
                    chat_history = []
                    for msg in st.session_state.messages[:-1]: # Exclude the current prompt we just appended
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        else:
                            chat_history.append(AIMessage(content=msg["content"]))
                            
                    context_docs = []
                    
                    def generate_response():
                        for chunk in rag_chain.stream({"input": prompt_text, "chat_history": chat_history}):
                            if "context" in chunk:
                                context_docs.extend(chunk["context"])
                            if "answer" in chunk:
                                yield chunk["answer"]
                    
                    with st.spinner("Thinking..."):
                        response = st.write_stream(generate_response())
                    
                    if context_docs:
                        unique_sources = []
                        seen_chunks = set()
                        for doc in context_docs:
                            source = doc.metadata.get("source", "Unknown Source")
                            page = doc.metadata.get("page")
                            src_str = f"{source} (Page {page})" if page is not None else f"{source}"
                            chunk_text = doc.page_content
                            if chunk_text not in seen_chunks:
                                unique_sources.append({"source": src_str, "content": chunk_text})
                                seen_chunks.add(chunk_text)
                except OffTopicException:
                    response = "I am a professional document assistant and cannot answer off-topic queries."
                    st.markdown(response)
                except ValueError as ve:
                    response = f"An error occurred: {ve}"
                    st.markdown(response)
                except Exception as e:
                    response = f"An error occurred: {e}"
                    st.markdown(response)

            if unique_sources:
                with st.expander("🔍 View Sources Cited"):
                    for src in unique_sources:
                        st.markdown(f"**{src['source']}**")
                        preview = src['content'][:300] + "..." if len(src['content']) > 300 else src['content']
                        st.info(preview)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response, "sources": unique_sources})
