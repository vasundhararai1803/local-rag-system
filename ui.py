import streamlit as st
import os
import uuid
from src.engine import LocalRAGEngine
from src.exceptions import OffTopicException

# 1. Page Configuration & Custom CSS Injection
st.set_page_config(page_title="Enterprise Local RAG", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; color: #1A1D20; }
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E9ECEF; }
    .stChatMessage { border-radius: 12px; background-color: #FFFFFF; margin-bottom: 10px; padding: 15px; border: 1px solid #E9ECEF; }
    .user-card { background-color: #F1F5F9; border: 1px solid #CBD5E1; padding: 15px; border-radius: 12px; margin-left: auto; max-width: 80%; }
    .file-chip { background-color: #E2E8F0; border-radius: 20px; padding: 4px 12px; display: inline-block; font-size: 0.8rem; margin-bottom: 8px; color: #475569; }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_files" not in st.session_state:
    st.session_state.active_files = set()

# Sidebar Setup & Active Context Library
st.sidebar.title("Settings")
st.sidebar.markdown("### Active Context")
if st.session_state.active_files:
    for f in st.session_state.active_files:
        st.sidebar.markdown(f"- {f}")
    if st.sidebar.button("Clear Vector Database", use_container_width=True):
        st.session_state.active_files.clear()
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.messages = []
        st.rerun()
else:
    st.sidebar.info("No documents ingested yet.")
st.sidebar.markdown("---")

# Instantiate Engine safely
@st.cache_resource
def get_engine():
    return LocalRAGEngine(session_id=st.session_state.session_id)

try:
    engine = get_engine()
except Exception:
    st.error("Could not connect to backend vector database or LLM services. Please verify Docker containers are running.")
    st.stop()

# Header Display
st.title("Local Enterprise RAG Platform")
st.caption(f"Session Token Secured: `{st.session_state.session_id}`")
st.markdown("---")

# Render Timeline
clicked_prompt = None
if not st.session_state.messages:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### Welcome to Local Enterprise RAG")
    st.markdown("Select a starter prompt or type your own below.")
    col1, col2, col3 = st.columns(3)
    if col1.button("Summarize key findings", use_container_width=True):
        clicked_prompt = "Summarize the key findings in the documents."
    if col2.button("Extract financial data", use_container_width=True):
        clicked_prompt = "Extract any financial data and format as a table."
    if col3.button("Find action items", use_container_width=True):
        clicked_prompt = "List all action items mentioned in the text."
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                for f in msg.get("files", []):
                    st.caption(f"File: {f}")
                st.markdown(msg["content"])
            else:
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("View Grounding Sources"):
                        for i, doc_text in enumerate(msg["sources"][:3]):
                            st.markdown(f"**Source {i+1}**")
                            st.info(doc_text[:300] + "...")

# Bottom Command Interface Tray
st.markdown("<br><br>", unsafe_allow_html=True)
with st.container():
    st.markdown("### Context Control & Prompt Execution")
    
    # --- FIX: Remove the invalid accept_file parameter entirely ---
    uploaded_files = st.file_uploader(
        "Drop context files here to ingest into your secure Qdrant session container", 
        accept_multiple_files=True, 
        label_visibility="collapsed"
    )

    # Keep chat_input standard and clean
    prompt_data = st.chat_input("Ask a question about your local documents...")

    if prompt_data or clicked_prompt:
        active_filenames = []
        prompt_text = prompt_data if prompt_data else clicked_prompt
        
        # 1. Process files instantly if staged
        if uploaded_files:
            for f in uploaded_files:
                active_filenames.append(f.name)
                st.session_state.active_files.add(f.name)
                try:
                    file_path = os.path.join(engine.user_data_dir, f.name)
                    with open(file_path, "wb") as f_out:
                        f_out.write(f.read())
                except Exception as e:
                    st.error(f"Failed to save file {f.name}: {str(e)}")
                    st.stop()
            
            # Re-run the engine's ingestion pipeline
            engine.vectorstore = engine._load_and_verify_documents()
            if engine.vectorstore != "NO_DOCS":
                engine.retriever = engine.vectorstore.as_retriever(search_kwargs={"k": 3}) # defaulting k
                engine._setup_lcel_graph()
                    
        # 2. Commit user state instantly to screen canvas
        if prompt_text or active_filenames:
            st.session_state.messages.append({
                "role": "user",
                "content": prompt_text,
                "files": active_filenames
            })
            st.rerun()

# Execute model text inference streaming if user just updated the message timeline
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_prompt = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        context_docs = []
        
        try:
            with st.spinner("Analyzing isolated Qdrant vector space..."):
                for chunk in engine.stream({"input": last_prompt}):
                    if "context" in chunk:
                        context_docs.extend(chunk["context"])
                    if "answer" in chunk:
                        full_response += chunk["answer"]
                        placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
            
            sources = []
            if context_docs:
                with st.expander("View Grounding Sources"):
                    for i, doc in enumerate(context_docs[:3]):
                        st.markdown(f"**Source {i+1}**")
                        st.info(doc.page_content[:300] + "...")
                        sources.append(doc.page_content)
                        
            st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": sources})
        except OffTopicException:
            placeholder.warning("This query is outside the scope of your ingested document context bounds.")
        except Exception as e:
            placeholder.error(f"Network Connection Lost during live stream: {str(e)}")