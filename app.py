import streamlit as st
from PyPDF2 import PdfReader
import re
import hashlib
import os
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# SaaS Styling Custom CSS
def inject_custom_css():
    css = """
    <style>
    /* Custom Font (Outfit) */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title styling */
    h1 {
        color: #4CAF50 !important;
        font-size: 38px !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar styling to prevent scrollbar */
    section[data-testid="stSidebar"] {
        overflow-y: hidden !important; /* Hide sidebar scroll */
    }
    
    /* User Message Bubble background and padding */
    div[data-testid="stChatMessage"][data-testid="user"] {
        background-color: #1E293B !important;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
    }
    
    /* Assistant Message Bubble background, left border, and padding */
    div[data-testid="stChatMessage"][data-testid="assistant"] {
        background-color: #111827 !important;
        padding: 15px;
        border-radius: 12px;
        border-left: 4px solid #10B981 !important;
        margin-bottom: 10px;
    }
    
    /* Explicitly make text inside chat bubbles light gray for maximum visibility */
    div[data-testid="stChatMessage"] p, 
    div[data-testid="stChatMessage"] span, 
    div[data-testid="stChatMessage"] li, 
    div[data-testid="stChatMessage"] ul, 
    div[data-testid="stChatMessage"] ol,
    div[data-testid="stChatMessage"] div,
    div[data-testid="stChatMessage"] strong,
    div[data-testid="stChatMessage"] em,
    div[data-testid="stChatMessage"] h1,
    div[data-testid="stChatMessage"] h2,
    div[data-testid="stChatMessage"] h3 {
        color: #F8FAFC !important;
    }
    
    /* Custom buttons styling (e.g. Clear Chat) */
    .stButton > button {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: bold !important;
    }
    .stButton > button:hover {
        background-color: #45A049 !important;
    }
    
    /* File uploader high contrast labels */
    [data-testid="stFileUploader"] label {
        color: #E2E8F0 !important;
        font-weight: 600;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )
    chunks = splitter.split_documents(documents)
    return chunks

@st.cache_resource
def load_langchain_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

def create_vector_store(chunks):
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=load_langchain_embeddings()
    )
    return vector_store

@st.cache_resource
def load_llm():
    llm = ChatOllama(
        model="llama3",
        temperature=0
    )
    return llm

def format_context_with_sources(retrieved_results):
    context_parts = []
    for doc, score in retrieved_results:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        context_parts.append(f"--- Source: {source}, Page {page} ---\n{doc.page_content}")
    return "\n\n".join(context_parts)

def retrieval_agent(vector_store, query, chat_history):
    # Contextualize query if there is history
    if chat_history:
        history_str = ""
        for msg in chat_history[-3:]: # Keep last 3 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"
            
        rewrite_prompt = f"""
Given the conversation history and the latest user question, formulate a standalone question that can be understood without the conversation history. Do NOT answer the question, just reformulate it and return ONLY the standalone question.

Conversation History:
{history_str}

Latest Question:
{query}

Standalone Question:
"""
        llm = load_llm()
        standalone_query = llm.invoke(rewrite_prompt).content.strip()
        standalone_query = standalone_query.replace('"', '').replace("'", "").strip()
    else:
        standalone_query = query

    results = vector_store.similarity_search_with_score(
        standalone_query,
        k=3
    )
    return results, standalone_query

def answer_agent(query, retrieved_results, chat_history):
    context = format_context_with_sources(retrieved_results)
    
    # Format dialogue history
    history_str = ""
    for msg in chat_history[-5:]: # Keep last 5 messages for context
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    prompt = f"""
You are a helpful document assistant.

Answer the user's question ONLY using the provided context. If the question contains pronouns or relative references (like "its", "they", "this", "that", "the former", or "the latter"), refer to the Conversation History to understand what the user is referring to.

For every factual claim you make, you MUST cite the source document and page number in brackets, for example: [Filename.pdf, Page 4]. Do not invent any sources or page numbers that are not explicitly shown in the Context headers.

Context:
{context}

Conversation History:
{history_str}

Question:
{query}

Answer:
"""
    llm = load_llm()
    response = llm.invoke(prompt)
    return response.content

def verification_agent(query, candidate_answer, retrieved_results):
    context = format_context_with_sources(retrieved_results)
    verification_prompt = f"""
You are a verification assistant.
Your job is to check if the proposed answer is fully supported by the context.

Context:
{context}

Question:
{query}

Proposed Answer:
{candidate_answer}

Instruction:
Review the Proposed Answer against the Context. If the proposed answer contains any statements that are NOT supported by the Context, correct the answer to remove or modify those unsupported statements. Make sure all citations (e.g., [Filename.pdf, Page X]) match the sources in the context exactly. Return ONLY the verified and corrected answer. If it is already fully supported, return the Proposed Answer exactly. Do not add any introductory or meta text, just return the verified answer.

Verified Answer:
"""
    llm = load_llm()
    response = llm.invoke(verification_prompt)
    return response.content

# Helper to compute MD5 hash of input files to detect changes
def get_inputs_hash(uploaded_files, local_path):
    hash_parts = []
    if uploaded_files:
        for f in uploaded_files:
            hash_parts.append(f"{f.name}_{f.size}")
    if local_path:
        if os.path.exists(local_path):
            hash_parts.append(f"{local_path}_{os.path.getmtime(local_path)}")
    return hashlib.md5("".join(hash_parts).encode()).hexdigest()

# Helper to extract pages and return chunked Document objects
def extract_and_chunk_all(uploaded_files, local_path):
    all_docs = []
    
    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                pdf_reader = PdfReader(uploaded_file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        cleaned = clean_text(page_text)
                        all_docs.append(Document(
                            page_content=cleaned,
                            metadata={"source": uploaded_file.name, "page": page_num + 1}
                        ))
            except Exception as e:
                st.sidebar.error(f"Error parsing {uploaded_file.name}: {e}")
                
    # Process local path
    if local_path:
        if os.path.exists(local_path):
            try:
                pdf_reader = PdfReader(local_path)
                filename = os.path.basename(local_path)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        cleaned = clean_text(page_text)
                        all_docs.append(Document(
                            page_content=cleaned,
                            metadata={"source": filename, "page": page_num + 1}
                        ))
            except Exception as e:
                st.sidebar.error(f"Error parsing local path {local_path}: {e}")
                
    if not all_docs:
        return []
        
    return create_chunks(all_docs)


# Main Layout Setup
st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "files_hash" not in st.session_state:
    st.session_state.files_hash = ""
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []


# Sidebar Configuration (Tighter, Scroll-free SaaS Dashboard layout controls)
with st.sidebar:
    st.title("📚 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    local_path = st.text_input(
        "Or specify local PDF path",
        value=""
    )

    # Detect changes and process files
    current_hash = get_inputs_hash(uploaded_files, local_path)
    if current_hash != st.session_state.files_hash:
        st.session_state.files_hash = current_hash
        st.session_state.messages = []  # Clear history on document context change
        
        if current_hash:
            with st.spinner("Processing & indexing..."):
                chunks = extract_and_chunk_all(uploaded_files, local_path)
                st.session_state.chunks = chunks
                if chunks:
                    st.session_state.vector_store = create_vector_store(chunks)
                else:
                    st.session_state.vector_store = None
        else:
            st.session_state.chunks = []
            st.session_state.vector_store = None

    st.divider()
    st.subheader("Statistics")
    
    chunk_count = len(st.session_state.chunks)
    source_files = list(set([doc.metadata.get("source") for doc in st.session_state.chunks]))
    document_count = len(source_files)
    
    # Side-by-side metrics to save space
    col1, col2 = st.columns(2)
    col1.metric("Chunks", chunk_count)
    col2.metric("Documents", document_count)
    
    if st.session_state.vector_store is not None:
        with st.expander("Active Source Files", expanded=False):
            for src in source_files:
                st.write(f"- {src}")
                
    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main View Elements
st.title("🤖 Agentic RAG Assistant")

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Render confidence score badge (SaaS layout standard containers)
            avg_confidence = message.get("confidence_pct", 0.0)
            if avg_confidence >= 80.0:
                st.success(f"Confidence: {avg_confidence:.1f}%")
            elif avg_confidence >= 60.0:
                st.warning(f"Confidence: {avg_confidence:.1f}%")
            else:
                st.error(f"Confidence: {avg_confidence:.1f}%")
            
            # Render verified answer
            st.markdown(message["content"])
            
            # Render sources used inside collapsible details card
            if "sources" in message:
                with st.expander("Retrieved Sources Used"):
                    for src in message["sources"]:
                        st.info(f"📄 Source: {src['source']} | Page {src['page']}")
                        st.caption(src["content"])
        else:
            st.markdown(message["content"])

# Render Greeting if Database Empty
if st.session_state.vector_store is None:
    st.markdown(
        """
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 50vh; text-align: center;">
            <div style="font-size: 5rem; margin-bottom: 1rem;">🤖</div>
            <h2 style="font-family: 'Outfit', sans-serif; font-weight: 700; color: #f8fafc; margin-bottom: 0.5rem;">Ready to Query Your Documents?</h2>
            <p style="color: #94a3b8; max-width: 480px; font-size: 0.95rem; line-height: 1.6; margin-bottom: 2rem;">
                Upload one or multiple PDF files or specify a local PDF path in the sidebar. Once indexing completes, you can run semantic search and multi-agent validation.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Accept Chat Input
query = st.chat_input("Ask a question about the indexed documents...")

if query and st.session_state.vector_store is not None:
    # Render user query
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Process Assistant Response
    with st.chat_message("assistant"):
        # Interactive Multi-Agent execution via st.status
        with st.status("Agentic pipeline executing...", expanded=True) as status:
            st.write("🔍 **Retrieval Agent**: Contextualizing query & searching vector store...")
            retrieved_docs, reformulated_query = retrieval_agent(st.session_state.vector_store, query, st.session_state.messages[:-1])
            if reformulated_query != query:
                st.write(f"↪️ Reformulated relative query to: *\"{reformulated_query}\"*")
            st.write(f"✓ Found {len(retrieved_docs)} relevant source passages.")
            
            st.write("🧠 **Answer Agent**: Generating candidate response with source citations...")
            candidate_answer = answer_agent(query, retrieved_docs, st.session_state.messages[:-1])
            
            st.write("🛡️ **Verification Agent**: Auditing claims and checking grounding...")
            verified_answer = verification_agent(query, candidate_answer, retrieved_docs)
            
            status.update(label="Response generated and verified!", state="complete", expanded=False)
            
        # Calculate Confidence Scores
        total_confidence = 0.0
        scores_to_save = []
        for doc, score in retrieved_docs:
            confidence = (1.0 - (score / 2.0)) * 100
            confidence = max(0.0, min(100.0, confidence))
            total_confidence += confidence
            scores_to_save.append({
                "content": doc.page_content,
                "score": score,
                "confidence": confidence,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown")
            })
            
        avg_confidence = total_confidence / len(retrieved_docs) if retrieved_docs else 0.0
        
        # Render confidence level box
        if avg_confidence >= 80.0:
            st.success(f"Confidence: {avg_confidence:.1f}%")
        elif avg_confidence >= 60.0:
            st.warning(f"Confidence: {avg_confidence:.1f}%")
        else:
            st.error(f"Confidence: {avg_confidence:.1f}%")
            
        # Render verified response
        st.markdown(verified_answer)
        
        # Render sources used
        with st.expander("Retrieved Sources Used"):
            for src in scores_to_save:
                st.info(f"📄 Source: {src['source']} | Page {src['page']}")
                st.caption(src["content"])
                
        # Append response to memory
        st.session_state.messages.append({
            "role": "assistant",
            "content": verified_answer,
            "confidence_pct": avg_confidence,
            "sources": scores_to_save
        })
        
        # Force rerun to cleanly paint session state changes
        st.rerun()
elif query and st.session_state.vector_store is None:
    st.sidebar.warning("Please index documents first before querying.")
