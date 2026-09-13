import os
import sys
import time

# Add the parent directory to sys.path so we can import the locally built vectorforge module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# On Windows, if compiled with MinGW, we need the DLL directory
if os.name == 'nt':
    try:
        os.add_dll_directory(r"C:\msys64\ucrt64\bin")
    except Exception:
        pass

import streamlit as st
import fitz # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq
from dotenv import load_dotenv

import vectorforge

# Setup
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="VectorForge Hybrid RAG", page_icon="⚡", layout="wide")
st.title("⚡ VectorForge Hybrid RAG (Enterprise Demo)")
st.write("Demonstrating Hybrid Search (RRF), Dynamic CRUD, and Metadata Filtering!")

# Caching the embedding model so it doesn't reload on every UI interaction
@st.cache_resource(show_spinner="Loading Embedding Model (Downloading on first run)...")
def load_embedder():
    # 384-dimensional embeddings
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_tokenizer():
    # Sparse keyword tokenizer
    return TfidfVectorizer()

# Categories to bitmask mapping
CATEGORY_MAP = {
    "General": 1,
    "Finance": 2,
    "Technical": 4,
    "Legal": 8
}

# Initialize session state for our custom VectorDB and Chat History
if "vdb_hybrid" not in st.session_state:
    dim = 384
    st.session_state.vdb_vamana = vectorforge.VamanaIndex(dimension=dim, max_degree=32)
    st.session_state.vdb_delta = vectorforge.DeltaIndex(st.session_state.vdb_vamana)
    st.session_state.vdb_sparse = vectorforge.SparseIndex()
    st.session_state.vdb_hybrid = vectorforge.HybridIndex(st.session_state.vdb_delta, st.session_state.vdb_sparse)
    
    st.session_state.chunks = []
    st.session_state.doc_metadata = [] # Store text and metadata for rendering
    st.session_state.chunk_id_counter = 0
    st.session_state.fitted_tfidf = False

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for PDF upload & CRUD
with st.sidebar:
    st.header("1. Document Library (CRUD)")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    category = st.selectbox("Category (Metadata Filter)", list(CATEGORY_MAP.keys()))
    mask = CATEGORY_MAP[category]
    
    if uploaded_file is not None and st.button("Process & Add to Index"):
        with st.spinner("Extracting text..."):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            
            chunk_size = 1000
            overlap = 200
            new_chunks = []
            start = 0
            while start < len(text):
                new_chunks.append(text[start:start+chunk_size])
                start += chunk_size - overlap
                
        with st.spinner(f"Embedding & Tokenizing {len(new_chunks)} chunks..."):
            embedder = load_embedder()
            tokenizer = load_tokenizer()
            
            # Embed Dense
            embeddings = embedder.encode(new_chunks)
            
            # Accumulate all chunks for TF-IDF fitting (simplified for demo)
            st.session_state.chunks.extend(new_chunks)
            sparse_matrix = tokenizer.fit_transform(st.session_state.chunks)
            st.session_state.fitted_tfidf = True
            
            # Insert into VectorForge Delta (CRUD) and Sparse
            for i, emb in enumerate(embeddings):
                c_id = st.session_state.chunk_id_counter
                
                # Add to Dense (with metadata mask)
                st.session_state.vdb_delta.add(c_id, emb.tolist())
                
                # Add to Sparse
                row = sparse_matrix[c_id]
                sparse_dict = {int(k): float(v) for k, v in zip(row.indices, row.data)}
                st.session_state.vdb_sparse.add(c_id, sparse_dict)
                
                # Store metadata for display
                st.session_state.doc_metadata.append({"id": c_id, "text": new_chunks[i], "category": category})
                st.session_state.chunk_id_counter += 1
                
            st.success(f"Added {len(new_chunks)} chunks to {category}!")

    st.divider()
    st.header("2. Search Options")
    filter_categories = st.multiselect("Filter by Categories", list(CATEGORY_MAP.keys()))
    search_mask = 0
    for cat in filter_categories:
        search_mask |= CATEGORY_MAP[cat]

# Chat Interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the indexed documents..."):
    if not GROQ_API_KEY or GROQ_API_KEY == "gsk_your_api_key_here":
        st.error("Please add your GROQ_API_KEY to the .env file in the rag_demo folder!")
        st.stop()
        
    if st.session_state.chunk_id_counter == 0:
        st.error("Please upload and index a PDF first!")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Searching VectorForge (Hybrid RRF)..."):
            # Embed the user query
            embedder = load_embedder()
            tokenizer = load_tokenizer()
            
            query_emb = embedder.encode(prompt)
            
            # Sparse Query
            sparse_q_dict = {}
            if st.session_state.fitted_tfidf:
                q_vec = tokenizer.transform([prompt])
                sparse_q_dict = {int(k): float(v) for k, v in zip(q_vec.indices, q_vec.data)}
            
            # Search VectorForge Hybrid
            opts = vectorforge.SearchOptions()
            opts.top_k = 3
            opts.filter_mask = search_mask
            
            results = st.session_state.vdb_hybrid.search(query_emb.tolist(), sparse_q_dict, opts)
            
            # Retrieve the chunks
            context_chunks = []
            for res in results:
                for meta in st.session_state.doc_metadata:
                    if meta["id"] == res.id:
                        context_chunks.append(f"[{meta['category']}] {meta['text']}")
                        break
                        
            context_str = "\n\n---\n\n".join(context_chunks)
            
            with st.expander("🔍 View Retrieved Context (RRF Merged)"):
                if not results:
                    st.write("No results found. Check your category filters!")
                for i, res in enumerate(results):
                    st.markdown(f"**Result {i+1} (ID: {res.id}, RRF Distance: {res.distance:.4f})**")
                    st.write(context_chunks[i] if i < len(context_chunks) else "Unknown")
            
        if results:
            with st.spinner("Generating answer with Groq..."):
                client = Groq(api_key=GROQ_API_KEY)
                
                system_prompt = (
                    "You are a helpful assistant. Use the following context retrieved from a document "
                    "to answer the user's question. If you don't know the answer based on the context, "
                    "just say you don't know.\n\nContext:\n" + context_str
                )
                
                completion = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_completion_tokens=1024,
                    top_p=1,
                    stream=True,
                    stop=None
                )
                
                response_placeholder = st.empty()
                full_response = ""
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
