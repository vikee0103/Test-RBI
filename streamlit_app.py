"""
Production-ready Streamlit application for RBI PDF Q&A with RAG.
Interactive interface with advanced features for document management and querying.
"""

import streamlit as st
import logging
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple, Any

from config import RAGConfig, ModelProvider, EmbeddingModel
from rag_engine import RAGEngine, DocumentProcessor, ChromaVectorStore
from llm_client import LLMClient, PromptBuilder
from aws_login import AWSPortalClient


# Configure Streamlit
st.set_page_config(
    page_title="RBI Advisory Q&A System",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize or retrieve session state variables"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "bedrock_client" not in st.session_state:
        st.session_state.bedrock_client = None
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None
    if "llm_client" not in st.session_state:
        st.session_state.llm_client = None
    if "config" not in st.session_state:
        st.session_state.config = RAGConfig()
    if "query_history" not in st.session_state:
        st.session_state.query_history = []


initialize_session_state()


# ============================================================================
# SIDEBAR - AUTHENTICATION & CONFIGURATION
# ============================================================================

def render_authentication_section():
    """Render AWS authentication section"""
    st.sidebar.markdown("## 🔐 AWS Bedrock Authentication")
    
    if not st.session_state.authenticated:
        with st.sidebar.form("auth_form", clear_on_submit=False):
            username = st.text_input("AWS Portal Username")
            password = st.text_input("AWS Portal Password", type="password")
            account_id = st.text_input("AWS Account ID")
            region = st.selectbox(
                "Region",
                ["us-west-2", "us-east-1", "eu-west-1", "ap-southeast-1"],
                index=0
            )
            
            submitted = st.form_submit_button("🔗 Connect to AWS", use_container_width=True)
        
        if submitted:
            if not all([username, password, account_id]):
                st.sidebar.error("⚠️ Please fill all authentication fields")
            else:
                try:
                    with st.spinner("🔄 Authenticating with AWS..."):
                        # Initialize AWS client
                        aws_client = AWSPortalClient(username=username, password=password)
                        token = aws_client.get_token()
                        creds = aws_client.fetch_sts_creds(token, account_id)
                        bedrock_client = aws_client.create_client(creds, "bedrock-runtime", region)
                        
                        st.session_state.authenticated = True
                        st.session_state.bedrock_client = bedrock_client
                        st.session_state.rag_engine = RAGEngine(bedrock_client, st.session_state.config)
                        st.session_state.llm_client = LLMClient(bedrock_client, st.session_state.config)
                        
                        st.sidebar.success("✅ Connected to AWS Bedrock")
                        logger.info("Successfully authenticated with AWS Bedrock")
                        
                except Exception as e:
                    st.sidebar.error(f"❌ Authentication failed: {str(e)}")
                    logger.error(f"Authentication error: {e}")
    
    else:
        st.sidebar.success("✅ Connected to AWS Bedrock")
        if st.sidebar.button("🔓 Disconnect", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.bedrock_client = None
            st.session_state.rag_engine = None
            st.session_state.llm_client = None
            st.rerun()


def render_configuration_section():
    """Render configuration settings"""
    st.sidebar.markdown("## ⚙️ Configuration")
    
    with st.sidebar.expander("📊 RAG Settings", expanded=False):
        st.session_state.config.chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=500,
            max_value=3000,
            value=st.session_state.config.chunk_size,
            step=100
        )
        st.session_state.config.chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=500,
            value=st.session_state.config.chunk_overlap,
            step=50
        )
        st.session_state.config.top_k_retrieval = st.slider(
            "Top-K Results",
            min_value=1,
            max_value=10,
            value=st.session_state.config.top_k_retrieval,
            step=1
        )
    
    with st.sidebar.expander("🤖 Model Settings", expanded=False):
        model_choice = st.selectbox(
            "LLM Model",
            list(ModelProvider),
            index=0
        )
        st.session_state.config.llm_model = model_choice.value
        
        st.session_state.config.max_tokens = st.slider(
            "Max Tokens",
            min_value=256,
            max_value=2048,
            value=st.session_state.config.max_tokens,
            step=128
        )
        
        st.session_state.config.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.config.temperature,
            step=0.05
        )
    
    with st.sidebar.expander("📁 Document Settings", expanded=False):
        st.session_state.config.pdf_folder_path = st.text_input(
            "PDF Folder Path",
            value=st.session_state.config.pdf_folder_path
        )


def render_document_management_section():
    """Render document management controls"""
    st.sidebar.markdown("## 📚 Document Management")
    
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        if st.button("📥 Ingest PDFs", use_container_width=True):
            if not st.session_state.authenticated:
                st.sidebar.error("❌ Please authenticate first")
            elif not os.path.exists(st.session_state.config.pdf_folder_path):
                st.sidebar.error("❌ Folder not found")
            else:
                try:
                    with st.spinner("📂 Processing PDFs..."):
                        result = st.session_state.rag_engine.ingest_documents(
                            st.session_state.config.pdf_folder_path
                        )
                        
                        if result["success"]:
                            st.sidebar.success(
                                f"✅ {result['chunks_created']} chunks created from "
                                f"{result['total_documents']} documents"
                            )
                            logger.info(f"Ingestion successful: {result}")
                        else:
                            st.sidebar.error(f"❌ {result['message']}")
                            logger.error(f"Ingestion failed: {result}")
                except Exception as e:
                    st.sidebar.error(f"❌ Error: {str(e)}")
                    logger.error(f"Ingestion error: {e}")
    
    with col2:
        if st.button("🔄 Clear Index", use_container_width=True):
            try:
                st.session_state.rag_engine.vector_store.clear_collection()
                st.sidebar.success("✅ Index cleared")
                logger.info("Index cleared successfully")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")
                logger.error(f"Clear error: {e}")
    
    with col3:
        if st.button("📊 Stats", use_container_width=True):
            try:
                stats = st.session_state.rag_engine.get_stats()
                st.sidebar.info(f"📊 Documents in index: {stats.get('document_count', 0)}")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")


# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

def render_header():
    """Render application header"""
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1>📋 RBI Advisory Q&A System</h1>
        <p style='color: #666; font-size: 1.1rem;'>
            Intelligent document retrieval and analysis using RAG
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_question_section():
    """Render Q&A section"""
    st.markdown("## ❓ Ask Questions About RBI Documents")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_area(
            "Your Question:",
            placeholder="e.g., What are the latest KYC requirements under RBI guidelines?",
            height=100,
            label_visibility="collapsed"
        )
    
    with col2:
        st.write("")  # Spacing
        submit_button = st.button("🔍 Search", use_container_width=True, type="primary")
    
    return question, submit_button


def process_question(question: str):
    """Process user question and generate answer"""
    if not st.session_state.authenticated:
        st.error("❌ Please authenticate with AWS Bedrock first")
        return
    
    if not question.strip():
        st.error("❌ Please enter a question")
        return
    
    if st.session_state.rag_engine.get_stats().get("document_count", 0) == 0:
        st.error("❌ No documents in index. Please ingest PDFs first.")
        return
    
    try:
        with st.spinner("🔍 Retrieving relevant documents..."):
            # Retrieve relevant documents
            retrieved_docs = st.session_state.rag_engine.retrieve_relevant_docs(
                question,
                top_k=st.session_state.config.top_k_retrieval
            )
            
            if not retrieved_docs:
                st.warning("⚠️ No relevant documents found for your query")
                return
        
        with st.spinner("🤖 Generating answer..."):
            # Generate answer
            result = st.session_state.llm_client.answer_question(question, retrieved_docs)
        
        # Display results
        display_answer_section(result, retrieved_docs)
        
        # Add to history
        st.session_state.query_history.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": result["answer"],
            "confidence": result["confidence"]
        })
        
        logger.info(f"Question processed successfully: {question[:50]}...")
        
    except Exception as e:
        st.error(f"❌ Error processing question: {str(e)}")
        logger.error(f"Question processing error: {e}")


def display_answer_section(result: Dict, retrieved_docs: List[Tuple[str, Dict, float]]):
    """Display answer and retrieval information"""
    
    # Answer
    st.markdown("### 🤖 Answer")
    st.markdown(result["answer"])
    
    # Confidence and sources
    col1, col2, col3 = st.columns(3)
    with col1:
        confidence_percent = result["confidence"] * 100
        st.metric(
            "Confidence Score",
            f"{confidence_percent:.1f}%",
            delta=f"{confidence_percent - 50:.1f}%" if confidence_percent > 50 else None
        )
    with col2:
        st.metric("Sources Used", len(result["sources"]))
    with col3:
        st.metric("Context Chunks", len(retrieved_docs))
    
    # Citations
    st.markdown("### 📚 Citations")
    for citation in result["citations"]:
        with st.expander(
            f"📄 {citation['source']} (Chunk #{citation['chunk_id']}) "
            f"- Score: {citation['relevance_score']:.3f}"
        ):
            # Find and display the actual content
            for doc_text, metadata, score in retrieved_docs:
                if (metadata.get("filename") == citation["source"] and 
                    metadata.get("chunk_id") == citation["chunk_id"]):
                    st.write(doc_text)
                    break
    
    # Divider
    st.markdown("---")


def render_query_history_section():
    """Render query history"""
    if st.session_state.query_history:
        st.markdown("## 📜 Query History")
        
        for i, query in enumerate(reversed(st.session_state.query_history[-5:]), 1):
            with st.expander(f"Query {i}: {query['question'][:50]}..."):
                st.write(f"**Question:** {query['question']}")
                st.write(f"**Answer:** {query['answer'][:200]}...")
                st.write(f"**Confidence:** {query['confidence']:.1%}")
                st.write(f"**Time:** {query['timestamp']}")


# ============================================================================
# DOCUMENT BROWSER
# ============================================================================

def render_document_browser():
    """Render document browser section"""
    st.markdown("## 📖 Document Browser")
    
    try:
        stats = st.session_state.rag_engine.get_stats()
        doc_count = stats.get("document_count", 0)
        
        if doc_count == 0:
            st.info("📭 No documents indexed yet. Please ingest PDFs first.")
            return
        
        st.info(f"📊 Index contains {doc_count} chunks")
        
        # Show instructions
        with st.expander("ℹ️ How to browse documents"):
            st.write("""
            1. Use semantic search to find relevant chunks
            2. Adjust retrieval settings in the sidebar
            3. View extracted text and metadata
            """)
        
    except Exception as e:
        st.error(f"Error retrieving document stats: {str(e)}")


# ============================================================================
# MAIN APPLICATION LOGIC
# ============================================================================

def main():
    """Main application logic"""
    # Render sidebar
    with st.sidebar:
        render_authentication_section()
        
        if st.session_state.authenticated:
            st.divider()
            render_configuration_section()
            st.divider()
            render_document_management_section()
    
    # Render main content
    render_header()
    
    if not st.session_state.authenticated:
        st.info(
            "👋 Welcome! Please authenticate with AWS Bedrock using the sidebar to get started."
        )
        st.markdown("""
        ### Features
        - 📤 **Upload & Process**: Ingest RBI PDF documents
        - 🔍 **Semantic Search**: Find relevant document chunks
        - 🤖 **AI Powered**: Get answers using Claude 3 Sonnet
        - 📚 **Citation Tracking**: See exact document sources
        - ⚙️ **Configurable**: Adjust RAG parameters
        """)
    else:
        # Question section
        question, submit_button = render_question_section()
        
        if submit_button:
            process_question(question)
        
        # Divider
        st.divider()
        
        # Additional sections
        render_document_browser()
        st.divider()
        render_query_history_section()


if __name__ == "__main__":
    main()
