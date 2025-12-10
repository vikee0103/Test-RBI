"""
Configuration module for RBI PDF Q&A RAG System.
Centralizes all settings and model configurations.
"""

import os
from enum import Enum
from typing import Dict, List
from dataclasses import dataclass


class ModelProvider(Enum):
    """Supported LLM providers"""
    CLAUDE = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    CLAUDE_OPUS = "anthropic.claude-opus-v1:0"
    TITAN = "amazon.titan-text-lite-v1"


class EmbeddingModel(Enum):
    """Supported embedding models"""
    TITAN_EMBED_V2 = "amazon.titan-embed-text-v2:0"
    TITAN_EMBED_V1 = "amazon.titan-embed-text-v1:0"


@dataclass
class RAGConfig:
    """Core RAG configuration"""
    chunk_size: int = 1500
    chunk_overlap: int = 200
    top_k_retrieval: int = 5
    max_tokens: int = 1024
    temperature: float = 0.2
    embedding_model: str = EmbeddingModel.TITAN_EMBED_V2.value
    llm_model: str = ModelProvider.CLAUDE.value
    
    # Chroma specific
    chroma_db_dir: str = "./chroma_db"
    chroma_collection_name: str = "rbi_documents"
    
    # Logging and persistence
    log_dir: str = "./logs"
    metadata_cache_file: str = "./cache/metadata.json"
    
    # Document processing
    pdf_folder_path: str = "./rbi_pdfs"
    supported_formats: List[str] = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [".pdf"]
        
        # Create directories if they don't exist
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.chroma_db_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_cache_file), exist_ok=True)


class PromptTemplates:
    """RAG prompt templates"""
    
    SYSTEM_PROMPT = """You are an expert assistant specializing in RBI (Reserve Bank of India) 
advisories, regulations, and compliance requirements. 

Your responsibilities:
1. Provide accurate, context-grounded answers based on the provided RBI documents
2. Always cite the source documents and specific sections when relevant
3. If information is not available in the provided context, clearly state: "This information is not found in the provided RBI documents"
4. Be precise and avoid speculation or external knowledge
5. Structure your response with clear sections and bullet points for readability
6. Highlight any compliance implications or critical requirements
"""

    CONTEXT_PROMPT = """Using the following RBI document excerpts as context, answer the question:

{context}

Question: {question}

Provide a detailed, compliance-focused response with proper citations. If the answer requires information 
not present in the context, explicitly state that."""

    CITATION_FORMAT = "(Source: {doc_name}, Section: {chunk_id})"


class AWS_CONFIG:
    """AWS-specific configurations"""
    DEFAULT_REGION = "us-west-2"
    BEDROCK_SERVICE = "bedrock-runtime"
    TIMEOUT = 300  # seconds
    MAX_RETRIES = 3
    RETRY_BACKOFF = 1.5


class LOG_CONFIG:
    """Logging configuration"""
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    FILE_LOG_FORMAT = "[%(asctime)s] %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
