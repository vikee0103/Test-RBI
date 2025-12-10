"""
Core RAG engine module with Chroma vector store integration.
Handles document processing, embedding, and retrieval.
"""

import os
import json
import logging
import time
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
import re

from PyPDF2 import PdfReader
import numpy as np
import chroma

from config import RAGConfig, PromptTemplates


logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles PDF extraction and text preprocessing"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text and metadata from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dict with text, metadata, and success status
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            reader = PdfReader(pdf_path)
            texts = []
            metadata = {
                "filename": os.path.basename(pdf_path),
                "path": pdf_path,
                "page_count": len(reader.pages),
                "processed_at": datetime.now().isoformat(),
                "source": "RBI_Document"
            }
            
            # Extract document title if available
            if reader.metadata:
                metadata["title"] = reader.metadata.get("/Title", "")
                metadata["author"] = reader.metadata.get("/Author", "")
            
            # Extract text from each page
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text() or ""
                    # Normalize whitespace but preserve structure
                    text = re.sub(r'\n\s*\n', '\n\n', text)
                    text = re.sub(r'[ \t]+', ' ', text)
                    texts.append(text.strip())
                except Exception as e:
                    self.logger.warning(f"Error extracting page {page_num} from {pdf_path}: {e}")
            
            full_text = "\n".join(texts)
            
            if not full_text.strip():
                raise ValueError(f"No text extracted from {pdf_path}")
            
            return {
                "success": True,
                "text": full_text,
                "metadata": metadata,
                "page_count": len(texts)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract PDF {pdf_path}: {e}")
            return {
                "success": False,
                "text": "",
                "metadata": {"filename": os.path.basename(pdf_path)},
                "error": str(e)
            }
    
    def chunk_text(self, text: str, doc_metadata: Dict) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks with metadata.
        Respects semantic boundaries (sentences, sections).
        
        Args:
            text: Full document text
            doc_metadata: Document metadata to attach
            
        Returns:
            List of chunk dicts with text and metadata
        """
        if not text:
            return []
        
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 < chunk_size:
                current_chunk = (current_chunk + " " + sentence).strip()
            else:
                if current_chunk:
                    chunk_metadata = doc_metadata.copy()
                    chunk_metadata["chunk_id"] = chunk_id
                    chunk_metadata["chunk_size"] = len(current_chunk)
                    
                    chunks.append({
                        "text": current_chunk,
                        "metadata": chunk_metadata
                    })
                    chunk_id += 1
                
                # Add overlap from end of previous chunk
                if overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = (overlap_text + " " + sentence).strip()
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk:
            chunk_metadata = doc_metadata.copy()
            chunk_metadata["chunk_id"] = chunk_id
            chunk_metadata["chunk_size"] = len(current_chunk)
            chunks.append({
                "text": current_chunk,
                "metadata": chunk_metadata
            })
        
        self.logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def process_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Process all PDFs in a folder.
        
        Args:
            folder_path: Path to folder containing PDFs
            
        Returns:
            List of processed chunks
        """
        if not os.path.exists(folder_path):
            self.logger.error(f"Folder not found: {folder_path}")
            return []
        
        all_chunks = []
        pdf_files = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {folder_path}")
            return []
        
        self.logger.info(f"Processing {len(pdf_files)} PDF files")
        
        for i, pdf_file in enumerate(pdf_files, 1):
            self.logger.info(f"[{i}/{len(pdf_files)}] Processing {pdf_file}")
            
            pdf_path = os.path.join(folder_path, pdf_file)
            extraction_result = self.extract_text_from_pdf(pdf_path)
            
            if extraction_result["success"]:
                chunks = self.chunk_text(
                    extraction_result["text"],
                    extraction_result["metadata"]
                )
                all_chunks.extend(chunks)
            else:
                self.logger.error(f"Failed to process {pdf_file}")
        
        self.logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks


class ChromaVectorStore:
    """Manages Chroma vector database operations"""
    
    def __init__(self, config: RAGConfig, bedrock_client):
        self.config = config
        self.bedrock_client = bedrock_client
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.collection = None
        self._initialize_chroma()
    
    def _initialize_chroma(self):
        """Initialize Chroma client and collection"""
        try:
            # Use persistent client
            self.client = chroma.PersistentClient(path=self.config.chroma_db_dir)
            self.logger.info(f"Initialized Chroma client at {self.config.chroma_db_dir}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Chroma: {e}")
            raise
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            # Try to get existing collection
            self.collection = self.client.get_or_create_collection(
                name=self.config.chroma_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.logger.info(f"Loaded/created collection: {self.config.chroma_collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to get/create collection: {e}")
            raise
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Add processed chunks to Chroma with embeddings.
        
        Args:
            chunks: List of chunk dicts with text and metadata
            
        Returns:
            Number of documents added
        """
        if not self.collection:
            self._get_or_create_collection()
        
        if not chunks:
            self.logger.warning("No chunks to add")
            return 0
        
        try:
            # Prepare data for Chroma
            ids = []
            documents = []
            metadatas = []
            embeddings = []
            
            self.logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self._embed_text(chunk["text"])
                
                # Prepare Chroma entry
                doc_id = f"{chunk['metadata']['filename']}_{chunk['metadata']['chunk_id']}"
                ids.append(doc_id)
                documents.append(chunk["text"])
                
                # Flatten metadata for Chroma (no nested dicts)
                flat_metadata = {
                    "filename": str(chunk["metadata"].get("filename", "")),
                    "chunk_id": str(chunk["metadata"].get("chunk_id", 0)),
                    "page_count": str(chunk["metadata"].get("page_count", "")),
                    "chunk_size": str(chunk["metadata"].get("chunk_size", 0)),
                    "source": "RBI_Document"
                }
                metadatas.append(flat_metadata)
                embeddings.append(embedding)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Embedded {i + 1}/{len(chunks)} chunks")
            
            # Add to Chroma
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            self.logger.info(f"Successfully added {len(chunks)} chunks to Chroma")
            return len(chunks)
            
        except Exception as e:
            self.logger.error(f"Error adding documents to Chroma: {e}")
            raise
    
    def search(self, query: str, top_k: int = None) -> List[Tuple[str, Dict, float]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (text, metadata, score) tuples
        """
        if not self.collection:
            self._get_or_create_collection()
        
        if top_k is None:
            top_k = self.config.top_k_retrieval
        
        try:
            # Get query embedding
            query_embedding = self._embed_text(query)
            
            # Search in Chroma
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["embeddings", "documents", "metadatas", "distances"]
            )
            
            # Format results
            retrieved = []
            if results["documents"] and len(results["documents"]) > 0:
                for doc, metadata, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ):
                    # Convert distance to similarity score (cosine distance -> similarity)
                    similarity_score = 1 - distance
                    retrieved.append((doc, metadata, similarity_score))
            
            return retrieved
            
        except Exception as e:
            self.logger.error(f"Error searching Chroma: {e}")
            return []
    
    def _embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using AWS Bedrock"""
        import json
        
        try:
            payload = {"inputText": text}
            response = self.bedrock_client.invoke_model(
                modelId=self.config.embedding_model,
                body=json.dumps(payload)
            )
            
            response_body = json.loads(response["body"].read().decode())
            embedding = response_body.get("embedding", [])
            
            if not embedding:
                self.logger.warning("Empty embedding returned")
                return [0.0] * 1024
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        if not self.collection:
            self._get_or_create_collection()
        
        try:
            count = self.collection.count()
            return {
                "collection_name": self.config.chroma_collection_name,
                "document_count": count,
                "status": "active"
            }
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def clear_collection(self):
        """Clear all documents from collection"""
        try:
            if self.collection:
                # Delete the collection and recreate it
                self.client.delete_collection(name=self.config.chroma_collection_name)
                self._get_or_create_collection()
                self.logger.info("Collection cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing collection: {e}")
            raise


class RAGEngine:
    """Main RAG orchestration engine"""
    
    def __init__(self, bedrock_client, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.bedrock_client = bedrock_client
        self.logger = logging.getLogger(__name__)
        
        self.processor = DocumentProcessor(self.config)
        self.vector_store = ChromaVectorStore(self.config, bedrock_client)
        self.metadata_cache = {}
    
    def ingest_documents(self, folder_path: str) -> Dict[str, Any]:
        """
        Ingest and index all PDFs in a folder.
        
        Args:
            folder_path: Path to folder containing PDFs
            
        Returns:
            Ingestion status and statistics
        """
        try:
            self.logger.info(f"Starting document ingestion from {folder_path}")
            
            # Process documents
            chunks = self.processor.process_folder(folder_path)
            
            if not chunks:
                return {
                    "success": False,
                    "message": "No documents processed",
                    "chunks_created": 0
                }
            
            # Add to vector store
            added_count = self.vector_store.add_documents(chunks)
            
            # Cache metadata
            for chunk in chunks:
                doc_id = chunk["metadata"]["filename"]
                if doc_id not in self.metadata_cache:
                    self.metadata_cache[doc_id] = chunk["metadata"]
            
            return {
                "success": True,
                "message": f"Successfully ingested {len(set(c['metadata']['filename'] for c in chunks))} documents",
                "chunks_created": added_count,
                "total_documents": len(set(c["metadata"]["filename"] for c in chunks))
            }
            
        except Exception as e:
            self.logger.error(f"Error during ingestion: {e}")
            return {
                "success": False,
                "message": str(e)
            }
    
    def retrieve_relevant_docs(self, query: str, top_k: int = None) -> List[Tuple[str, Dict, float]]:
        """Retrieve relevant documents for a query"""
        return self.vector_store.search(query, top_k)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG engine statistics"""
        return self.vector_store.get_collection_stats()
