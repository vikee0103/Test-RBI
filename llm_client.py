"""
LLM client module for AWS Bedrock integration.
Handles model invocations with error handling and retries.
"""

import json
import logging
import time
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum

from config import AWS_CONFIG, PromptTemplates, ModelProvider, RAGConfig


logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper for AWS Bedrock LLM operations"""
    
    def __init__(self, bedrock_client, config: RAGConfig = None):
        self.client = bedrock_client
        self.config = config or RAGConfig()
        self.logger = logging.getLogger(__name__)
    
    def _retry_on_error(self, func, max_retries: int = AWS_CONFIG.MAX_RETRIES):
        """Decorator-like retry logic"""
        last_error = None
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = AWS_CONFIG.RETRY_BACKOFF ** attempt
                    self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
        
        raise last_error
    
    def embed_text(self, text: str, model_id: str = None) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            model_id: Model ID (uses config default if not specified)
            
        Returns:
            Embedding vector
        """
        if model_id is None:
            model_id = self.config.embedding_model
        
        try:
            # Truncate text if too long (embeddings have token limits)
            max_chars = 10000
            if len(text) > max_chars:
                text = text[:max_chars]
            
            payload = {"inputText": text}
            
            def invoke():
                response = self.client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(payload)
                )
                return json.loads(response["body"].read().decode())
            
            response_body = self._retry_on_error(invoke)
            embedding = response_body.get("embedding", [])
            
            if not embedding:
                self.logger.warning("Empty embedding returned, using zero vector")
                return [0.0] * 1024
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error embedding text: {e}")
            raise
    
    def generate_answer(
        self,
        prompt: str,
        model_id: str = None,
        max_tokens: int = None,
        temperature: float = None
    ) -> str:
        """
        Generate answer using Claude model.
        
        Args:
            prompt: The prompt/question
            model_id: Model ID to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Generated answer text
        """
        if model_id is None:
            model_id = self.config.llm_model
        if max_tokens is None:
            max_tokens = self.config.max_tokens
        if temperature is None:
            temperature = self.config.temperature
        
        try:
            messages = [{"role": "user", "content": prompt}]
            inference_config = {
                "maxTokens": max_tokens,
                "temperature": temperature
            }
            
            def invoke():
                response = self.client.invoke_model(
                    modelId=model_id,
                    messages=messages,
                    inferenceConfig=inference_config
                )
                return response
            
            response = self._retry_on_error(invoke)
            
            # Extract text from response
            for content_block in response.get("content", []):
                if content_block.get("type") == "text":
                    return content_block.get("text", "").strip()
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            raise
    
    def build_rag_prompt(
        self,
        question: str,
        retrieved_docs: List[Tuple[str, Dict, float]],
        use_system_prompt: bool = True
    ) -> str:
        """
        Build a formatted RAG prompt with context and question.
        
        Args:
            question: User question
            retrieved_docs: List of (text, metadata, score) tuples
            use_system_prompt: Whether to include system prompt
            
        Returns:
            Formatted prompt
        """
        # Build context from retrieved docs
        context_parts = []
        
        for i, (doc_text, metadata, score) in enumerate(retrieved_docs, 1):
            filename = metadata.get("filename", "Unknown Document")
            chunk_id = metadata.get("chunk_id", "0")
            
            # Format citation
            citation = f"Source {i}: {filename} (Chunk #{chunk_id}, Relevance: {score:.2f})"
            
            context_parts.append(f"\n{'='*60}\n{citation}\n{'='*60}\n{doc_text}")
        
        context_block = "".join(context_parts)
        
        # Build final prompt
        if use_system_prompt:
            prompt = f"""{PromptTemplates.SYSTEM_PROMPT}

{PromptTemplates.CONTEXT_PROMPT.format(
    context=context_block,
    question=question
)}"""
        else:
            prompt = f"""{context_block}

Question: {question}

Please provide a detailed answer based on the above context, citing sources."""
        
        return prompt
    
    def answer_question(
        self,
        question: str,
        retrieved_docs: List[Tuple[str, Dict, float]]
    ) -> Dict[str, Any]:
        """
        Answer a question based on retrieved documents.
        
        Args:
            question: User question
            retrieved_docs: Retrieved documents from vector search
            
        Returns:
            Dict with answer, confidence, and citations
        """
        try:
            # Check if we have relevant documents
            if not retrieved_docs:
                return {
                    "answer": "No relevant RBI documents found to answer your question.",
                    "confidence": 0.0,
                    "citations": [],
                    "sources": []
                }
            
            # Build prompt
            prompt = self.build_rag_prompt(question, retrieved_docs)
            
            # Generate answer
            answer = self.generate_answer(prompt)
            
            # Extract citations
            citations = []
            sources = []
            for doc_text, metadata, score in retrieved_docs:
                citations.append({
                    "source": metadata.get("filename", "Unknown"),
                    "chunk_id": metadata.get("chunk_id", 0),
                    "relevance_score": float(score)
                })
                sources.append(metadata.get("filename", "Unknown"))
            
            # Calculate average confidence (relevance score)
            confidence = sum(score for _, _, score in retrieved_docs) / len(retrieved_docs)
            
            return {
                "answer": answer,
                "confidence": float(confidence),
                "citations": citations,
                "sources": list(set(sources))
            }
            
        except Exception as e:
            self.logger.error(f"Error answering question: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "confidence": 0.0,
                "citations": [],
                "sources": [],
                "error": str(e)
            }


class PromptBuilder:
    """Helper class for building structured prompts"""
    
    @staticmethod
    def build_context_block(
        documents: List[Tuple[str, Dict, float]],
        include_scores: bool = True
    ) -> str:
        """Build formatted context block from retrieved documents"""
        context_lines = []
        
        for i, (text, metadata, score) in enumerate(documents, 1):
            header = f"\n[Context {i}]"
            source = f"Document: {metadata.get('filename', 'N/A')}"
            chunk = f"Chunk ID: {metadata.get('chunk_id', 'N/A')}"
            
            if include_scores:
                relevance = f"Relevance Score: {score:.3f}"
                header = f"{header} - {relevance}"
            
            content = f"{header}\n{source}\n{chunk}\n\n{text}"
            context_lines.append(content)
        
        return "\n" + "="*70 + "".join(context_lines)
    
    @staticmethod
    def build_compliance_prompt(
        question: str,
        retrieved_docs: List[Tuple[str, Dict, float]]
    ) -> str:
        """Build compliance-focused prompt for RBI advisories"""
        context = PromptBuilder.build_context_block(retrieved_docs)
        
        return f"""As an RBI compliance expert, answer the following question using only the provided regulatory documents.

{context}

Question: {question}

Your response should:
1. Cite the specific regulatory clause or advisory
2. Highlight compliance requirements
3. Include any applicable deadlines or conditions
4. Note any exceptions or special provisions

Answer:"""
    
    @staticmethod
    def build_interpretation_prompt(
        question: str,
        retrieved_docs: List[Tuple[str, Dict, float]]
    ) -> str:
        """Build interpretation-focused prompt for clarifications"""
        context = PromptBuilder.build_context_block(retrieved_docs)
        
        return f"""Provide a clear interpretation of the RBI regulation based on the provided context.

{context}

Question: {question}

Please explain:
1. The main requirement or guideline
2. Who it applies to
3. Practical implications
4. Related regulatory references

Answer:"""
