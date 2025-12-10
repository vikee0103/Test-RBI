"""
Utility functions module for RBI PDF Q&A System.
Provides helpers for logging, file operations, and data processing.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import hashlib


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def setup_logging(log_dir: str = "./logs", level: str = "INFO") -> logging.Logger:
    """
    Setup comprehensive logging configuration.
    
    Args:
        log_dir: Directory for log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("RBI_RAG_System")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler
    log_file = os.path.join(
        log_dir,
        f"rbi_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatters
    file_formatter = logging.Formatter(
        '[%(asctime)s] %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def get_pdf_files(folder_path: str) -> List[str]:
    """
    Get all PDF files from a folder.
    
    Args:
        folder_path: Path to folder
        
    Returns:
        List of PDF file paths
    """
    if not os.path.exists(folder_path):
        return []
    
    pdf_files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(folder_path, file))
    
    return sorted(pdf_files)


def save_metadata(metadata: Dict[str, Any], file_path: str):
    """
    Save metadata to JSON file.
    
    Args:
        metadata: Dictionary to save
        file_path: Path to save file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def load_metadata(file_path: str) -> Dict[str, Any]:
    """
    Load metadata from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary with metadata
    """
    if not os.path.exists(file_path):
        return {}
    
    with open(file_path, 'r') as f:
        return json.load(f)


def get_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """
    Calculate SHA256 hash of a file.
    Useful for detecting document changes.
    
    Args:
        file_path: Path to file
        chunk_size: Size of chunks to read
        
    Returns:
        Hex digest of SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()


# ============================================================================
# TEXT PROCESSING
# ============================================================================

def chunk_text_simple(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Simple text chunking by character count.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length - len(suffix)]
    return truncated.rsplit(' ', 1)[0] + suffix


def extract_sentences(text: str) -> List[str]:
    """
    Extract sentences from text.
    
    Args:
        text: Text to process
        
    Returns:
        List of sentences
    """
    import re
    
    # Split by sentence delimiters
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


# ============================================================================
# DATA VALIDATION & FORMATTING
# ============================================================================

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        'chunk_size',
        'top_k_retrieval',
        'embedding_model',
        'llm_model'
    ]
    
    for field in required_fields:
        if field not in config:
            return False
    
    # Validate value ranges
    if not (100 < config['chunk_size'] < 10000):
        return False
    
    if not (1 <= config['top_k_retrieval'] <= 20):
        return False
    
    return True


def format_answer_with_citations(
    answer: str,
    citations: List[Dict[str, Any]]
) -> str:
    """
    Format answer with citations.
    
    Args:
        answer: The answer text
        citations: List of citation dictionaries
        
    Returns:
        Formatted answer with citations
    """
    formatted = answer + "\n\n**Citations:**\n"
    
    for i, citation in enumerate(citations, 1):
        source = citation.get('source', 'Unknown')
        chunk = citation.get('chunk_id', 0)
        score = citation.get('relevance_score', 0.0)
        
        formatted += f"\n[{i}] {source} (Chunk #{chunk}) - Relevance: {score:.2%}"
    
    return formatted


# ============================================================================
# METRICS & STATISTICS
# ============================================================================

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate statistics for a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Dictionary with statistics
    """
    if not values:
        return {}
    
    import statistics
    
    return {
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'stdev': statistics.stdev(values) if len(values) > 1 else 0,
        'min': min(values),
        'max': max(values),
        'count': len(values)
    }


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = seconds / 60
        return f"{minutes:.1f}m"


def estimate_tokens(text: str, model: str = "claude") -> int:
    """
    Estimate token count for text.
    Rough estimation: 1 token ≈ 4 characters
    
    Args:
        text: Text to estimate
        model: Model type (affects ratio)
        
    Returns:
        Estimated token count
    """
    # Rough approximations
    if model == "claude":
        # Claude: ~1 token per 3-4 characters
        return len(text) // 4
    else:
        # Titan/Generic: ~1 token per 4 characters
        return len(text) // 4


# ============================================================================
# DOCUMENT PROCESSING HELPERS
# ============================================================================

def extract_metadata_from_filename(filename: str) -> Dict[str, str]:
    """
    Extract metadata from filename.
    Assumes format: YYYY_MM_DD_DocumentType_Description.pdf
    
    Args:
        filename: Filename to parse
        
    Returns:
        Dictionary with extracted metadata
    """
    from pathlib import Path
    
    stem = Path(filename).stem
    parts = stem.split('_')
    
    metadata = {
        'filename': filename,
        'original_name': stem
    }
    
    if len(parts) >= 3:
        metadata['date'] = f"{parts[0]}-{parts[1]}-{parts[2]}"
    
    if len(parts) >= 4:
        metadata['type'] = parts[3]
    
    return metadata


def merge_metadata(base: Dict, updates: Dict) -> Dict:
    """
    Merge metadata dictionaries, with updates taking precedence.
    
    Args:
        base: Base metadata
        updates: Updates to apply
        
    Returns:
        Merged metadata
    """
    merged = base.copy()
    merged.update(updates)
    return merged


# ============================================================================
# ERROR HANDLING
# ============================================================================

class RAGSystemError(Exception):
    """Base exception for RAG system"""
    pass


class DocumentProcessingError(RAGSystemError):
    """Raised when document processing fails"""
    pass


class EmbeddingError(RAGSystemError):
    """Raised when embedding generation fails"""
    pass


class RetrievalError(RAGSystemError):
    """Raised when document retrieval fails"""
    pass


def safe_json_load(json_str: str, default: Any = None) -> Any:
    """
    Safely load JSON string with fallback.
    
    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


# ============================================================================
# DEBUGGING UTILITIES
# ============================================================================

def print_config(config: Dict[str, Any], prefix: str = ""):
    """
    Print configuration in readable format.
    
    Args:
        config: Configuration dictionary
        prefix: Prefix for each line
    """
    print(f"\n{prefix}Configuration:")
    for key, value in config.items():
        if not isinstance(value, (dict, list)):
            print(f"{prefix}  {key}: {value}")
        else:
            print(f"{prefix}  {key}: [complex]")


def debug_retrieval(
    query: str,
    retrieved_docs: List[Tuple[str, Dict, float]],
    logger: logging.Logger = None
):
    """
    Debug information about retrieval results.
    
    Args:
        query: Original query
        retrieved_docs: Retrieved documents
        logger: Logger instance
    """
    log_func = logger.info if logger else print
    
    log_func(f"Query: {query}")
    log_func(f"Retrieved {len(retrieved_docs)} documents:")
    
    for i, (text, metadata, score) in enumerate(retrieved_docs, 1):
        log_func(f"  [{i}] {metadata.get('filename', 'Unknown')} - Score: {score:.3f}")
        log_func(f"      Chunk #{metadata.get('chunk_id', 0)} - {len(text)} chars")
