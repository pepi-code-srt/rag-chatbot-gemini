"""
Configuration management for RAG Chatbot
Handles environment variables, constants, and settings
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# ============================================================================
# PATHS & DIRECTORIES
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CHROMADB_DIR = DATA_DIR / "chromadb"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, DOCUMENTS_DIR, CHROMADB_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# API KEYS & CREDENTIALS
# ============================================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
NOMIC_API_KEY = os.getenv("NOMIC_API_KEY", "")

if not GOOGLE_API_KEY:
    # Warning instead of raise to allow import without keys
    print("⚠️  GOOGLE_API_KEY not found. Add to .env file")

# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================

CHUNK_CONFIG = {
    "chunk_size": 512,              # Tokens per chunk
    "overlap": 50,                  # Token overlap between chunks
    "method": "semantic",           # "semantic", "recursive", "fixed"
    "min_chunk_size": 256,          # Minimum chunk size
    "max_chunk_size": 1024,         # Maximum chunk size
    "similarity_threshold": 0.85,   # For semantic chunking
}

# ============================================================================
# EMBEDDING CONFIGURATION
# ============================================================================

EMBEDDING_CONFIG = {
    "model_name": "models/text-embedding-004",
    "embedding_dim": 768,
    "batch_size": 32,
    "device": "cpu",                 # Google API doesn't need local GPU
    "normalize_embeddings": True,
}

# ============================================================================
# RETRIEVAL CONFIGURATION (Hybrid Search)
# ============================================================================

RETRIEVAL_CONFIG = {
    "retrieval_type": "hybrid",     # "hybrid", "vector", "bm25"
    "top_k": 5,                     # Number of chunks to retrieve
    "ensemble_weights": {
        "vector": 0.6,              # Weight for vector search
        "bm25": 0.4,                # Weight for keyword search
    },
    "similarity_threshold": 0.5,    # Minimum similarity score
    "use_reranking": True,          # Rerank results for better quality
    "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}

# ============================================================================
# GENERATION CONFIGURATION (Gemini API)
# ============================================================================

GENERATION_CONFIG = {
    "backend": "google",           # "google" or "ollama"
    "model_name": "models/gemini-3-flash-preview",
    "temperature": 0.7,             # Lower = more deterministic
    "max_output_tokens": 1024,
    "top_p": 0.95,
    "top_k": 40,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}

# ============================================================================
# RAGAS EVALUATION CONFIGURATION
# ============================================================================

EVALUATION_CONFIG = {
    "evaluate_faithfulness": True,      # LLM response matches context
    "evaluate_relevance": True,         # Response relevant to query
    "evaluate_coherence": True,         # Response is well-structured
    "faithfulness_threshold": 0.8,      # Minimum score to accept response
    "relevance_threshold": 0.75,
    "use_hallucination_gate": True,
    "hallucination_filter_fraction": 0.15,  # Filter bottom 15% of responses
}

# ============================================================================
# CHROMADB CONFIGURATION
# ============================================================================

CHROMADB_CONFIG = {
    "persist_directory": str(CHROMADB_DIR),
    "collection_name": "rag_documents",
    "metadata_schema": {
        "source": str,              # Document source file
        "page": int,                # Page number
        "chunk_index": int,         # Chunk sequence number
    },
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    "level": "INFO",                # DEBUG, INFO, WARNING, ERROR
    "log_file": LOGS_DIR / "rag_chatbot.log",
    "max_bytes": 10485760,          # 10MB
    "backup_count": 5,
    "format": "structured_json",    # "structured_json" or "text"
}

# ============================================================================
# FASTAPI CONFIGURATION
# ============================================================================

FASTAPI_CONFIG = {
    "title": "RAG Chatbot API",
    "description": "Production-ready RAG system with Gemini, LangChain, ChromaDB",
    "version": "1.0.0",
    "host": "0.0.0.0",
    "port": 8000,
    "reload": False,
    "workers": 4,
}

# ============================================================================
# CONSTANTS
# ============================================================================

SUPPORTED_DOCUMENT_TYPES = [".pdf", ".txt", ".docx", ".md"]
MAX_DOCUMENT_SIZE_MB = 50
MAX_DOCUMENTS_PER_BATCH = 100
CACHE_TTL_SECONDS = 3600  # 1 hour

# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

SYSTEM_PROMPT = """You are an expert AI assistant specializing in document analysis and question answering.

Your responsibilities:
1. Answer questions based ONLY on the provided context
2. If the context doesn't contain relevant information, say so clearly
3. Provide accurate, concise, and well-structured responses
4. Cite specific parts of the context when possible
5. Indicate confidence level in your response (high, medium, low)

Guidelines:
- Be factual and avoid speculation
- If you're uncertain, acknowledge it
- Provide structured responses with clear sections when appropriate
- Keep responses concise but informative"""

EVALUATION_PROMPT = """Evaluate the following response for hallucination and factual consistency.

Question: {question}
Context: {context}
Response: {response}

Rate the response on:
1. Faithfulness (0-1): How well does the response align with the context?
2. Relevance (0-1): How relevant is the response to the question?
3. Hallucination (0-1): How likely is this response to contain made-up information?

Provide a score and brief explanation for each."""

# ============================================================================
# PERFORMANCE TUNING
# ============================================================================

PERFORMANCE_CONFIG = {
    "enable_caching": True,
    "cache_size_mb": 1000,
    "enable_async_indexing": True,
    "batch_indexing_size": 50,
    "timeout_seconds": 30,
}

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration on startup"""
    errors = []
    
    # Check directories
    for directory in [DOCUMENTS_DIR, CHROMADB_DIR]:
        if not directory.exists():
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create directory {directory}: {e}")
    
    # Check chunk configuration
    if CHUNK_CONFIG["chunk_size"] < CHUNK_CONFIG["min_chunk_size"]:
        errors.append("chunk_size must be >= min_chunk_size")
    
    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        raise ValueError("Invalid configuration")
    
    print("✅ Configuration validated successfully")

# Run validation on import
if __name__ != "__main__":
    validate_config()
