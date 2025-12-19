"""
Embedding Generation
Creates vector embeddings for documents and queries
"""

import os
import numpy as np
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
import logging
import nomic
from nomic import embed

# Import NOMIC_API_KEY from config to check availability
# We use a try-except block to avoid circular imports if config imports this file
try:
    from config import NOMIC_API_KEY
except ImportError:
    NOMIC_API_KEY = os.getenv("NOMIC_API_KEY", "")

logger = logging.getLogger(__name__)


class LocalNomicEmbeddings(Embeddings):
    """
    Nomic embeddings using local sentence-transformers (CPU/GPU)
    Free, open-source, 768-dimensional embeddings running locally.
    """
    
    def __init__(
        self, 
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        batch_size: int = 32,
        device: str = "cpu"
    ):
        """
        Initialize local embeddings
        
        Args:
            model_name: HuggingFace model identifier
            batch_size: Number of texts to embed at once
            device: "cuda" for GPU, "cpu" for CPU
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        
        logger.info(f"Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        logger.info(f"Local embedding model loaded on {device}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents
        """
        # Prefix for Nomic v1.5 matryoshka models if needed, but usually handled by model
        # For nomic-embed-text-v1.5, it's recommended to add "search_document: " prefix for docs
        # provided we are using the official Nomic model.
        # However, sentence-transformers usage usually handles it if we are careful.
        # Nomic v1.5 recommends prefixes. Let's add them if they are standard.
        # Actually, sentence-transformers implementations often don't enforce prefixes unless specified.
        # We will stick to raw encoding for local to match previous behavior, or safe defaults.
        
        # Note: Nomic v1.5 recommends "search_document: " for documents and "search_query: " for queries
        # We'll apply this for better quality if it is indeed v1.5
        
        prefixed_texts = [f"search_document: {t}" for t in texts]
        
        embeddings = self.model.encode(
            prefixed_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True  # L2 normalization
        )
        
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query
        """
        prefixed_text = f"search_query: {text}"
        
        embedding = self.model.encode(
            [prefixed_text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embedding[0].tolist()


class RemoteNomicEmbeddings(Embeddings):
    """
    Nomic embeddings using Nomic API (Remote)
    Uses nomic python client.
    """
    
    def __init__(
        self, 
        model_name: str = "nomic-embed-text-v1.5",
        api_key: Optional[str] = None
    ):
        """
        Initialize remote embeddings
        
        Args:
            model_name: Nomic API model identifier
            api_key: Nomic API Key
        """
        self.model_name = model_name
        self.api_key = api_key or NOMIC_API_KEY
        
        if not self.api_key:
            raise ValueError("Nomic API Key is required for RemoteNomicEmbeddings")
        
        # Login to Nomic
        nomic.login(self.api_key)
        logger.info(f"Initialized Remote Nomic Embeddings with model: {model_name}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents using Nomic API
        """
        try:
            output = embed.text(
                texts=texts,
                model=self.model_name,
                task_type="search_document"
            )
            return output['embeddings']
        except Exception as e:
            logger.error(f"Error calling Nomic API: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query using Nomic API
        """
        try:
            output = embed.text(
                texts=[text],
                model=self.model_name,
                task_type="search_query"
            )
            return output['embeddings'][0]
        except Exception as e:
            logger.error(f"Error calling Nomic API: {e}")
            raise



from langchain_google_genai import GoogleGenerativeAIEmbeddings
try:
    from config import GOOGLE_API_KEY
except ImportError:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

def get_embeddings(
    model_name: str = "models/text-embedding-004",
    **kwargs
) -> Embeddings:
    """
    Get Google Gemini embeddings
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY needed for Google Embeddings")
        
    logger.info(f"Using Google Embeddings with model: {model_name}")
    return GoogleGenerativeAIEmbeddings(
        model=model_name, 
        google_api_key=GOOGLE_API_KEY
    )


# Example usage
if __name__ == "__main__":
    # Test factory
    try:
        embeddings = get_embeddings()
        print(f"Using embedding class: {type(embeddings).__name__}")
        
        docs = ["This is a test document."]
        query = "test query"
        
        doc_emb = embeddings.embed_documents(docs)
        print(f"Document embedding shape: {len(doc_emb)}x{len(doc_emb[0])}")
        
        query_emb = embeddings.embed_query(query)
        print(f"Query embedding length: {len(query_emb)}")
    except Exception as e:
        print(f"Embedding generation failed: {e}")
