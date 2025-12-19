"""
Semantic Chunking Implementation
Splits documents based on semantic similarity, not just fixed size
"""

import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import TextSplitter
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class SemanticChunker(TextSplitter):
    """
    Splits documents into semantically meaningful chunks
    
    Algorithm:
    1. Split text into sentences
    2. Embed each sentence
    3. Compare similarity between consecutive sentences
    4. Create new chunk when similarity drops below threshold
    """
    
    def __init__(
        self,
        embedding_model: str = "nomic-embed-text",
        chunk_size: int = 512,
        overlap: int = 50,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1500
    ):
        """
        Initialize semantic chunker
        
        Args:
            embedding_model: Model name for embeddings
            chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks
            similarity_threshold: Cosine similarity threshold for splitting
            min_chunk_size: Minimum chunk size in tokens
            max_chunk_size: Maximum chunk size in tokens
        """
        # Lazy import to avoid circular dependency
        from src.embedding import get_embeddings
        self.embedding_model = get_embeddings(model_name=embedding_model)
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
    def _sentence_tokenize(self, text: str) -> List[str]:
        """Split text into sentences"""
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_embeddings(self, sentences: List[str]) -> List[List[float]]:
        """Get embeddings for a list of sentences"""
        return self.embedding_model.embed_documents(sentences)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        return dot_product / (norm_vec1 * norm_vec2)
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation: ~1 token per 4 characters"""
        return len(text) // 4
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into semantic chunks
        
        Algorithm:
        1. Split into sentences
        2. Group sentences with similar embeddings
        3. Merge groups to reach target chunk size
        4. Return chunks
        """
        if not text:
            return []
        
        # Step 1: Tokenize into sentences
        sentences = self._sentence_tokenize(text)
        if not sentences:
            return []
        
        # Step 2: Get embeddings for all sentences
        logger.info(f"Embedding {len(sentences)} sentences...")
        embeddings = self._get_embeddings(sentences)
        
        # Step 3: Determine chunk boundaries based on similarity
        chunk_boundaries = [0]  # Start of first chunk
        
        for i in range(len(embeddings) - 1):
            similarity = self._cosine_similarity(
                embeddings[i], 
                embeddings[i + 1]
            )
            
            # New chunk if:
            # - Similarity drops below threshold
            # - Current chunk exceeds size limit
            current_chunk_tokens = self._estimate_tokens(
                " ".join(sentences[chunk_boundaries[-1]:i + 1])
            )
            
            if (similarity < self.similarity_threshold or 
                current_chunk_tokens > self.max_chunk_size):
                chunk_boundaries.append(i + 1)
        
        # Add end boundary
        if chunk_boundaries[-1] != len(sentences):
            chunk_boundaries.append(len(sentences))
        
        # Step 4: Create chunks from boundaries
        chunks = []
        for i in range(len(chunk_boundaries) - 1):
            start_idx = chunk_boundaries[i]
            end_idx = chunk_boundaries[i + 1]
            
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)
            chunk_tokens = self._estimate_tokens(chunk_text)
            
            # Skip if too small
            if chunk_tokens < self.min_chunk_size and i < len(chunk_boundaries) - 2:
                continue
            
            chunks.append(chunk_text)
        
        logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split a list of documents"""
        split_docs = []
        
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            
            for i, chunk_text in enumerate(chunks):
                # Preserve metadata
                metadata = doc.metadata.copy()
                metadata["chunk_index"] = i
                metadata["original_doc_id"] = doc.metadata.get("source", "unknown")
                
                split_docs.append(
                    Document(
                        page_content=chunk_text,
                        metadata=metadata
                    )
                )
        
        return split_docs


def chunk_documents(
    documents: List[Document],
    method: str = "semantic",
    **kwargs
) -> List[Document]:
    """
    Chunk documents using specified method
    
    Args:
        documents: List of Document objects
        method: "semantic", "recursive", or "fixed"
        **kwargs: Additional arguments for chunker
    
    Returns:
        List of chunked documents
    """
    
    if method == "semantic":
        chunker = SemanticChunker(**kwargs)
        return chunker.split_documents(documents)
    
    elif method == "recursive":
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        chunker = RecursiveCharacterTextSplitter(**kwargs)
        return chunker.split_documents(documents)
    
    elif method == "fixed":
        from langchain.text_splitter import CharacterTextSplitter
        chunker = CharacterTextSplitter(**kwargs)
        return chunker.split_documents(documents)
    
    else:
        raise ValueError(f"Unknown chunking method: {method}")


# Example usage
if __name__ == "__main__":
    from config import CHUNK_CONFIG
    
    sample_text = """
    Machine learning is a subset of artificial intelligence.
    It enables computers to learn from data without being explicitly programmed.
    Deep learning is a more advanced technique that uses neural networks.
    Neural networks are inspired by biological brain structures.
    """
    
    chunker = SemanticChunker(**CHUNK_CONFIG)
    chunks = chunker.split_text(sample_text)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(chunk)
