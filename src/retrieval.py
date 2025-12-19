"""
Hybrid Search Retrieval
Combines BM25 (keyword search) + Vector Search (semantic)
"""

from typing import List, Tuple, Dict, Any
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder 
from langchain_classic.retrievers.document_compressors.base import DocumentCompressorPipeline
from langchain_community.document_transformers import LongContextReorder
import logging

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and vector search
    """
    
    def __init__(
        self,
        vectorstore: Chroma,
        documents: List[Document],
        ensemble_weights: Dict[str, float] = None,
        use_reranking: bool = True,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
        **kwargs
    ):
        """
        Initialize hybrid retriever
        
        Args:
            vectorstore: ChromaDB vector store
            documents: Original documents for BM25
            ensemble_weights: Weights for vector and BM25 search
            use_reranking: Whether to rerank results
            rerank_model: Cross-encoder model for reranking
            top_k: Number of results to return
            **kwargs: Additional configuration
        """
        
        if ensemble_weights is None:
            ensemble_weights = {"vector": 0.6, "bm25": 0.4}
        
        self.vectorstore = vectorstore
        self.documents = documents
        self.ensemble_weights = ensemble_weights
        self.use_reranking = use_reranking
        self.top_k = top_k
        
        # Initialize BM25 retriever
        logger.info("Initializing BM25 retriever...")
        self.bm25_retriever = BM25Retriever.from_documents(
            documents,
            k=top_k
        )
        
        # Initialize vector retriever
        self.vector_retriever = vectorstore.as_retriever(
            search_kwargs={"k": top_k}
        )
        
        # Initialize ensemble retriever
        logger.info("Creating ensemble retriever...")
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[
                ensemble_weights["vector"],
                ensemble_weights["bm25"]
            ]
        )
        
        # Initialize reranker if requested
        if use_reranking:
            logger.info(f"Loading reranking model: {rerank_model}")
            model = HuggingFaceCrossEncoder(model_name=rerank_model)
            self.reranker = CrossEncoderReranker(
                model=model,
                top_n=top_k
            )
        else:
            self.reranker = None
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        return_scores: bool = True
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents for query
        
        Args:
            query: Query string
            top_k: Number of results (uses default if None)
            return_scores: Whether to return similarity scores
        
        Returns:
            List of (Document, score) tuples
        """
        
        if top_k is None:
            top_k = self.top_k
        
        # Step 1: Ensemble retrieval
        logger.info(f"Running ensemble retrieval for: {query[:50]}...")
        results = self.ensemble_retriever.invoke(query)
        
        if not results:
            logger.warning(f"No results found for query: {query}")
            return []
        
        # Step 2: Reranking (optional)
        if self.use_reranking and self.reranker:
            logger.info(f"Reranking {len(results)} documents...")
            results = self.reranker.compress_documents(results, query)
        
        # Step 3: Calculate similarity scores
        if return_scores:
            # Get query embedding from vector store
            query_embedding = self.vectorstore.embeddings.embed_query(query)
            
            scored_results = []
            for doc in results[:top_k]:
                # Simple cosine similarity
                doc_embedding = self.vectorstore.embeddings.embed_documents(
                    [doc.page_content]
                )[0]
                
                similarity = self._cosine_similarity(
                    query_embedding,
                    doc_embedding
                )
                scored_results.append((doc, similarity))
            
            return scored_results
        
        return [(doc, 1.0) for doc in results[:top_k]]
    
    @staticmethod
    def _cosine_similarity(vec1, vec2):
        """Calculate cosine similarity"""
        import numpy as np
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        return float(dot_product / (norm_vec1 * norm_vec2))
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: int = None,
        add_surrounding_context: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve documents with additional context
        
        Args:
            query: Query string
            top_k: Number of results
            add_surrounding_context: Include surrounding chunks
        
        Returns:
            Dictionary with results and metadata
        """
        
        results = self.retrieve(query, top_k=top_k, return_scores=True)
        
        if not results:
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "avg_similarity": 0.0
            }
        
        # Prepare response
        response_results = []
        similarities = []
        
        for doc, similarity in results:
            similarities.append(similarity)
            
            result_item = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(similarity)
            }
            
            # Add surrounding context if requested
            if add_surrounding_context:
                chunk_idx = doc.metadata.get("chunk_index", 0)
                source = doc.metadata.get("source", "unknown")
                
                # Find previous and next chunks
                prev_chunk = self._find_adjacent_chunk(source, chunk_idx - 1)
                next_chunk = self._find_adjacent_chunk(source, chunk_idx + 1)
                
                result_item["context"] = {
                    "previous_chunk": prev_chunk,
                    "next_chunk": next_chunk
                }
            
            response_results.append(result_item)
        
        return {
            "query": query,
            "results": response_results,
            "total_results": len(response_results),
            "avg_similarity": sum(similarities) / len(similarities),
            "retrieval_method": "hybrid"
        }
    
    def _find_adjacent_chunk(self, source: str, chunk_idx: int) -> str:
        """Find an adjacent chunk from the same source"""
        for doc in self.documents:
            if (doc.metadata.get("source") == source and
                doc.metadata.get("chunk_index") == chunk_idx):
                return doc.page_content
        return ""


# Example usage
if __name__ == "__main__":
    # from config import RETRIEVAL_CONFIG
    
    # This would be used with actual ChromaDB and documents
    # retriever = HybridRetriever(vectorstore, documents, **RETRIEVAL_CONFIG)
    # results = retriever.retrieve("What is machine learning?")
    pass
