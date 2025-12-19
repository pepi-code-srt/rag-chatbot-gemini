"""
Ragas Evaluation Framework
Evaluate hallucination, faithfulness, and relevance
"""

from typing import List, Dict, Any, Tuple
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas import evaluate
from datasets import Dataset
import logging

logger = logging.getLogger(__name__)


class RagasEvaluator:
    """
    Evaluate RAG responses using Ragas framework
    """
    
    def __init__(
        self,
        llm: Any,
        embeddings: Any,
        faithfulness_threshold: float = 0.8,
        relevance_threshold: float = 0.75,
        hallucination_filter_fraction: float = 0.15
    ):
        """
        Initialize evaluator
        
        Args:
            llm: LangChain LLM object
            embeddings: LangChain Embeddings object
            faithfulness_threshold: Minimum faithfulness score
            relevance_threshold: Minimum relevance score
            hallucination_filter_fraction: Fraction of responses to filter
        """
        self.llm = llm
        self.embeddings = embeddings
        self.faithfulness_threshold = faithfulness_threshold
        self.relevance_threshold = relevance_threshold
        self.hallucination_filter_fraction = hallucination_filter_fraction
    
    def evaluate_single_response(
        self,
        query: str,
        context: List[str],
        response: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single query-context-response triplet
        """
        
        # Prepare data for Ragas
        data_dict = {
            "question": [query],
            "contexts": [[c for c in context]], # Nested list for Ragas
            "answer": [response]
        }
        
        dataset = Dataset.from_dict(data_dict)
        
        try:
            logger.info("Evaluating response with Ragas...")
            
            # Run evaluation
            score = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    # context_recall, # Requires ground_truth which we don't have usually
                    # context_precision,
                ],
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            # Extract scores
            faithfulness_score = float(score["faithfulness"][0])
            relevance_score = float(score["answer_relevancy"][0])
            # context_recall_score = float(score["context_recall"][0])
            # context_precision_score = float(score["context_precision"][0])
            
            # Determine if hallucination detected
            is_hallucinating = faithfulness_score < self.faithfulness_threshold
            
            logger.info(
                f"Evaluation complete - "
                f"Faithfulness: {faithfulness_score:.2f}, "
                f"Relevance: {relevance_score:.2f}"
            )
            
            return {
                "query": query,
                "response": response,
                "scores": {
                    "faithfulness": faithfulness_score,
                    "answer_relevancy": relevance_score,
                    # "context_recall": context_recall_score,
                    # "context_precision": context_precision_score,
                },
                "thresholds": {
                    "faithfulness": self.faithfulness_threshold,
                    "relevance": self.relevance_threshold,
                },
                "hallucination_detected": is_hallucinating,
                "confidence": self._calculate_confidence(
                    faithfulness_score,
                    relevance_score
                ),
                "status": "success"
            }
        
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {
                "query": query,
                "response": response,
                "error": str(e),
                "status": "error",
                "hallucination_detected": True  # Fail safe
            }
    
    def batch_evaluate(
        self,
        queries: List[str],
        contexts: List[List[str]],
        responses: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate multiple responses in batch
        
        Args:
            queries: List of queries
            contexts: List of context document lists
            responses: List of responses
        
        Returns:
            Batch evaluation results
        """
        
        data_dict = {
            "question": queries,
            "contexts": contexts,
            "answer": responses
        }
        
        dataset = Dataset.from_dict(data_dict)
        
        try:
            logger.info(f"Running batch evaluation on {len(queries)} items...")
            
            score = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_recall,
                    context_precision,
                ]
            )
            
            # Process results
            results = {
                "total_queries": len(queries),
                "scores": {
                    "faithfulness": score["faithfulness"],
                    "answer_relevancy": score["answer_relevancy"],
                    "context_recall": score["context_recall"],
                    "context_precision": score["context_precision"],
                },
                "averages": {
                    "faithfulness": float(score["faithfulness"].mean()),
                    "answer_relevancy": float(score["answer_relevancy"].mean()),
                    "context_recall": float(score["context_recall"].mean()),
                    "context_precision": float(score["context_precision"].mean()),
                },
                "hallucination_rate": float(
                    (score["faithfulness"] < self.faithfulness_threshold).mean()
                ),
            }
            
            logger.info(
                f"Batch evaluation complete - "
                f"Avg Faithfulness: {results['averages']['faithfulness']:.2f}"
            )
            
            return results
        
        except Exception as e:
            logger.error(f"Error in batch evaluation: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    @staticmethod
    def _calculate_confidence(
        faithfulness: float,
        relevance: float
    ) -> str:
        """
        Calculate confidence level based on scores
        
        Args:
            faithfulness: Faithfulness score (0-1)
            relevance: Relevance score (0-1)
        
        Returns:
            "High", "Medium", or "Low"
        """
        
        avg_score = (faithfulness + relevance) / 2
        
        if avg_score >= 0.8:
            return "High"
        elif avg_score >= 0.6:
            return "Medium"
        else:
            return "Low"
    
    def should_filter_response(self, evaluation_result: Dict) -> bool:
        """
        Determine if response should be filtered based on evaluation
        
        Args:
            evaluation_result: Result from evaluate_single_response
        
        Returns:
            True if response should be filtered
        """
        
        if evaluation_result.get("status") == "error":
            return True
        
        if evaluation_result.get("hallucination_detected"):
            return True
        
        scores = evaluation_result.get("scores", {})
        if (scores.get("faithfulness", 1.0) < self.faithfulness_threshold or
            scores.get("answer_relevancy", 1.0) < self.relevance_threshold):
            return True
        
        return False


# Example usage
if __name__ == "__main__":
    evaluator = RagasEvaluator()
    
    # Example evaluation
    result = evaluator.evaluate_single_response(
        query="What is machine learning?",
        context=[
            "Machine learning is a subset of artificial intelligence."
        ],
        response="Machine learning is a subset of AI that enables computers to learn from data."
    )
    
    print(result)
