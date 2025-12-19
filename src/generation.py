"""
LLM Generation with Gemini API
Generates responses based on retrieved context
"""

from typing import List, Dict, Any, Tuple
import google.generativeai as genai
from langchain_core.documents import Document
from config import GENERATION_CONFIG, SYSTEM_PROMPT
import logging

logger = logging.getLogger(__name__)


class GeminiGenerator:
    """
    Generate responses using Google Gemini API
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "models/gemini-flash-latest",
        temperature: float = 0.3,
        max_output_tokens: int = 1024,
        system_prompt: str = SYSTEM_PROMPT,
        **kwargs
    ):
        """
        Initialize Gemini generator
        
        Args:
            api_key: Google API key
            model_name: Model identifier
            temperature: Response temperature (0-1)
            max_output_tokens: Maximum output tokens
            system_prompt: System instructions
            **kwargs: Additional generation parameters (top_p, top_k, etc.)
        """
        
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.system_prompt = system_prompt
        self.generation_params = kwargs
        
        logger.info(f"Initialized Gemini generator with model: {model_name}")
    
    def _format_context(self, documents: List[Tuple[Document, float]]) -> str:
        """
        Format retrieved documents into context string
        
        Args:
            documents: List of (Document, score) tuples
        
        Returns:
            Formatted context string
        """
        
        context_parts = []
        
        for i, (doc, score) in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            chunk_idx = doc.metadata.get("chunk_index", 0)
            
            context_parts.append(
                f"[Source {i}: {source}, Chunk {chunk_idx} (Relevance: {score:.2%})]\n"
                f"{doc.page_content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def generate(
        self,
        query: str,
        context_documents: List[Tuple[Document, float]],
        include_reasoning: bool = False
    ) -> Dict[str, Any]:
        """
        Generate response using Gemini
        
        Args:
            query: User query
            context_documents: Retrieved documents with scores
            include_reasoning: Include reasoning in response
        
        Returns:
            Generation result with response and metadata
        """
        
        # Format context
        formatted_context = self._format_context(context_documents)
        
        # Prepare prompt
        if include_reasoning:
            prompt = f"""{self.system_prompt}

CONTEXT:
{formatted_context}

QUESTION: {query}

Please provide:
1. Your reasoning based on the context
2. Your answer
3. Confidence level (High/Medium/Low)
4. Any caveats or limitations"""
        else:
            prompt = f"""{self.system_prompt}

CONTEXT:
{formatted_context}

QUESTION: {query}

Answer based on the provided context. If the context doesn't contain relevant information, say so clearly."""
        
        try:
            logger.info("Calling Gemini API...")
            
            model = genai.GenerativeModel(self.model_name)
            
            # Merge defaults with kwargs
            gen_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "top_p": self.generation_params.get("top_p", 0.95),
                "top_k": self.generation_params.get("top_k", 40),
            }
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(**gen_config)
            )
            
            logger.info("Gemini response generated successfully")
            
            return {
                "query": query,
                "response": response.text,
                "model": self.model_name,
                "context_documents": len(context_documents),
                "generation_config": gen_config,
                "status": "success"
            }
        
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return {
                "query": query,
                "response": f"Error generating response: {str(e)}",
                "error": str(e),
                "status": "error"
            }
    
    def stream_generate(
        self,
        query: str,
        context_documents: List[Tuple[Document, float]]
    ):
        """
        Stream response from Gemini (for real-time updates)
        
        Args:
            query: User query
            context_documents: Retrieved documents
        
        Yields:
            Response chunks
        """
        
        formatted_context = self._format_context(context_documents)
        prompt = f"""{self.system_prompt}

CONTEXT:
{formatted_context}

QUESTION: {query}"""
        
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                prompt,
                stream=True,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                )
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        
        except Exception as e:
            logger.error(f"Error in stream generation: {e}")
            yield f"Error: {str(e)}"


from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

class OllamaGenerator:
    """
    Generator using local Ollama models (no API key required)
    """
    
    def __init__(
        self,
        model_name: str = "models/gemini-flash-latest",
        temperature: float = 0.7,
        system_prompt: str = SYSTEM_PROMPT,
        **kwargs
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        
        logger.info(f"Initializing Ollama Generator with model: {model_name}")
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            **kwargs
        )
        
    def generate(
        self,
        query: str,
        context_documents: List[Tuple[Document, float]],
        include_reasoning: bool = False
    ) -> Dict[str, Any]:
        """
        Generate response using Ollama
        """
        try:
            formatted_context = self._format_context(context_documents)
            
            # Simple prompt construction
            prompt_content = f"""{self.system_prompt}

CONTEXT:
{formatted_context}

QUESTION: {query}

Answer based on the provided context."""

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt_content)
            ]
            
            logger.info(f"Calling Ollama model: {self.model_name}...")
            response = self.llm.invoke(messages)
            
            return {
                "query": query,
                "response": response.content,
                "model": self.model_name,
                "context_documents": len(context_documents),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return {
                "query": query,
                "response": f"Error generating response: {str(e)}",
                "error": str(e),
                "status": "error"
            }

    def _format_context(self, context_documents: List[Tuple[Document, float]]) -> str:
        """Format context documents for the prompt"""
        context_parts = []
        for i, (doc, score) in enumerate(context_documents, 1):
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(f"Source {i} ({source}):\n{doc.page_content}\n")
        
        return "\n".join(context_parts)

