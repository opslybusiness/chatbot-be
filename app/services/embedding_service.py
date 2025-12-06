"""
Embedding service with support for multiple embedding providers
"""
import os
from abc import ABC, abstractmethod
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Embedding provider types
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "jina")  # "jina" or "gemini"


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts into vectors.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Dimension size (e.g., 768, 768, etc.)
        """
        pass


class JinaEmbedder(BaseEmbedder):
    """Jina AI embedding provider using API"""
    
    def __init__(self, api_key: str = None, model: str = "jina-embeddings-v3", task: str = "text-matching", dimensions: int = 768):
        """
        Initialize Jina embedder.
        
        Args:
            api_key: Jina AI API key. If None, reads from JINA_API_KEY env var.
            model: Name of the Jina model (default: jina-embeddings-v3)
            task: Task type for embeddings (default: text-matching)
            dimensions: Embedding dimensions (default: 768)
        """
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("JINA_API_KEY environment variable is required for Jina embeddings")
        
        self.model = model
        self.task = task
        self.dimensions = dimensions
        self.api_url = "https://api.jina.ai/v1/embeddings"
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using Jina AI API"""
        import requests
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "task": self.task,
            "dimensions": self.dimensions,
            "input": texts
        }
        
        try:
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            # The API returns data in the format: {"data": [{"embedding": [...], ...}, ...]}
            if "data" in result:
                embeddings = [item["embedding"] for item in result["data"]]
                return embeddings
            else:
                raise ValueError(f"Unexpected response format from Jina API: {result}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error calling Jina API: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimensions


class GeminiEmbedder(BaseEmbedder):
    """Google Gemini embedding provider"""
    
    def __init__(self, model_name: str = "models/embedding-001"):
        """
        Initialize Gemini embedder.
        
        Args:
            model_name: Name of the Gemini embedding model
        """
        import google.generativeai as genai
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini embeddings")
        
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self._initialized = False
    
    def _ensure_initialized(self):
        """Ensure Gemini API is configured"""
        if not self._initialized:
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
            self._initialized = True
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using Gemini embedding API"""
        import google.generativeai as genai
        
        self._ensure_initialized()
        
        embeddings = []
        for text in texts:
            # Gemini embedding API
            # Use "retrieval_document" for documents, "retrieval_query" for queries
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            # The result is a dict with 'embedding' key
            if 'embedding' in result:
                embeddings.append(result['embedding'])
            else:
                # Handle different response formats
                embeddings.append(result)
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension for Gemini models"""
        # Gemini embedding-001: 768 dimensions
        # You can also get it dynamically by embedding a test string
        return 768


class EmbeddingService:
    """Service class that provides embedding functionality with provider selection"""
    
    def __init__(self, provider: str = None):
        """
        Initialize embedding service with specified provider.
        
        Args:
            provider: "jina" or "gemini". If None, uses EMBEDDING_PROVIDER env var.
        """
        provider = provider or EMBEDDING_PROVIDER
        self.provider = provider.lower()
        
        if self.provider == "jina":
            model = os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v3")
            task = os.getenv("JINA_EMBEDDING_TASK", "text-matching")
            dimensions = int(os.getenv("JINA_EMBEDDING_DIMENSIONS", "768"))
            self.embedder = JinaEmbedder(model=model, task=task, dimensions=dimensions)
        elif self.provider == "gemini":
            model_name = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
            self.embedder = GeminiEmbedder(model_name=model_name)
        else:
            raise ValueError(
                f"Unknown embedding provider: {provider}. "
                "Supported providers: 'jina', 'gemini'"
            )
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        return self.embedder.embed_texts(texts)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.embedder.get_embedding_dimension()
    
    def get_provider(self) -> str:
        """Get the current embedding provider"""
        return self.provider


# Global instance (singleton pattern)
_embedding_service = None


def get_embedding_service(provider: str = None) -> EmbeddingService:
    """
    Get or create the global embedding service instance.
    
    Args:
        provider: Optional provider override
        
    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    if _embedding_service is None or (provider and _embedding_service.get_provider() != provider):
        _embedding_service = EmbeddingService(provider=provider)
    return _embedding_service


# Convenience function for backward compatibility
def embed_texts(texts: List[str], provider: str = None) -> List[List[float]]:
    """
    Embed texts using the configured embedding provider.
    This function maintains backward compatibility with the old embed_texts function.
    
    Args:
        texts: List of text strings to embed
        provider: Optional provider override ("jina" or "gemini")
        
    Returns:
        List of embedding vectors
    """
    service = get_embedding_service(provider=provider)
    return service.embed_texts(texts)

