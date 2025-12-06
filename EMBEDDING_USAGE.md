# Embedding Service Usage Guide

The embedding service supports multiple embedding providers. You can easily switch between Jina AI and Gemini embeddings.

## Configuration

### Environment Variables

Add to your `.env` file:

```bash
# Select embedding provider: "jina" or "gemini"
EMBEDDING_PROVIDER=jina  # or "gemini"

# For Jina AI embeddings
JINA_API_KEY=your_jina_api_key
JINA_EMBEDDING_MODEL=jina-embeddings-v3  # Optional, defaults to jina-embeddings-v3
JINA_EMBEDDING_TASK=text-matching  # Optional, defaults to text-matching
JINA_EMBEDDING_DIMENSIONS=768  # Optional, defaults to 768

# For Gemini embeddings
GOOGLE_API_KEY=your_google_api_key
GEMINI_EMBEDDING_MODEL=models/embedding-001  # Optional, defaults to embedding-001
```

## Usage Examples

### Using the Embedding Service Directly

```python
from app.services.embedding_service import get_embedding_service

# Get service with default provider (from env var)
service = get_embedding_service()

# Or specify provider explicitly
jina_service = get_embedding_service(provider="jina")
gemini_service = get_embedding_service(provider="gemini")

# Embed texts
texts = ["Hello world", "How are you?"]
embeddings = service.embed_texts(texts)

# Get embedding dimension
dimension = service.get_embedding_dimension()
print(f"Embedding dimension: {dimension}")
```

### Using the Convenience Function

```python
from app.services.embedding_service import embed_texts

# Uses default provider from EMBEDDING_PROVIDER env var
embeddings = embed_texts(["Hello", "World"])

# Or specify provider
embeddings = embed_texts(["Hello", "World"], provider="gemini")
```

### Backward Compatibility

The existing `embeddings_util.py` still works and automatically uses the configured provider:

```python
from embeddings_util import embed_texts

# This now uses the embedding service based on EMBEDDING_PROVIDER
embeddings = embed_texts(["Hello", "World"])
```

## Switching Providers

### Method 1: Environment Variable (Recommended)

Change `EMBEDDING_PROVIDER` in your `.env` file:
```bash
EMBEDDING_PROVIDER=gemini  # Switch to Gemini
```

### Method 2: Programmatically

```python
from app.services.embedding_service import get_embedding_service

# Create service with specific provider
service = get_embedding_service(provider="jina")
embeddings = service.embed_texts(["text"])
```

## Provider Comparison

### Jina AI Embeddings
- **Pros**: 
  - No local model storage needed
  - API-based embeddings with consistent quality
  - Supports multiple languages
  - Configurable dimensions and tasks
- **Cons**: 
  - Requires API key
  - API rate limits
  - Network latency
  - Potential costs

### Gemini Embeddings
- **Pros**: 
  - No local model storage needed
  - Consistent API-based embeddings
  - Managed by Google
- **Cons**: 
  - Requires API key
  - API rate limits
  - Network latency
  - Potential costs

## Embedding Dimensions

Different models produce different dimensions:
- Jina `jina-embeddings-v3`: 768 dimensions (default, configurable)
- Gemini `embedding-001`: 768 dimensions

**Important**: If you switch providers, make sure the embedding dimensions match, or you'll need to re-embed all your documents in the database.

## Example: Switching Providers

```python
# 1. Use Jina embeddings
from app.services.embedding_service import get_embedding_service

jina_service = get_embedding_service(provider="jina")
embeddings_jina = jina_service.embed_texts(["Hello"])

# 2. Switch to Gemini
gemini_service = get_embedding_service(provider="gemini")
embeddings_gemini = gemini_service.embed_texts(["Hello"])

# Both will work, but dimensions must match for database compatibility
```

## Notes

- The embedding service uses a singleton pattern - the first call initializes the service
- For production, stick with one provider to maintain consistency
- If switching providers, you may need to re-embed all documents in your database
- Both Jina and Gemini use API calls, so ensure stable network connectivity
