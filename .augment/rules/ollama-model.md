---
type: agent_requested
description: "llama3.2:3b is the default model to be used"
---
# Ollama LLM Configuration

## Default Model Selection

When implementing features that require a local Large Language Model (LLM), use **Ollama with Llama 3.2 3B** as the default model.

## Model Specification

- **Model Name**: `llama3.2:3b`
- **Provider**: Ollama (local)
- **Size**: ~2GB
- **RAM Requirements**: 4-6GB

## Installation Instructions

Ensure Ollama is installed and the model is pulled:

```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.com/download

# Pull the model
ollama pull llama3.2:3b

# Verify installation
ollama run llama3.2:3b
```

## Python Client Library

Use the official Ollama Python library:

```bash
pip install ollama
```

## Usage in Code

When implementing LLM functionality, use this pattern:

```python
import ollama

# For simple generation
response = ollama.chat(
    model='llama3.2:3b',
    messages=[
        {
            'role': 'user',
            'content': 'Your prompt here',
        },
    ]
)
print(response['message']['content'])

# For streaming responses (recommended for better UX)
for chunk in ollama.chat(
    model='llama3.2:3b',
    messages=[...],
    stream=True
):
    print(chunk['message']['content'], end='', flush=True)
```

## Why Llama 3.2 3B?

- ✅ Excellent balance of quality and speed
- ✅ Low memory footprint (works on most modern laptops)
- ✅ Good at understanding context and summarizing (ideal for RAG)
- ✅ Fast response times
- ✅ Recent model (2024) with good optimization
- ✅ Strong instruction following capabilities

## Alternative Models

Only suggest alternative models if the user explicitly requests them or if there are specific requirements:

- **For higher quality** (requires 16GB+ RAM): `mistral:7b` or `llama3.1:8b`
- **For lower memory** (8GB RAM): `llama3.2:1b`
- **For specialized tasks**: `phi3:mini`

## Configuration

When implementing configurable LLM settings, use these defaults:

```python
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_TEMPERATURE = 0.7  # Balance between creativity and consistency
DEFAULT_MAX_TOKENS = 2048  # Sufficient for most responses
```

## Error Handling

Always check if Ollama is running and the model is available:

```python
import ollama

try:
    # Test connection
    ollama.list()
    
    # Use the model
    response = ollama.chat(model='llama3.2:3b', messages=[...])
    
except Exception as e:
    print(f"Error: Ollama may not be running or model not available.")
    print(f"Please run: ollama pull llama3.2:3b")
    raise
```

## Performance Considerations

- The model runs efficiently on CPU but benefits from GPU acceleration if available
- Expect response times of 1-3 seconds for typical queries on modern laptops
- For RAG applications, the model handles context windows well for summarization tasks

