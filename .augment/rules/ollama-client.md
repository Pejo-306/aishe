---
type: agent_requested
description: "ollama client requirements and description"
keywords: ["ollama", "llm", "server", "client", "python", "ai"]
---

# Ollama Python Library

## Library Choice

**Always use the official Ollama Python library** from https://github.com/ollama/ollama-python for integrating with Ollama.

This is the official Python client maintained by Ollama for interacting with local and cloud LLM models.

## Prerequisites

Before using the library:
1. **Ollama must be installed and running** on your system
2. **Pull a model** to use: `ollama pull <model>` (e.g., `ollama pull gemma3`)
3. See https://ollama.com for available models

## Installation

```bash
pip install ollama
```

## Core Concepts

Ollama provides two main interaction patterns:
1. **Chat**: Conversational interface with message history
2. **Generate**: Single prompt completion without conversation context

Both support:
- Synchronous and asynchronous operations
- Streaming and non-streaming responses
- Local and cloud models

## Best Practices

### 1. Use Chat for Conversational Interactions

For multi-turn conversations, always use the `chat` API:

```python
from ollama import chat, ChatResponse

response: ChatResponse = chat(
    model='gemma3',
    messages=[
        {
            'role': 'user',
            'content': 'Why is the sky blue?',
        },
    ]
)

# Access response content
print(response['message']['content'])
# or use dot notation
print(response.message.content)
```

### 2. Use Generate for Single Completions

For one-off completions without conversation context:

```python
from ollama import generate

response = generate(
    model='gemma3',
    prompt='Why is the sky blue?'
)

print(response['response'])
```

### 3. Use Streaming for Long Responses

Enable streaming to get responses as they're generated:

```python
from ollama import chat

stream = chat(
    model='gemma3',
    messages=[{'role': 'user', 'content': 'Explain quantum physics in detail'}],
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
```

### 4. Use AsyncClient for Concurrent Operations

For better performance with multiple requests:

```python
import asyncio
from ollama import AsyncClient

async def chat_async():
    message = {'role': 'user', 'content': 'Why is the sky blue?'}
    response = await AsyncClient().chat(model='gemma3', messages=[message])
    return response.message.content

# Run async function
result = asyncio.run(chat_async())
```

### 5. Use Async Streaming for Real-Time Responses

Combine async with streaming for optimal performance:

```python
import asyncio
from ollama import AsyncClient

async def stream_chat():
    message = {'role': 'user', 'content': 'Explain machine learning'}
    async for part in await AsyncClient().chat(
        model='gemma3',
        messages=[message],
        stream=True
    ):
        print(part['message']['content'], end='', flush=True)

asyncio.run(stream_chat())
```

### 6. Use Custom Client for Configuration

Configure client with custom settings:

```python
from ollama import Client

client = Client(
    host='http://localhost:11434',
    headers={'x-custom-header': 'value'},
    timeout=30.0,  # Custom timeout
)

response = client.chat(
    model='gemma3',
    messages=[{'role': 'user', 'content': 'Hello'}]
)
```

### 7. Handle Errors Gracefully

Always handle potential errors:

```python
import ollama

model = 'gemma3'

try:
    response = ollama.chat(model=model, messages=[
        {'role': 'user', 'content': 'Hello'}
    ])
    print(response.message.content)
except ollama.ResponseError as e:
    print(f'Error: {e.error}')
    if e.status_code == 404:
        print(f'Model {model} not found. Pulling...')
        ollama.pull(model)
        # Retry after pulling
        response = ollama.chat(model=model, messages=[
            {'role': 'user', 'content': 'Hello'}
        ])
except Exception as e:
    print(f'Unexpected error: {e}')
```

### 8. Maintain Conversation Context

Keep message history for multi-turn conversations:

```python
from ollama import chat

messages = []

def chat_with_context(user_message: str) -> str:
    """Chat while maintaining conversation history."""
    # Add user message
    messages.append({
        'role': 'user',
        'content': user_message
    })

    # Get response
    response = chat(model='gemma3', messages=messages)

    # Add assistant response to history
    messages.append({
        'role': 'assistant',
        'content': response.message.content
    })

    return response.message.content

# Use it
print(chat_with_context("What is Python?"))
print(chat_with_context("Can you give me an example?"))  # Maintains context
```

### 9. Use Embeddings for Semantic Search

Generate embeddings for text similarity:

```python
from ollama import embed

# Single text
embedding = embed(
    model='gemma3',
    input='The sky is blue because of rayleigh scattering'
)

# Batch embeddings (more efficient)
embeddings = embed(
    model='gemma3',
    input=[
        'The sky is blue because of rayleigh scattering',
        'Grass is green because of chlorophyll',
        'Water is wet because of hydrogen bonds'
    ]
)

# Use embeddings for similarity search
print(f"Generated {len(embeddings['embeddings'])} embeddings")
```

### 10. Use System Messages for Behavior Control

Set system prompts to control model behavior:

```python
from ollama import chat

response = chat(
    model='gemma3',
    messages=[
        {
            'role': 'system',
            'content': 'You are a helpful assistant that speaks like a pirate.'
        },
        {
            'role': 'user',
            'content': 'Tell me about Python programming.'
        }
    ]
)

print(response.message.content)
```

## Model Management

### List Available Models

```python
from ollama import list

models = list()
for model in models['models']:
    print(f"Model: {model['name']}")
    print(f"Size: {model['size']}")
    print(f"Modified: {model['modified_at']}")
```

### Pull Models

```python
from ollama import pull

# Pull a model
pull('gemma3')

# Pull with streaming progress
for progress in pull('gemma3', stream=True):
    print(f"Progress: {progress}")
```

### Show Model Information

```python
from ollama import show

info = show('gemma3')
print(f"Model: {info['modelfile']}")
print(f"Parameters: {info['parameters']}")
```

### Create Custom Models

```python
from ollama import create

# Create a custom model with system prompt
create(
    model='my-assistant',
    from_='gemma3',
    system="You are a helpful coding assistant specialized in Python."
)
```

### Delete Models

```python
from ollama import delete

delete('old-model')
```

## Cloud Models

### Using Cloud Models via Local Ollama

For larger models offloaded to Ollama's cloud:

```python
from ollama import Client

# One-time: Sign in via CLI
# ollama signin

# Pull cloud model
# ollama pull gpt-oss:120b-cloud

client = Client()

messages = [{'role': 'user', 'content': 'Why is the sky blue?'}]

for part in client.chat('gpt-oss:120b-cloud', messages=messages, stream=True):
    print(part.message.content, end='', flush=True)
```

### Using Cloud API Directly

Access cloud models via ollama.com API:

```python
import os
from ollama import Client

# Set API key: export OLLAMA_API_KEY=your_api_key

client = Client(
    host='https://ollama.com',
    headers={'Authorization': f'Bearer {os.environ.get("OLLAMA_API_KEY")}'}
)

messages = [{'role': 'user', 'content': 'Why is the sky blue?'}]

for part in client.chat('gpt-oss:120b', messages=messages, stream=True):
    print(part.message.content, end='', flush=True)
```

## Advanced Patterns

### Retry Logic with Exponential Backoff

```python
import time
from ollama import chat, ResponseError

def chat_with_retry(model: str, messages: list, max_retries: int = 3) -> str:
    """Chat with automatic retry on failure."""
    for attempt in range(max_retries):
        try:
            response = chat(model=model, messages=messages)
            return response.message.content
        except ResponseError as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s...")
            time.sleep(wait_time)
```

### Concurrent Requests with AsyncClient

```python
import asyncio
from ollama import AsyncClient

async def process_multiple_prompts(prompts: list[str]) -> list[str]:
    """Process multiple prompts concurrently."""
    client = AsyncClient()

    async def process_one(prompt: str) -> str:
        response = await client.chat(
            model='gemma3',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response.message.content

    # Run all prompts concurrently
    results = await asyncio.gather(*[process_one(p) for p in prompts])
    return results

# Use it
prompts = [
    "What is Python?",
    "What is JavaScript?",
    "What is Rust?"
]
results = asyncio.run(process_multiple_prompts(prompts))
```

### Streaming with Progress Tracking

```python
from ollama import chat

def chat_with_progress(model: str, messages: list) -> str:
    """Chat with progress indication."""
    stream = chat(model=model, messages=messages, stream=True)

    full_response = ""
    chunk_count = 0

    for chunk in stream:
        content = chunk['message']['content']
        full_response += content
        chunk_count += 1

        # Show progress every 10 chunks
        if chunk_count % 10 == 0:
            print(f"[Received {chunk_count} chunks...]", end='\r')

    print()  # New line after progress
    return full_response
```

### Context Window Management

```python
from ollama import chat

def chat_with_context_limit(
    model: str,
    messages: list,
    max_messages: int = 10
) -> str:
    """Chat while limiting context window size."""
    # Keep only the last N messages (plus system message if present)
    system_messages = [m for m in messages if m['role'] == 'system']
    recent_messages = [m for m in messages if m['role'] != 'system'][-max_messages:]

    limited_messages = system_messages + recent_messages

    response = chat(model=model, messages=limited_messages)
    return response.message.content
```



## Testing

### Mock Ollama for Unit Tests

```python
from unittest.mock import Mock, patch
from ollama import ChatResponse

def test_chat_function():
    """Test function that uses Ollama chat."""
    mock_response = Mock(spec=ChatResponse)
    mock_response.message.content = "Mocked response"

    with patch('ollama.chat', return_value=mock_response):
        # Your code that uses ollama.chat
        result = my_chat_function("test prompt")
        assert result == "Mocked response"
```

### Integration Tests with Real Ollama

```python
import pytest
from ollama import chat, list, ResponseError

@pytest.fixture
def ensure_model():
    """Ensure test model is available."""
    models = list()
    model_names = [m['name'] for m in models['models']]

    if 'gemma3' not in model_names:
        pytest.skip("Test model 'gemma3' not available")

def test_chat_integration(ensure_model):
    """Integration test with real Ollama."""
    response = chat(
        model='gemma3',
        messages=[{'role': 'user', 'content': 'Say hello'}]
    )

    assert response.message.content
    assert len(response.message.content) > 0
```

### Test Async Functions

```python
import pytest
from ollama import AsyncClient

@pytest.mark.asyncio
async def test_async_chat():
    """Test async chat functionality."""
    client = AsyncClient()
    response = await client.chat(
        model='gemma3',
        messages=[{'role': 'user', 'content': 'Hello'}]
    )

    assert response.message.content
```

## Common Pitfalls and Solutions

### 1. Model Not Found Error

**Problem**: Getting 404 errors when trying to use a model.

**Solution**: Always check if model exists and pull if needed:

```python
from ollama import chat, pull, list, ResponseError

def safe_chat(model: str, messages: list) -> str:
    """Chat with automatic model pulling."""
    try:
        response = chat(model=model, messages=messages)
        return response.message.content
    except ResponseError as e:
        if e.status_code == 404:
            print(f"Model {model} not found. Pulling...")
            pull(model)
            response = chat(model=model, messages=messages)
            return response.message.content
        raise
```

### 2. Connection Refused Error

**Problem**: Cannot connect to Ollama server.

**Solution**: Verify Ollama is running and check connection:

```python
from ollama import Client, list

def check_ollama_connection(host: str = 'http://localhost:11434') -> bool:
    """Check if Ollama server is accessible."""
    try:
        client = Client(host=host)
        client.list()
        return True
    except Exception as e:
        print(f"Cannot connect to Ollama at {host}: {e}")
        return False

# Use it
if not check_ollama_connection():
    print("Please start Ollama: ollama serve")
```

### 3. Memory Issues with Large Contexts

**Problem**: Running out of memory with long conversations.

**Solution**: Implement context window management:

```python
def truncate_messages(messages: list, max_tokens: int = 4000) -> list:
    """Truncate messages to fit within token limit."""
    # Keep system messages
    system_msgs = [m for m in messages if m['role'] == 'system']
    other_msgs = [m for m in messages if m['role'] != 'system']

    # Estimate tokens (rough: 1 token ≈ 4 chars)
    total_chars = sum(len(m['content']) for m in other_msgs)
    estimated_tokens = total_chars / 4

    if estimated_tokens > max_tokens:
        # Keep only recent messages
        chars_to_keep = max_tokens * 4
        truncated = []
        current_chars = 0

        for msg in reversed(other_msgs):
            msg_chars = len(msg['content'])
            if current_chars + msg_chars <= chars_to_keep:
                truncated.insert(0, msg)
                current_chars += msg_chars
            else:
                break

        return system_msgs + truncated

    return messages
```

### 4. Slow Response Times

**Problem**: Responses taking too long.

**Solution**: Use streaming and async for better perceived performance:

```python
import asyncio
from ollama import AsyncClient

async def fast_chat(model: str, prompt: str) -> str:
    """Get response with streaming for faster perceived performance."""
    client = AsyncClient()

    response_parts = []
    async for part in await client.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        stream=True
    ):
        content = part['message']['content']
        response_parts.append(content)
        # Show progress to user
        print(content, end='', flush=True)

    print()  # New line
    return ''.join(response_parts)
```

### 5. Inconsistent Response Quality

**Problem**: Getting varying quality in responses.

**Solution**: Use system prompts and temperature control:

```python
from ollama import chat

def consistent_chat(prompt: str, temperature: float = 0.7) -> str:
    """Chat with consistent behavior."""
    response = chat(
        model='gemma3',
        messages=[
            {
                'role': 'system',
                'content': 'You are a precise and consistent assistant. '
                          'Always provide clear, factual answers.'
            },
            {
                'role': 'user',
                'content': prompt
            }
        ],
        options={
            'temperature': temperature,  # Lower = more consistent
            'top_p': 0.9,
            'top_k': 40
        }
    )
    return response.message.content
```

## Configuration Options

### Model Parameters

Control model behavior with options:

```python
from ollama import chat

response = chat(
    model='gemma3',
    messages=[{'role': 'user', 'content': 'Hello'}],
    options={
        'temperature': 0.8,      # Creativity (0.0-1.0)
        'top_p': 0.9,           # Nucleus sampling
        'top_k': 40,            # Top-k sampling
        'num_predict': 100,     # Max tokens to generate
        'stop': ['\n\n'],       # Stop sequences
        'seed': 42,             # Reproducibility
    }
)
```

### Client Configuration

```python
from ollama import Client

client = Client(
    host='http://localhost:11434',
    timeout=60.0,           # Request timeout in seconds
    headers={
        'User-Agent': 'MyApp/1.0',
        'X-Custom-Header': 'value'
    }
)
```

## Performance Tips

### 1. Reuse Client Instances

```python
from ollama import Client

# Good: Reuse client
client = Client()
for prompt in prompts:
    response = client.chat(model='gemma3', messages=[{'role': 'user', 'content': prompt}])

# Bad: Create new client each time
for prompt in prompts:
    response = Client().chat(model='gemma3', messages=[{'role': 'user', 'content': prompt}])
```

### 2. Use Batch Operations

```python
from ollama import embed

# Good: Batch embeddings
texts = ['text1', 'text2', 'text3']
embeddings = embed(model='gemma3', input=texts)

# Bad: Individual embeddings
for text in texts:
    embedding = embed(model='gemma3', input=text)
```

### 3. Use Async for Concurrent Requests

```python
import asyncio
from ollama import AsyncClient

async def process_batch(prompts: list[str]) -> list[str]:
    """Process multiple prompts concurrently."""
    client = AsyncClient()

    tasks = [
        client.chat(model='gemma3', messages=[{'role': 'user', 'content': p}])
        for p in prompts
    ]

    responses = await asyncio.gather(*tasks)
    return [r.message.content for r in responses]
```

### 4. Stream for Long Responses

```python
from ollama import chat

# Good: Stream long responses
stream = chat(
    model='gemma3',
    messages=[{'role': 'user', 'content': 'Write a long essay'}],
    stream=True
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)

# Bad: Wait for entire response
response = chat(
    model='gemma3',
    messages=[{'role': 'user', 'content': 'Write a long essay'}]
)
print(response.message.content)
```

## Summary

- **Use the official Ollama Python library** for all Ollama interactions
- **Use chat for conversations**, generate for single completions
- **Enable streaming** for long responses and better UX
- **Use AsyncClient** for concurrent operations
- **Handle errors gracefully** with proper error handling
- **Maintain conversation context** for multi-turn interactions
- **Use embeddings** for semantic search and similarity
- **Configure model parameters** for consistent behavior
- **Implement retry logic** for production reliability
- **Test with mocks** for unit tests, real Ollama for integration tests
- **Manage context windows** to avoid memory issues
- **Reuse client instances** for better performance
- **Use batch operations** when processing multiple items
- **Monitor model availability** and pull models as needed

## Quick Reference

```python
# Basic chat
from ollama import chat
response = chat(model='gemma3', messages=[{'role': 'user', 'content': 'Hello'}])

# Streaming
stream = chat(model='gemma3', messages=[...], stream=True)
for chunk in stream:
    print(chunk['message']['content'], end='')

# Async
from ollama import AsyncClient
response = await AsyncClient().chat(model='gemma3', messages=[...])

# Custom client
from ollama import Client
client = Client(host='http://localhost:11434')

# Error handling
try:
    response = chat(model='gemma3', messages=[...])
except ollama.ResponseError as e:
    if e.status_code == 404:
        ollama.pull('gemma3')

# Embeddings
from ollama import embed
embeddings = embed(model='gemma3', input=['text1', 'text2'])

# Model management
from ollama import list, pull, delete, show
models = list()
pull('gemma3')
info = show('gemma3')
delete('old-model')
```
