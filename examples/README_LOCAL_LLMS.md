# Using Argentum with Local LLMs

This example demonstrates how to use Argentum with local LLM servers that implement the OpenAI API format.

## Supported Local Servers

Argentum works with any OpenAI-compatible server, including:

- **llama.cpp server** - Fast C++ inference
- **vLLM** - High-throughput serving
- **LocalAI** - Drop-in OpenAI replacement
- **Ollama** (with OpenAI compatibility mode)
- **LM Studio** - GUI-based local server
- **text-generation-webui** (with OpenAI extension)
- **FastChat** - Chatbot serving platform

## Quick Start

### 1. Start Your Local Server

Make sure your OpenAI-compatible server is running. For example, if using llama.cpp:

```bash
# Example with llama.cpp server
./server -m models/your-model.gguf --port 5000
```

The server should expose an endpoint like: `http://localhost:5000/v1/chat/completions`

### 2. Run the Example

```bash
# From the argentum root directory
python examples/local_llm_example.py
```

### 3. Configure for Your Setup

Edit `local_llm_example.py` and adjust:

```python
local_provider = OpenAIProvider(
    model="your-model-name",  # Whatever your server expects
    api_key="not-needed",     # Most local servers accept any value
    base_url="http://localhost:5000/v1",  # Your server's base URL
)
```

## Using with Different Scenarios

You can use local LLMs with any Argentum scenario:

### Advisory Panel Example

```python
from argentum.scenarios.advisory import CTOAdvisoryPanel
from argentum.llm.provider import OpenAIProvider

# Configure local provider
local_provider = OpenAIProvider(
    model="local-model",
    api_key="dummy",
    base_url="http://localhost:5000/v1",
)

# Create advisory panel
panel = CTOAdvisoryPanel(
    question="Should we migrate to microservices?",
    advisors=["security", "finance", "engineering"],
    provider=local_provider,
)

# Run the panel
result = await panel.run()
```

### Government Debate Example

```python
from argentum.scenarios.debate import GovernmentDebate
from argentum.llm.provider import OpenAIProvider

local_provider = OpenAIProvider(
    model="local-model",
    api_key="dummy",
    base_url="http://localhost:5000/v1",
)

debate = GovernmentDebate(
    topic="Universal Basic Income",
    ministers=["finance", "health", "environment"],
    provider=local_provider,
)

result = await debate.run()
```

## Performance Notes

- **Slower than cloud APIs**: Local LLMs typically run slower, especially on CPU
- **Lower quality**: Smaller local models may produce less coherent multi-agent dialogues
- **Memory intensive**: Running LLMs locally requires significant RAM/VRAM
- **Recommended models**: Try 7B+ parameter models (e.g., Llama 3, Mistral, Qwen)

## Troubleshooting

### Connection Refused
- Make sure your server is running: `curl http://localhost:5000/v1/models`
- Check the port number matches your server configuration
- Verify the `/v1` path is correct for your server

### Slow Responses
- Use GPU acceleration if available
- Reduce `max_tokens` in generation calls
- Try a smaller or quantized model
- Enable batching in your server if supported

### Invalid API Key Errors
- Some servers require a dummy key: try `"dummy"`, `"not-needed"`, or `"sk-dummy"`
- Check your server's documentation for authentication requirements

### Model Not Found
- List available models: `curl http://localhost:5000/v1/models`
- Use the exact model name from the list
- Some servers accept any model name and use their loaded model

## Environment Variable Alternative

You can also use environment variables instead of parameters:

```python
import os

os.environ["OPENAI_BASE_URL"] = "http://localhost:5000/v1"
os.environ["OPENAI_API_KEY"] = "dummy"

# This will use the environment variables
provider = OpenAIProvider(model="local-model")
```

## Next Steps

- Try different models and compare outputs
- Adjust temperature and other parameters
- Create custom agent personas optimized for your model
- Monitor token usage and response times
- Experiment with different orchestration patterns
