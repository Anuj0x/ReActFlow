# AgentFlow

A modern, high-performance async framework for building intelligent ReAct agents with reasoning and acting capabilities. Unifies ReAct prompting and OpenAI function calling in a single, efficient architecture.


**Created by [Anuj0x](https://github.com/Anuj0x)** - Expert in AI, ML, and modern frameworks

## ✨ Key Features

- 🚀 **Unified Async Architecture**: Single executor supporting both ReAct reasoning and native OpenAI function calling
- ⚡ **High Performance**: Async/await throughout with concurrent tool execution and request limiting
- 🌐 **REST API**: Production-ready FastAPI web service with automatic OpenAPI documentation
- 📊 **Structured Logging**: Enterprise-grade logging with context tracking using structlog
- 🛠️ **Built-in Tools**: Ready-to-use Wikipedia, Calculator, Date, and Web Search tools
- 🎯 **Type Safety**: Full type hints, dataclasses, and Pydantic validation
- ⚙️ **Environment Config**: Flexible configuration via environment variables
- 📈 **Metrics & Monitoring**: Built-in execution time tracking and performance insights

## 🏗️ Project Structure

```
src/
├── core/           # Unified agent components
│   ├── agent.py    # Agent & Tool dataclasses
│   ├── config.py   # Environment-based config
│   └── executor.py # Single async executor
├── api/            # FastAPI web application
├── tools/          # Built-in async tools
└── utils/          # Structured logging
```

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-openai-api-key"
```

### 2. Basic Usage
```python
import asyncio
from src.core import Agent, AgentConfig, ReActExecutor
from src.tools import get_default_tools

async def main():
    agent = Agent(
        name="Assistant",
        instructions="Help with research and calculations",
        tools=get_default_tools()
    )

    config = AgentConfig()
    executor = ReActExecutor(config)

    result = await executor.execute_react_style(
        agent=agent,
        query="What is Python's age in years?"
    )

    print(f"Answer: {result.final_answer}")

asyncio.run(main())
```

### 3. Start Web API
```bash
uvicorn src.api.app:app --reload
```
API available at `http://localhost:8000/docs`

## 📡 API Usage

```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "agent": {
        "name": "ResearchBot",
        "instructions": "Conduct research and calculations",
        "tools": [
            {
                "name": "WikipediaSearch",
                "description": "Search Wikipedia",
                "parameters": {
                    "type": "object",
                    "properties": {"search_query": {"type": "string"}},
                    "required": ["search_query"]
                }
            }
        ]
    },
    "query": "Explain quantum computing in simple terms"
})

print(response.json()["final_answer"])
```

## ⚙️ Configuration

```bash
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o
MAX_INTERACTIONS=10
TOKEN_LIMIT=5000
LOG_LEVEL=INFO
```

## 🔧 Built-in Tools

- **WikipediaSearch**: Async Wikipedia queries
- **Calculator**: Mathematical operations
- **Date_of_today**: Current date/time
- **WebSearch**: DuckDuckGo instant answers

## 📊 Performance

- **Unified Architecture**: ~30% less code than dual-executor approach
- **Async Execution**: Non-blocking tool calls and API requests
- **Concurrent Limits**: Configurable semaphore-based request control
- **Memory Efficient**: No threading overhead, pure asyncio

## 🎯 Technical Highlights

- **Creator**: [Anuj0x](https://github.com/Anuj0x) - Programming, AI/ML, Deep Learning, Generative Models, Attention Mechanisms, Reinforcement Learning, Multimodal AI, Computer Vision, Vector Databases, LLM Agents, Forecasting, Algorithmic Optimization, Blockchain, DevOps, AI Hardware, Quantum Computing, Web Frameworks
- **Architecture**: Modern dataclasses, async generators, structured concurrency
- **Reliability**: Comprehensive error handling, graceful degradation
- **Monitoring**: Execution metrics, token usage, performance tracking
- **Integration**: RESTful API, environment config, tool extensibility

## 🤝 Usage

```bash
# Development
python examples/modern_search/main.py

# Production API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Testing
python test_framework.py
```

## 📄 License

MIT License - see LICENSE file for details.

---

## 💡 Alternative Project Names & Descriptions

**1. "ReActFlow"**
*Modern async ReAct agent framework with unified execution*

**2. "AgentCore"**
*High-performance ReAct agent system with REST API and built-in tools*

**3. "AsyncReAct"**
*Async-first ReAct framework combining reasoning and tool execution*

**4. "BrainFlow"**
*Intelligent agent framework with unified ReAct architecture and async execution*

**5. "ThinkAct"**
*Reasoning and acting agent platform with modern async architecture*

The framework demonstrates concurrent tool execution, structured agent state management, and efficient API integration while maintaining the core ReAct algorithm's reasoning capabilities.
