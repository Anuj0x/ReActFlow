#!/usr/bin/env python3
"""Modern async example demonstrating the improved ReAct framework."""

import asyncio
import os
from pathlib import Path

# Add src to path for easy importing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core import Agent, AgentConfig, ReActExecutor, Message
from tools.builtins import get_default_tools
from utils.logging import setup_logging


async def demo_simple_query():
    """Demonstrate simple query execution with ReAct style."""
    print("🔍 Demo 1: Simple Query with ReAct Reasoning")
    print("=" * 50)

    # Create agent with search and calculation capabilities
    agent = Agent(
        name="ResearchAssistant",
        instructions="""You are a research assistant that can search for information and perform calculations.
You must think step by step and use available tools to gather information before providing final answers.""",
        tools=get_default_tools()
    )

    # Set up executor
    config = AgentConfig()
    executor = ReActExecutor(config)

    # Execute query
    query = "What is the population of Japan and how does it compare to Germany?"
    print(f"Query: {query}")
    print("🤔 Reasoning and acting...")

    result = await executor.execute_react_style(
        agent=agent,
        query=query
    )

    print(f"✅ Final Answer: {result.final_answer}")
    print(".2f")
    print()


async def demo_function_calling():
    """Demonstrate modern OpenAI function calling approach."""
    print("⚡ Demo 2: Function Calling Approach")
    print("=" * 50)

    agent = Agent(
        name="DataAnalyzer",
        instructions="""You are a data analyst. Use available tools to gather and analyze information.
Always provide factual, evidence-based responses.""",
        tools=get_default_tools(),
        tool_choice="auto"  # Let model choose when to use tools
    )

    config = AgentConfig()
    executor = ReActExecutor(config)

    # Start conversation
    messages = [
        Message(role="user", content="Find information about Albert Einstein's birth year and calculate how old he would be today.")
    ]

    print("User:", messages[0].content)
    print("🤖 Processing with function calling...")

    result = await executor.execute(
        agent=agent,
        messages=messages,
    )

    # Display conversation
    for i, msg in enumerate(result.messages):
        role_name = msg.role.upper() if msg.role != "assistant" else "🤖 ASSISTANT"
        print(f"{role_name}: {msg.content}")
        if msg.tool_calls:
            for call in msg.tool_calls:
                func_name = call.get("function", {}).get("name", "unknown")
                print(f"  🛠️ Tool Call: {func_name}")

    print(f"📊 Execution time: {result.execution_time:.2f}s")
    print()


async def demo_error_handling():
    """Demonstrate error handling and recovery."""
    print("🛡️ Demo 3: Error Handling and Recovery")
    print("=" * 50)

    # Create agent with limited tools
    agent = Agent(
        name="CalculatorBot",
        instructions="You are a calculator. Use tools to perform mathematical operations.",
        tools=[]  # No tools initially
    )

    config = AgentConfig()
    executor = ReActExecutor(config)

    # Try to use calculator without tools
    messages = [
        Message(role="user", content="Calculate 25 * 17 + 10")
    ]

    print("User:", messages[0].content)
    print("🤖 Agent has no tools configured...")

    result = await executor.execute(
        agent=agent,
        messages=messages,
    )

    print("Result:", result.messages[-1].content if result.messages else "No response")
    if result.error:
        print(f"⚠️ Error: {result.error}")
    print()


async def run_all_demos():
    """Run all demonstrations."""
    print("🚀 ReAct Agent Framework - Modern Implementation Demo")
    print("=" * 60)
    print()

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return

    # Setup logging
    setup_logging(log_level="INFO", enable_structured=True)

    try:
        await demo_simple_query()
        await demo_function_calling()
        await demo_error_handling()

        print("✅ All demos completed successfully!")
        print("\nKey improvements demonstrated:")
        print("• Unified async execution engine")
        print("• Built-in error handling and recovery")
        print("• Structured logging and performance monitoring")
        print("• Modern type-safe dataclasses")
        print("• Both ReAct and function calling approaches")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set a dummy API key for demonstration if not set
    if not os.getenv("OPENAI_API_KEY"):
        print("Note: Set OPENAI_API_KEY to run actual demos")
        print("Using demo mode with mock responses...")
        # In a real scenario, you'd set a valid API key
        # os.environ["OPENAI_API_KEY"] = "your-key-here"

    asyncio.run(run_all_demos())
