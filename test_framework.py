#!/usr/bin/env python3
"""Test script to verify the modernized ReAct framework works correctly."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_basic_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports...")

    try:
        from core import Agent, AgentConfig, ReActExecutor, Tool, Message
        from tools.builtins import get_default_tools, calculator_tool
        from utils.logging import setup_logging
        from api.app import app

        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


async def test_agent_creation():
    """Test agent creation and tool management."""
    print("Testing agent creation...")

    try:
        from core import Agent
        from tools.builtins import get_default_tools

        # Create agent with tools
        agent = Agent(
            name="TestAgent",
            instructions="Test agent instructions",
            tools=get_default_tools()
        )

        # Verify agent properties
        assert agent.name == "TestAgent"
        assert len(agent.tools) == 4  # Should have 4 default tools
        assert agent.find_tool("Calculator") is not None
        assert agent.find_tool("NonExistentTool") is None

        # Test tool management
        assert agent.add_tool(agent.find_tool("WikipediaSearch"))
        assert agent.remove_tool("Calculator")

        print("✅ Agent creation and tool management working")
        return True
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False


async def test_configuration():
    """Test configuration management."""
    print("Testing configuration...")

    try:
        from core import AgentConfig

        config = AgentConfig()

        # Test defaults
        assert config.openai_model == "gpt-4o"
        assert config.max_interactions == 10
        assert config.enable_caching is False

        # Test computed properties (without actually needing OpenAI key)
        assert hasattr(config, 'semaphore')
        assert hasattr(config, 'get_openai_client_kwargs')

        print("✅ Configuration management working")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


async def test_tools():
    """Test built-in tools functionality."""
    print("Testing tools...")

    try:
        from tools.builtins import calculator

        # Test calculator tool
        result_add = calculator("add", 5, 3)
        result_multiply = calculator("multiply", 5, 3)
        result_divide = calculator("divide", 10, 2)

        assert result_add == 8
        assert result_multiply == 15
        assert result_divide == 5

        # Test error handling
        try:
            calculator("divide", 10, 0)
            assert False, "Should have raised an error for division by zero"
        except ValueError:
            pass  # Expected

        print("✅ Tool functionality working")
        return True
    except Exception as e:
        print(f"❌ Tool test failed: {e}")
        return False


async def test_message_handling():
    """Test message creation and handling."""
    print("Testing message handling...")

    try:
        from core import Message

        # Create messages
        user_msg = Message(
            role="user",
            content="Hello world"
        )

        assistant_msg = Message(
            role="assistant",
            content="Hi there!",
            tool_calls=[{"id": "call_123", "function": {"name": "test"}}]
        )

        # Test conversions
        openai_format = user_msg.to_openai_format()
        assert openai_format["role"] == "user"
        assert openai_format["content"] == "Hello world"

        openai_format_with_tools = assistant_msg.to_openai_format()
        assert "tool_calls" in openai_format_with_tools

        print("✅ Message handling working")
        return True
    except Exception as e:
        print(f"❌ Message handling test failed: {e}")
        return False


async def run_tests():
    """Run all tests."""
    print("🧪 Testing ReAct Agent Framework")
    print("=" * 40)

    tests = [
        test_basic_imports,
        test_agent_creation,
        test_configuration,
        test_tools,
        test_message_handling,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            failed += 1

    print(f"\n📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed! The framework is ready to use.")
        print("\n🚀 To try the framework:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set API key: export OPENAI_API_KEY=your_key")
        print("3. Run example: python examples/modern_search/main.py")
        print("4. Start web API: uvicorn src.api.app:app --reload")
    else:
        print("⚠️ Some tests failed. Check the errors above.")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
