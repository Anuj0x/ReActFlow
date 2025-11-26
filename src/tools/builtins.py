"""Built-in tools for ReAct agents with modern implementations."""

import asyncio
from datetime import datetime, date
from typing import Any, Dict, Optional
import wikipedia
import httpx

from ..core import Tool


async def wikipedia_search(search_query: str) -> str:
    """
    Search Wikipedia for information.

    Args:
        search_query: The search query to look up on Wikipedia

    Returns:
        Wikipedia search results
    """
    try:
        # Run in thread pool to avoid blocking
        def _search():
            return wikipedia.summary(search_query, sentences=3, auto_suggest=True)

        # Use asyncio to run in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _search)
        return f"Wikipedia result for '{search_query}': {result}"

    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found. Options: {', '.join(e.options[:5])}"
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{search_query}'"
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"


def calculator(operation: str, a: float, b: float) -> float:
    """
    Perform mathematical calculations.

    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        a: First number
        b: Second number

    Returns:
        Result of the calculation
    """
    try:
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Division by zero")
            return a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")
    except Exception as e:
        raise ValueError(f"Calculator error: {str(e)}")


async def date_today() -> str:
    """
    Get the current date.

    Returns:
        Current date in YYYY-MM-DD format
    """
    try:
        today = date.today()
        return today.strftime("%Y-%m-%d")
    except Exception as e:
        return f"Error getting date: {str(e)}"


async def web_search(query: str) -> str:
    """
    Perform a simple web search using DuckDuckGo Instant Answer API.

    Args:
        query: Search query

    Returns:
        Web search results
    """
    try:
        # Use DuckDuckGo Instant Answer API (free, no API key needed)
        url = "https://api.duckduckgo.com"
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        # Extract relevant information
        answer = data.get("Answer")
        abstract = data.get("AbstractText")

        if answer:
            return f"Direct answer: {answer}"
        elif abstract:
            return f"Abstract: {abstract}"
        else:
            return f"Search completed for '{query}'. No direct answer found, but search was successful."

    except httpx.RequestError as e:
        return f"Web search request failed: {str(e)}"
    except Exception as e:
        return f"Web search error: {str(e)}"


# Pre-configured tool instances
WIKIPEDIA_TOOL = Tool(
    name="WikipediaSearch",
    func=wikipedia_search,
    description="Search Wikipedia for information about any topic",
    parameters={
        "type": "object",
        "properties": {
            "search_query": {
                "type": "string",
                "description": "The search query to look up on Wikipedia"
            }
        },
        "required": ["search_query"]
    }
)

CALCULATOR_TOOL = Tool(
    name="Calculator",
    func=calculator,
    description="Perform mathematical calculations (add, subtract, multiply, divide)",
    parameters={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "Mathematical operation to perform"
            },
            "a": {
                "type": "number",
                "description": "First number"
            },
            "b": {
                "type": "number",
                "description": "Second number"
            }
        },
        "required": ["operation", "a", "b"]
    }
)

DATE_TOOL = Tool(
    name="Date_of_today",
    func=date_today,
    description="Get the current date",
    parameters={
        "type": "object",
        "properties": {},
        "required": []
    }
)

WEB_SEARCH_TOOL = Tool(
    name="WebSearch",
    func=web_search,
    description="Perform a web search for current information",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            }
        },
        "required": ["query"]
    }
)

# Convenience functions to get tool instances
def wikipedia_search_tool() -> Tool:
    """Get Wikipedia search tool."""
    return WIKIPEDIA_TOOL

def calculator_tool() -> Tool:
    """Get calculator tool."""
    return CALCULATOR_TOOL

def date_today_tool() -> Tool:
    """Get date tool."""
    return DATE_TOOL

def web_search_tool() -> Tool:
    """Get web search tool."""
    return WEB_SEARCH_TOOL

def get_default_tools() -> list[Tool]:
    """Get a list of commonly useful tools."""
    return [
        wikipedia_search_tool(),
        calculator_tool(),
        date_today_tool(),
        web_search_tool()
    ]
