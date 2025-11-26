"""Modern agent implementation using dataclasses and better type hints."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
import json


@dataclass
class Tool:
    """Represents a tool that an agent can use."""
    name: str
    func: Callable
    description: str
    parameters: Optional[Dict[str, Any]] = None

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert tool to OpenAI function schema."""
        schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }

        if self.parameters:
            schema["function"]["parameters"]["properties"] = self.parameters.get("properties", {})
            schema["function"]["parameters"]["required"] = self.parameters.get("required", [])

        return schema


@dataclass
class Agent:
    """Modern agent implementation with better type safety."""
    name: str = "Agent"
    model: str = "gpt-4o"
    instructions: Union[str, Callable[[Dict[str, Any]], str]] = "You are a helpful agent."
    tools: List[Tool] = field(default_factory=list)
    tool_choice: Optional[str] = None
    parallel_tool_calls: bool = True

    def get_instructions(self, context_variables: Optional[Dict[str, Any]] = None) -> str:
        """Get agent instructions, resolving callable if needed."""
        if context_variables is None:
            context_variables = {}

        if callable(self.instructions):
            return self.instructions(context_variables)
        return self.instructions

    def get_tools_json_schema(self) -> List[Dict[str, Any]]:
        """Get all tools as JSON schema for OpenAI API."""
        return [tool.to_json_schema() for tool in self.tools]

    def find_tool(self, name: str) -> Optional[Tool]:
        """Find a tool by name."""
        return next((tool for tool in self.tools if tool.name == name), None)

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent."""
        self.tools.append(tool)

    def remove_tool(self, name: str) -> bool:
        """Remove a tool by name. Returns True if removed."""
        original_length = len(self.tools)
        self.tools = [tool for tool in self.tools if tool.name != name]
        return len(self.tools) < original_length


@dataclass
class Message:
    """Represents a chat message."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI API message format."""
        msg = {
            "role": self.role,
            "content": self.content
        }

        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name

        return msg


@dataclass
class ExecutionResult:
    """Result of an agent execution."""
    messages: List[Message]
    final_answer: Optional[str] = None
    agent: Optional[Agent] = None
    context_variables: Dict[str, Any] = field(default_factory=dict)
    token_usage: Optional[Dict[str, int]] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None
