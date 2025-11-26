"""Unified, modern async ReAct executor with enhanced error handling."""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import structlog

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage

from .agent import Agent, Message, Tool, ExecutionResult
from .config import AgentConfig

# Configure structured logging
logger = structlog.get_logger(__name__)


@dataclass
class ThoughtProcess:
    """Represents one step in the ReAct reasoning process."""
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    confidence: Optional[float] = None


class ReActExecutor:
    """Modern, unified async ReAct executor with hybrid approach."""

    def __init__(self, config: AgentConfig, client: Optional[AsyncOpenAI] = None):
        self.config = config
        self.client = client or AsyncOpenAI(**config.get_openai_client_kwargs())
        self.logger = logger.bind(executor_id=id(self))

    async def execute(
        self,
        agent: Agent,
        messages: List[Message],
        context_variables: Optional[Dict[str, Any]] = None,
        max_turns: int = None,
        streaming: bool = False
    ) -> ExecutionResult:
        """Execute agent conversation with modern async patterns."""
        start_time = time.time()
        context_variables = context_variables or {}

        if max_turns is None:
            max_turns = self.config.max_interactions

        # Convert messages to OpenAI format
        conversation_history = [msg.to_openai_format() for msg in messages]
        new_messages = []

        try:
            current_agent = agent
            turn_count = 0

            while turn_count < max_turns:
                turn_count += 1
                self.logger.info("Starting turn", turn=turn_count, agent=current_agent.name)

                # Create inference request
                request_params = await self._create_inference_request(
                    current_agent, conversation_history, context_variables
                )

                # Get completion from OpenAI
                async with self.config.semaphore:
                    completion = await self.client.chat.completions.create(**request_params)

                response_message = completion.choices[0].message

                # Add assistant message to history
                assistant_msg = Message(
                    role="assistant",
                    content=response_message.content or "",
                    tool_calls=[call.model_dump() for call in response_message.tool_calls] if response_message.tool_calls else None
                )
                conversation_history.append(assistant_msg.to_openai_format())
                new_messages.append(assistant_msg)

                # If no tool calls, we're done
                if not response_message.tool_calls:
                    self.logger.info("No tool calls, ending execution")
                    break

                # Process tool calls
                tool_results = await self._execute_tool_calls(
                    response_message.tool_calls,
                    current_agent,
                    context_variables
                )

                # Add tool results to conversation
                conversation_history.extend(tool_results)
                new_messages.extend([
                    Message(role=msg["role"], content=msg["content"],
                           tool_call_id=msg.get("tool_call_id"))
                    for msg in tool_results
                ])

            execution_time = time.time() - start_time
            return ExecutionResult(
                messages=new_messages,
                agent=current_agent,
                context_variables=context_variables,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error("Execution failed", error=str(e), execution_time=execution_time)
            return ExecutionResult(
                messages=new_messages,
                error=str(e),
                execution_time=execution_time
            )

    async def _create_inference_request(
        self,
        agent: Agent,
        history: List[Dict[str, Any]],
        context_variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create OpenAI API request parameters."""
        # Add system instructions
        instructions = agent.get_instructions(context_variables)
        system_messages = [{"role": "system", "content": instructions}]

        # Get available tools
        tools = agent.get_tools_json_schema()

        request_params = {
            "model": agent.model,
            "messages": system_messages + history,
            "temperature": self.config.openai_temperature,
            "max_tokens": self.config.openai_max_tokens,
        }

        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = agent.tool_choice
            if agent.parallel_tool_calls:
                request_params["parallel_tool_calls"] = agent.parallel_tool_calls

        return request_params

    async def _execute_tool_calls(
        self,
        tool_calls: List[Any],
        agent: Agent,
        context_variables: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute tool calls and return results."""
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            self.logger.info("Executing tool", tool_name=tool_name, args=tool_args)

            try:
                # Find and execute tool
                tool = agent.find_tool(tool_name)
                if not tool:
                    raise ValueError(f"Tool {tool_name} not found")

                # Execute tool (handle both sync and async functions)
                if asyncio.iscoroutinefunction(tool.func):
                    result = await tool.func(**tool_args)
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, tool.func, **tool_args
                    )

                result_str = str(result)

            except Exception as e:
                result_str = f"Error executing tool {tool_name}: {str(e)}"
                self.logger.error("Tool execution failed", tool_name=tool_name, error=str(e))

            # Create tool result message
            tool_result = {
                "role": "tool",
                "content": result_str,
                "tool_call_id": tool_call.id
            }
            results.append(tool_result)

        return results

    async def execute_react_style(
        self,
        agent: Agent,
        query: str,
        context_variables: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute in traditional ReAct style with explicit reasoning steps."""
        start_time = time.time()
        context_variables = context_variables or {}

        try:
            thought_process = []
            current_agent = agent

            # Initial messages
            messages = [Message(role="user", content=query)]
            conversation_history = [messages[0].to_openai_format()]

            # ReAct loop
            while len(thought_process) < self.config.max_interactions:
                # Thought step
                thought = await self._generate_thought(query, thought_process, current_agent)
                thought_process.append(ThoughtProcess(thought=thought))

                # Action step
                tool_choice = await self._choose_action(query, thought_process, current_agent)

                if not tool_choice["tool_name"]:
                    # Final answer
                    final_answer = await self._generate_final_answer(query, thought_process, current_agent)
                    thought_process[-1].confidence = 1.0
                    break

                thought_process[-1].action = tool_choice["tool_name"]
                thought_process[-1].action_input = tool_choice["parameters"]

                # Execute action
                try:
                    result = await self._execute_single_tool(
                        tool_choice["tool_name"],
                        tool_choice["parameters"],
                        current_agent
                    )
                    thought_process[-1].observation = str(result)
                except Exception as e:
                    thought_process[-1].observation = f"Error: {str(e)}"

                # Check if we should continue
                should_continue = await self._evaluate_continuation(query, thought_process, current_agent)
                if not should_continue["continue"]:
                    final_answer = should_continue.get("final_answer")
                    thought_process[-1].confidence = should_continue.get("confidence", 0.5)
                    break

            execution_time = time.time() - start_time
            final_answer = final_answer or "Could not determine a final answer."

            return ExecutionResult(
                messages=messages,
                final_answer=final_answer,
                agent=current_agent,
                context_variables=context_variables,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error("ReAct execution failed", error=str(e))
            return ExecutionResult(
                messages=[],
                error=str(e),
                execution_time=execution_time
            )

    async def _generate_thought(
        self, query: str, thought_process: List[ThoughtProcess], agent: Agent
    ) -> str:
        """Generate a reasoning thought."""
        context = self._build_context_history(thought_process)

        prompt = f"""Answer the following request: {query}

{context}

First think step by step about what to do next. Plan your approach carefully."""

        messages = [
            {"role": "system", "content": agent.get_instructions()},
            {"role": "user", "content": prompt}
        ]

        response = await self.client.chat.completions.create(
            model=agent.model,
            messages=messages,
            temperature=self.config.openai_temperature,
            max_tokens=self.config.openai_max_tokens,
        )

        return response.choices[0].message.content or ""

    async def _choose_action(
        self, query: str, thought_process: List[ThoughtProcess], agent: Agent
    ) -> Dict[str, Any]:
        """Choose which tool to use next."""
        context = self._build_context_history(thought_process)
        tools_desc = "\n".join([f"- {tool.name}: {tool.description}" for tool in agent.tools])

        prompt = f"""Given this request: {query}

{context}

Available tools:
{tools_desc}

Choose the next tool to use. If no tool is needed, respond with "FINAL_ANSWER".

Response format:
{{
    "tool_name": "ToolName",
    "parameters": {{"param1": "value1"}}
}}"""

        messages = [
            {"role": "system", "content": agent.get_instructions()},
            {"role": "user", "content": prompt}
        ]

        response = await self.client.chat.completions.create(
            model=agent.model,
            messages=messages,
            temperature=self.config.openai_temperature,
            max_tokens=self.config.openai_max_tokens,
        )

        content = response.choices[0].message.content or "{}"
        try:
            choice = json.loads(content)
            if choice.get("tool_name") == "FINAL_ANSWER":
                return {"tool_name": None, "parameters": {}}
            return choice
        except json.JSONDecodeError:
            return {"tool_name": None, "parameters": {}}

    async def _execute_single_tool(
        self, tool_name: str, parameters: Dict[str, Any], agent: Agent
    ) -> Any:
        """Execute a single tool."""
        tool = agent.find_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")

        if asyncio.iscoroutinefunction(tool.func):
            return await tool.func(**parameters)
        else:
            return await asyncio.get_event_loop().run_in_executor(None, tool.func, **parameters)

    async def _evaluate_continuation(
        self, query: str, thought_process: List[ThoughtProcess], agent: Agent
    ) -> Dict[str, Any]:
        """Evaluate whether to continue or provide final answer."""
        context = self._build_context_history(thought_process)

        prompt = f"""Given this request: {query}

{context}

Do we have enough information to provide a final answer?
Rate your confidence in providing a final answer (0.0 to 1.0).

If confidence >= 0.8, provide the final answer.
If confidence < 0.5, suggest continuing.
Otherwise, you decide.

Response format:
{{
    "continue": false,
    "final_answer": "The answer is...",
    "confidence": 0.9
}}"""

        messages = [
            {"role": "system", "content": agent.get_instructions()},
            {"role": "user", "content": prompt}
        ]

        response = await self.client.chat.completions.create(
            model=agent.model,
            messages=messages,
            temperature=self.config.openai_temperature,
            max_tokens=self.config.openai_max_tokens,
        )

        content = response.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"continue": True, "confidence": 0.5}

    async def _generate_final_answer(
        self, query: str, thought_process: List[ThoughtProcess], agent: Agent
    ) -> str:
        """Generate final answer."""
        context = self._build_context_history(thought_process)

        prompt = f"""Given this request: {query}

{context}

Provide a comprehensive final answer."""

        messages = [
            {"role": "system", "content": agent.get_instructions()},
            {"role": "user", "content": prompt}
        ]

        response = await self.client.chat.completions.create(
            model=agent.model,
            messages=messages,
            temperature=self.config.openai_temperature,
            max_tokens=self.config.openai_max_tokens,
        )

        return response.choices[0].message.content or "Could not generate final answer."

    def _build_context_history(self, thought_process: List[ThoughtProcess]) -> str:
        """Build context history from thought process."""
        if not thought_process:
            return "Context History:\n---"

        lines = ["Context History:"]
        for i, thought in enumerate(thought_process, 1):
            lines.append(f"Step {i}:")
            lines.append(f"  Thought: {thought.thought}")
            if thought.action:
                lines.append(f"  Action: {thought.action} with {thought.action_input}")
            if thought.observation:
                lines.append(f"  Observation: {thought.observation}")
            if thought.confidence is not None:
                lines.append(f"  Confidence: {thought.confidence}")
        lines.append("---")
        return "\n".join(lines)
