"""FastAPI application with modern endpoints for ReAct agents."""

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import structlog

from ..core import Agent, AgentConfig, ReActExecutor, Message, Tool
from ..utils.logging import setup_logging

# Set up structured logging
setup_logging()
logger = structlog.get_logger(__name__)

# Global executor instance
executor: Optional[ReActExecutor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global executor

    # Startup
    try:
        config = AgentConfig()
        executor = ReActExecutor(config)
        logger.info("Application started successfully")
    except Exception as e:
        logger.error("Failed to start application", error=str(e))
        raise

    yield

    # Shutdown
    logger.info("Application shutting down")


# Create FastAPI app
app = FastAPI(
    title="ReAct Agent API",
    description="Modern async API for ReAct reasoning agents",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API
class ToolCreateRequest(BaseModel):
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Tool parameters schema")


class AgentCreateRequest(BaseModel):
    name: str = Field(..., description="Agent name")
    instructions: str = Field(..., description="Agent instructions")
    model: Optional[str] = Field("gpt-4o", description="OpenAI model")
    tools: List[ToolCreateRequest] = Field(default_factory=list, description="Agent tools")


class ExecuteRequest(BaseModel):
    agent: AgentCreateRequest
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="Conversation messages")
    context_variables: Optional[Dict[str, Any]] = Field(None, description="Context variables")
    max_turns: Optional[int] = Field(None, description="Maximum interaction turns")
    use_react_style: bool = Field(False, description="Use traditional ReAct style execution")


class ExecuteQueryRequest(BaseModel):
    agent: AgentCreateRequest
    query: str = Field(..., description="User query")
    context_variables: Optional[Dict[str, Any]] = Field(None, description="Context variables")
    use_react_style: bool = Field(False, description="Use traditional ReAct style execution")


class MessageResponse(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ExecutionResponse(BaseModel):
    messages: List[MessageResponse]
    final_answer: Optional[str] = None
    agent_name: Optional[str] = None
    context_variables: Dict[str, Any] = Field(default_factory=dict)
    token_usage: Optional[Dict[str, int]] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None


def create_agent_from_request(agent_req: AgentCreateRequest) -> Agent:
    """Create Agent instance from API request."""
    tools = []
    for tool_req in agent_req.tools:
        tool = Tool(
            name=tool_req.name,
            func=lambda **kwargs: f"Tool {tool_req.name} executed with {kwargs}",  # Placeholder
            description=tool_req.description,
            parameters=tool_req.parameters
        )
        tools.append(tool)

    return Agent(
        name=agent_req.name,
        model=agent_req.model,
        instructions=agent_req.instructions,
        tools=tools
    )


def convert_messages_to_response(messages: List[Message]) -> List[MessageResponse]:
    """Convert internal messages to API response format."""
    return [
        MessageResponse(
            role=msg.role,
            content=msg.content,
            tool_calls=msg.tool_calls,
            tool_call_id=msg.tool_call_id,
            name=msg.name
        )
        for msg in messages
    ]


@app.post("/execute", response_model=ExecutionResponse)
async def execute_agent(request: ExecuteRequest, background_tasks: BackgroundTasks):
    """Execute agent with conversation history."""
    global executor
    if not executor:
        raise HTTPException(status_code=503, detail="Agent executor not initialized")

    try:
        # Create agent from request
        agent = create_agent_from_request(request.agent)

        # Convert messages
        messages = [
            Message(
                role=msg["role"],
                content=msg["content"],
                tool_calls=msg.get("tool_calls"),
                tool_call_id=msg.get("tool_call_id"),
                name=msg.get("name")
            )
            for msg in request.messages
        ]

        # Execute
        if request.use_react_style:
            # For ReAct style, we need a query
            if not messages or messages[0].role != "user":
                raise HTTPException(status_code=400, detail="ReAct style requires initial user message")
            result = await executor.execute_react_style(
                agent=agent,
                query=messages[0].content,
                context_variables=request.context_variables
            )
        else:
            result = await executor.execute(
                agent=agent,
                messages=messages,
                context_variables=request.context_variables,
                max_turns=request.max_turns
            )

        return ExecutionResponse(
            messages=convert_messages_to_response(result.messages),
            final_answer=result.final_answer,
            agent_name=result.agent.name if result.agent else None,
            context_variables=result.context_variables,
            token_usage=result.token_usage,
            execution_time=result.execution_time,
            error=result.error
        )

    except Exception as e:
        logger.error("Execution failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")


@app.post("/query", response_model=ExecutionResponse)
async def execute_query(request: ExecuteQueryRequest):
    """Execute agent with a simple query."""
    global executor
    if not executor:
        raise HTTPException(status_code=503, detail="Agent executor not initialized")

    try:
        # Create agent from request
        agent = create_agent_from_request(request.agent)

        # Execute
        if request.use_react_style:
            result = await executor.execute_react_style(
                agent=agent,
                query=request.query,
                context_variables=request.context_variables
            )
        else:
            # Convert query to message for standard execution
            messages = [Message(role="user", content=request.query)]
            result = await executor.execute(
                agent=agent,
                messages=messages,
                context_variables=request.context_variables
            )

        return ExecutionResponse(
            messages=convert_messages_to_response(result.messages),
            final_answer=result.final_answer,
            agent_name=result.agent.name if result.agent else None,
            context_variables=result.context_variables,
            token_usage=result.token_usage,
            execution_time=result.execution_time,
            error=result.error
        )

    except Exception as e:
        logger.error("Query execution failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global executor
    return {
        "status": "healthy" if executor else "unhealthy",
        "executor_ready": executor is not None
    }


@app.get("/config")
async def get_config():
    """Get current configuration (without sensitive data)."""
    config = AgentConfig()
    return {
        "openai_model": config.openai_model,
        "max_interactions": config.max_interactions,
        "token_limit": config.token_limit,
        "enable_caching": config.enable_caching,
        "log_level": config.log_level
    }
