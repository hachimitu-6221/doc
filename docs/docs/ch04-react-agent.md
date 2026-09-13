# 第 4 章 · 实现基础 ReAct Agent

> 对应原书 第 4 章（小节号在各代码块中标注）。本章文档：`docs/ch04-react-agent.md`。

## 本章构建目标

这是全书代码量最大的一章：4.2 节给出 agent 架构总览，4.3–4.6 节依次实现 ExecutionContext、工具抽象、LLM 通信层和 Agent 主类，4.7 节加结构化输出，4.8 节用 GAIA 测试。

本章结束后，`scratch_agents` 包的核心骨架（types / context / tools.base / llm / agent）全部就位，已经能跑通「思考→调用工具→观察结果→再思考」的 ReAct 循环。后面第 5–9 章都是在这个骨架上叠加能力。

注意：`agent.py`、`tools/base.py`、`context.py` 在第 5/6/8/9 章还会被**整体替换**。每个文件的清单都是「本章结束后的完整内容」，直接整文件覆盖即可，不用自己合并 diff。

## 4.3 节：ExecutionContext——agent 的中央存储

`types.py` 定义通信的基本类型（Listing 4.2 Message、4.3 Event），`context.py` 定义 ExecutionContext（Listing 4.4）——执行期间所有事件的存储，以及返回给调用方的 AgentResult。本章的 context 是最小版本：还没有会话（ch06）、代码环境（ch08）、agent 转移（ch09）这些字段。

#### 📄 `scratch_agents/types.py`

- **状态**：ch04（首次创建，之后不变）
- **对应书内容**：Listing 4.2（Message / ToolCall / ToolResult）、4.3（Event）
- **行数**：45 行（清单为完整文件内容，从第 1 行到第 45 行）

<!-- FILE: scratch_agents/types.py -->
<!-- STATE: ch04 -->

````python
"""Core types for the scratch_agents framework."""

from __future__ import annotations

import uuid
from typing import List, Literal, Union

from pydantic import BaseModel, Field
from datetime import datetime


class Message(BaseModel):
    """A text message in the conversation."""
    type: Literal["message"] = "message"
    role: Literal["system", "user", "assistant"]
    content: str


class ToolCall(BaseModel):
    """LLM's request to execute a tool."""
    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    name: str
    arguments: dict


class ToolResult(BaseModel):
    """Result from tool execution."""
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    status: Literal["success", "error"]
    content: list


ContentItem = Union[Message, ToolCall, ToolResult]


class Event(BaseModel):
    """A recorded occurrence during agent execution."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    author: str  # "user" or agent name
    content: List[ContentItem] = Field(default_factory=list)
````


#### 📄 `scratch_agents/context.py`

- **状态**：ch04（中间态，ch06 会被替换）
- **对应书内容**：Listing 4.4（ExecutionContext、AgentResult）
- **行数**：37 行（清单为完整文件内容，从第 1 行到第 37 行）
- **说明**：ch04 版只有 execution_id / events / current_step / state / final_result；PendingToolCall、ToolConfirmation 和 status/pending_tool_calls 是 ch06 加的。

<!-- FILE: scratch_agents/context.py -->
<!-- STATE: ch04 -->

````python
"""Execution context and result types for the scratch_agents framework."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from scratch_agents.types import Event


@dataclass
class ExecutionContext:
    """Central storage for all execution state."""

    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: List[Event] = field(default_factory=list)
    current_step: int = 0
    state: Dict[str, Any] = field(default_factory=dict)
    final_result: Optional[str | BaseModel] = None

    def add_event(self, event: Event):
        """Append an event to the execution history."""
        self.events.append(event)

    def increment_step(self):
        """Move to the next execution step."""
        self.current_step += 1


@dataclass
class AgentResult:
    """Result of an agent execution."""
    output: Any  # str | BaseModel
    context: ExecutionContext
````


## 4.4 节：工具抽象（BaseTool / FunctionTool）

`tools/base.py`（Listing 4.5 BaseTool、4.6 FunctionTool + `@tool` 装饰器）把任意 Python 函数包装成统一接口。ch04 版本还没有 `requires_confirmation`（ch06）和 `sandbox_executable`（ch08）。

`tools/mcp.py` 把 MCP server 的工具转换成 FunctionTool（Listing 4.7 load_mcp_tools、4.8 单个 MCP 工具包装），并附带第 3 章的 `mcp_tools_to_openai_format`（Listing 3.23）和 `mcp_connection` 上下文管理器。

#### 📄 `scratch_agents/tools/base.py`

- **状态**：ch04（中间态，ch06 会被替换）
- **对应书内容**：Listing 4.5（BaseTool）、4.6（FunctionTool、@tool）
- **行数**：93 行（清单为完整文件内容，从第 1 行到第 93 行）

<!-- FILE: scratch_agents/tools/base.py -->
<!-- STATE: ch04 -->

````python
"""Base tool abstraction for the scratch_agents framework."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

from pydantic import BaseModel

from scratch_agents.tools.helpers import format_tool_definition, function_to_input_schema
from scratch_agents.context import ExecutionContext


class BaseTool(ABC):
    """Abstract base class for all tools."""

    def __init__(
        self,
        name: str = None,
        description: str = None,
        tool_definition: Dict[str, Any] = None,
    ):
        self.name = name or self.__class__.__name__
        self.description = description or self.__doc__ or ""
        self._tool_definition = tool_definition

    @property
    def tool_definition(self) -> Dict[str, Any] | None:
        return self._tool_definition

    @abstractmethod
    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        pass

    async def __call__(self, context: ExecutionContext, **kwargs) -> Any:
        return await self.execute(context, **kwargs)


class FunctionTool(BaseTool):
    """Wraps a Python function as a BaseTool."""

    def __init__(
        self,
        func: Callable,
        name: str = None,
        description: str = None,
        tool_definition: Dict[str, Any] = None,
    ):
        self.func = func
        self.needs_context = "context" in inspect.signature(func).parameters

        resolved_name = name or func.__name__
        resolved_desc = description or (func.__doc__ or "").strip()

        super().__init__(
            name=resolved_name,
            description=resolved_desc,
            tool_definition=tool_definition,
        )

        if self._tool_definition is None:
            self._tool_definition = self._generate_definition()

    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        """Execute the wrapped function."""
        if self.needs_context:
            result = self.func(context=context, **kwargs)
        else:
            result = self.func(**kwargs)

        if inspect.iscoroutine(result):
            return await result
        return result

    def _generate_definition(self) -> Dict[str, Any]:
        """Generate tool definition from function signature."""
        parameters = function_to_input_schema(self.func)
        return format_tool_definition(self.name, self.description, parameters)


def tool(func=None, *, name=None, description=None):
    """Decorator to create a FunctionTool from a function."""
    def decorator(f):
        return FunctionTool(
            func=f,
            name=name,
            description=description,
        )

    if func is not None:
        return decorator(func)
    return decorator
````


#### 📄 `scratch_agents/tools/mcp.py`

- **状态**：ch04（首次创建，之后不变）
- **对应书内容**：Listing 4.7（load_mcp_tools）、4.8（_create_mcp_tool）、3.23（mcp_tools_to_openai_format）
- **行数**：97 行（清单为完整文件内容，从第 1 行到第 97 行）

<!-- FILE: scratch_agents/tools/mcp.py -->
<!-- STATE: ch04 -->

````python
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from scratch_agents.tools.base import BaseTool, FunctionTool
from scratch_agents.tools.helpers import format_tool_definition


def _extract_text_content(result) -> str:
    """Extract plain text from an MCP CallToolResult."""
    parts = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts)


def _create_mcp_tool(mcp_tool, connection: dict) -> FunctionTool:
    """Create a FunctionTool that wraps an MCP tool."""

    async def call_mcp(**kwargs):
        async with stdio_client(StdioServerParameters(**connection)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(mcp_tool.name, kwargs)
                return _extract_text_content(result)

    tool_definition = {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.inputSchema,
        },
    }

    return FunctionTool(
        func=call_mcp,
        name=mcp_tool.name,
        description=mcp_tool.description,
        tool_definition=tool_definition,
    )


async def load_mcp_tools(connection: dict) -> list[BaseTool]:
    """Load tools from an MCP server and convert to FunctionTools.

    Matches CH04 Listing 4.7. Each MCP tool becomes a FunctionTool that
    re-establishes the connection on each invocation.
    """
    tools: list[BaseTool] = []

    async with stdio_client(StdioServerParameters(**connection)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()

            for mcp_tool in mcp_tools.tools:
                func_tool = _create_mcp_tool(mcp_tool, connection)
                tools.append(func_tool)

    return tools


def mcp_tools_to_openai_format(mcp_tools) -> list[dict]:
    """Convert MCP tool definitions to OpenAI tool format (CH03 Listing 3.23)."""
    return [
        format_tool_definition(
            name=tool.name,
            description=tool.description,
            parameters=tool.inputSchema,
        )
        for tool in mcp_tools.tools
    ]


@asynccontextmanager
async def mcp_connection(connection: dict):
    """Context manager for maintaining an MCP server connection.

    Usage:
        async with mcp_connection({"command": "npx", "args": [...]}) as session:
            tools = await session.list_tools()
            result = await session.call_tool("tool_name", arguments={...})
    """
    server_params = StdioServerParameters(
        command=connection["command"],
        args=connection.get("args", []),
        env=connection.get("env"),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session
````


#### 📄 `scratch_agents/tools/__init__.py`

- **状态**：ch04（中间态，ch09 会更新）
- **对应书内容**：无（包导出约定）
- **行数**：6 行（清单为完整文件内容，从第 1 行到第 6 行）
- **说明**：在 ch03 版基础上追加 base 与 mcp 的导出。

<!-- FILE: scratch_agents/tools/__init__.py -->
<!-- STATE: ch04 -->

````python
"""Tool modules for the scratch_agents framework."""

from scratch_agents.tools.base import BaseTool, FunctionTool, tool
from scratch_agents.tools.search import search_web
from scratch_agents.tools.calculator import calculator
from scratch_agents.tools.mcp import load_mcp_tools, mcp_connection, mcp_tools_to_openai_format
````


## 4.5 节：LLM 通信层

`llm.py` 是 provider 适配层：LlmRequest（Listing 4.9，决定发什么）、LlmResponse（Listing 4.10）、LlmClient（Listing 4.11，内部用 LiteLLM 的 `acompletion`）。`build_messages()`（Listing 4.12）把 Request 展平成 API 消息格式，`LlmClient._parse_response()`（Listing 4.13）把响应解析回 ToolCall/Message。`ask()` 是第 6 章长期记忆抽取结构化结果时要用到的便捷方法。

#### 📄 `scratch_agents/llm.py`

- **状态**：ch04（首次创建，之后不变）
- **对应书内容**：Listing 4.9（LlmRequest）、4.10（LlmResponse）、4.11（LlmClient）、4.12（build_messages）、4.13（_parse_response）
- **行数**：172 行（清单为完整文件内容，从第 1 行到第 172 行）

<!-- FILE: scratch_agents/llm.py -->
<!-- STATE: ch04 -->

````python
"""LLM communication layer for the scratch_agents framework."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Type, Union

from litellm import acompletion
from pydantic import BaseModel, Field

from scratch_agents.types import ContentItem, Message, ToolCall, ToolResult
from scratch_agents.tools.base import BaseTool


class LlmRequest(BaseModel):
    """Request object for LLM calls."""
    model_config = {"arbitrary_types_allowed": True}

    instructions: List[str] = Field(default_factory=list)
    contents: List[ContentItem] = Field(default_factory=list)
    tools: List[BaseTool] = Field(default_factory=list)
    tool_choice: Optional[str] = None
    model_id: Optional[str] = None

    def append_instructions(self, text: str) -> None:
        """Append a single instruction string to the instructions list."""
        self.instructions.append(text)


class LlmResponse(BaseModel):
    """Response object from LLM calls."""
    content: List[ContentItem] = Field(default_factory=list)
    error_message: Optional[str] = None
    usage_metadata: Dict[str, Any] = Field(default_factory=dict)


def build_messages(request: LlmRequest) -> List[dict]:
    """Convert LlmRequest to API message format."""
    messages = []

    for instruction in request.instructions:
        messages.append({"role": "system", "content": instruction})

    for item in request.contents:
        if isinstance(item, Message):
            messages.append({"role": item.role, "content": item.content})

        elif isinstance(item, ToolCall):
            tool_call_dict = {
                "id": item.tool_call_id,
                "type": "function",
                "function": {
                    "name": item.name,
                    "arguments": json.dumps(item.arguments),
                },
            }
            if messages and messages[-1]["role"] == "assistant":
                messages[-1].setdefault("tool_calls", []).append(tool_call_dict)
            else:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call_dict],
                })

        elif isinstance(item, ToolResult):
            messages.append({
                "role": "tool",
                "tool_call_id": item.tool_call_id,
                "content": str(item.content[0]) if item.content else "",
            })

    return messages


class LlmClient:
    """Client for LLM API calls using LiteLLM."""

    def __init__(self, model: str, **config):
        self.model = model
        self.config = config

    async def generate(self, request: LlmRequest) -> LlmResponse:
        """Generate a response from the LLM."""
        try:
            messages = build_messages(request)
            tools = [t.tool_definition for t in request.tools] if request.tools else None

            response = await acompletion(
                model=self.model,
                messages=messages,
                tools=tools,
                **({"tool_choice": request.tool_choice} if request.tool_choice else {}),
                **self.config,
            )

            return self._parse_response(response)
        except Exception as e:
            return LlmResponse(error_message=str(e))

    async def ask(
        self,
        prompt: str,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> Union[str, BaseModel]:
        """Convenience method for one-shot prompts with optional structured output."""
        if response_format is not None:
            schema_text = json.dumps(response_format.model_json_schema())
            instruction = (
                f"{prompt}\n\nRespond ONLY with valid JSON matching this schema:\n"
                f"{schema_text}"
            )
        else:
            instruction = prompt

        request = LlmRequest(
            model_id=self.model,
            instructions=[instruction],
            contents=[Message(role="user", content="Please respond.")],
        )
        response = await self.generate(request)
        if response.error_message:
            raise RuntimeError(f"LLM request failed: {response.error_message}")

        text = ""
        for item in response.content:
            if isinstance(item, Message):
                text = item.content
                break

        if response_format is None:
            return text

        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].lstrip()
        return response_format.model_validate_json(cleaned.strip())

    def _build_messages(self, request: LlmRequest) -> List[dict]:
        """Backwards-compatible thin wrapper around module-level build_messages."""
        return build_messages(request)

    def _parse_response(self, response) -> LlmResponse:
        """Convert API response to LlmResponse."""
        choice = response.choices[0]
        content_items = []

        if choice.message.content:
            content_items.append(Message(
                role="assistant",
                content=choice.message.content,
            ))

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                content_items.append(ToolCall(
                    tool_call_id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))

        return LlmResponse(
            content=content_items,
            usage_metadata={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
        )
````


## 4.6 节：Agent 主类（run / step / think / act）

`agent.py` 是 ReAct 循环本体：构造时 `_setup_tools()` 注册工具（Listing 4.16），`run()` 驱动循环（Listing 4.18/4.20），`step()` 完成一次 think-act（Listing 4.20/4.22/4.23），`think()` 调 LLM，`act()` 执行工具调用并把 ToolResult 记为事件。完成检测在 `_is_final_response()` / `_extract_final_result()`（Listing 4.19）。

ch04 版本只支持 `output_type` 一个构造参数；callbacks（ch05）、session/memory（ch06）、code_execution/skills（ch08）、sub_agents/transfer（ch09）都还没有。

最后建两个 `__init__.py`，让 `from scratch_agents import Agent` 可用；`utils.py`（`display_trace()`，Listing 4.34）用于打印执行轨迹调试。

#### 📄 `scratch_agents/agent.py`

- **状态**：ch04（中间态，ch05 会被替换）
- **对应书内容**：Listing 4.15–4.23（Agent 类与 ReAct 循环）、4.16（_setup_tools）、4.17（AgentResult 在 context.py 中）、4.19（完成检测）
- **行数**：251 行（清单为完整文件内容，从第 1 行到第 251 行）

<!-- FILE: scratch_agents/agent.py -->
<!-- STATE: ch04 -->

````python
"""Core Agent class for the scratch_agents framework."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Type

from pydantic import BaseModel

from scratch_agents.llm import LlmClient, LlmRequest, LlmResponse
from scratch_agents.types import Event, Message, ToolCall, ToolResult
from scratch_agents.tools.base import BaseTool, FunctionTool, tool
from scratch_agents.tools.helpers import format_tool_definition
from scratch_agents.context import AgentResult, ExecutionContext

logger = logging.getLogger(__name__)


class Agent:
    """Tool-calling agent with ReAct loop."""

    def __init__(
        self,
        model: LlmClient,
        tools: List[BaseTool] | None = None,
        instructions: str = "",
        max_steps: int = 10,
        name: str = "agent",
        description: str = "",
        output_type: Optional[Type[BaseModel]] = None,
    ):
        self.model = model
        self.instructions = instructions
        self.max_steps = max_steps
        self.name = name
        self.description = description
        self.output_type = output_type
        self.output_tool_name: str | None = None
        self.tools = self._setup_tools(tools or [])

    # ------------------------------------------------------------------ #
    # Core loop
    # ------------------------------------------------------------------ #

    async def run(
        self,
        user_input: str | None = None,
        context: ExecutionContext | None = None,
        verbose: bool = False,
    ) -> AgentResult:
        """Execute the agent."""
        if context is None:
            context = ExecutionContext()

        if user_input:
            user_event = Event(
                execution_id=context.execution_id,
                author="user",
                content=[Message(role="user", content=user_input)],
            )
            context.add_event(user_event)

        while not context.final_result and context.current_step < self.max_steps:
            await self.step(context, verbose=verbose)

            if context.events:
                last_event = context.events[-1]
                if self._is_final_response(last_event):
                    context.final_result = self._extract_final_result(last_event)

        return AgentResult(output=context.final_result, context=context)

    async def step(
        self,
        context: ExecutionContext,
        verbose: bool = False,
    ) -> None:
        """Perform one think-act cycle."""
        llm_request = self._prepare_llm_request(context)
        llm_response = await self.think(llm_request)

        if verbose:
            self._log_response(llm_response)

        response_event = Event(
            execution_id=context.execution_id,
            author=self.name,
            content=llm_response.content,
        )
        context.add_event(response_event)

        tool_calls = [c for c in llm_response.content if isinstance(c, ToolCall)]
        if tool_calls:
            await self.act(context, tool_calls)

        context.increment_step()

    async def think(self, llm_request: LlmRequest) -> LlmResponse:
        """Call the LLM to decide the next action."""
        return await self.model.generate(llm_request)

    async def act(
        self,
        context: ExecutionContext,
        tool_calls: List[ToolCall],
    ) -> None:
        """Execute the tools requested by the LLM."""
        tools_dict = {t.name: t for t in self.tools}
        results = []

        for tool_call in tool_calls:
            if tool_call.name not in tools_dict:
                results.append(ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status="error",
                    content=[f"Tool '{tool_call.name}' not found"],
                ))
                continue

            tool_obj = tools_dict[tool_call.name]

            try:
                output = await tool_obj(context, **tool_call.arguments)
                results.append(ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status="success",
                    content=[output],
                ))
            except Exception as e:
                results.append(ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status="error",
                    content=[str(e)],
                ))

        if results:
            tool_event = Event(
                execution_id=context.execution_id,
                author=self.name,
                content=results,
            )
            context.add_event(tool_event)

    # ------------------------------------------------------------------ #
    # Internal methods
    # ------------------------------------------------------------------ #

    def _prepare_llm_request(self, context: ExecutionContext) -> LlmRequest:
        """Build an LlmRequest from the current context."""
        flat_contents = []
        for event in context.events:
            flat_contents.extend(event.content)

        instructions = []
        if self.instructions:
            instructions.append(self.instructions)

        if self.output_tool_name:
            tool_choice = "required"
        elif self.tools:
            tool_choice = "auto"
        else:
            tool_choice = None

        return LlmRequest(
            instructions=instructions,
            contents=flat_contents,
            tools=self.tools,
            tool_choice=tool_choice,
        )

    def _is_final_response(self, event: Event) -> bool:
        """Check if this event contains a final response."""
        if self.output_tool_name:
            for item in event.content:
                if (
                    isinstance(item, ToolResult)
                    and item.name == self.output_tool_name
                    and item.status == "success"
                ):
                    return True
            return False

        has_tool_calls = any(isinstance(c, ToolCall) for c in event.content)
        has_tool_results = any(isinstance(c, ToolResult) for c in event.content)
        return not has_tool_calls and not has_tool_results

    def _extract_final_result(self, event: Event) -> Any:
        """Extract the final result from an event."""
        if self.output_tool_name:
            for item in event.content:
                if (
                    isinstance(item, ToolResult)
                    and item.name == self.output_tool_name
                    and item.status == "success"
                    and item.content
                ):
                    return item.content[0]

        for item in event.content:
            if isinstance(item, Message) and item.role == "assistant":
                return item.content
        return None

    def _setup_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
        """Prepare the tools list, including dynamic tools."""
        tools = list(tools)

        if self.output_type is not None:
            output_schema = self.output_type.model_json_schema()
            output_schema.pop("title", None)
            output_schema.pop("$defs", None)

            tool_definition = format_tool_definition(
                "final_answer",
                "Return the final structured answer matching the required schema.",
                {
                    "type": "object",
                    "properties": {"output": output_schema},
                    "required": ["output"],
                },
            )

            captured_type = self.output_type

            def _parse_output(output) -> str:
                if isinstance(output, dict):
                    return captured_type.model_validate(output)
                return output

            final_answer_tool = FunctionTool(
                func=_parse_output,
                name="final_answer",
                description="Return the final structured answer matching the required schema.",
                tool_definition=tool_definition,
            )
            tools.append(final_answer_tool)
            self.output_tool_name = "final_answer"

        return tools

    def _log_response(self, response: LlmResponse):
        """Log LLM response for verbose mode."""
        for item in response.content:
            if isinstance(item, Message):
                logger.info(f"[{self.name}] {item.content}")
            elif isinstance(item, ToolCall):
                logger.info(f"[{self.name}] Tool call: {item.name}({item.arguments})")
````


#### 📄 `scratch_agents/utils.py`

- **状态**：ch04（首次创建，之后不变）
- **对应书内容**：Listing 4.34（display_trace）
- **行数**：17 行（清单为完整文件内容，从第 1 行到第 17 行）

<!-- FILE: scratch_agents/utils.py -->
<!-- STATE: ch04 -->

````python
"""Utility helpers for displaying agent execution traces."""

from scratch_agents.types import Message, ToolCall, ToolResult


def display_trace(context) -> None:
    """Print a readable trace of events recorded in an ExecutionContext."""
    for index, event in enumerate(context.events, start=1):
        print(f"\n[{index}] {event.author}")
        for item in event.content:
            if isinstance(item, Message):
                print(f"  {item.role}: {item.content}")
            elif isinstance(item, ToolCall):
                print(f"  tool call: {item.name}({item.arguments})")
            elif isinstance(item, ToolResult):
                preview = str(item.content[0]) if item.content else ""
                print(f"  tool result: {item.name} -> {preview[:500]}")
````


#### 📄 `scratch_agents/__init__.py`

- **状态**：ch04（中间态，ch06 会更新）
- **对应书内容**：无（包导出约定）
- **行数**：6 行（清单为完整文件内容，从第 1 行到第 6 行）

<!-- FILE: scratch_agents/__init__.py -->
<!-- STATE: ch04 -->

````python
from scratch_agents.types import (
    Message, ToolCall, ToolResult, Event, ContentItem
)
from scratch_agents.context import ExecutionContext, AgentResult
from scratch_agents.llm import LlmClient, LlmRequest, LlmResponse
from scratch_agents.agent import Agent
````


## 4.7 节：结构化输出（tools as output formatters）

思路：给 Agent 传入 `output_type`（Pydantic 模型）后，`_setup_tools()` 动态生成一个 `final_answer` 工具（Listing 4.24–4.25），并把 `tool_choice` 设为 `required` 强迫模型调用它（Listing 4.26）；完成检测改为识别 `final_answer` 的 ToolResult（Listing 4.27–4.28）。这些逻辑已经包含在下面的 ch04 版 `agent.py` 中。

## 本章改动一览

| 文件 | 状态 | 行数 |
|---|---|---|
| `scratch_agents/types.py` | ch04（首次创建，之后不变） | 45 |
| `scratch_agents/context.py` | ch04（中间态，ch06 会被替换） | 37 |
| `scratch_agents/tools/base.py` | ch04（中间态，ch06 会被替换） | 93 |
| `scratch_agents/tools/mcp.py` | ch04（首次创建，之后不变） | 97 |
| `scratch_agents/tools/__init__.py` | ch04（中间态，ch09 会更新） | 6 |
| `scratch_agents/llm.py` | ch04（首次创建，之后不变） | 172 |
| `scratch_agents/agent.py` | ch04（中间态，ch05 会被替换） | 251 |
| `scratch_agents/utils.py` | ch04（首次创建，之后不变） | 17 |
| `scratch_agents/__init__.py` | ch04（中间态，ch06 会更新） | 6 |

> ch04 版 agent.py 相对最终版的差异（快照头注记）：__init__ 只有 output_type；run() 无 session/confirmation/code_env/transfer；step() 无 before_llm_callbacks；act() 无 callbacks 与 confirmation；_setup_tools() 只注册 final_answer；_prepare_llm_request() 无 sandbox/skills prompt。

## 本章自测

```bash
python -c "from scratch_agents import Agent, ExecutionContext, LlmClient; print('ch04 ok')"
```
