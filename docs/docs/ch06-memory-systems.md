# 第 6 章 · 给 Agent 加记忆

> 对应原书 第 6 章（小节号在各代码块中标注）。本章文档：`docs/ch06-memory-systems.md`。

## 本章构建目标

第 6 章解决上下文与记忆：6.2 节讲执行期间的上下文管理（滑动窗口 / token 计数 / 压缩 / 摘要，即 before-LLM 回调），6.3 节讲会话与状态管理（Session、暂停/恢复、人在环审批 HITL），6.4 节讲跨会话的长期记忆（ChromaDB 向量库 + 结构化记忆抽取）。

本章动的文件最多：新增 `memory/` 子包和记忆工具，整体替换 `context.py`、`tools/base.py`、`agent.py` 三个核心文件，并更新包 `__init__.py`。

## 6.2 节：执行期间的上下文管理

`memory/context_optimizer.py`：`create_optimizer_callback()`（Listing 6.1，超过 token 阈值才触发优化）、`count_tokens()`（Listing 6.4，tiktoken 计数）、`apply_sliding_window()`（Listing 6.2）、`apply_compaction()`（Listing 6.5，把工具结果替换成引用消息）、`apply_summarization()`（Listing 6.6，用 LLM 摘要替换旧消息）、`ContextOptimizer`（Listing 6.7，分层组合压缩+摘要）。

#### 📄 `scratch_agents/memory/context_optimizer.py`

- **状态**：ch06（首次创建，之后不变）
- **对应书内容**：Listing 6.1–6.7（上下文优化策略全集）
- **行数**：294 行（清单为完整文件内容，从第 1 行到第 294 行）

<!-- FILE: scratch_agents/memory/context_optimizer.py -->
<!-- STATE: ch06 -->

````python
"""Context optimization strategies: sliding window, compaction, summarization."""

from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from scratch_agents.context import ExecutionContext
from scratch_agents.types import ContentItem, Message, ToolCall, ToolResult

if TYPE_CHECKING:
    from scratch_agents.llm import LlmClient, LlmRequest, LlmResponse


def create_optimizer_callback(apply_optimization, threshold: int = 50000):
    """Factory function that creates a callback applying optimization strategy."""

    async def callback(
        context: ExecutionContext,
        request: "LlmRequest",
    ) -> Optional["LlmResponse"]:
        token_count = count_tokens(request)

        if token_count < threshold:
            return None

        # Support both sync and async functions
        result = apply_optimization(context, request)
        if inspect.isawaitable(result):
            await result
        return None

    return callback


def count_tokens(request: "LlmRequest") -> int:
    """Calculate total token count of LlmRequest."""
    import tiktoken

    from scratch_agents.llm import build_messages

    try:
        encoding = tiktoken.encoding_for_model(request.model_id or "gpt-5.5")
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")

    messages = build_messages(request)
    total_tokens = 0

    for message in messages:
        total_tokens += 4  # per-message overhead

        if message.get("content"):
            total_tokens += len(encoding.encode(str(message["content"])))

        if message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                func = tool_call.get("function", {})
                if func.get("name"):
                    total_tokens += len(encoding.encode(func["name"]))
                if func.get("arguments"):
                    total_tokens += len(encoding.encode(func["arguments"]))

    if request.tools:
        for tool in request.tools:
            tool_def = tool.tool_definition
            if tool_def:
                total_tokens += len(encoding.encode(json.dumps(tool_def)))

    return total_tokens


def apply_sliding_window(
    context: ExecutionContext,
    request: "LlmRequest",
    window_size: int = 20,
) -> None:
    """Sliding window that keeps only the most recent N messages."""
    contents = request.contents

    # Find user message position
    user_message_idx = None
    for i, item in enumerate(contents):
        if isinstance(item, Message) and item.role == "user":
            user_message_idx = i
            break

    if user_message_idx is None:
        return

    # Preserve up to user message
    preserved = contents[: user_message_idx + 1]

    # Keep only the most recent N from remaining items
    remaining = contents[user_message_idx + 1 :]
    if len(remaining) > window_size:
        remaining = remaining[-window_size:]

    request.contents = preserved + remaining


# Tools to compress ToolCall arguments
TOOLCALL_COMPACTION_RULES = {
    "create_file": "[Content saved to file]",
}

# Tools to compress ToolResult content
TOOLRESULT_COMPACTION_RULES = {
    "read_file": "File content from {file_path}. Re-read if needed.",
    "search_web": "Search results processed. Query: {query}. Re-search if needed.",
    "tavily_search": "Search results processed. Query: {query}. Re-search if needed.",
}


def apply_compaction(context: ExecutionContext, request: "LlmRequest") -> None:
    """Compress tool calls and results into reference messages."""
    tool_call_args: Dict[str, Dict] = {}
    compacted = []

    for item in request.contents:
        if isinstance(item, ToolCall):
            tool_call_args[item.tool_call_id] = item.arguments

            if item.name in TOOLCALL_COMPACTION_RULES:
                compressed_args = {
                    k: TOOLCALL_COMPACTION_RULES[item.name] if k == "content" else v
                    for k, v in item.arguments.items()
                }
                compacted.append(
                    ToolCall(
                        tool_call_id=item.tool_call_id,
                        name=item.name,
                        arguments=compressed_args,
                    )
                )
            else:
                compacted.append(item)

        elif isinstance(item, ToolResult):
            if item.name in TOOLRESULT_COMPACTION_RULES:
                args = tool_call_args.get(item.tool_call_id, {})
                template = TOOLRESULT_COMPACTION_RULES[item.name]
                compressed_content = template.format(
                    file_path=args.get("file_path", args.get("path", "unknown")),
                    query=args.get("query", "unknown"),
                )
                compacted.append(
                    ToolResult(
                        tool_call_id=item.tool_call_id,
                        name=item.name,
                        status=item.status,
                        content=[compressed_content],
                    )
                )
            else:
                compacted.append(item)
        else:
            compacted.append(item)

    request.contents = compacted


SUMMARIZATION_PROMPT = """You are summarizing an AI agent's work progress.

Given the following execution history, extract:
1. Key findings: Important information discovered
2. Tools used: List of tools that were called
3. Current status: What has been accomplished and what remains

Be concise. Focus on information that will help the agent continue its work.

Execution History:
{history}

Provide a structured summary."""


async def apply_summarization(
    context: ExecutionContext,
    request: "LlmRequest",
    llm_client: "LlmClient",
    keep_recent: int = 5,
) -> None:
    """Replace old messages with a summary."""
    contents = request.contents

    # Find user message position
    user_idx = None
    for i, item in enumerate(contents):
        if isinstance(item, Message) and item.role == "user":
            user_idx = i
            break

    if user_idx is None:
        return

    # Check previous summary position
    last_summary_idx = context.state.get("last_summary_idx", user_idx)

    # Calculate summarization target range
    summary_start = last_summary_idx + 1
    summary_end = len(contents) - keep_recent

    if summary_end <= summary_start:
        return

    preserved_start = contents[: last_summary_idx + 1]
    preserved_end = contents[summary_end:]
    to_summarize = contents[summary_start:summary_end]

    # Generate summary
    history_text = format_history_for_summary(to_summarize)
    summary = await generate_summary(llm_client, history_text)

    # Add summary to instructions
    request.append_instructions(f"[Previous work summary]\n{summary}")

    # Keep only preserved portions
    request.contents = preserved_start + preserved_end

    # Record summary position
    context.state["last_summary_idx"] = len(preserved_start) - 1


def format_history_for_summary(items: List[ContentItem]) -> str:
    """Convert ContentItem list to text for summarization."""
    lines = []
    for item in items:
        if isinstance(item, Message):
            lines.append(f"[{item.role}]: {item.content[:500]}...")
        elif isinstance(item, ToolCall):
            lines.append(f"[Tool Call]: {item.name}({item.arguments})")
        elif isinstance(item, ToolResult):
            content_preview = str(item.content[0])[:200] if item.content else ""
            lines.append(f"[Tool Result]: {item.name} -> {content_preview}...")
    return "\n".join(lines)


async def generate_summary(llm_client: "LlmClient", history: str) -> str:
    """Generate history summary using LLM."""
    from scratch_agents.llm import LlmRequest as LR

    request = LR(
        instructions=[SUMMARIZATION_PROMPT.format(history=history)],
        contents=[Message(role="user", content="Please summarize.")],
    )

    response = await llm_client.generate(request)

    for item in response.content:
        if isinstance(item, Message):
            return item.content

    return ""


class ContextOptimizer:
    """Hierarchical context optimization strategy."""

    def __init__(
        self,
        llm_client: "LlmClient",
        token_threshold: int = 50000,
        enable_compaction: bool = True,
        enable_summarization: bool = True,
        keep_recent: int = 5,
    ):
        self.llm_client = llm_client
        self.token_threshold = token_threshold
        self.enable_compaction = enable_compaction
        self.enable_summarization = enable_summarization
        self.keep_recent = keep_recent

    async def __call__(
        self,
        context: ExecutionContext,
        request: "LlmRequest",
    ) -> Optional["LlmResponse"]:
        """Register as before_llm_callback."""
        if count_tokens(request) < self.token_threshold:
            return None

        if self.enable_compaction:
            apply_compaction(context, request)
            if count_tokens(request) < self.token_threshold:
                return None

        if self.enable_summarization:
            await apply_summarization(
                context, request, self.llm_client, self.keep_recent
            )

        return None
````


## 6.3 节：会话与状态管理（Session / HITL）

`memory/session.py`：Session（Listing 6.8）、BaseSessionManager（6.9）、InMemorySessionManager（6.10）。

HITL 需要新的数据结构和工具扩展：`context.py` 增加 `PendingToolCall` / `ToolConfirmation` / `AgentResult.status` / `pending_tool_calls`（Listing 6.15–6.17），`tools/base.py` 增加 `requires_confirmation` 和 `get_confirmation_message()`（Listing 6.18–6.19）。三个文件都整体替换为 ch06 版。

#### 📄 `scratch_agents/memory/session.py`

- **状态**：ch06（首次创建，之后不变）
- **对应书内容**：Listing 6.8（Session）、6.9（BaseSessionManager）、6.10（InMemorySessionManager）
- **行数**：84 行（清单为完整文件内容，从第 1 行到第 84 行）

<!-- FILE: scratch_agents/memory/session.py -->
<!-- STATE: ch06 -->

````python
"""Session management for multi-turn conversations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from scratch_agents.types import Event


class Session(BaseModel):
    """Container for persistent conversation state across multiple run() calls."""

    session_id: str
    user_id: str | None = None
    events: list[Event] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class BaseSessionManager(ABC):
    """Abstract base class for session management."""

    @abstractmethod
    async def create(
        self,
        session_id: str,
        user_id: str | None = None,
    ) -> Session:
        """Create a new session."""
        pass

    @abstractmethod
    async def get(self, session_id: str) -> Session | None:
        """Retrieve a session by ID. Returns None if not found."""
        pass

    @abstractmethod
    async def save(self, session: Session) -> None:
        """Persist session changes to storage."""
        pass

    async def get_or_create(
        self,
        session_id: str,
        user_id: str | None = None,
    ) -> Session:
        """Get existing session or create new one."""
        session = await self.get(session_id)
        if session is None:
            session = await self.create(session_id, user_id)
        return session


class InMemorySessionManager(BaseSessionManager):
    """In-memory session storage for development and testing."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    async def create(
        self,
        session_id: str,
        user_id: str | None = None,
    ) -> Session:
        """Create a new session."""
        if session_id in self._sessions:
            raise ValueError(f"Session {session_id} already exists")

        session = Session(session_id=session_id, user_id=user_id)
        self._sessions[session_id] = session
        return session

    async def get(self, session_id: str) -> Session | None:
        """Retrieve a session by ID."""
        return self._sessions.get(session_id)

    async def save(self, session: Session) -> None:
        """Save session to storage."""
        self._sessions[session.session_id] = session
````


#### 📄 `scratch_agents/context.py`

- **状态**：ch06（中间态，ch08 会被替换）
- **对应书内容**：Listing 6.11（ExecutionContext 增加 session/session_manager/memory_manager）、6.15（PendingToolCall）、6.16（ToolConfirmation）、6.17（AgentResult 增加状态）
- **行数**：58 行（清单为完整文件内容，从第 1 行到第 58 行）
- **说明**：ch09 还会再加 transfer_to / transfer_tools；本版注释里标了 `# CH06 session` 等来源。

<!-- FILE: scratch_agents/context.py -->
<!-- STATE: ch06 -->

````python
"""Execution context and result types for the scratch_agents framework."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from scratch_agents.types import Event, ToolCall


@dataclass
class ExecutionContext:
    """Central storage for all execution state."""

    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: List[Event] = field(default_factory=list)
    current_step: int = 0
    state: Dict[str, Any] = field(default_factory=dict)
    final_result: Optional[str | BaseModel] = None
    # NEW: CH06 session
    session: Optional[Any] = None
    session_manager: Optional[Any] = None

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
    # NEW: CH06
    status: str = "complete"  # "complete" | "pending_confirmation" | "error"
    pending_tool_calls: list = field(default_factory=list)


# NEW: CH06 Human-in-the-loop types
class PendingToolCall(BaseModel):
    """A tool call awaiting user confirmation."""
    tool_call: ToolCall
    confirmation_message: str


class ToolConfirmation(BaseModel):
    """User's response to a pending tool call."""
    tool_call_id: str
    approved: bool
    modified_arguments: dict | None = None
    reason: str | None = None
````


#### 📄 `scratch_agents/tools/base.py`

- **状态**：ch06（中间态，ch08 会被替换）
- **对应书内容**：Listing 6.18（BaseTool 增加确认参数）、6.19（@tool 支持 requires_confirmation）
- **行数**：110 行（清单为完整文件内容，从第 1 行到第 110 行）
- **说明**：相对 ch04：新增 requires_confirmation、confirmation_message_template、get_confirmation_message()；sandbox_executable 要到 ch08。

<!-- FILE: scratch_agents/tools/base.py -->
<!-- STATE: ch06 -->

````python
"""Base tool abstraction for the scratch_agents framework."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

from pydantic import BaseModel

from scratch_agents.context import ExecutionContext
from scratch_agents.tools.helpers import format_tool_definition, function_to_input_schema


class BaseTool(ABC):
    """Abstract base class for all tools."""

    def __init__(
        self,
        name: str = None,
        description: str = None,
        tool_definition: Dict[str, Any] = None,
        # NEW: CH06
        requires_confirmation: bool = False,
        confirmation_message_template: str = "",
    ):
        self.name = name or self.__class__.__name__
        self.description = description or self.__doc__ or ""
        self._tool_definition = tool_definition
        self.requires_confirmation = requires_confirmation
        self.confirmation_message_template = confirmation_message_template

    @property
    def tool_definition(self) -> Dict[str, Any] | None:
        return self._tool_definition

    # NEW: CH06
    def get_confirmation_message(self, arguments: dict) -> str:
        return self.confirmation_message_template.format(name=self.name, arguments=arguments)

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
        # NEW: CH06
        requires_confirmation: bool = False,
        confirmation_message_template: str = "",
    ):
        self.func = func
        self.needs_context = "context" in inspect.signature(func).parameters

        resolved_name = name or func.__name__
        resolved_desc = description or (func.__doc__ or "").strip()

        super().__init__(
            name=resolved_name,
            description=resolved_desc,
            tool_definition=tool_definition,
            requires_confirmation=requires_confirmation,
            confirmation_message_template=confirmation_message_template,
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


def tool(func=None, *, name=None, description=None,
         requires_confirmation=False, confirmation_message=None):
    """Decorator to create a FunctionTool from a function."""
    def decorator(f):
        return FunctionTool(
            func=f,
            name=name,
            description=description,
            requires_confirmation=requires_confirmation,
            confirmation_message_template=confirmation_message or "",
        )

    if func is not None:
        return decorator(func)
    return decorator
````


## 6.3 节续：Agent 的暂停 / 恢复 / 确认处理

`agent.py` 整体替换为 ch06 版：`run()` 支持 session 加载/保存和确认恢复（Listing 6.12、6.20–6.22），`act()` 遇到 `requires_confirmation` 的工具时暂停并返回 `pending_confirmation`（Listing 6.20–6.22），新增 `_process_confirmations()`。`before_llm_callbacks` 也在本章加入 `step()`（配合 6.2 节的优化器回调）。

#### 📄 `scratch_agents/agent.py`

- **状态**：ch06（中间态，ch08 会被替换）
- **对应书内容**：Listing 6.12（session 集成）、6.20–6.22（run/step/act 的确认处理）、6.22 内 _process_confirmations
- **行数**：445 行（清单为完整文件内容，从第 1 行到第 445 行）
- **说明**：相对 ch05（快照头注记）：__init__ 增加 session_manager/memory_manager/before_llm_callbacks；run() 增加 session 存取、确认处理、记忆保存；step() 执行 before_llm_callbacks；act() 检查 requires_confirmation；_setup_tools() 注册记忆工具；新增 _process_confirmations()。

<!-- FILE: scratch_agents/agent.py -->
<!-- STATE: ch06 -->

````python
"""Core Agent class for the scratch_agents framework."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Type

from pydantic import BaseModel

from scratch_agents.llm import LlmClient, LlmRequest, LlmResponse
from scratch_agents.types import Event, Message, ToolCall, ToolResult
from scratch_agents.tools.base import BaseTool, FunctionTool, tool
from scratch_agents.tools.helpers import format_tool_definition
from scratch_agents.context import (
    AgentResult,
    ExecutionContext,
    PendingToolCall,
    ToolConfirmation,
)

if TYPE_CHECKING:
    from scratch_agents.memory.long_term import TaskMemoryManager
    from scratch_agents.memory.session import BaseSessionManager

logger = logging.getLogger(__name__)


class Agent:
    """Tool-calling agent with ReAct loop, callbacks, sessions, and memory."""

    def __init__(
        self,
        model: LlmClient,
        tools: List[BaseTool] | None = None,
        instructions: str = "",
        max_steps: int = 10,
        name: str = "agent",
        description: str = "",
        output_type: Optional[Type[BaseModel]] = None,
        # CH05 callbacks
        before_tool_callbacks: list[Callable] | None = None,
        after_tool_callbacks: list[Callable] | None = None,
        # NEW: CH06
        session_manager: Optional["BaseSessionManager"] = None,
        memory_manager: Optional["TaskMemoryManager"] = None,
        before_llm_callbacks: list[Callable] | None = None,
    ):
        self.model = model
        self.instructions = instructions
        self.max_steps = max_steps
        self.name = name
        self.description = description
        self.output_type = output_type
        self.output_tool_name: str | None = None
        self.before_tool_callbacks = before_tool_callbacks or []
        self.after_tool_callbacks = after_tool_callbacks or []
        self.before_llm_callbacks = before_llm_callbacks or []
        self.session_manager = session_manager
        self.memory_manager = memory_manager
        self.tools = self._setup_tools(tools or [])

    # ------------------------------------------------------------------ #
    # Core loop
    # ------------------------------------------------------------------ #

    async def run(
        self,
        user_input: str | None = None,
        context: ExecutionContext | None = None,
        session_id: str | None = None,
        tool_confirmations: list[ToolConfirmation] | None = None,
        verbose: bool = False,
    ) -> AgentResult:
        """Execute the agent."""

        # NEW: Load or create session
        session = None
        if session_id and self.session_manager:
            session = await self.session_manager.get_or_create(session_id)

        if context is None:
            context = ExecutionContext(
                session=session,
                session_manager=self.session_manager,
            )
            # Restore previous events from session
            if session:
                context.events = list(session.events)
                context.state = dict(session.state)

        # NEW: Handle tool confirmations (human-in-the-loop resume)
        if tool_confirmations:
            await self._process_confirmations(context, tool_confirmations)

        if user_input:
            user_event = Event(
                execution_id=context.execution_id,
                author="user",
                content=[Message(role="user", content=user_input)],
            )
            context.add_event(user_event)

        while not context.final_result and context.current_step < self.max_steps:
            result = await self.step(context, verbose=verbose)

            # NEW: Check for pending tool calls (human-in-the-loop)
            if result and result.status == "pending_confirmation":
                if session and self.session_manager:
                    session.events = list(context.events)
                    session.state = dict(context.state)
                    await self.session_manager.save(session)
                return result

            if context.events:
                last_event = context.events[-1]
                if self._is_final_response(last_event):
                    context.final_result = self._extract_final_result(last_event)

        # NEW: Save memory
        if self.memory_manager:
            try:
                await self.memory_manager.save(context)
            except Exception as e:
                logger.warning(f"Failed to save memory: {e}")

        # NEW: Save session
        if session and self.session_manager:
            session.events = list(context.events)
            session.state = dict(context.state)
            await self.session_manager.save(session)

        return AgentResult(output=context.final_result, context=context)

    async def step(
        self,
        context: ExecutionContext,
        verbose: bool = False,
    ) -> AgentResult | None:
        """Perform one think-act cycle."""
        llm_request = self._prepare_llm_request(context)

        # NEW: Run before-LLM callbacks
        for callback in self.before_llm_callbacks:
            cb_result = callback(context, llm_request)
            if hasattr(cb_result, "__await__"):
                cb_result = await cb_result
            if isinstance(cb_result, LlmResponse):
                llm_response = cb_result
                break
        else:
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
            result = await self.act(context, tool_calls)
            if result and result.status == "pending_confirmation":
                return result

        context.increment_step()
        return None

    async def think(self, llm_request: LlmRequest) -> LlmResponse:
        """Call the LLM to decide the next action."""
        return await self.model.generate(llm_request)

    async def act(
        self,
        context: ExecutionContext,
        tool_calls: List[ToolCall],
    ) -> AgentResult | None:
        """Execute the tools requested by the LLM."""
        tools_dict = {t.name: t for t in self.tools}
        results = []
        pending = []

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

            # NEW: Check if tool requires confirmation
            if tool_obj.requires_confirmation:
                msg = tool_obj.get_confirmation_message(tool_call.arguments)
                pending.append(PendingToolCall(
                    tool_call=tool_call,
                    confirmation_message=msg,
                ))
                continue

            # before_tool callbacks (CH05)
            skip = False
            for cb in self.before_tool_callbacks:
                cb_result = cb(context, tool_call)
                if hasattr(cb_result, "__await__"):
                    cb_result = await cb_result
                if cb_result is not None:
                    results.append(ToolResult(
                        tool_call_id=tool_call.tool_call_id,
                        name=tool_call.name,
                        status="success",
                        content=[cb_result],
                    ))
                    skip = True
                    break

            if skip:
                continue

            try:
                output = await tool_obj(context, **tool_call.arguments)
                tool_result = ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status="success",
                    content=[output],
                )
            except Exception as e:
                tool_result = ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status="error",
                    content=[str(e)],
                )

            # after_tool callbacks (CH05)
            for cb in self.after_tool_callbacks:
                cb_result = cb(context, tool_result)
                if hasattr(cb_result, "__await__"):
                    cb_result = await cb_result
                if cb_result is not None:
                    tool_result = cb_result

            results.append(tool_result)

        # NEW: If there are pending confirmations, pause execution
        if pending:
            context.state["pending_tool_calls"] = [
                p.model_dump() for p in pending
            ]
            if results:
                tool_event = Event(
                    execution_id=context.execution_id,
                    author=self.name,
                    content=results,
                )
                context.add_event(tool_event)
            return AgentResult(
                output=None,
                context=context,
                status="pending_confirmation",
                pending_tool_calls=pending,
            )

        if results:
            tool_event = Event(
                execution_id=context.execution_id,
                author=self.name,
                content=results,
            )
            context.add_event(tool_event)

        return None

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

        # NEW: Add memory tool
        if self.memory_manager:
            from scratch_agents.tools.memory_tool import MemoryTool
            tools.append(MemoryTool(self.memory_manager))

        return tools

    # NEW: CH06
    async def _process_confirmations(
        self,
        context: ExecutionContext,
        confirmations: list[ToolConfirmation],
    ):
        """Process tool confirmations from human-in-the-loop."""
        raw_pending = context.state.pop("pending_tool_calls", [])
        pending = [PendingToolCall.model_validate(d) for d in raw_pending]

        tools_dict = {t.name: t for t in self.tools}
        results = []

        for pending_call in pending:
            tc = pending_call.tool_call
            confirmation = next(
                (c for c in confirmations if c.tool_call_id == tc.tool_call_id),
                None,
            )

            if confirmation and confirmation.approved:
                args = confirmation.modified_arguments or tc.arguments
                tool_obj = tools_dict.get(tc.name)
                if tool_obj:
                    try:
                        output = await tool_obj(context, **args)
                        results.append(ToolResult(
                            tool_call_id=tc.tool_call_id,
                            name=tc.name,
                            status="success",
                            content=[output],
                        ))
                    except Exception as e:
                        results.append(ToolResult(
                            tool_call_id=tc.tool_call_id,
                            name=tc.name,
                            status="error",
                            content=[str(e)],
                        ))
            else:
                results.append(ToolResult(
                    tool_call_id=tc.tool_call_id,
                    name=tc.name,
                    status="error",
                    content=["User denied the tool execution."],
                ))

        if results:
            tool_event = Event(
                execution_id=context.execution_id,
                author=self.name,
                content=results,
            )
            context.add_event(tool_event)

    def _log_response(self, response: LlmResponse):
        """Log LLM response for verbose mode."""
        for item in response.content:
            if isinstance(item, Message):
                logger.info(f"[{self.name}] {item.content}")
            elif isinstance(item, ToolCall):
                logger.info(f"[{self.name}] Tool call: {item.name}({item.arguments})")
````


## 6.4 节：长期记忆（ChromaDB）

`memory/long_term.py`：TaskMemory（Listing 6.27）与 DuplicateCheckResult（6.28）两个 Pydantic 模型、抽取/查重 prompt、`TaskMemoryManager`（6.30–6.35：ChromaDB 集合、LLM 结构化抽取、查重、保存、检索）。

`tools/memory_tool.py`：`MemoryTool`（Listing 6.36/6.37）——`recall_memory` 工具，并在 `process_llm_request()` 里自动把相关记忆注入 LLM 请求（Listing 6.38 的钩子，定义在 tools/base.py 的 BaseTool 上）。

最后更新包 `__init__.py` 导出 PendingToolCall / ToolConfirmation，并建 `memory/__init__.py`。

#### 📄 `scratch_agents/memory/long_term.py`

- **状态**：ch06（首次创建，之后不变）
- **对应书内容**：Listing 6.27（TaskMemory）、6.28（DuplicateCheckResult）、6.30–6.35（TaskMemoryManager）
- **行数**：202 行（清单为完整文件内容，从第 1 行到第 202 行）

<!-- FILE: scratch_agents/memory/long_term.py -->
<!-- STATE: ch06 -->

````python
"""Long-term memory with ChromaDB for task memory storage and retrieval."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from pydantic import BaseModel, Field

from scratch_agents.context import ExecutionContext
from scratch_agents.types import Event, Message, ToolCall, ToolResult

if TYPE_CHECKING:
    from scratch_agents.llm import LlmClient


class TaskMemory(BaseModel):
    """Structured memory for GAIA problem-solving records."""

    task_summary: str = Field(description="What the problem asked")
    approach: str = Field(description="Methods and tools used to solve it")
    final_answer: str = Field(description="The agent's submitted answer")
    is_correct: bool = Field(description="Whether the answer was correct")
    error_analysis: str | None = Field(
        default=None,
        description="Why the attempt failed, if it did",
    )

    def to_embedding_text(self) -> str:
        """Generate text for vector search."""
        return f"Task: {self.task_summary}"


class DuplicateCheckResult(BaseModel):
    """Result of duplicate check."""

    decision: str = Field(description="ADD (new information) or SKIP (duplicate)")
    reason: str = Field(description="Explanation for the decision")


TASK_MEMORY_EXTRACTION_PROMPT = """Analyze the following execution history and extract a structured task memory.

Execution History:
{execution_history}

Extract:
- task_summary: What the problem asked
- approach: Methods and tools used to solve it
- final_answer: The agent's submitted answer
- is_correct: Whether the answer was correct (true or false)
- error_analysis: If incorrect, explain why; otherwise leave null
"""


DUPLICATE_CHECK_PROMPT = """Compare the new memory against existing memories to determine if it's a duplicate.

Existing memories:
{existing_memories}

New memory:
{new_memory}

Respond with one of:
- ADD: This is new information that should be stored
- SKIP: Similar information already exists, no need to store

Judgment criteria:
- Same problem with different approach or different result counts as new information
- Same problem with same approach and same result is a duplicate
"""


class TaskMemoryManager:
    """Memory manager for GAIA problem-solving learning."""

    def __init__(
        self,
        llm_client: "LlmClient",
        collection_name: str = "task_memories",
    ):
        self.llm_client = llm_client

        # ChromaDB setup
        self.client = chromadb.Client()
        embedding_fn = OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
        )

    async def _extract_memory(self, execution_history: str) -> TaskMemory | None:
        """Extract structured memory from execution history."""
        prompt = TASK_MEMORY_EXTRACTION_PROMPT.format(
            execution_history=execution_history
        )
        try:
            return await self.llm_client.ask(
                prompt=prompt,
                response_format=TaskMemory,
            )
        except Exception as e:
            print(f"Memory extraction failed: {e}")
            return None

    def _format_execution_history(self, events: list[Event]) -> str:
        """Convert event list to text."""
        lines = []
        for event in events:
            for item in event.content:
                if isinstance(item, Message):
                    lines.append(f"[{item.role}]: {item.content}")
                elif isinstance(item, ToolCall):
                    lines.append(f"[Tool Call]: {item.name}({item.arguments})")
                elif isinstance(item, ToolResult):
                    content_preview = str(item.content[0])[:500] if item.content else ""
                    lines.append(f"[Tool Result]: {item.name} -> {content_preview}")
        return "\n".join(lines)

    async def _is_duplicate(
        self,
        new_memory: TaskMemory,
        existing_results: dict,
    ) -> bool:
        """Determine if a new memory duplicates an existing one."""
        if not existing_results["metadatas"] or not existing_results["metadatas"][0]:
            return False

        existing_texts = []
        for meta in existing_results["metadatas"][0]:
            existing_texts.append(
                f"task_summary: {meta.get('task_summary')}, "
                f"- approach: {meta.get('approach')}, "
                f"is_correct: {meta.get('is_correct')}"
            )

        prompt = DUPLICATE_CHECK_PROMPT.format(
            existing_memories="\n".join(existing_texts),
            new_memory=(
                f"task_summary: {new_memory.task_summary}, "
                f"approach: {new_memory.approach}, "
                f"is_correct: {new_memory.is_correct}"
            ),
        )

        try:
            result = await self.llm_client.ask(
                prompt=prompt,
                response_format=DuplicateCheckResult,
            )
            return result.decision == "SKIP"
        except Exception:
            return False

    async def save(self, context: ExecutionContext) -> str | None:
        """Extract and save memory from execution context.

        Returns:
            memory_id if saved, None if ignored as duplicate
        """
        # 1. Convert execution history to text
        execution_history = self._format_execution_history(context.events)

        # 2. Extract structured memory using LLM
        memory = await self._extract_memory(execution_history)
        if memory is None:
            return None

        # 3. Duplicate check using the same text used for storage
        existing = self.collection.query(
            query_texts=[memory.to_embedding_text()],
            n_results=3,
        )
        if await self._is_duplicate(memory, existing):
            return None

        # 4. Store in ChromaDB
        memory_id = str(uuid.uuid4())
        metadata = memory.model_dump()
        # ChromaDB metadata cannot store None values
        metadata = {k: ("" if v is None else v) for k, v in metadata.items()}
        self.collection.add(
            ids=[memory_id],
            documents=[memory.to_embedding_text()],
            metadatas=[metadata],
        )
        return memory_id

    async def search(self, query: str, top_k: int = 5) -> list[TaskMemory]:
        """Search for memories related to the query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        if not results["metadatas"] or not results["metadatas"][0]:
            return []

        return [TaskMemory(**meta) for meta in results["metadatas"][0]]
````


#### 📄 `scratch_agents/tools/memory_tool.py`

- **状态**：ch06（首次创建，之后不变）
- **对应书内容**：Listing 6.36（MemoryTool 定义）、6.37（execute/search）
- **行数**：81 行（清单为完整文件内容，从第 1 行到第 81 行）

<!-- FILE: scratch_agents/tools/memory_tool.py -->
<!-- STATE: ch06 -->

````python
"""Memory tool with automatic injection of relevant past experiences (Listing 6.37)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scratch_agents.context import ExecutionContext
from scratch_agents.tools.base import BaseTool
from scratch_agents.types import Message

if TYPE_CHECKING:
    from scratch_agents.llm import LlmRequest
    from scratch_agents.memory.long_term import TaskMemory


class MemoryTool(BaseTool):
    """Tool that injects relevant past memories into the LLM request automatically."""

    def __init__(self):
        super().__init__(
            name="recall_memory",
            description=(
                "Search for past problem-solving records. "
                "Use this to check if similar problems were solved before."
            ),
            tool_definition=None,  # Automatic injection only
        )

    async def execute(self, context: ExecutionContext, query: str = "") -> str:
        """Search memories and return formatted results."""
        if context.memory_manager is None:
            return ""
        memories = await context.memory_manager.search(query, top_k=3)
        if not memories:
            return ""
        return self._format_memories(memories)

    def _format_memories(self, memories: list["TaskMemory"]) -> str:
        """Format memories for display."""
        results = []
        for i, mem in enumerate(memories, 1):
            status = "Correct" if mem.is_correct else "Incorrect"
            text = (
                f"[Record {i}]\n"
                f"- Problem: {mem.task_summary}\n"
                f"- Approach: {mem.approach}\n"
                f"- Answer: {mem.final_answer}\n"
                f"- Result: {status}"
            )
            if not mem.is_correct and mem.error_analysis:
                text += f"\n- Error analysis: {mem.error_analysis}"
            results.append(text)
        return "\n\n".join(results)

    async def process_llm_request(
        self,
        context: ExecutionContext,
        request: "LlmRequest",
    ) -> None:
        """Inject relevant memories before LLM call."""
        if context.memory_manager is None:
            return

        user_msgs = [
            c for c in request.contents
            if isinstance(c, Message) and c.role == "user"
        ]
        if not user_msgs:
            return

        result = await self.execute(context, user_msgs[-1].content)
        if not result:
            return

        request.append_instructions(
            "The following are records from similar problems solved in the past:\n"
            "<PAST_EXPERIENCES>\n"
            f"{result}\n"
            "</PAST_EXPERIENCES>\n"
            "Reference successful approaches and avoid approaches that led to failures."
        )
````


#### 📄 `scratch_agents/memory/__init__.py`

- **状态**：ch06（首次创建，之后不变）
- **对应书内容**：无（包导出约定）
- **行数**：7 行（清单为完整文件内容，从第 1 行到第 7 行）

<!-- FILE: scratch_agents/memory/__init__.py -->
<!-- STATE: ch06 -->

````python
from scratch_agents.memory.session import Session, BaseSessionManager, InMemorySessionManager
from scratch_agents.memory.context_optimizer import (
    create_optimizer_callback, count_tokens,
    apply_sliding_window, apply_compaction, apply_summarization,
    ContextOptimizer
)
from scratch_agents.memory.long_term import TaskMemory, TaskMemoryManager
````


#### 📄 `scratch_agents/__init__.py`

- **状态**：ch06（更新，最终版）
- **对应书内容**：无（包导出约定）
- **行数**：8 行（清单为完整文件内容，从第 1 行到第 8 行）
- **说明**：在 ch04 版基础上，context 导入增加 PendingToolCall / ToolConfirmation。这是 `__init__.py` 的最终形态，ch07–ch10 不再改动。

<!-- FILE: scratch_agents/__init__.py -->
<!-- STATE: ch06 -->

````python
from scratch_agents.types import (
    Message, ToolCall, ToolResult, Event, ContentItem
)
from scratch_agents.context import (
    ExecutionContext, AgentResult, PendingToolCall, ToolConfirmation
)
from scratch_agents.llm import LlmClient, LlmRequest, LlmResponse
from scratch_agents.agent import Agent
````


## 本章改动一览

| 文件 | 状态 | 行数 |
|---|---|---|
| `scratch_agents/memory/context_optimizer.py` | ch06（首次创建，之后不变） | 294 |
| `scratch_agents/memory/session.py` | ch06（首次创建，之后不变） | 84 |
| `scratch_agents/context.py` | ch06（中间态，ch08 会被替换） | 58 |
| `scratch_agents/tools/base.py` | ch06（中间态，ch08 会被替换） | 110 |
| `scratch_agents/agent.py` | ch06（中间态，ch08 会被替换） | 445 |
| `scratch_agents/memory/long_term.py` | ch06（首次创建，之后不变） | 202 |
| `scratch_agents/tools/memory_tool.py` | ch06（首次创建，之后不变） | 81 |
| `scratch_agents/memory/__init__.py` | ch06（首次创建，之后不变） | 7 |
| `scratch_agents/__init__.py` | ch06（更新，最终版） | 8 |

## 本章自测

```bash
python -c "from scratch_agents.memory import Session, ContextOptimizer; from scratch_agents import PendingToolCall, ToolConfirmation; print('ch06 ok')"
```
