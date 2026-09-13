# 第 8 章 · 代码执行与沙箱、skills

> 对应原书 第 8 章（小节号在各代码块中标注）。本章文档：`docs/ch08-code-execution.md`。

## 本章构建目标

第 8 章给 agent 一台「计算机」：8.2 节接入 E2B 沙箱并在 Agent 里管理其生命周期，8.3 节让工具可移植进沙箱（sandbox_executable），8.4 节实现 workspace 文件/命令工具，8.5 节实现 skills（把 SKILL.md 技能包上传沙箱、渐进式披露）。

本章整体替换 `tools/base.py`、`context.py`、`agent.py` 为 ch08 版，新增 `tools/code_execution.py` 和 `skills.py`。

说明：书 8.4 节的 workspace 工具（Listing 8.24/8.25）在 notebook 里以 FunctionTool 包装的形式现场定义，不进包；包内的 `execute_python` / `bash_tool` / `upload_file` 三个工具在 Listing 8.9–8.13 一带。

## 8.2 节：E2B 沙箱接入

`tools/code_execution.py`：`execute_python`（调 `sandbox.run_code()`）、`bash_tool`（调 `sandbox.commands.run()`）、`upload_file`（调 `sandbox.files.write`）。三个工具都从 `context.code_env` 取沙箱实例——沙箱对象由 Agent 在 `run()` 开始时创建、结束时销毁（Listing 8.12），所以对工具来说沙箱是「凭空出现」的。

#### 📄 `scratch_agents/tools/code_execution.py`

- **状态**：ch08（首次创建，最终版）
- **对应书内容**：Listing 8.9–8.13（execute_python、bash_tool、upload_file）
- **行数**：75 行（清单为完整文件内容，从第 1 行到第 75 行）

<!-- FILE: scratch_agents/tools/code_execution.py -->
<!-- STATE: ch08 -->

````python
"""Code execution tools using E2B sandbox."""

import json

from scratch_agents.tools.base import tool
from scratch_agents.context import ExecutionContext


@tool(
    name="execute_python",
    description="Execute Python code in a sandboxed environment. "
                "Use this to perform calculations, data processing, or any Python operations."
)
def execute_python(context: ExecutionContext, code: str) -> str:
    """Execute Python code in the E2B sandbox.

    Args:
        context: Execution context with code_env
        code: Python code to execute
    """
    if context.code_env is None:
        raise RuntimeError("No code execution environment available.")
    sandbox = context.code_env
    execution = sandbox.run_code(code)
    return json.dumps(execution.to_json(), indent=2, ensure_ascii=False)


@tool(name="bash_tool", description="Execute a bash command in a sandboxed environment.")
def bash_tool(context: ExecutionContext, command: str) -> str:
    """Execute a bash command in the E2B sandbox.

    Args:
        context: Execution context with code_env
        command: Bash command to execute
    """
    if context.code_env is None:
        raise RuntimeError("No code execution environment available.")
    sandbox = context.code_env

    try:
        result = sandbox.commands.run(command)
        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"STDERR: {result.stderr}")
        return "\n".join(output_parts) if output_parts else "Command completed (no output)"
    except Exception as e:
        return f"Command error: {str(e)}"


@tool
def upload_file(context: ExecutionContext, local_path: str, sandbox_path: str = None) -> str:
    """Upload a local file to the E2B sandbox.

    Args:
        context: Execution context with code_env
        local_path: Path to the local file
        sandbox_path: Destination path in sandbox (defaults to /home/user/)
    """
    import os

    sandbox = context.code_env
    if sandbox is None:
        return "Error: No code execution environment available"

    if sandbox_path is None:
        sandbox_path = f"/home/user/{os.path.basename(local_path)}"

    try:
        with open(local_path, "rb") as f:
            sandbox.files.write(sandbox_path, f.read())
        return f"File uploaded to {sandbox_path}"
    except Exception as e:
        return f"Upload error: {str(e)}"
````


#### 📄 `scratch_agents/context.py`

- **状态**：ch08（中间态，ch09 会被替换）
- **对应书内容**：Listing 8.x（ExecutionContext 增加 code_env 字段）
- **行数**：57 行（清单为完整文件内容，从第 1 行到第 57 行）
- **说明**：相对 ch06 只增加 `code_env`；transfer_to / transfer_tools 是 ch09 的。

<!-- FILE: scratch_agents/context.py -->
<!-- STATE: ch08 -->

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
    # CH06 session
    session: Optional[Any] = None
    session_manager: Optional[Any] = None
    # NEW: CH08 code execution
    code_env: Optional[Any] = None  # E2B Sandbox

    def add_event(self, event: Event):
        """Append an event to the execution history."""
        self.events.append(event)

    def increment_step(self):
        """Move to the next execution step."""
        self.current_step += 1


@dataclass
class AgentResult:
    """Result of an agent execution."""
    output: Any
    context: ExecutionContext
    status: str = "complete"
    pending_tool_calls: list = field(default_factory=list)


class PendingToolCall(BaseModel):
    """A tool call awaiting user confirmation."""
    tool_call: ToolCall
    confirmation_message: str


class ToolConfirmation(BaseModel):
    """User's response to a pending tool call."""
    tool_call_id: str
    approved: bool
    modified_arguments: dict | None = None
````


## 8.3 节：工具移植进沙箱 + Agent 生命周期管理

`tools/base.py` 增加 `sandbox_executable` 标记和 `get_source_code()`（Listing 8.15/8.16），`@tool` 装饰器同步支持该参数。

`agent.py` 整体替换为 ch08 版：`__init__` 增加 `code_execution` / `skills_path`；`run()` 用 try/finally 管理沙箱（创建→注册沙箱工具→运行→kill）；`_setup_tools()` 自动追加 `execute_python` 并收集 `sandbox_executable` 工具；`_prepare_llm_request()` 追加沙箱工具说明与 skills 提示（Listing 8.18–8.21、8.31）。

#### 📄 `scratch_agents/tools/base.py`

- **状态**：ch08（最终版）
- **对应书内容**：Listing 8.15（sandbox_executable 参数）、8.16（get_source_code）
- **行数**：171 行（清单为完整文件内容，从第 1 行到第 171 行）
- **说明**：tools/base.py 的最终形态：在 ch08 快照（见 notebooks/ch08/base.py）基础上有少量整理——默认确认消息模板（ch06 的 DEFAULT_CONFIRMATION_TEMPLATE）、process_llm_request 钩子（ch06 Listing 6.37/6.38，供记忆工具注入）、更完整的 docstring。本章直接采用最终版。

<!-- FILE: scratch_agents/tools/base.py -->
<!-- STATE: ch08 -->

````python
"""Base tool abstraction for the scratch_agents framework."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel

from scratch_agents.tools.helpers import format_tool_definition, function_to_input_schema
from scratch_agents.context import ExecutionContext

if TYPE_CHECKING:
    from scratch_agents.llm import LlmRequest


class BaseTool(ABC):
    """Abstract base class for all tools."""

    DEFAULT_CONFIRMATION_TEMPLATE = (
        "The agent wants to execute '{name}' with arguments: {arguments}. "
        "Do you approve?"
    )

    def __init__(
        self,
        name: str = None,
        description: str = None,
        tool_definition: Dict[str, Any] = None,
        requires_confirmation: bool = False,
        confirmation_message_template: str | None = None,
    ):
        self.name = name or self.__class__.__name__
        self.description = description or self.__doc__ or ""
        self._tool_definition = tool_definition
        self.requires_confirmation = requires_confirmation
        self.confirmation_message_template = (
            confirmation_message_template
            if confirmation_message_template
            else self.DEFAULT_CONFIRMATION_TEMPLATE
        )

    @property
    def tool_definition(self) -> Dict[str, Any] | None:
        return self._tool_definition

    def get_confirmation_message(self, arguments: dict) -> str:
        return self.confirmation_message_template.format(name=self.name, arguments=arguments)

    async def process_llm_request(
        self,
        context: "ExecutionContext",
        request: "LlmRequest",
    ) -> None:
        """Hook for tools to modify the LlmRequest before it is sent (Listing 6.37/6.38)."""
        return None

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
        sandbox_executable: bool = False,
        requires_confirmation: bool = False,
        confirmation_message_template: str = "",
    ):
        self.func = func
        self.needs_context = "context" in inspect.signature(func).parameters
        self.sandbox_executable = sandbox_executable

        if sandbox_executable and self.needs_context:
            raise ValueError(
                f"Tool '{func.__name__}' cannot be sandbox_executable "
                "because it requires 'context' parameter."
            )

        resolved_name = name or func.__name__
        resolved_desc = description or (func.__doc__ or "").strip()

        # Must set name/description before _generate_definition uses them
        super().__init__(
            name=resolved_name,
            description=resolved_desc,
            tool_definition=tool_definition,
            requires_confirmation=requires_confirmation,
            confirmation_message_template=confirmation_message_template,
        )

        # Generate definition after super().__init__ so self.name is available
        if self._tool_definition is None:
            self._tool_definition = self._generate_definition()

    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        """Execute the wrapped function."""
        if self.needs_context:
            result = self.func(context=context, **kwargs)
        else:
            result = self.func(**kwargs)

        # Handle both sync and async functions
        if inspect.iscoroutine(result):
            return await result
        return result

    def _generate_definition(self) -> Dict[str, Any]:
        """Generate tool definition from function signature."""
        parameters = function_to_input_schema(self.func)
        return format_tool_definition(self.name, self.description, parameters)

    def get_source_code(self) -> str:
        """Get the source code of the wrapped function (CH08 sandbox)."""
        if not self.sandbox_executable:
            raise ValueError(f"Tool '{self.name}' is not marked as sandbox_executable")
        source = inspect.getsource(self.func)
        lines = source.split('\n')
        filtered_lines = []
        skip_decorator = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('@tool'):
                skip_decorator = True
                if '(' not in stripped or ')' in stripped:
                    skip_decorator = False
                continue
            if skip_decorator:
                if ')' in stripped:
                    skip_decorator = False
                continue
            filtered_lines.append(line)
        return '\n'.join(filtered_lines)


def tool(func=None, *, name=None, description=None, sandbox_executable=False,
         requires_confirmation=False, confirmation_message=None):
    """Decorator to create a FunctionTool from a function.

    Can be used with or without arguments:
        @tool
        def my_func(...): ...

        @tool(name="custom_name", description="Custom description")
        def my_func(...): ...
    """
    def decorator(f):
        return FunctionTool(
            func=f,
            name=name,
            description=description,
            sandbox_executable=sandbox_executable,
            requires_confirmation=requires_confirmation,
            confirmation_message_template=confirmation_message or "",
        )

    if func is not None:
        # Called without arguments: @tool
        return decorator(func)
    # Called with arguments: @tool(name=...)
    return decorator
````


#### 📄 `scratch_agents/agent.py`

- **状态**：ch08（中间态，ch09 会被替换）
- **对应书内容**：Listing 8.12（沙箱生命周期）、8.18–8.20（注册沙箱工具）、8.21（沙箱工具提示）、8.31（skills 提示）、8.32（上传 skills 到沙箱）
- **行数**：496 行（清单为完整文件内容，从第 1 行到第 496 行）
- **说明**：相对 ch06（快照头注记）：__init__ 增加 code_execution/skills_path；run() 增加 code_env 建立/清理；_setup_tools() 追加 execute_python 与沙箱工具收集；_prepare_llm_request() 追加沙箱/skills 提示；新增 _setup_code_env / _register_sandbox_tools / _get_sandbox_tools_prompt。sub_agents/transfer 还没有。

<!-- FILE: scratch_agents/agent.py -->
<!-- STATE: ch08 -->

````python
"""Core Agent class for the scratch_agents framework."""

from __future__ import annotations

import json
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
    """Tool-calling agent with ReAct loop, callbacks, sessions, memory,
    code execution, and skills."""

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
        # CH06 session & memory
        session_manager: Optional["BaseSessionManager"] = None,
        memory_manager: Optional["TaskMemoryManager"] = None,
        before_llm_callbacks: list[Callable] | None = None,
        # NEW: CH08 code execution
        code_execution: str | None = None,  # "e2b"
        skills_path: str | None = None,
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
        self.code_execution = code_execution
        self.skills_path = skills_path
        self._sandbox_tools: List[FunctionTool] = []
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

        session = None
        if session_id and self.session_manager:
            session = await self.session_manager.get_or_create(session_id)

        if context is None:
            context = ExecutionContext(
                session=session,
                session_manager=self.session_manager,
            )
            # Restore previous events from session
            if session and session.events:
                context.events = list(session.events)

        if tool_confirmations:
            await self._process_confirmations(context, tool_confirmations)

        if user_input:
            user_event = Event(
                execution_id=context.execution_id,
                author="user",
                content=[Message(role="user", content=user_input)],
            )
            context.add_event(user_event)

        # NEW: Set up code execution environment
        if self.code_execution == "e2b" and context.code_env is None:
            await self._setup_code_env(context)

        try:
            while not context.final_result and context.current_step < self.max_steps:
                result = await self.step(context, verbose=verbose)

                if result and result.status == "pending":
                    if session and self.session_manager:
                        session.events = list(context.events)
                        await self.session_manager.save(session)
                    return result

                if context.events:
                    last_event = context.events[-1]
                    if self._is_final_response(last_event):
                        context.final_result = self._extract_final_result(last_event)

            if self.memory_manager:
                try:
                    await self.memory_manager.save(context)
                except Exception as e:
                    logger.warning(f"Failed to save memory: {e}")

            if session and self.session_manager:
                session.events = list(context.events)
                await self.session_manager.save(session)

            return AgentResult(output=context.final_result, context=context)
        finally:
            # NEW: Clean up sandbox
            if context.code_env is not None:
                context.code_env.kill()

    async def step(
        self,
        context: ExecutionContext,
        verbose: bool = False,
    ) -> AgentResult | None:
        """Perform one think-act cycle."""
        llm_request = self._prepare_llm_request(context)

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
            if result and result.status == "pending":
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

            if tool_obj.requires_confirmation:
                msg = tool_obj.get_confirmation_message(tool_call.arguments)
                pending.append(PendingToolCall(
                    tool_call=tool_call,
                    confirmation_message=msg,
                ))
                continue

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

            for cb in self.after_tool_callbacks:
                cb_result = cb(context, tool_result)
                if hasattr(cb_result, "__await__"):
                    cb_result = await cb_result
                if cb_result is not None:
                    tool_result = cb_result

            results.append(tool_result)

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
                status="pending",
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

        # NEW: Add sandbox tools prompt
        sandbox_prompt = self._get_sandbox_tools_prompt()
        if sandbox_prompt:
            instructions.append(sandbox_prompt)

        # NEW: Add skills prompt
        if self.skills_path:
            try:
                from scratch_agents.skills import discover_skills, generate_skills_prompt
                skills = discover_skills(self.skills_path)
                skills_prompt = generate_skills_prompt(skills)
                if skills_prompt:
                    instructions.append(skills_prompt)
            except Exception:
                pass

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
        if self.output_tool_name:
            for item in event.content:
                if (isinstance(item, ToolResult)
                    and item.name == self.output_tool_name
                    and item.status == "success"):
                    return True
            return False
        has_tool_calls = any(isinstance(c, ToolCall) for c in event.content)
        has_tool_results = any(isinstance(c, ToolResult) for c in event.content)
        return not has_tool_calls and not has_tool_results

    def _extract_final_result(self, event: Event) -> Any:
        if self.output_tool_name:
            for item in event.content:
                if (isinstance(item, ToolResult)
                    and item.name == self.output_tool_name
                    and item.status == "success" and item.content):
                    return item.content[0]
        for item in event.content:
            if isinstance(item, Message) and item.role == "assistant":
                return item.content
        return None

    def _setup_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
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

        # NEW: Collect sandbox-executable tools
        invalid_sandbox_tools = []
        for t in tools:
            if not isinstance(t, FunctionTool) or not t.sandbox_executable:
                continue
            if self.code_execution != "e2b":
                invalid_sandbox_tools.append(t.name)
                continue
            self._sandbox_tools.append(t)

        if invalid_sandbox_tools:
            raise ValueError(
                f"Tools {invalid_sandbox_tools} are marked as sandbox_executable "
                "but code_execution is not enabled."
            )

        # NEW: Add code execution tool
        if self.code_execution == "e2b":
            from scratch_agents.tools.code_execution import execute_python
            tools.append(execute_python)

        if self.memory_manager:
            from scratch_agents.tools.memory_tool import MemoryTool
            tools.append(MemoryTool(self.memory_manager))

        return tools

    # NEW: CH08 methods
    async def _setup_code_env(self, context: ExecutionContext):
        """Set up E2B sandbox environment."""
        try:
            from e2b_code_interpreter import Sandbox
            sandbox = Sandbox.create(timeout=300)
            context.code_env = sandbox
            self._register_sandbox_tools(sandbox)

            if self.skills_path:
                from scratch_agents.skills import discover_skills
                skills = discover_skills(self.skills_path)
                for skill_info in skills:
                    sandbox.files.write(
                        f"/home/user/skills/{skill_info.name}/{skill_info.path.name}",
                        skill_info.path.read_text(),
                    )
        except Exception as e:
            logger.warning(f"Failed to set up code execution environment: {e}")

    def _register_sandbox_tools(self, sandbox) -> None:
        """Register sandbox-executable tools in the sandbox."""
        tool_sources = []
        for t in self._sandbox_tools:
            source = t.get_source_code()
            tool_sources.append(source)
        combined_source = "\n\n".join(tool_sources)
        result = sandbox.run_code(combined_source)
        if result.error:
            raise RuntimeError(f"Failed to register sandbox tools: {result.error}")

    def _get_sandbox_tools_prompt(self) -> str:
        """Generate prompt describing sandbox-executable tools."""
        if not self._sandbox_tools:
            return ""
        tool_definitions = [t.tool_definition for t in self._sandbox_tools]
        tools_json = json.dumps(tool_definitions, indent=2, ensure_ascii=False)
        return (
            "\n\n## Sandbox-Executable Tools\n"
            "The following functions are pre-registered in the sandbox "
            "and can be called directly in your Python code:\n"
            f"{tools_json}"
        )

    async def _process_confirmations(
        self, context: ExecutionContext, confirmations: list[ToolConfirmation],
    ):
        raw_pending = context.state.pop("pending_tool_calls", [])
        pending = [PendingToolCall.model_validate(d) for d in raw_pending]
        tools_dict = {t.name: t for t in self.tools}
        results = []
        for pending_call in pending:
            tc = pending_call.tool_call
            confirmation = next(
                (c for c in confirmations if c.tool_call_id == tc.tool_call_id), None)
            if confirmation and confirmation.approved:
                args = confirmation.modified_arguments or tc.arguments
                tool_obj = tools_dict.get(tc.name)
                if tool_obj:
                    try:
                        output = await tool_obj(context, **args)
                        results.append(ToolResult(
                            tool_call_id=tc.tool_call_id, name=tc.name,
                            status="success", content=[output]))
                    except Exception as e:
                        results.append(ToolResult(
                            tool_call_id=tc.tool_call_id, name=tc.name,
                            status="error", content=[str(e)]))
            else:
                results.append(ToolResult(
                    tool_call_id=tc.tool_call_id, name=tc.name,
                    status="error", content=["User denied the tool execution."]))
        if results:
            tool_event = Event(
                execution_id=context.execution_id, author=self.name, content=results)
            context.add_event(tool_event)

    def _log_response(self, response: LlmResponse):
        for item in response.content:
            if isinstance(item, Message):
                logger.info(f"[{self.name}] {item.content}")
            elif isinstance(item, ToolCall):
                logger.info(f"[{self.name}] Tool call: {item.name}({item.arguments})")
````


## 8.5 节：Agent Skills

`skills.py`：SkillInfo（Listing 8.27）、`parse_frontmatter()`（8.28，解析 SKILL.md 的 YAML 头）、`load_skill()` / `discover_skills()`（8.29，扫描技能目录）、`generate_skills_prompt()`（8.30，生成系统提示中的技能说明段落）。

技能目录的上传到沙箱发生在 `agent.py` 的 `_setup_code_env()` 中（Listing 8.32），ch08 版 agent.py 已包含。

#### 📄 `scratch_agents/skills.py`

- **状态**：ch08（首次创建，最终版）
- **对应书内容**：Listing 8.27–8.32（skills 机制）
- **行数**：100 行（清单为完整文件内容，从第 1 行到第 100 行）

<!-- FILE: scratch_agents/skills.py -->
<!-- STATE: ch08 -->

````python
"""Skill discovery and management for code execution agents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SkillInfo:
    """Information about a discovered skill."""
    name: str
    description: str
    path: Path


def parse_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter from markdown file."""
    pattern = r'^---\s*\n(.*?)\n---'
    match = re.match(pattern, content, re.DOTALL)
    if not match:
        return {}
    result = {}
    for line in match.group(1).split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            result[key.strip()] = value.strip().strip('"\'')
    return result


def load_skill(skill_dir: Path) -> SkillInfo | None:
    """Load skill info from a directory.

    Looks for a SKILL.md file with metadata about the skill.
    """
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return None

    content = skill_md.read_text(encoding='utf-8')
    frontmatter = parse_frontmatter(content)

    name = frontmatter.get('name')
    description = frontmatter.get('description')
    if not name or not description:
        return None

    return SkillInfo(name=name, description=description, path=skill_dir)


def discover_skills(skills_path: str | Path) -> list[SkillInfo]:
    """Discover all skills in a directory.

    Each subdirectory is treated as a potential skill.
    """
    skills_dir = Path(skills_path)
    if not skills_dir.exists():
        return []

    skills = []
    for item in sorted(skills_dir.iterdir()):
        if item.is_dir() and not item.name.startswith("."):
            skill = load_skill(item)
            if skill:
                skills.append(skill)
    return skills


def generate_skills_prompt(
    skills: list[SkillInfo],
    sandbox_path: str = "/home/user/skills",
) -> str:
    """Generate a prompt describing available skills.

    This prompt is added to the agent's instructions to inform it
    about available skills that can be used in code execution.
    """
    if not skills:
        return ""

    lines = [
        "## Available Skills",
        "The following skills are available in the sandbox environment:",
        "",
    ]

    for skill in skills:
        lines.append(f"### {skill.name}")
        lines.append(f"- Description: {skill.description}")
        lines.append(f"- Path: {sandbox_path}/{skill.name}/")
        lines.append(f"- Read the SKILL.md for usage instructions: {sandbox_path}/{skill.name}/SKILL.md")
        lines.append("")

    lines.append(
        "You can import and use these skills in your Python code. "
        "Read the SKILL.md file first to understand how to use each skill."
    )

    return "\n".join(lines)
````


## 本章改动一览

| 文件 | 状态 | 行数 |
|---|---|---|
| `scratch_agents/tools/code_execution.py` | ch08（首次创建，最终版） | 75 |
| `scratch_agents/context.py` | ch08（中间态，ch09 会被替换） | 57 |
| `scratch_agents/tools/base.py` | ch08（最终版） | 171 |
| `scratch_agents/agent.py` | ch08（中间态，ch09 会被替换） | 496 |
| `scratch_agents/skills.py` | ch08（首次创建，最终版） | 100 |

## 本章自测

```bash
python -c "from scratch_agents.tools.code_execution import execute_python; from scratch_agents.skills import discover_skills; print('ch08 ok')"
```
