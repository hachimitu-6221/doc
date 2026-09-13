# 第 9 章 · 多 Agent 系统编排

> 对应原书 第 9 章（小节号在各代码块中标注）。本章文档：`docs/ch09-multi-agent-systems.md`。

## 本章构建目标

第 9 章讲多 agent：9.3 节用代码定义执行顺序（顺序 / 并行 / 循环工作流），9.4 节把 agent 包装成工具（Agent as Tool），9.5 节实现 agent 转移（transfer，LLM 主动把任务转给其他 agent），9.6 节用 A2A 协议跨网络协作。

本章把 `agent.py` / `context.py` 推进到**最终形态**（ch09 版 = 最终版），并新增 `workflows/` 子包、`tools/agent_tool.py`、`transfer.py`、`remote.py`、`a2a_server.py`。

## 9.3 节：三种工作流

`workflows/sequential.py`（Listing 9.5，依次执行、传递 context）、`workflows/parallel.py`（Listing 9.6，并发执行、合并事件与输出；分支未完成时抛 `ParallelWorkflowIncomplete` 保留现场）、`workflows/loop.py`（Listing 9.7，循环直到停止条件）。

三个工作流类都继承 Agent，所以可以像 agent 一样被 `.run()`。

#### 📄 `scratch_agents/workflows/__init__.py`

- **状态**：ch09（首次创建，最终版）
- **对应书内容**：无（包导出约定）
- **行数**：3 行（清单为完整文件内容，从第 1 行到第 3 行）

<!-- FILE: scratch_agents/workflows/__init__.py -->
<!-- STATE: ch09 -->

````python
from scratch_agents.workflows.sequential import SequentialWorkflow
from scratch_agents.workflows.parallel import ParallelWorkflow, ParallelWorkflowIncomplete
from scratch_agents.workflows.loop import LoopWorkflow
````


#### 📄 `scratch_agents/workflows/sequential.py`

- **状态**：ch09（首次创建，最终版）
- **对应书内容**：Listing 9.5（SequentialWorkflow）
- **行数**：52 行（清单为完整文件内容，从第 1 行到第 52 行）

<!-- FILE: scratch_agents/workflows/sequential.py -->
<!-- STATE: ch09 -->

````python
"""Sequential workflow: run agents one after another."""

from __future__ import annotations

from typing import List

from scratch_agents.agent import Agent
from scratch_agents.context import AgentResult, ExecutionContext


class SequentialWorkflow(Agent):
    """Run agents sequentially, passing context from one to the next."""

    def __init__(
        self,
        agents: List[Agent],
        name: str = "sequential_workflow",
    ):
        self.agents = agents
        self.name = name

    async def run(
        self,
        user_input: str | None = None,
        context: ExecutionContext | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> AgentResult:
        """Execute all agents in sequence."""
        if context is None:
            context = ExecutionContext()

        result = None
        for i, agent in enumerate(self.agents):
            if context is not None:
                context.final_result = None
                context.current_step = 0

            if i == 0:
                result = await agent.run(
                    user_input=user_input,
                    context=context,
                    verbose=verbose,
                )
            else:
                result = await agent.run(
                    context=context,
                    verbose=verbose,
                )
            context = result.context

        return result
````


#### 📄 `scratch_agents/workflows/parallel.py`

- **状态**：ch09（首次创建，最终版）
- **对应书内容**：Listing 9.6（ParallelWorkflow、ParallelWorkflowIncomplete）
- **行数**：99 行（清单为完整文件内容，从第 1 行到第 99 行）

<!-- FILE: scratch_agents/workflows/parallel.py -->
<!-- STATE: ch09 -->

````python
"""Parallel workflow: run agents concurrently."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import List

from scratch_agents.agent import Agent
from scratch_agents.context import AgentResult, ExecutionContext


class ParallelWorkflowIncomplete(RuntimeError):
    """A branch needs attention; retain results for inspection or manual resume."""

    def __init__(self, branch_results, branch_errors=()):
        self.branch_results = tuple(branch_results)
        self.branch_errors = tuple(branch_errors)
        states = ", ".join(
            f"{name}: {result.status}"
            for name, result in self.branch_results if result.status != "complete"
        )
        super().__init__(f"Parallel workflow did not complete ({states}). "
                         "Inspect branch_results; automatic parallel resume is not supported.")


class ParallelWorkflow(Agent):
    """Run agents in parallel and combine results."""

    def __init__(
        self,
        agents: List[Agent],
        name: str = "parallel_workflow",
    ):
        self.agents = agents
        self.name = name

    async def run(
        self,
        user_input: str | None = None,
        context: ExecutionContext | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> AgentResult:
        """Execute all agents concurrently."""
        # Each branch owns its execution state. Sharing context would race on
        # final_result/current_step and duplicate events during the merge.
        seed = context or ExecutionContext()
        existing_event_count = len(seed.events)
        branches = [
            ExecutionContext(
                events=deepcopy(seed.events),
                state=deepcopy(seed.state),
                memory_manager=seed.memory_manager,
            )
            for _ in self.agents
        ]

        results = await asyncio.gather(
            *[agent.run(user_input, context=branch, verbose=verbose)
              for agent, branch in zip(self.agents, branches)],
            return_exceptions=True,
        )
        # Retain ordinary failures alongside successful/pending results.
        errors = []
        for i, result in enumerate(results):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, Exception):
                errors.append((self.agents[i].name, result))
                results[i] = AgentResult(output=None, context=branches[i], status="error")

        # A merged transcript cannot represent independent pending approvals.
        # Never discard them or label partial work complete. Completed branches
        # may already have performed work; callers must not blindly rerun them.
        if any(result.status != "complete" for result in results):
            raise ParallelWorkflowIncomplete(
                [(agent.name, result) for agent, result in zip(self.agents, results)], errors
            )

        merged_context = ExecutionContext(events=deepcopy(seed.events), state=deepcopy(seed.state))

        seen_user_event = False
        for result in results:
            new_events = result.context.events[existing_event_count:]
            for event in new_events:
                if event.author == "user":
                    if not seen_user_event:
                        merged_context.add_event(event)
                        seen_user_event = True
                else:
                    merged_context.add_event(event)

        # Combine outputs
        combined_output = "\n\n".join(
            f"[{agent.name}]\n{result.output}"
            for agent, result in zip(self.agents, results)
        )
        return AgentResult(output=combined_output, context=merged_context)
````


#### 📄 `scratch_agents/workflows/loop.py`

- **状态**：ch09（首次创建，最终版）
- **对应书内容**：Listing 9.7（LoopWorkflow）
- **行数**：65 行（清单为完整文件内容，从第 1 行到第 65 行）

<!-- FILE: scratch_agents/workflows/loop.py -->
<!-- STATE: ch09 -->

````python
"""Loop workflow: run agents repeatedly until a stop condition is met."""

from __future__ import annotations

from typing import Callable, List

from scratch_agents.agent import Agent
from scratch_agents.context import AgentResult, ExecutionContext

StopCondition = Callable[[AgentResult, int], bool]


class LoopWorkflow(Agent):
    """Run agents in a loop until stop condition is met."""

    def __init__(
        self,
        agents: List[Agent],
        stop_condition: StopCondition | None = None,
        max_iterations: int = 10,
        name: str = "loop_workflow",
    ):
        self.agents = agents
        self.stop_condition = stop_condition
        self.max_iterations = max_iterations
        self.name = name

    async def run(
        self,
        user_input: str | None = None,
        context: ExecutionContext | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> AgentResult:
        """Execute agents in a loop."""
        if context is None:
            context = ExecutionContext()

        result = None
        is_first_agent = True

        for iteration in range(1, self.max_iterations + 1):
            for agent in self.agents:
                if context is not None:
                    context.final_result = None
                    context.current_step = 0

                if is_first_agent:
                    result = await agent.run(
                        user_input=user_input,
                        context=context,
                        verbose=verbose,
                    )
                    is_first_agent = False
                else:
                    result = await agent.run(
                        context=context,
                        verbose=verbose,
                    )
                context = result.context

            if result and self.stop_condition and self.stop_condition(result, iteration):
                break

        return result
````


## 9.4 节：Agent as Tool

`tools/agent_tool.py`：AgentTool（Listing 9.8）把 Agent 适配成 BaseTool——工具名/描述取 agent 的 name/description，execute() 里调用子 agent 的 run() 并返回其输出。

#### 📄 `scratch_agents/tools/agent_tool.py`

- **状态**：ch09（首次创建，最终版）
- **对应书内容**：Listing 9.8（AgentTool）
- **行数**：54 行（清单为完整文件内容，从第 1 行到第 54 行）

<!-- FILE: scratch_agents/tools/agent_tool.py -->
<!-- STATE: ch09 -->

````python
"""AgentTool: wrap an Agent as a tool for use by other agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Type

from pydantic import BaseModel

from scratch_agents.tools.base import BaseTool
from scratch_agents.tools.helpers import format_tool_definition
from scratch_agents.context import ExecutionContext

if TYPE_CHECKING:
    from scratch_agents.agent import Agent


class AgentTool(BaseTool):
    """Adapter that wraps an Agent as a tool."""

    def __init__(
        self,
        agent: "Agent",
        input_schema: Type[BaseModel] | None = None,
    ):
        self.agent = agent
        self.input_schema = input_schema

        if input_schema:
            parameters = input_schema.model_json_schema()
            parameters.pop("$defs", None)
            parameters.pop("title", None)
        else:
            parameters = {
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": "The task or question to delegate"
                    }
                },
                "required": ["request"]
            }

        tool_def = format_tool_definition(agent.name, agent.description, parameters)
        super().__init__(name=agent.name, description=agent.description, tool_definition=tool_def)

    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        if self.input_schema:
            validated = self.input_schema.model_validate(kwargs)
            request = validated.model_dump_json(exclude_none=True)
        else:
            request = kwargs.get("request", str(kwargs))
        result = await self.agent.run(request)
        return result.output
````


## 9.5 节：Agent Transfer

`transfer.py`：`create_transfer_tool()`（Listing 9.14）生成 `transfer_to_agent` 工具；转移目标列表由 Agent 的方法计算——`_get_transfer_targets()`（Listing 9.12，子 agent / 父 agent / 兄弟 agent）、`_find_agent()` / `_find_in_subtree()`（Listing 9.13，按名字在 agent 树中查找）。

Agent 侧的变化（都在 ch09 版 agent.py 里）：构造时 `sub_agents` 校验并设置 parent（Listing 9.11），`_setup_tools()` 自动注册转移工具（Listing 9.15），`run()` 循环里检测 `context.transfer_to` 并切换到目标 agent（Listing 9.18）。`context.py` 相应增加 `transfer_to` / `transfer_tools` 字段。

#### 📄 `scratch_agents/transfer.py`

- **状态**：ch09（首次创建，最终版）
- **对应书内容**：Listing 9.14（create_transfer_tool）；9.11–9.13 对应 agent.py 的方法
- **行数**：53 行（清单为完整文件内容，从第 1 行到第 53 行）

<!-- FILE: scratch_agents/transfer.py -->
<!-- STATE: ch09 -->

````python
"""Agent transfer tool for multi-agent routing."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from scratch_agents.tools.base import FunctionTool, tool
from scratch_agents.context import ExecutionContext

if TYPE_CHECKING:
    from scratch_agents.agent import Agent


def create_transfer_tool(target_agents: List["Agent"]) -> FunctionTool:
    """Create transfer tool from list of transferable agents."""

    # Compose agent info
    target_names = [agent.name for agent in target_agents]
    agent_descriptions = []
    for agent in target_agents:
        desc = agent.description or agent.instructions[:100].replace('\n', ' ')
        if len(desc) > 100:
            desc = desc[:100] + "..."
        agent_descriptions.append(f"  - {agent.name}: {desc}")

    agent_info = "\n".join(agent_descriptions)

    @tool(
        name="transfer_to_agent",
        description=f"""Transfers work to another agent.

Use this tool when the current question belongs to another agent's specialty.

Available agents:
{agent_info}
"""
    )
    def transfer_to_agent(context: ExecutionContext, agent_name: str) -> str:
        """Agent transfer tool."""
        if agent_name not in target_names:
            return f"Error: '{agent_name}' is not valid. Available: {target_names}"

        # Only apply first transfer request
        if context.transfer_to is None:
            context.transfer_to = agent_name
            return f"Transferring to {agent_name}..."
        else:
            return f"Transfer already requested to {context.transfer_to}"

    # Add enum constraint
    transfer_to_agent.tool_definition["function"]["parameters"]["properties"]["agent_name"]["enum"] = target_names

    return transfer_to_agent
````


#### 📄 `scratch_agents/context.py`

- **状态**：ch09（最终版）
- **对应书内容**：Listing 9.x（transfer_to / transfer_tools 字段）
- **行数**：69 行（清单为完整文件内容，从第 1 行到第 69 行）
- **说明**：context.py 的最终形态：execution_id/events/current_step/state/final_result + session/session_manager/memory_manager（ch06）+ code_env（ch08）+ transfer_to/transfer_tools（ch09），以及 AgentResult / PendingToolCall / ToolConfirmation。

<!-- FILE: scratch_agents/context.py -->
<!-- STATE: ch09 -->

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
    # CH06 session
    session: Optional[Any] = None
    session_manager: Optional[Any] = None
    # CH06 long-term memory
    memory_manager: Optional[Any] = None
    # CH08 code execution
    code_env: Optional[Any] = None  # E2B Sandbox
    # CH09 agent transfer
    transfer_to: Optional[str] = None
    transfer_tools: Dict[str, Any] = field(default_factory=dict)

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
    status: str = "complete"  # "complete" | "pending_confirmation" | "error"
    pending_tool_calls: list = field(default_factory=list)


class PendingToolCall(BaseModel):
    """A tool call awaiting user confirmation (CH06 human-in-the-loop)."""
    tool_call: "ToolCall"
    confirmation_message: str


class ToolConfirmation(BaseModel):
    """User's response to a pending tool call (CH06 human-in-the-loop)."""
    tool_call_id: str
    approved: bool
    modified_arguments: dict | None = None
    reason: str | None = None


# Avoid circular import — resolve forward reference
from scratch_agents.types import ToolCall  # noqa: E402

PendingToolCall.model_rebuild()
````


#### 📄 `scratch_agents/agent.py`

- **状态**：ch09（最终版，与 scratch_agents/agent.py 完全一致）
- **对应书内容**：Listing 9.11（_validate_and_set_sub_agents）、9.12（_get_transfer_targets）、9.13（_find_agent）、9.15（_setup_tools 注册转移工具）、9.18（run() 转移处理）
- **行数**：679 行（清单为完整文件内容，从第 1 行到第 679 行）
- **说明**：agent.py 的最终形态：在 ch08 版基础上增加 sub_agents / disallow_transfer_to_peers 构造参数、run() 的转移处理、转移工具自动注册和三个多 agent 辅助方法。测试套件（tests/test_regressions.py 与 tests/test_notebooks.py）的整体断言都建立在这一版之上。

<!-- FILE: scratch_agents/agent.py -->
<!-- STATE: ch09 -->

````python
"""Core Agent class for the scratch_agents framework."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Type

from pydantic import BaseModel

from scratch_agents.llm import LlmClient, LlmRequest, LlmResponse
from scratch_agents.tools.base import BaseTool, FunctionTool, tool
from scratch_agents.tools.helpers import format_tool_definition
from scratch_agents.context import (
    AgentResult,
    ExecutionContext,
    PendingToolCall,
    ToolConfirmation,
)
from scratch_agents.types import (
    Event,
    Message,
    ToolCall,
    ToolResult,
)

if TYPE_CHECKING:
    from scratch_agents.memory.long_term import TaskMemoryManager
    from scratch_agents.memory.session import BaseSessionManager

logger = logging.getLogger(__name__)


class Agent:
    """Tool-calling agent with ReAct loop.

    Supports structured output, callbacks, sessions, memory,
    code execution, skills, and multi-agent patterns.
    """

    def __init__(
        self,
        model: LlmClient,
        tools: List[BaseTool] | None = None,
        instructions: str = "",
        max_steps: int = 10,
        name: str = "agent",
        description: str = "",
        # CH04 structured output
        output_type: Optional[Type[BaseModel]] = None,
        # CH05 callbacks
        before_tool_callbacks: list[Callable] | None = None,
        after_tool_callbacks: list[Callable] | None = None,
        # CH06 session & memory
        session_manager: Optional["BaseSessionManager"] = None,
        memory_manager: Optional["TaskMemoryManager"] = None,
        before_llm_callbacks: list[Callable] | None = None,
        # CH08 code execution
        code_execution: str | None = None,  # "e2b"
        skills_path: str | None = None,
        # CH09 multi-agent
        sub_agents: list["Agent"] | None = None,
        disallow_transfer_to_peers: bool = False,
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
        self.sub_agents = sub_agents or []
        self.disallow_transfer_to_peers = disallow_transfer_to_peers
        self.parent: Agent | None = None
        self._sandbox_tools: List[FunctionTool] = []
        self.tools = self._setup_tools(tools or [])

        # Set up sub-agent relationships
        if self.sub_agents:
            self._validate_and_set_sub_agents()

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

        # Load or create session
        session = None
        if session_id and self.session_manager:
            session = await self.session_manager.get_or_create(session_id)

        # Create or reuse context
        if context is None:
            context = ExecutionContext(
                session=session,
                session_manager=self.session_manager,
                memory_manager=self.memory_manager,
            )
            # Restore previous events from session
            if session:
                context.events = list(session.events)
                context.state = dict(session.state)
        elif context.memory_manager is None:
            context.memory_manager = self.memory_manager

        # Handle tool confirmations (human-in-the-loop resume)
        if tool_confirmations:
            await self._process_confirmations(context, tool_confirmations)

        # Add user input as the first event
        if user_input:
            user_event = Event(
                execution_id=context.execution_id,
                author="user",
                content=[Message(role="user", content=user_input)],
            )
            context.add_event(user_event)

        # Set up code execution environment if needed
        if self.code_execution == "e2b" and context.code_env is None:
            await self._setup_code_env(context)

        try:
            # Execute steps until completion or max steps reached
            while not context.final_result and context.current_step < self.max_steps:
                result = await self.step(context, verbose=verbose)

                # Check for pending tool calls (human-in-the-loop)
                if result and result.status == "pending_confirmation":
                    if session and self.session_manager:
                        session.events = list(context.events)
                        session.state = dict(context.state)
                        await self.session_manager.save(session)
                    return result

                # Check if the last event is a final response
                if context.events:
                    last_event = context.events[-1]
                    if self._is_final_response(last_event):
                        context.final_result = self._extract_final_result(last_event)

                # Check for agent transfer
                if context.transfer_to:
                    target_name = context.transfer_to
                    context.transfer_to = None
                    target = self._find_agent(target_name)
                    if target:
                        return await target.run(context=context, verbose=verbose)

            # Save memory if manager is available
            if self.memory_manager:
                try:
                    await self.memory_manager.save(context)
                except Exception as e:
                    logger.warning(f"Failed to save memory: {e}")

            # Save session (sync events back)
            if session and self.session_manager:
                session.events = list(context.events)
                session.state = dict(context.state)
                await self.session_manager.save(session)

            return AgentResult(output=context.final_result, context=context)
        finally:
            if context.code_env is not None:
                sandbox = context.code_env
                context.code_env = None
                sandbox.kill()

    async def step(
        self,
        context: ExecutionContext,
        verbose: bool = False,
    ) -> AgentResult | None:
        """Perform one think-act cycle."""

        # Prepare what to send to the LLM
        llm_request = await self._prepare_llm_request(context)

        # Run before-LLM callbacks
        for callback in self.before_llm_callbacks:
            cb_result = callback(context, llm_request)
            if hasattr(cb_result, "__await__"):
                cb_result = await cb_result
            if isinstance(cb_result, LlmResponse):
                # Callback provided a response, skip LLM call
                llm_response = cb_result
                break
        else:
            # Get LLM's decision
            llm_response = await self.think(llm_request)

        if llm_response.error_message:
            raise RuntimeError(f"LLM request failed: {llm_response.error_message}")

        if verbose:
            self._log_response(llm_response)

        # Record LLM response as an event
        response_event = Event(
            execution_id=context.execution_id,
            author=self.name,
            content=llm_response.content,
        )
        context.add_event(response_event)

        # Execute tools if the LLM requested any
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

            # Check if tool requires confirmation (human-in-the-loop)
            if tool_obj.requires_confirmation:
                msg = tool_obj.get_confirmation_message(tool_call.arguments)
                pending.append(PendingToolCall(
                    tool_call=tool_call,
                    confirmation_message=msg,
                ))
                continue

            # Run before-tool callbacks
            skip = False
            for cb in self.before_tool_callbacks:
                cb_result = cb(context, tool_call)
                if hasattr(cb_result, "__await__"):
                    cb_result = await cb_result
                if cb_result is not None:
                    # Callback returned a replacement result
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

            # Execute the tool
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

            # Run after-tool callbacks
            for cb in self.after_tool_callbacks:
                cb_result = cb(context, tool_result)
                if hasattr(cb_result, "__await__"):
                    cb_result = await cb_result
                if cb_result is not None:
                    tool_result = cb_result

            results.append(tool_result)

        # If there are pending confirmations, pause execution
        if pending:
            # Store pending calls in context state
            context.state["pending_tool_calls"] = [
                p.model_dump() for p in pending
            ]
            # Still record any results we have
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

        # Record tool results
        if results:
            tool_event = Event(
                execution_id=context.execution_id,
                author=self.name,
                content=results,
            )
            context.add_event(tool_event)

        # Handle transfer_to (CH09)
        for result in results:
            if result.name == "transfer_to_agent" and result.status == "success":
                # The transfer tool sets context.transfer_to
                pass

        return None

    # ------------------------------------------------------------------ #
    # Internal methods
    # ------------------------------------------------------------------ #

    async def _prepare_llm_request(self, context: ExecutionContext) -> LlmRequest:
        """Build an LlmRequest from the current context."""
        # Flatten events into content items
        flat_contents = []
        for event in context.events:
            flat_contents.extend(event.content)

        # Build instructions
        instructions = []
        if self.instructions:
            instructions.append(self.instructions)

        # Add sandbox tools prompt (CH08)
        sandbox_prompt = self._get_sandbox_tools_prompt()
        if sandbox_prompt:
            instructions.append(sandbox_prompt)

        # Add skills prompt if available (CH08)
        if self.skills_path:
            try:
                from scratch_agents.skills import discover_skills, generate_skills_prompt
                skills = discover_skills(self.skills_path)
                skills_prompt = generate_skills_prompt(skills)
                if skills_prompt:
                    instructions.append(skills_prompt)
            except Exception:
                pass

        # Filter tools that should be exposed to the LLM (Listing 6.38)
        llm_tools = [t for t in self.tools if t.tool_definition is not None]

        # Determine tool choice strategy
        if self.output_tool_name:
            tool_choice = "required"
        elif llm_tools:
            tool_choice = "auto"
        else:
            tool_choice = None

        request = LlmRequest(
            model_id=self.model.model,
            instructions=instructions,
            contents=flat_contents,
            tools=llm_tools,
            tool_choice=tool_choice,
        )

        # Let tools modify the request (Listing 6.38)
        for tool_obj in self.tools:
            await tool_obj.process_llm_request(context, request)

        return request

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
        tools = [
            t if isinstance(t, BaseTool) else FunctionTool(t)
            for t in tools
        ]

        # Add structured output tool (CH04)
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

        # Collect sandbox-executable tools (CH08)
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

        # Add code execution tool (CH08)
        if self.code_execution == "e2b":
            from scratch_agents.tools.code_execution import execute_python
            tools.append(execute_python)

        # Add transfer tool (CH09)
        if self.sub_agents:
            from scratch_agents.transfer import create_transfer_tool
            transfer_tool = create_transfer_tool(self.sub_agents)
            tools.append(transfer_tool)

        # Add memory tool if memory_manager is available (CH06)
        if self.memory_manager:
            from scratch_agents.tools.memory_tool import MemoryTool
            tools.append(MemoryTool())

        return tools

    async def _setup_code_env(self, context: ExecutionContext):
        """Set up E2B sandbox environment (CH08)."""
        try:
            from e2b_code_interpreter import Sandbox
            sandbox = Sandbox.create(timeout=300)
            context.code_env = sandbox

            # Upload sandbox-executable tools
            self._register_sandbox_tools(sandbox)

            # Upload skills if available
            if self.skills_path:
                from scratch_agents.skills import discover_skills
                skills = discover_skills(self.skills_path)
                for skill_info in skills:
                    for path in skill_info.path.rglob("*"):
                        if path.is_file():
                            relative = path.relative_to(skill_info.path).as_posix()
                            sandbox.files.write(
                                f"/home/user/skills/{skill_info.name}/{relative}",
                                path.read_bytes(),
                            )
        except Exception as e:
            if context.code_env is not None:
                sandbox = context.code_env
                context.code_env = None
                sandbox.kill()
            raise RuntimeError("Failed to set up code execution environment") from e

    def _register_sandbox_tools(self, sandbox) -> None:
        """Register sandbox-executable tools by running their source in the sandbox (CH08)."""
        tool_sources = []
        for t in self._sandbox_tools:
            source = t.get_source_code()
            tool_sources.append(source)
        combined_source = "\n\n".join(tool_sources)
        result = sandbox.run_code(combined_source)
        if result.error:
            raise RuntimeError(f"Failed to register sandbox tools: {result.error}")

    def _get_sandbox_tools_prompt(self) -> str:
        """Generate prompt describing sandbox-executable tools (CH08)."""
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

    # CH09 multi-agent helpers
    def _get_transfer_targets(self) -> list["Agent"]:
        """List of targets the current agent can transfer to (Listing 9.12)."""
        targets: list["Agent"] = []

        # 1. Children
        targets.extend(self.sub_agents)

        # 2. Parent and siblings
        if self.parent:
            targets.append(self.parent)

            # 3. Siblings (optional)
            if not self.disallow_transfer_to_peers:
                for sibling in self.parent.sub_agents:
                    if sibling.name != self.name:
                        targets.append(sibling)
        return targets

    def _find_agent(self, name: str) -> "Agent" | None:
        """Search by name across the entire agent tree (Listing 9.13)."""
        root = self
        while root.parent:
            root = root.parent
        return root._find_in_subtree(name)

    def _find_in_subtree(self, name: str) -> "Agent" | None:
        """Search in current agent and subtree (Listing 9.13)."""
        if self.name == name:
            return self
        for sub in self.sub_agents:
            if found := sub._find_in_subtree(name):
                return found
        return None

    def _validate_and_set_sub_agents(self) -> None:
        """Validate name/parent duplicates in sub_agents and set parent (Listing 9.11)."""
        seen_names = set()
        for sub in self.sub_agents:
            if sub.name in seen_names:
                raise ValueError(f"Duplicate sub-agent name: '{sub.name}'")
            seen_names.add(sub.name)

            if sub.parent is not None:
                raise ValueError(
                    f"Agent '{sub.name}' already has parent '{sub.parent.name}'"
                )
            sub.parent = self

    async def _process_confirmations(
        self,
        context: ExecutionContext,
        confirmations: list[ToolConfirmation],
    ):
        """Process tool confirmations from human-in-the-loop (CH06)."""
        raw_pending = context.state.pop("pending_tool_calls", [])
        pending = [PendingToolCall.model_validate(d) for d in raw_pending]

        tools_dict = {t.name: t for t in self.tools}
        results = []

        for pending_call in pending:
            tc = pending_call.tool_call
            # Find matching confirmation
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


## 9.6 节：A2A 跨网络协作

`a2a_server.py`：AgentExecutor 基类与示例 MathAgentExecutor——把 agent 包装成 A2A server 的 executor（书 9.6.2 节）。

`remote.py`：RemoteAgent——A2A 客户端（书 9.6.3 节），从 `/.well-known/agent.json` 读取 agent card，向 `/tasks/send` 发 JSON-RPC 请求。

最后更新 `tools/__init__.py` 为最终版（追加 AgentTool 导出）。

#### 📄 `scratch_agents/a2a_server.py`

- **状态**：ch09（首次创建，最终版）
- **对应书内容**：书 9.6.2 节（A2A server 端，无编号 Listing）
- **行数**：41 行（清单为完整文件内容，从第 1 行到第 41 行）

<!-- FILE: scratch_agents/a2a_server.py -->
<!-- STATE: ch09 -->

````python
"""A2A server adapter for exposing an Agent as an A2A service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scratch_agents.agent import Agent


class AgentExecutor:
    """Base class for A2A agent executors."""

    async def execute(self, context: Any, event_queue: Any) -> None:
        raise NotImplementedError


class MathAgentExecutor(AgentExecutor):
    """Example A2A executor wrapping an Agent."""

    def __init__(self, agent: "Agent"):
        self.agent = agent

    async def execute(self, context: Any, event_queue: Any) -> None:
        """Execute the agent and push results to the event queue."""
        # Extract user message from A2A context
        user_input = ""
        if hasattr(context, "message"):
            for part in context.message.get("parts", []):
                if part.get("type") == "text":
                    user_input = part["text"]
                    break

        result = await self.agent.run(user_input=user_input)

        # Push result to event queue
        if event_queue and result.output:
            await event_queue.put({
                "type": "artifact",
                "parts": [{"type": "text", "text": str(result.output)}],
            })
````


#### 📄 `scratch_agents/remote.py`

- **状态**：ch09（首次创建，最终版）
- **对应书内容**：书 9.6.3 节（A2A 客户端，无编号 Listing）
- **行数**：71 行（清单为完整文件内容，从第 1 行到第 71 行）

<!-- FILE: scratch_agents/remote.py -->
<!-- STATE: ch09 -->

````python
"""Remote agent support via A2A protocol."""

from __future__ import annotations

from typing import Any

import httpx

from scratch_agents.context import AgentResult, ExecutionContext


class RemoteAgent:
    """Client for interacting with remote agents via A2A protocol."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.name = ""
        self.description = ""
        self._load_agent_info()

    def _load_agent_info(self) -> None:
        """Load agent info from the remote server's agent card."""
        try:
            response = httpx.get(f"{self.base_url}/.well-known/agent.json")
            if response.status_code == 200:
                info = response.json()
                self.name = info.get("name", "remote_agent")
                self.description = info.get("description", "")
        except Exception:
            self.name = "remote_agent"

    async def run(
        self,
        user_input: str,
        context: ExecutionContext | None = None,
        verbose: bool = False,
    ) -> AgentResult:
        """Send a request to the remote agent."""
        if context is None:
            context = ExecutionContext()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/tasks/send",
                json={
                    "jsonrpc": "2.0",
                    "method": "tasks/send",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [{"type": "text", "text": user_input}],
                        }
                    },
                },
                timeout=120.0,
            )

            result = response.json()
            output = ""

            if "result" in result:
                task_result = result["result"]
                if "artifacts" in task_result:
                    parts = []
                    for artifact in task_result["artifacts"]:
                        for part in artifact.get("parts", []):
                            if part.get("type") == "text":
                                parts.append(part["text"])
                    output = "\n".join(parts)

            return AgentResult(output=output, context=context)
````


#### 📄 `scratch_agents/tools/__init__.py`

- **状态**：ch09（最终版）
- **对应书内容**：无（包导出约定）
- **行数**：5 行（清单为完整文件内容，从第 1 行到第 5 行）
- **说明**：在 ch04 版基础上追加 AgentTool 导出。tools/__init__.py 的最终形态。

<!-- FILE: scratch_agents/tools/__init__.py -->
<!-- STATE: ch09 -->

````python
from scratch_agents.tools.base import BaseTool, FunctionTool, tool
from scratch_agents.tools.search import search_web
from scratch_agents.tools.calculator import calculator
from scratch_agents.tools.mcp import load_mcp_tools, mcp_connection, mcp_tools_to_openai_format
from scratch_agents.tools.agent_tool import AgentTool
````


## 本章改动一览

| 文件 | 状态 | 行数 |
|---|---|---|
| `scratch_agents/workflows/__init__.py` | ch09（首次创建，最终版） | 3 |
| `scratch_agents/workflows/sequential.py` | ch09（首次创建，最终版） | 52 |
| `scratch_agents/workflows/parallel.py` | ch09（首次创建，最终版） | 99 |
| `scratch_agents/workflows/loop.py` | ch09（首次创建，最终版） | 65 |
| `scratch_agents/tools/agent_tool.py` | ch09（首次创建，最终版） | 54 |
| `scratch_agents/transfer.py` | ch09（首次创建，最终版） | 53 |
| `scratch_agents/context.py` | ch09（最终版） | 69 |
| `scratch_agents/agent.py` | ch09（最终版，与 scratch_agents/agent.py 完全一致） | 679 |
| `scratch_agents/a2a_server.py` | ch09（首次创建，最终版） | 41 |
| `scratch_agents/remote.py` | ch09（首次创建，最终版） | 71 |
| `scratch_agents/tools/__init__.py` | ch09（最终版） | 5 |

## 本章自测

```bash
python -c "from scratch_agents.workflows import SequentialWorkflow, ParallelWorkflow, LoopWorkflow; from scratch_agents.transfer import create_transfer_tool; print('ch09 ok')"
```
