# 第 7 章 · 规划与反思

> 对应原书 第 7 章（小节号在各代码块中标注）。本章文档：`docs/ch07-planning-and-reflection.md`。

## 本章构建目标

第 7 章让 agent「有时间思考」：7.2 节的 planning 工具让模型把复杂任务拆成 Task 清单，7.3 节的 reflection 工具让模型定期自检、从失败中恢复。两者都是普通的 @tool 工具，不改 Agent 内核，本章只新增一个文件。

## 7.2 / 7.3 节：planning 工具与 reflection 工具

`planning.py`：Task 模型（Listing 7.1，三种状态三种 Markdown 呈现）、`create_tasks()`（Listing 7.2，planning 工具本体，docstring 里写明了何时用/何时不用）、`reflection()`（Listing 7.4，反思工具）。

用法（notebook 中演示，Listing 7.3 / 7.6 / 7.7）：`Agent(tools=[create_tasks, reflection])`，模型自行决定何时调用。

#### 📄 `scratch_agents/planning.py`

- **状态**：ch07（首次创建，最终版）
- **对应书内容**：Listing 7.1（Task）、7.2（create_tasks）、7.4（reflection）
- **行数**：77 行（清单为完整文件内容，从第 1 行到第 77 行）

<!-- FILE: scratch_agents/planning.py -->
<!-- STATE: ch07 -->

````python
"""Planning and reflection tools for the agent."""

from typing import List, Literal

from pydantic import BaseModel

from scratch_agents.tools.base import tool


class Task(BaseModel):
    """A task in the agent's plan."""
    content: str
    status: Literal["pending", "in_progress", "completed"]

    def __str__(self):
        if self.status == "pending":
            return f"[ ] {self.content}"
        elif self.status == "in_progress":
            return f"[>] **{self.content}**"
        elif self.status == "completed":
            return f"[x] ~~{self.content}~~"
        return self.content


@tool
def create_tasks(tasks: List[Task]) -> str:
    """Create or update a task plan.

    WHEN TO USE:
    - Complex queries requiring multiple steps of research
    - Questions that need to combine information from different sources

    WHEN NOT TO USE:
    - Simple questions answerable with a single search
    - Tasks with obvious, straightforward procedures

    HOW TO USE:
    - Regenerate the entire task list with updated statuses
    - Mark completed tasks as 'completed'
    - Mark the next task to work on as 'in_progress'
    - Keep future tasks as 'pending'
    """
    result = []
    for task in tasks:
        result.append(str(Task.model_validate(task)))
    return "\n".join(result)


@tool
def reflection(analysis: str, need_replan: bool = False) -> str:
    """Pause and analyze progress before continuing.

    WHEN TO USE:
    1. PROGRESS REVIEW - After completing a meaningful step
       "Kipchoge's record found: 2:01:09. Moving to moon distance research."

    2. ERROR ANALYSIS - When a tool fails or returns unexpected results
       "Wikipedia tool failed. Cause: service unavailable. Alternative: use web search."

    3. RESULT SYNTHESIS - When combining information from multiple sources
       "Two different moon distances found. Problem asks for closest approach, so using perigee: 356,500km."

    4. SELF CHECK - Before providing final answer
       "Have all required data: marathon pace 20.81km/h, moon distance 356,500km. Ready to calculate."

    WHEN NOT TO USE:
    - After every single tool call (excessive overhead)
    - During simple, straightforward operations
    - When everything is proceeding as expected

    Args:
        analysis: Your assessment of current situation and next direction
        need_replan: Set True if the current plan needs modification
    """
    if need_replan:
        return f"Reflection recorded (REPLAN NEEDED): {analysis}"
    return f"Reflection recorded: {analysis}"
````


## 本章改动一览

| 文件 | 状态 | 行数 |
|---|---|---|
| `scratch_agents/planning.py` | ch07（首次创建，最终版） | 77 |

## 本章自测

```bash
python -c "import asyncio; from scratch_agents.planning import create_tasks; from scratch_agents import ExecutionContext; print(asyncio.run(create_tasks(ExecutionContext(), tasks=[{'content': 'Find evidence', 'status': 'pending'}])))"
```
