# 第 3 章 · 工具使用（Tool Use）

> 对应原书 第 3 章（小节号在各代码块中标注）。本章文档：`docs/ch03-tool-use.md`。

## 本章构建目标

书第 3 章讲工具：3.1–3.2 节解释 function calling 的原理，3.3 节手写一个搜索工具并把它转换成 OpenAI 格式的 tool definition，3.4 节讲 MCP 协议。

本章要建的工具函数属于「第 3 章的实验品」，最终项目里它们被收进 `scratch_agents/tools/`：`helpers.py`（函数签名 → JSON Schema 的转换、tool definition 格式化、工具执行器，即 Listing 3.16–3.19）、`search.py`（Tavily 搜索，Listing 3.12–3.15）、`calculator.py`（Listing 3.1–3.2）。

`tools/base.py` 的统一工具抽象（BaseTool / FunctionTool）是第 4 章的内容（4.4 节），本章不用建。

## 3.3 节：工具实现与 tool definition 转换

先建 `tools/__init__.py`（ch03 版本，只导出搜索和计算器），然后依次建三个模块。

`helpers.py` 是本章的核心：`function_to_input_schema()`（Listing 3.17）用 `inspect` + 类型注解生成参数 JSON Schema；`function_to_tool_definition()`（Listing 3.18）补上名称和 docstring 描述；`tool_execution()`（Listing 3.19）按工具名分发执行——第 3 章的端到端示例（Listing 3.20–3.21 的 mini agent loop）会用到它。

#### 📄 `scratch_agents/tools/__init__.py`

- **状态**：ch03（中间态，ch04 会更新）
- **对应书内容**：无（包导出约定）
- **行数**：4 行（清单为完整文件内容，从第 1 行到第 4 行）

<!-- FILE: scratch_agents/tools/__init__.py -->
<!-- STATE: ch03 -->

````python
"""Tool modules for the scratch_agents framework."""

from scratch_agents.tools.search import search_web
from scratch_agents.tools.calculator import calculator
````


#### 📄 `scratch_agents/tools/helpers.py`

- **状态**：ch03（首次创建，之后不变）
- **对应书内容**：Listing 3.16（抽取元数据）、3.17（function_to_input_schema）、3.18（function_to_tool_definition）、3.19（tool_execution）；`format_tool_definition` 对应 Listing 3.1 的 schema 封装
- **行数**：114 行（清单为完整文件内容，从第 1 行到第 114 行）

<!-- FILE: scratch_agents/tools/helpers.py -->
<!-- STATE: ch03 -->

````python
import inspect
import json
from typing import get_type_hints


def function_to_input_schema(func) -> dict:
    """Convert a function's signature to a JSON Schema for tool parameters.

    Inspects type hints and docstring to generate the schema.
    """
    try:
        hints = get_type_hints(func)
    except Exception:
        # Fallback for closures where get_type_hints can't resolve annotations
        hints = {
            name: param.annotation
            for name, param in inspect.signature(func).parameters.items()
            if param.annotation is not inspect.Parameter.empty
        }
    sig = inspect.signature(func)

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name in ("self", "context"):
            continue

        prop = {}
        hint = hints.get(name)

        if hint == str:
            prop["type"] = "string"
        elif hint == int:
            prop["type"] = "integer"
        elif hint == float:
            prop["type"] = "number"
        elif hint == bool:
            prop["type"] = "boolean"
        elif hint == list or (hasattr(hint, "__origin__") and hint.__origin__ is list):
            prop["type"] = "array"
            # Try to get item type
            if hasattr(hint, "__args__") and hint.__args__:
                item_type = hint.__args__[0]
                if item_type == str:
                    prop["items"] = {"type": "string"}
                elif item_type == int:
                    prop["items"] = {"type": "integer"}
                elif hasattr(item_type, "model_json_schema"):
                    prop["items"] = item_type.model_json_schema()
        elif hasattr(hint, "model_json_schema"):
            # Pydantic model
            prop = hint.model_json_schema()
        else:
            prop["type"] = "string"

        # Add description from docstring if available
        prop["description"] = f"Parameter: {name}"

        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema


def format_tool_definition(name: str, description: str, parameters: dict) -> dict:
    """Format a tool definition in the OpenAI function calling format."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
    }


def function_to_tool_definition(func) -> dict:
    """Convert a Python function to an OpenAI-format tool definition.

    Uses the function name, docstring, and type hints.
    """
    name = func.__name__
    description = inspect.getdoc(func) or f"Function: {name}"
    parameters = function_to_input_schema(func)
    return format_tool_definition(name, description, parameters)


def tool_execution(tool_box: dict, tool_call) -> str:
    """Execute a tool call using a tool_box mapping.

    Args:
        tool_box: Dict mapping tool names to callables
        tool_call: Tool call object with function.name and function.arguments
    """
    func_name = tool_call.function.name
    if func_name not in tool_box:
        return f"Error: Unknown tool '{func_name}'"

    try:
        args = json.loads(tool_call.function.arguments)
        result = tool_box[func_name](**args)
        return str(result)
    except Exception as e:
        return f"Error executing {func_name}: {str(e)}"
````


#### 📄 `scratch_agents/tools/calculator.py`

- **状态**：ch03（首次创建，之后不变）
- **对应书内容**：Listing 3.1（calculator_tool_definition 的原型）、3.2（calculator 函数体）
- **行数**：20 行（清单为完整文件内容，从第 1 行到第 20 行）

<!-- FILE: scratch_agents/tools/calculator.py -->
<!-- STATE: ch03 -->

````python
def calculator(operator: str, first_number: float, second_number: float) -> float:
    """Perform basic arithmetic operations.

    Args:
        operator: Arithmetic operation - add, subtract, multiply, or divide
        first_number: First number
        second_number: Second number
    """
    if operator == "add":
        return first_number + second_number
    elif operator == "subtract":
        return first_number - second_number
    elif operator == "multiply":
        return first_number * second_number
    elif operator == "divide":
        if second_number == 0:
            raise ValueError("Cannot divide by zero")
        return first_number / second_number
    else:
        raise ValueError(f"Unknown operator: {operator}")
````


#### 📄 `scratch_agents/tools/search.py`

- **状态**：ch03（首次创建，之后不变）
- **对应书内容**：Listing 3.12（基础搜索）、3.14（增加 max_results/topic/time_range 参数）、3.15（错误处理）
- **行数**：32 行（清单为完整文件内容，从第 1 行到第 32 行）

<!-- FILE: scratch_agents/tools/search.py -->
<!-- STATE: ch03 -->

````python
from tavily import TavilyClient
import os


def search_web(
    query: str,
    max_results: int = 5,
    topic: str = "general",
    time_range: str | None = None,
) -> list | str:
    """Search the web using Tavily API.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        topic: Search topic - 'general' or 'news'
        time_range: Time range filter (e.g., 'day', 'week', 'month', 'year')
    """
    try:
        client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        kwargs = {
            "query": query,
            "max_results": max_results,
            "topic": topic,
        }
        if time_range:
            kwargs["time_range"] = time_range

        response = client.search(**kwargs)
        return response.get("results", [])
    except Exception as e:
        return f"Search error: {str(e)}"
````


## 3.4 节：MCP（第 3 章视角）

书 3.4 节在 notebook 里演示了连接 MCP server、`mcp_tools_to_openai_format()` 转换（Listing 3.23）。最终项目里这个转换函数放在 `scratch_agents/tools/mcp.py` 中，该文件的其余部分（`load_mcp_tools` / `mcp_connection`）属于第 4 章 Listing 4.7–4.8，我们在 ch04 文档中一并给出。

## 本章改动一览

| 文件 | 状态 | 行数 |
|---|---|---|
| `scratch_agents/tools/__init__.py` | ch03（中间态，ch04 会更新） | 4 |
| `scratch_agents/tools/helpers.py` | ch03（首次创建，之后不变） | 114 |
| `scratch_agents/tools/calculator.py` | ch03（首次创建，之后不变） | 20 |
| `scratch_agents/tools/search.py` | ch03（首次创建，之后不变） | 32 |

## 本章自测

```bash
python -c "from scratch_agents.tools.calculator import calculator; print(calculator('multiply', 6, 7))"
```
