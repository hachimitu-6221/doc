# 第 5 章 · 用 RAG 构建知识库 + 文件工具 + 回调

> 对应原书 第 5 章（小节号在各代码块中标注）。本章文档：`docs/ch05-rag-and-file-tools.md`。

## 本章构建目标

第 5 章解决「内部数据怎么用」：5.2–5.3 节实现向量检索（embedding、chunking、vector search），5.4 节实现文件系统工具（读 zip/csv/xlsx、列目录、读文本）并接回 agent，5.5 节用回调（callback）在不改 Agent 内核的前提下扩展行为——审批危险工具、压缩搜索结果。

## 5.3 节：向量检索

`rag.py` 三个函数：`get_embeddings()`（Listing 5.1，调 OpenAI embedding API）、`fixed_length_chunking()`（Listing 5.3，固定长度分块，带校验）、`vector_search()`（Listing 5.5，余弦相似度检索）。

#### 📄 `scratch_agents/rag.py`

- **状态**：ch05（首次创建，之后不变）
- **对应书内容**：Listing 5.1（get_embeddings）、5.3（fixed_length_chunking）、5.5（vector_search）
- **行数**：47 行（清单为完整文件内容，从第 1 行到第 47 行）

<!-- FILE: scratch_agents/rag.py -->
<!-- STATE: ch05 -->

````python
"""RAG functionality: embeddings, chunking, and vector search."""

from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_embeddings(texts, model="text-embedding-3-small") -> np.ndarray:
    """Convert text to embedding vectors."""
    client = OpenAI()
    if isinstance(texts, str):
        texts = [texts]

    response = client.embeddings.create(input=texts, model=model)
    return np.array([item.embedding for item in response.data])


def fixed_length_chunking(text, chunk_size=500, overlap=50) -> list[str]:
    """Split text into fixed-length chunks."""
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else end

    return chunks


def vector_search(query, chunks, chunk_embeddings, top_k=3) -> list:
    """Find the most similar chunks to the query."""
    query_embedding = get_embeddings(query)
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'chunk': chunks[idx],
            'similarity': similarities[idx],
        })
    return results
````


## 5.4 节：文件系统工具

`tools/file_tools.py`：常量定义扩展名分组后，是 `unzip_file`（Listing 5.15）、`list_files`（5.16）、`read_file`（5.17，按扩展名分发）、`read_media_file`（5.20，图片/音频/PDF 走多模态模型），以及文本/CSV/Excel 的读取辅助函数（5.18/5.19）。书 5.4.3 节（Listing 5.21）演示了如何把这些函数用 FunctionTool 包装后注册进 agent，那个注册动作发生在 notebook 里，包内不新增代码。

#### 📄 `scratch_agents/tools/file_tools.py`

- **状态**：ch05（首次创建，之后不变）
- **对应书内容**：Listing 5.15–5.21（文件工具全集）
- **行数**：222 行（清单为完整文件内容，从第 1 行到第 222 行）

<!-- FILE: scratch_agents/tools/file_tools.py -->
<!-- STATE: ch05 -->

````python
"""File system tools for the agent."""

import base64
import zipfile
from pathlib import Path

TEXT_EXTENSIONS = ['.txt', '.py', '.js', '.json', '.md', '.html',
                   '.css', '.xml', '.yaml', '.yml', '.log', '.sh']
SPREADSHEET_EXTENSIONS = ['.xlsx', '.xls', '.csv']
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']
AUDIO_EXTENSIONS = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm']
PDF_EXTENSIONS = ['.pdf']


def unzip_file(zip_path: str, extract_to: str = None) -> str:
    """Extract a zip file to the specified directory."""
    zip_path = Path(zip_path)

    if not zip_path.exists():
        return f"File not found: {zip_path}"

    # Default extraction path: create folder with zip filename
    if extract_to is None:
        extract_to = zip_path.parent / zip_path.stem
    else:
        extract_to = Path(extract_to)

    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        zip_ref.extractall(extract_to)

    # Format results
    result = f"Extracted {len(file_list)} files to {extract_to}/\n\n"
    result += "Contents:\n"
    for f in file_list[:20]:
        result += f"  - {f}\n"
    if len(file_list) > 20:
        result += f"  ... and {len(file_list) - 20} more files\n"

    return result


def list_files(path: str = ".") -> str:
    """List files and directories in the given path."""
    path = Path(path)

    if not path.exists():
        return f"Path not found: {path}"

    if not path.is_dir():
        return f"Not a directory: {path}"

    items = []
    for item in sorted(path.iterdir()):
        if item.name.startswith('.'):
            continue

        if item.is_dir():
            items.append(f"{item.name}/")
        else:
            items.append(f"{item.name}")

    # Sort directories first
    dirs = [i for i in items if i.endswith('/')]
    files = [i for i in items if not i.endswith('/')]

    result = f"Directory: {path}\n"
    for item in dirs + files:
        result += f"  {item}\n"

    return result


def read_file(file_path: str, start_line: int = 1, end_line: int = -1) -> str:
    """Read file content. Supports txt, py, json, md, csv, xlsx."""
    path = Path(file_path)

    if not path.exists():
        return f"File not found: {file_path}"

    ext = path.suffix.lower()

    if ext in TEXT_EXTENSIONS:
        return _read_text_file(file_path, start_line, end_line)
    elif ext == '.csv':
        return _read_csv(file_path)
    elif ext in SPREADSHEET_EXTENSIONS:
        return _read_excel(file_path)
    else:
        return _read_text_file(file_path, start_line, end_line)


def read_media_file(file_path: str, query: str) -> str:
    """Analyze an image, audio, or PDF file using LLM."""
    ext = Path(file_path).suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        return _analyze_image(file_path, query)
    elif ext in AUDIO_EXTENSIONS:
        return _analyze_audio(file_path, query)
    elif ext in PDF_EXTENSIONS:
        return _analyze_pdf(file_path, query)
    else:
        return f"Unsupported media format: {ext}"


def _read_text_file(file_path: str, start_line: int, end_line: int) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Adjust line numbers (1-indexed to 0-indexed)
    start_idx = max(0, start_line - 1)
    end_idx = len(lines) if end_line == -1 else min(end_line, len(lines))

    selected_lines = lines[start_idx:end_idx]

    result = []
    for i, line in enumerate(selected_lines, start=start_line):
        result.append(f"{i:4d} | {line.rstrip()}")
    return '\n'.join(result)


def _read_csv(file_path: str) -> str:
    import pandas as pd
    df = pd.read_csv(file_path)
    return df.to_markdown(index=False)


def _read_excel(file_path: str) -> str:
    import pandas as pd
    df = pd.read_excel(file_path)
    return df.to_markdown(index=False)


def _analyze_image(file_path: str, query: str) -> str:
    from openai import OpenAI

    with open(file_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    ext = Path(file_path).suffix.lower().lstrip('.')
    media_type = "image/jpeg" if ext == "jpg" else f"image/{ext}"

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {
                    "url": f"data:{media_type};base64,{image_data}"
                }}
            ]
        }]
    )
    return response.choices[0].message.content


def _analyze_audio(file_path: str, query: str) -> str:
    from openai import OpenAI

    with open(file_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode("utf-8")

    ext = Path(file_path).suffix.lower().lstrip('.')

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-audio",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "input_audio", "input_audio": {
                    "data": audio_data,
                    "format": ext
                }}
            ]
        }]
    )
    return response.choices[0].message.content


def _analyze_pdf(file_path: str, query: str) -> str:
    import fitz  # PyMuPDF
    from openai import OpenAI

    doc = fitz.open(file_path)

    # Extract text for context
    text_content = ""
    for page in doc:
        text_content += page.get_text()

    # Convert pages to images
    images = []
    for page in doc[:5]:  # First 5 pages
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes("png")
        images.append(base64.b64encode(img_bytes).decode('utf-8'))

    # Build content with text and images
    content = [{
        "type": "text",
        "text": f"{query}\n\nExtracted text:\n{text_content[:3000]}"
    }]

    for img_b64 in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        })

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": content}]
    )
    return response.choices[0].message.content
````


## 5.5 节：回调（callbacks）

`callbacks.py`：`approval_callback()`（Listing 5.26，危险工具执行前询问用户）和 `search_compressor()`（Listing 5.29–5.30，用第 5.3 节的向量检索把超长搜索结果压缩成 top-3 相关分块）。

为了让回调生效，Agent 需要新增构造参数 `before_tool_callbacks` / `after_tool_callbacks`，并在 `act()` 里执行它们（书 Listing 5.24–5.25）。所以本章要**整体替换** `agent.py` 为 ch05 版：

#### 📄 `scratch_agents/callbacks.py`

- **状态**：ch05（首次创建，之后不变）
- **对应书内容**：Listing 5.26（approval_callback）、5.29（search_compressor）、5.30（_extract_search_query）
- **行数**：64 行（清单为完整文件内容，从第 1 行到第 64 行）

<!-- FILE: scratch_agents/callbacks.py -->
<!-- STATE: ch05 -->

````python
"""Tool callbacks for the agent: approval and compression."""

from scratch_agents.rag import fixed_length_chunking, get_embeddings, vector_search
from scratch_agents.context import ExecutionContext
from scratch_agents.types import ToolCall, ToolResult, Message


DANGEROUS_TOOLS = ["delete_file", "send_email", "execute_sql"]


def approval_callback(context: ExecutionContext, tool_call: ToolCall):
    """Requests user approval before executing dangerous tools."""
    if tool_call.name not in DANGEROUS_TOOLS:
        return None

    print(f"\n⚠️ Dangerous tool execution requested")
    print(f"Tool: {tool_call.name}")
    print(f"Arguments: {tool_call.arguments}")

    response = input("Do you want to execute? (y/n): ").lower().strip()

    if response == 'y':
        print("✅ Approved. Executing...\n")
        return None
    else:
        print("❌ Denied. Skipping execution.\n")
        return f"User denied execution of {tool_call.name}"


def search_compressor(context: ExecutionContext, tool_result: ToolResult):
    """Callback that compresses web search results."""
    if tool_result.name != "search_web":
        return None

    original_content = tool_result.content[0]

    if len(original_content) < 2000:
        return None

    query = _extract_search_query(context, tool_result.tool_call_id)
    if not query:
        return None

    chunks = fixed_length_chunking(original_content, chunk_size=500, overlap=50)
    embeddings = get_embeddings(chunks)
    results = vector_search(query, chunks, embeddings, top_k=3)

    compressed = "\n\n".join([r['chunk'] for r in results])

    return ToolResult(
        tool_call_id=tool_result.tool_call_id,
        name=tool_result.name,
        status="success",
        content=[compressed]
    )


def _extract_search_query(context: ExecutionContext, tool_call_id: str) -> str:
    """Extract the original search query from context."""
    for event in context.events:
        for item in event.content:
            if isinstance(item, ToolCall) and item.name == "search_web" and item.tool_call_id == tool_call_id:
                return item.arguments.get("query", "")
    return ""
````


#### 📄 `scratch_agents/agent.py`

- **状态**：ch05（中间态，ch06 会被替换）
- **对应书内容**：Listing 5.24（回调参数加入 __init__）、5.25（act() 内执行回调）
- **行数**：285 行（清单为完整文件内容，从第 1 行到第 285 行）
- **说明**：相对 ch04 的改动（快照头注记）：__init__ 增加 before/after_tool_callbacks；act() 增加回调执行逻辑。其余与 ch04 相同。

<!-- FILE: scratch_agents/agent.py -->
<!-- STATE: ch05 -->

````python
"""Core Agent class for the scratch_agents framework."""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Type

from pydantic import BaseModel

from scratch_agents.llm import LlmClient, LlmRequest, LlmResponse
from scratch_agents.types import Event, Message, ToolCall, ToolResult
from scratch_agents.tools.base import BaseTool, FunctionTool, tool
from scratch_agents.tools.helpers import format_tool_definition
from scratch_agents.context import AgentResult, ExecutionContext

logger = logging.getLogger(__name__)


class Agent:
    """Tool-calling agent with ReAct loop and callbacks."""

    def __init__(
        self,
        model: LlmClient,
        tools: List[BaseTool] | None = None,
        instructions: str = "",
        max_steps: int = 10,
        name: str = "agent",
        description: str = "",
        output_type: Optional[Type[BaseModel]] = None,
        # NEW: CH05 callbacks
        before_tool_callbacks: list[Callable] | None = None,
        after_tool_callbacks: list[Callable] | None = None,
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
        """Execute the tools requested by the LLM, with callback support."""
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

            # NEW: before_tool callbacks
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

            # NEW: after_tool callbacks
            for cb in self.after_tool_callbacks:
                cb_result = cb(context, tool_result)
                if hasattr(cb_result, "__await__"):
                    cb_result = await cb_result
                if cb_result is not None:
                    tool_result = cb_result

            results.append(tool_result)

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


## 本章改动一览

| 文件 | 状态 | 行数 |
|---|---|---|
| `scratch_agents/rag.py` | ch05（首次创建，之后不变） | 47 |
| `scratch_agents/tools/file_tools.py` | ch05（首次创建，之后不变） | 222 |
| `scratch_agents/callbacks.py` | ch05（首次创建，之后不变） | 64 |
| `scratch_agents/agent.py` | ch05（中间态，ch06 会被替换） | 285 |

## 本章自测

```bash
python -c "from scratch_agents.rag import fixed_length_chunking; print(fixed_length_chunking('abcdef', 3, 1)); from scratch_agents.callbacks import approval_callback; print('ch05 ok')"
```
