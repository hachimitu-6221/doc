# 从零构建《Build an AI Agent from Scratch》的 `scratch_agents` 包

这份文档集把 Manning 出版的 *Build an AI Agent (From Scratch)*（Hur & Song）按章节顺序
翻译成「逐步构建指南」：每一章告诉你**写什么代码、写到哪个文件、本章结束后该文件的完整内容是什么**，
照做下来得到的 `scratch_agents/` 包与本仓库最终版本逐字节一致，可以通过仓库自带的全部测试。

## 与书章节的对应关系

| 文档 | 书章节 | 主题 | 本章新增/替换的文件 |
|---|---|---|---|
| [ch02](ch02-llm-api-basics.md) | 第 2 章 | LLM API 基础 | `eval/gaia.py`、`eval/__init__.py` |
| [ch03](ch03-tool-use.md) | 第 3 章 | 工具使用 | `tools/helpers.py`、`tools/search.py`、`tools/calculator.py` |
| [ch04](ch04-react-agent.md) | 第 4 章 | 基础 ReAct Agent | `types.py`、`context.py`、`tools/base.py`、`tools/mcp.py`、`llm.py`、`agent.py`、`utils.py`、两个 `__init__.py` |
| [ch05](ch05-rag-and-file-tools.md) | 第 5 章 | RAG / 文件工具 / 回调 | `rag.py`、`tools/file_tools.py`、`callbacks.py`、替换 `agent.py` |
| [ch06](ch06-memory-systems.md) | 第 6 章 | 记忆系统 | `memory/` 子包、`tools/memory_tool.py`、替换 `context.py`、`tools/base.py`、`agent.py` |
| [ch07](ch07-planning-and-reflection.md) | 第 7 章 | 规划与反思 | `planning.py` |
| [ch08](ch08-code-execution.md) | 第 8 章 | 代码执行 / 沙箱 / skills | `tools/code_execution.py`、`skills.py`、替换 `tools/base.py`、`context.py`、`agent.py` |
| [ch09](ch09-multi-agent-systems.md) | 第 9 章 | 多 Agent 编排 | `workflows/` 子包、`tools/agent_tool.py`、`transfer.py`、`remote.py`、`a2a_server.py`、替换 `context.py`、`agent.py` |
| [ch10](ch10-evaluation.md) | 第 10 章 | 评估 | `eval/prompts.py` |

书第 1 章是概念章，没有代码。

## 每个代码清单怎么读

每个文件块长这样：

    #### 📄 `scratch_agents/agent.py`
    - **状态**：ch04（中间态，ch05 会被替换）
    - **对应书内容**：Listing 4.15–4.23 …
    - **行数**：259 行（清单为完整文件内容，从第 1 行到第 259 行）
    <!-- FILE: scratch_agents/agent.py -->
    ```python
    （完整文件）
    ```

规则只有三条：

1. **清单永远是完整文件**。把代码块整体写入对应路径、整体覆盖旧内容即可，不需要手工合并 diff。
2. **同一文件多次出现时，后出现的章节胜出**。例如 `agent.py` 在 ch04–ch09 共出现 5 次，
   每次都用本章清单整体覆盖——这正是「crafting a compiler」式的增量构建。
3. 文档中 `<!-- FILE: ... -->` 标记仅供脚本识别，照抄代码时忽略它。

## 两条使用路径

**路径 A：学习**。按 ch02 → ch10 的顺序读文档、敲代码，每章末尾有自测命令
（大多是 `python -c "import ..."` 级别的冒烟检查）。想对照书时，文档里标了每段代码对应的
书小节号和 Listing 号（如 Listing 4.11）。

**路径 B：直接拼装**。仓库附带 [assemble.py](assemble.py)，可以从文档中机械地提取每个文件的
最终清单并拼装成完整的包：

```bash
python docs/assemble.py --list     # 查看每个文件取自哪一章
python docs/assemble.py --out /tmp/assembled   # 拼成一个目录
python docs/assemble.py --check    # 与仓库 ./scratch_agents 逐字节比对
```

## 验证：跑测试

按文档拼出的包（或仓库自带的包）这样验证（需要 Python ≥ 3.13，推荐用 `uv`）：

```bash
uv sync --locked --extra test
uv run --extra test pytest -q
```

测试分为两组，都使用模拟的外部服务（不需要任何 API key）：

- `tests/test_notebooks.py`：把 `notebooks/ch02`–`ch10` 的 9 个 notebook 在干净的
  Jupyter kernel 里逐格执行（LLM 响应、E2B 沙箱、MCP、Tavily、HuggingFace 下载全部被
  `tests/notebook_services.py` 里的离线替身替代），并按章节追加断言；
  其中 ch08 还会被参数化地注入「损坏的工具」验证错误可见性。
- `tests/test_regressions.py`：直接 import `scratch_agents` 做单元/回归测试
  （并行工作流的上下文隔离、LLM 错误不伪装成成功、沙箱清理与 skill 上传、
  计划工具接受 JSON 参数、分块器拒绝不前进的窗口、HITL 审批防注入等），
  其中部分用例会同时执行 notebook 里的对应单元格，保证「书上的代码」和「包里的代码」行为一致。

两组测试全绿即说明：你照着文档（或脚本拼装）得到的包与仓库最终状态一致，且与书中的教学代码语义一致。

## 目录结构（构建完成后的包）

```
scratch_agents/
├── __init__.py          # 包导出（ch04 建，ch06 更新）
├── types.py             # Message/ToolCall/ToolResult/Event（ch04）
├── context.py           # ExecutionContext/AgentResult/HITL 类型（ch04→ch06→ch08→ch09 逐章演进）
├── llm.py               # LlmRequest/LlmResponse/LlmClient（ch04）
├── agent.py             # ReAct 主循环（ch04→ch05→ch06→ch08→ch09 逐章演进，共 5 版）
├── utils.py             # display_trace（ch04）
├── rag.py               # embedding/chunking/vector search（ch05）
├── callbacks.py         # approval_callback/search_compressor（ch05）
├── planning.py          # Task/create_tasks/reflection（ch07）
├── skills.py            # SKILL.md 技能发现（ch08）
├── transfer.py          # transfer_to_agent 工具（ch09）
├── remote.py            # A2A 客户端（ch09）
├── a2a_server.py        # A2A server 端适配（ch09）
├── tools/
│   ├── __init__.py      # ch03 建，ch04/ch09 更新
│   ├── helpers.py       # 签名→JSON Schema 转换（ch03）
│   ├── base.py          # BaseTool/FunctionTool/@tool（ch04→ch06→ch08 逐章演进）
│   ├── search.py        # Tavily 搜索（ch03）
│   ├── calculator.py    # 计算器（ch03）
│   ├── mcp.py           # MCP 工具加载（ch04）
│   ├── file_tools.py    # 文件工具（ch05）
│   ├── code_execution.py# execute_python/bash_tool/upload_file（ch08）
│   ├── memory_tool.py   # recall_memory（ch06）
│   └── agent_tool.py    # AgentTool（ch09）
├── memory/
│   ├── session.py       # Session/SessionManager（ch06）
│   ├── context_optimizer.py  # 滑动窗口/压缩/摘要（ch06）
│   └── long_term.py     # ChromaDB 长期记忆（ch06）
├── workflows/
│   ├── sequential.py    # 顺序工作流（ch09）
│   ├── parallel.py      # 并行工作流（ch09）
│   └── loop.py          # 循环工作流（ch09）
└── eval/
    ├── gaia.py          # GAIA 基准评测（ch02，含 ch04 agent 版评测）
    └── prompts.py       # LLM-as-a-judge 评审 prompt（ch10）
```

## 提示

- 书中的大量实验代码（API 调用示例、GAIA 下载、A2A 演示等）在 `notebooks/chXX/*.ipynb` 里，
  不在 `scratch_agents` 包内；本文档只负责构建包。读书时 notebook 与文档对照效果最好。
- `notebooks/ch04`–`ch09` 目录下的 `agent.py`/`base.py`/`context.py` 快照文件，就是文档中
  各章中间态清单的来源（文档里已把快照头的说明性 docstring 换成了正式模块 docstring）。
