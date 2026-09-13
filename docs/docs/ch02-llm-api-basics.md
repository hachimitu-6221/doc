# 第 2 章 · LLM——Agent 的大脑

> 对应原书 第 2 章（小节号在各代码块中标注）。本章文档：`docs/ch02-llm-api-basics.md`。

## 本章构建目标

书第 2 章解决「用什么模型、怎么调 API」的问题：2.1 节讨论模型选择，2.2 节讲 OpenAI / Anthropic / LiteLLM 三种调用方式、无状态 API 的对话管理、结构化输出和异步调用，2.3 节讲 prompt 工程，2.4 节用 GAIA 基准说明「光有 LLM 不够，需要工具」。

本章**不产生 agent 框架的代码文件**。书中的实验代码（OpenAI 调用、GAIA 评测脚本）写在 notebook（`notebooks/ch02/ch02_llm_api_basics.ipynb`）里，对照书阅读、逐格运行即可。

但有一件事现在就要做：GAIA 评测脚本在书里是第 2 章的实验代码（Listing 2.10–2.17），在最终项目里它被模块化为 `scratch_agents/eval/gaia.py`。本章我们把它建好——注意 `evaluate_gaia_agent_single()` 在函数体内部才 `import Agent`（延迟导入），所以第 2 章阶段这个文件就能正常加载，Agent 要到第 4 章才存在。

### 动手之前：环境准备

- 创建项目根目录的 `pyproject.toml`，声明依赖（完整清单见书 2.2.1 节；本仓库的 `pyproject.toml` 已包含全部依赖，含 `openai`、`anthropic`、`litellm`、`pydantic`、`python-dotenv`、`tavily-python`、`mcp`、`fastmcp`、`chromadb`、`e2b-code-interpreter`、`scikit-learn`、`tiktoken`、`datasets` 等）。
- 创建 `.env` 文件（对照仓库 `.env.example`）：`OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`TAVILY_API_KEY`、`HF_TOKEN`、`E2B_API_KEY`。书中的 `.env` 加载即 Listing 2.1。

## 2.4 节：GAIA 评测脚本（唯一要建的包文件）

书 Listing 2.11–2.17 给出 GAIA 评测代码：`GaiaOutput` 结构化输出模型、按 provider 限流的信号量、`solve_problem()` 单题求解、`evaluate_gaia_single()` 单题评测、`run_experiment()` 批量实验。最终项目里这些被整理进 `scratch_agents/eval/gaia.py`，并追加了一个 `evaluate_gaia_agent_single()`（书 Listing 4.31 的 agent 版评测，函数内延迟导入 Agent）。

#### 📄 `scratch_agents/eval/__init__.py`

- **状态**：ch02（首次创建，之后不变）
- **对应书内容**：无（包导出约定）
- **行数**：1 行（清单为完整文件内容，从第 1 行到第 1 行）
- **说明**：从 `scratch_agents.eval` 直接导出 GAIA 评测函数。

<!-- FILE: scratch_agents/eval/__init__.py -->
<!-- STATE: ch02 -->

````python
from scratch_agents.eval.gaia import GaiaOutput, is_correct, solve_problem, evaluate_gaia_single, run_experiment
````


#### 📄 `scratch_agents/eval/gaia.py`

- **状态**：ch02（首次创建，之后不变）
- **对应书内容**：Listing 2.11（GaiaOutput）、2.12（PROVIDER_SEMAPHORES）、2.14（is_correct）、2.15（evaluate_gaia_single）、2.16（run_experiment）；`evaluate_gaia_agent_single` 对应 Listing 4.31
- **行数**：164 行（清单为完整文件内容，从第 1 行到第 164 行）
- **说明**：第 2 章阶段只用到其中不依赖 Agent 的函数；`evaluate_gaia_agent_single` 在函数体内才 `from scratch_agents.agent import Agent`，所以本文件在 ch02 即可被 import。

<!-- FILE: scratch_agents/eval/gaia.py -->
<!-- STATE: ch02 -->

````python
import asyncio
from pydantic import BaseModel
from litellm import acompletion
from tqdm.asyncio import tqdm_asyncio

# Pydantic model for GAIA responses
class GaiaOutput(BaseModel):
    is_solvable: bool
    unsolvable_reason: str = ""
    final_answer: str = ""

# System prompt for GAIA evaluation
gaia_prompt = """You are a general AI assistant. I will ask you a question.
First, determine if you can solve this problem with your current capabilities and set "is_solvable" accordingly.
If you can solve it, set "is_solvable" to true and provide your answer in "final_answer".
If you cannot solve it, set "is_solvable" to false and explain why in "unsolvable_reason".
Your final answer should be a number OR as few words as possible OR a comma-separated list of numbers and/or strings.
If you are asked for a number, don't use a comma to write your number neither use units such as $ or percent sign unless specified otherwise.
If you are asked for a string, don't use articles, neither abbreviations (e.g., for cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma-separated list, apply the above rules depending on whether the element is a number or a string."""

# Provider-specific rate limiting
PROVIDER_SEMAPHORES = {
    "openai": asyncio.Semaphore(30),
    "anthropic": asyncio.Semaphore(10),
}

def get_provider(model: str) -> str:
    """Extract provider name from model string."""
    return "anthropic" if model.startswith("anthropic/") else "openai"

def is_correct(prediction: str | None, answer: str) -> bool:
    """Check exact match between prediction and answer (case-insensitive)."""
    if prediction is None:
        return False
    return prediction.strip().lower() == answer.strip().lower()

async def solve_problem(model: str, question: str) -> GaiaOutput:
    """Solve a single problem and return structured output."""
    provider = get_provider(model)

    async with PROVIDER_SEMAPHORES[provider]:
        response = await acompletion(
            model=model,
            messages=[
                {"role": "system", "content": gaia_prompt},
                {"role": "user", "content": question},
            ],
            response_format=GaiaOutput,
            num_retries=2,
        )
        finish_reason = response.choices[0].finish_reason
        content = response.choices[0].message.content

        if finish_reason == "refusal" or content is None:
            return GaiaOutput(
                is_solvable=False,
                unsolvable_reason=f"Model refused to answer (finish_reason: {finish_reason})",
                final_answer=""
            )
        return GaiaOutput.model_validate_json(content)

async def evaluate_gaia_single(problem: dict, model: str) -> dict:
    """Evaluate a single problem-model pair and return result."""
    try:
        output = await solve_problem(model, problem["Question"])
        return {
            "task_id": problem["task_id"],
            "model": model,
            "correct": is_correct(output.final_answer, problem["Final answer"]),
            "is_solvable": output.is_solvable,
            "prediction": output.final_answer,
            "answer": problem["Final answer"],
            "unsolvable_reason": output.unsolvable_reason,
        }
    except Exception as e:
        return {
            "task_id": problem["task_id"],
            "model": model,
            "correct": False,
            "is_solvable": None,
            "prediction": None,
            "answer": problem["Final answer"],
            "error": str(e),
        }


async def evaluate_gaia_agent_single(problem: dict, model: str, tools: list) -> dict:
    """Evaluate one problem using the full Agent loop and optional tools."""
    from scratch_agents.agent import Agent
    from scratch_agents.llm import LlmClient

    try:
        agent = Agent(
            model=LlmClient(model=model),
            tools=tools,
            instructions=gaia_prompt,
            output_type=GaiaOutput,
            max_steps=15,
        )
        result = await agent.run(problem["Question"])
        output = result.output
        if output is None:
            return {
                "task_id": problem["task_id"],
                "model": model,
                "correct": False,
                "is_solvable": None,
                "prediction": None,
                "answer": problem["Final answer"],
                "error": "Agent did not return a final GaiaOutput",
                "steps": result.context.current_step,
            }
        if not isinstance(output, GaiaOutput):
            output = GaiaOutput.model_validate(output)

        return {
            "task_id": problem["task_id"],
            "model": model,
            "correct": is_correct(output.final_answer, problem["Final answer"]),
            "is_solvable": output.is_solvable,
            "prediction": output.final_answer,
            "answer": problem["Final answer"],
            "unsolvable_reason": output.unsolvable_reason,
            "steps": result.context.current_step,
        }
    except Exception as e:
        return {
            "task_id": problem["task_id"],
            "model": model,
            "correct": False,
            "is_solvable": None,
            "prediction": None,
            "answer": problem["Final answer"],
            "error": str(e),
        }

async def run_experiment(
    problems: list[dict],
    models: list[str],
    tools: list | None = None,
) -> dict[str, list]:
    """Evaluate all models on all problems."""
    if tools is None:
        tasks = [
            evaluate_gaia_single(problem, model)
            for problem in problems
            for model in models
        ]
    else:
        tasks = [
            evaluate_gaia_agent_single(problem, model, tools)
            for problem in problems
            for model in models
        ]

    all_results = await tqdm_asyncio.gather(*tasks)

    # Group results by model
    results = {model: [] for model in models}
    for result in all_results:
        results[result["model"]].append(result)

    return results
````


## 本章改动一览

| 文件 | 状态 | 行数 |
|---|---|---|
| `scratch_agents/eval/__init__.py` | ch02（首次创建，之后不变） | 1 |
| `scratch_agents/eval/gaia.py` | ch02（首次创建，之后不变） | 164 |

## 本章自测

```bash
python -c "from scratch_agents.eval.gaia import GaiaOutput, run_experiment; print('ch02 ok')"
```
