# 第 10 章 · 评估 Agent

> 对应原书 第 10 章（小节号在各代码块中标注）。本章文档：`docs/ch10-evaluation.md`。

## 本章构建目标

第 10 章讲评估：10.1 节是可观测性（指标/追踪/日志、OpenTelemetry），10.2 节是数据集与评估标准，10.3 节是 LLM-as-a-judge 评分系统，10.4 节是 CI/CD 与质量飞轮。

本章在书中**没有编号 Listing**（概念为主），对包的改动只有一处：新增 `eval/prompts.py`，存放 LLM-as-a-judge 的评审 prompt（答案相关性、引用可靠性、需求符合度）与评审系统提示。GAIA 基准的代码在第 2 章已经建好（`eval/gaia.py`）。

## 10.3 节：LLM-as-a-judge 的评审 prompts

`eval/prompts.py`：`ANSWER_RELEVANCY_PROMPT`、`CITATION_RELIABILITY_PROMPT`、`REQUIREMENT_COMPLIANCE_PROMPT` 三个评分模板和 `EVALUATION_SYSTEM_PROMPT`。书中它们以正文代码块形式出现在 10.2–10.3 节（无 Listing 编号）。

至此全部 10 章结束，`scratch_agents` 包与仓库最终状态一致——用 README 里的验证步骤跑一遍完整测试。

#### 📄 `scratch_agents/eval/prompts.py`

- **状态**：ch10（首次创建，最终版）
- **对应书内容**：书 10.2–10.3 节（评审 prompt，无编号 Listing）
- **行数**：51 行（清单为完整文件内容，从第 1 行到第 51 行）

<!-- FILE: scratch_agents/eval/prompts.py -->
<!-- STATE: ch10 -->

````python
"""Evaluation prompts and schemas for CH10."""

# Overall Pass Rate calculation
# OPR = (# all-pass) / (total evaluations)

ANSWER_RELEVANCY_PROMPT = """Evaluate whether the agent's answer is relevant to the question asked.

Question: {question}
Agent's Answer: {answer}

Rate the relevancy on a scale of 1-5:
1 - Completely irrelevant
2 - Slightly relevant
3 - Somewhat relevant
4 - Mostly relevant
5 - Highly relevant

Provide your rating and a brief explanation."""


CITATION_RELIABILITY_PROMPT = """Evaluate whether the agent's citations and sources are reliable.

Question: {question}
Agent's Answer: {answer}
Sources Used: {sources}

Check:
1. Are the sources real and accessible?
2. Do the sources support the claims made?
3. Are the sources authoritative for the topic?

Rate reliability on a scale of 1-5 and explain."""


REQUIREMENT_COMPLIANCE_PROMPT = """Evaluate whether the agent's answer complies with the requirements.

Original Question: {question}
Requirements: {requirements}
Agent's Answer: {answer}

Check if the answer:
1. Addresses all parts of the question
2. Follows the specified format
3. Meets any constraints mentioned

Rate compliance on a scale of 1-5 and explain."""


EVALUATION_SYSTEM_PROMPT = """You are an evaluation judge for AI agent responses.
Provide fair, consistent, and detailed evaluations.
Always explain your reasoning."""
````


## 本章改动一览

| 文件 | 状态 | 行数 |
|---|---|---|
| `scratch_agents/eval/prompts.py` | ch10（首次创建，最终版） | 51 |

## 本章自测

```bash
python -c "from scratch_agents.eval.prompts import ANSWER_RELEVANCY_PROMPT; print('ch10 ok')"
```
