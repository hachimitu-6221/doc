**LLM 工具调用的执行循环（ReAct Loop）**
工具调用本质上是模型与 harness 之间基于约定文本格式的协作协议，完整流程如下：
1. **请求构建**：harness 将工具定义（名称、描述、参数的 JSON Schema）与对话历史（含用户输入）一同提交给推理服务；chat template 将其序列化为单条文本序列，作为模型的实际输入。模型仅通过 schema 文本获知可用工具，无法访问工具的实现代码。
2. **调用生成**：模型基于上下文，以普通文本生成的形式输出约定格式（如 `<tool_call>` 块）的调用请求，包含工具名称与参数值。每个调用被分配唯一 call ID。
3. **解析与执行**：harness 解析模型输出，识别调用请求后依次执行：查注册表定位执行器 → 按 schema 校验参数 → 权限检查 → 调用真实函数/外部服务。
4. **结果回注**：执行结果封装为 `tool` 角色的消息（通过 `tool_call_id` 与对应调用匹配），追加至对话历史。
5. **循环迭代**：更新后的历史再次输入模型；模型可基于结果发起后续调用，直至其输出不含工具调用的最终答复（或显式调用终止工具），循环结束。
- 关键认知：模型在整个流程中是被动的文本生成函数，所有主动性（请求发起、执行决策、循环控制）均在 harness 侧；模型对工具的一切"意图"只能通过生成约定格式的文本来表达。
```text
模型生成文本 → harness（外部软件）读到这段文本
 → 解析：这是合法 JSON 吗？工具名存在吗？
 → 校验：batch_size 是整数吗？数值在允许范围内吗？
 → 决定：这个操作有权限吗？要不要拒绝？
 → 通过了才执行，结果再喂回给模型
```

| 层次         | 问题     | 典型失败      |
| ---------- | ------ | --------- |
| Selection  | 选对了工具吗 | 漏调/多调     |
| Arguments  | 参数值对吗  | 字段/值错误    |
| Trajectory | 顺序对吗   | 依赖关系错     |
| Task       | 目标达成了吗 | 看似合理的错误答案 |

- prompt be like, 即为`list[dict[str, Any]]`
```
[
    # 1. 常驻指令（你是谁、规则、环境信息、技能目录）—— TODO 1.1.b 构造
    {"role": "system", "content": "You are a coding agent... <system_information>{...}</system_information>..."},

    # 2. 任务说明（这次要干什么）—— 也是 1.1.b
    {"role": "user", "content "print hello, world to the terminal"},

    # 3. 第 1 步：模型自己说的话（原样保留，含 tool_calls 字段）
    {"role": "assistant", "content": "Calling execute.",
     "_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "execute", "arguments": '{"command": "ls -la"}'}}]},

    # 4. 第 1 步的工具结果，role 是 tool，用 tool_call_id 指回上面那个调用
    {"role": "tool", "tool_call_id": "call_1",
     "content": "<output>total 4\ndrwxr-xr-x 2 root root ...output>\n<returncode>0</returncode>"},

    # 5. 第 2 步的 assistant 消息
    {"role": "assistant", "content": "Calling execute.",
     "tool_calls": [{"id": "call2", ..., "function": {"name": "execute", "arguments": '{"command": "cat <<EOF..."}'}}]},

    # 6. 第 2 步的工具结果
    {"role": "tool", "tool_call_id": "call_2", "content": "<output></output>\n<returncode>0</returncode>"},
]
```
