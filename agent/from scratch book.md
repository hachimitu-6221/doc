## ch_06
### memory
- `count_tokens()`：普通消息计算 `content` 的 token 数并加上每条消息默认的 4 个 token；工具调用额外计算工具名称和参数的 token 数；工具定义计算其 JSON 表示的 token 数；最后返回整个 `LlmRequest` 的总 token 数。
- `apply_sliding_window()`：找到第一条 `user` 消息，保留它之前的初始上下文（通常包括 `system` 消息）以及这条 `user` 消息；对后续消息只保留最近的 `window_size` 条，直接删除中间的旧消息。如果截断位置导致 `ToolResult` 被保留但对应的 `ToolCall` 被删除，则向前调整边界，确保工具调用和工具结果成对保留。该函数按消息条数截断，不按 token 数计算，也不会生成摘要。
### tool