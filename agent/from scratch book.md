## ch_06
### memory
- count_tokens(): 普通message就算content的token再＋4; 对于tool就是tool的名字, tool的参数, tool的定义加上默认的4, 最后返回llm_request的总token数
- apply_sliding_window()
### tool