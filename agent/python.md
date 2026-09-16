```python
@tool
def add(a, b):
    return a + b
```
等价于：
```python
def add(a, b):
    return a + b
add = tool(add)
```