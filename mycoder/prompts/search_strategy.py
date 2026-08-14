"""Search-strategy system-prompt segment (Phase 2 agentic search).

Pure tool-driven search: zero index, zero vector DB, zero embedding. The agent
navigates the codebase with grep_search / list_files / read_file alone, and
this segment tells it how to do that cheaply and correctly.
"""

SEARCH_STRATEGY_PROMPT = """
## 代码探索与搜索策略（硬性指引）

### 搜索优先级
1. **有明确的函数名/类名/报错信息** → `grep_search`（精确匹配，加 file_types 过滤）
2. **想找特定文件**（如配置文件、测试文件） → `list_files`（glob 模式）
3. **找到候选文件后** → 先用 `read_file` 读文件头部（前 50 行），了解 imports 和结构
4. **搜索结果太多** → 缩小 `path` 范围，或改用更具体的正则
5. **结果为零** → 检查拼写，尝试去掉参数或缩短关键词

### 示例工作流（查找 authenticate_user 的实现与调用方）
1. `grep_search(pattern="def authenticate_user", file_types="py")`
   → 发现 `src/auth/handler.py L23`
2. `read_file(file_path="src/auth/handler.py", start_line=1, end_line=50)`
   → 看到实现及其 imports
3. `grep_search(pattern="authenticate_user\\(", file_types="py")`
   → 找到所有调用方

### 硬性约束（禁止行为）
- ❌ 不要猜测文件路径，用 `list_files` 确认
- ❌ 不要一次 grep 整个项目搜常见词（如 "data"）——先用 `list_files` 缩小范围
- ❌ 不要连续读超过 3 个完整大文件（>300 行）——先读头尾筛选

### Token 预算量化指引
- 单次 `read_file` 不超过 300 行
- 一轮探索累计读取不超过 3 个文件（先筛选再精读）
- 单次 `grep_search` 结果超过 50 条时自动截断
"""
