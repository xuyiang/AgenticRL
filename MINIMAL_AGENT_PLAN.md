# 最小 ReAct Agent 实现计划

## 1. 目标

不依赖 LangChain，用 Python 标准库和一个可替换的 LLM 接口，实现一个约 100–300 行核心代码的最小 ReAct Agent。

首版需要完整展示以下闭环：

```text
Task / Observation
        ↓
LLM: Thought + Action
        ↓
Action parser / validator
        ↓
Tool execution
        ↓
Environment: Observation
        ↓
History update
        ↓
继续循环或提交 Final Answer
        ↓
Terminal verifier → terminal_reward
```

核心代码应当足够小，方便后续直接改造成 trajectory collection、SFT、offline RL 或在线 Agent RL 环境。

## 2. 非目标

- 首版不使用 LangChain、复杂工作流框架或多 Agent。
- 不做并行工具调用、流式输出、GUI 和向量数据库。
- 不把 Python 工具当作安全沙箱；它只用于本地可信实验。
- 不在首版实现 Tree of Thoughts 搜索，只保留可扩展的数据结构。
- 不更新模型参数；Reflexion memory 只是文本记忆。

## 3. 必读材料与阅读重点

### ReAct: Synergizing Reasoning and Acting

重点理解 Agent 的基本交互语法：

```text
Thought → Action → Observation → Thought → ... → Final
```

需要回答：

- 推理 token 和环境反馈如何交替出现？
- 工具结果为什么必须作为新的 observation 返回模型？
- trajectory 中哪些内容由 policy 生成，哪些内容来自 environment？

按照 ReAct 原论文的记号，在时刻 \(t\)，Agent 接收环境给出的
observation \(o_t \in \mathcal{O}\)，并根据当前 context \(c_t\)
选择 action：

$$
a_t \sim \pi_\theta(a_t \mid c_t)
$$

其中 context 是此前 action 与 observation 的完整序列：

$$
c_t=(o_1,a_1,\ldots,o_{t-1},a_{t-1},o_t)
$$

ReAct 将原本的环境 action space \(\mathcal{A}\) 扩展为：

$$
\hat{\mathcal{A}}=\mathcal{A}\cup\mathcal{L}
$$

其中 \(\mathcal{L}\) 是 language space。当
\(\hat{a}_t\in\mathcal{L}\) 时，它是 Thought 或 reasoning trace。
Thought 由 policy 生成，但不作用于外部环境，因此不会触发新的
observation，只更新 context：

$$
c_{t+1}=(c_t,\hat{a}_t),
\qquad \hat{a}_t\in\mathcal{L}
$$

当 \(a_t=(n_t,x_t)\in\mathcal{A}\) 时，它表示调用工具 \(n_t\)
及其参数 \(x_t\)。原始工具结果定义为：

$$
v_t=\mathcal{T}_{n_t}(x_t)
$$

环境将原始工具结果、错误或超时信息转换为下一条 observation：

$$
o_{t+1}=\mathrm{Serialize}
\left(v_t,\mathrm{status}_t,\mathrm{metadata}_t\right)
$$

执行环境 action 后，context 更新为：

$$
c_{t+1}=(c_t,a_t,o_{t+1}),
\qquad a_t\in\mathcal{A}
$$

因此，Thought 和工具 Action 都来自 policy；工具结果与
Observation 来自 environment。工程实现可以在同一次模型输出中
序列化 Thought 和 Action，但不能因此把 Thought 标记为 environment token。
  

### Reflexion

重点理解：

- 一次 trajectory 失败后，模型生成语言形式的 reflection。
- reflection 写入 memory，在下一次尝试中作为额外上下文。
- Reflexion 不进行梯度更新，也不修改模型参数。

首版只设计简单的 `memory.jsonl`：

```json
{"task_id":"task-001","failure":"答案错误","reflection":"下次先检查单位并调用计算器"}
```

### Toolformer

重点理解模型如何学习：

- 什么时候调用 API。
- 选择哪个 API。
- 如何生成合法参数。
- 工具返回结果如何影响后续 token。

在本项目中对应 action schema、参数验证和 tool result 注入。

### Tree of Thoughts

重点理解：

- branch：从一个状态采样多个候选 thought/action。
- evaluate：评价候选状态。
- backtrack：放弃低价值分支并回退。

首版只运行单一路径，但 trajectory 中需要稳定保存 `state`、`turn` 和奖励，方便后续把 loop 替换成树搜索。

## 4. 建议文件结构

```text
AgenticRL/
├── MINIMAL_AGENT_PLAN.md
├── minimal_agent.py       # 100–300 行核心 ReAct loop、工具与日志
├── tasks/
│   └── example.json       # 示例任务和 verifier 配置
├── runs/                  # 每次运行生成 trajectory JSON
├── memory.jsonl           # 失败后的语言反思
└── tests/
    └── test_agent.py      # parser、工具、超时、verifier 测试
```

为了保持可读性，第一版将核心逻辑放在一个 `minimal_agent.py` 中。测试和示例数据不计入 100–300 行限制。

## 5. LLM 输出协议

模型每轮只能输出一个 JSON object，不允许一轮调用多个工具。

调用工具：

```json
{
  "thought": "需要计算目录中 Python 文件的数量。",
  "action": {
    "name": "search_files",
    "arguments": {"pattern": "*.py"}
  }
}
```

提交答案：

```json
{
  "thought": "已经获得并检查了结果。",
  "action": {
    "name": "final",
    "arguments": {"answer": "共有 4 个 Python 文件。"}
  }
}
```

约束：

- `thought` 必须是字符串。
- `action.name` 必须属于工具白名单或等于 `final`。
- `action.arguments` 必须是 JSON object。
- parser 不从自由文本中猜测 JSON；解析失败就是 invalid action。
- 日志保留模型原始输出和解析后的 action，避免训练数据丢失。

## 6. Message 与 history 管理

内部使用轻量 message 结构：

```python
{"role": "system", "content": "..."}
{"role": "user", "content": "task ..."}
{"role": "assistant", "content": "{thought/action JSON}"}
{"role": "tool", "name": "calculator", "content": "42"}
```

每轮流程：

1. system prompt 声明协议、工具 schema、约束和 memory。
2. user message 提供 task 和初始 observation。
3. 模型生成 assistant message。
4. parser 提取 `thought` 和 `action`。
5. 环境执行 action，生成 tool observation。
6. assistant 原始输出和 tool observation 都追加到 history。
7. 达到 `final` 或最大轮数后停止。

history 不做隐式摘要。首版保持完整历史，便于检查 trajectory 与 token 边界。

## 7. 三个工具

### `calculator(expression: str)`

- 使用 `ast.parse(..., mode="eval")`。
- 只允许数字、四则运算、幂、取模和括号。
- 不使用 Python `eval`。
- 限制表达式长度、AST 节点数量和指数大小。

### `python(code: str)`

- 用独立 subprocess 执行 `python -I -c <code>`。
- 捕获 stdout、stderr 和退出码。
- 设置 2–5 秒 timeout。
- 限制最大输出长度。
- 明确说明：subprocess timeout 不是安全沙箱，不能运行不可信代码。

### `search_files(pattern: str, text: str | None = None)`

- 搜索范围固定为 workspace root。
- 用 `pathlib.Path.rglob` 查找文件名。
- 可选地在小型文本文件内搜索字符串。
- 拒绝逃逸 workspace 的路径。
- 限制匹配文件数、单文件大小和总输出长度。

统一工具返回结构：

```python
ToolResult(
    ok=True,
    output="...",
    error=None,
    elapsed_ms=12,
)
```

## 8. 异常与停止条件

### Tool error

- 捕获工具异常，不让主 loop 崩溃。
- observation 明确返回 `TOOL_ERROR`、错误类型和简短信息。
- 错误结果写入 trajectory，允许模型下一轮修正。

### Timeout

- Python 工具和 terminal verifier 使用 subprocess timeout。
- timeout 转换为普通 observation：`TOOL_TIMEOUT`。
- 超时进程必须被终止并回收。

### Invalid action

以下情况都视为 invalid action：

- JSON 无法解析。
- 缺少必要字段。
- 工具名不存在。
- arguments 类型或字段错误。
- 模型在 `final` 中没有提供 answer。

invalid action 不立即终止，而是返回格式错误 observation。连续无效 action 达到阈值后提前停止。

### 最大交互轮数

- 默认 `max_turns=8`。
- 每次 LLM completion 计为一轮。
- 达到上限后状态记为 `max_turns_exceeded`。
- 即使没有 final answer，也必须落盘完整 trajectory，并由 verifier 给出失败奖励。

## 9. Terminal success verifier

verifier 是环境的一部分，不是模型工具。模型不能直接控制 reward。

首版支持两种 verifier：

1. `exact_match`：标准化 final answer 后与 expected answer 比较。
2. `command`：运行预先配置的固定命令，例如 `pytest -q`，检查退出码。

安全约束：

- command 来自任务配置，不接受模型生成的命令。
- 设置 timeout 并捕获 stdout、stderr。
- verifier 结果在 episode 结束后写入日志。

返回结构：

```json
{
  "success": true,
  "reward": 1.0,
  "feedback": "exact match",
  "elapsed_ms": 4
}
```

默认 reward：

- 成功：`1.0`
- 失败、超时、无 final 或超出轮数：`0.0`

## 10. 完整 trajectory logging

每次 episode 保存为 `runs/<task_id>-<timestamp>.json`：

```json
{
  "task_id": "task-001",
  "task": "计算 17 * 23",
  "initial_observation": "可使用 calculator、python、search_files",
  "model": "configured-model",
  "started_at": "ISO-8601 timestamp",
  "turns": [
    {
      "turn": 1,
      "observation": "计算任务及可用工具",
      "reasoning": "应该调用计算器避免心算错误。",
      "raw_model_output": "{\"thought\":\"...\",\"action\":{...}}",
      "action": {
        "name": "calculator",
        "arguments": {"expression": "17 * 23"}
      },
      "tool_result": {
        "ok": true,
        "output": "391",
        "error": null,
        "elapsed_ms": 1
      }
    },
    {
      "turn": 2,
      "observation": "391",
      "reasoning": "结果已经得到，可以提交。",
      "raw_model_output": "{\"thought\":\"...\",\"action\":{...}}",
      "action": {
        "name": "final",
        "arguments": {"answer": "391"}
      },
      "tool_result": null
    }
  ],
  "final_answer": "391",
  "stop_reason": "final",
  "verifier": {
    "success": true,
    "reward": 1.0,
    "feedback": "exact match"
  },
  "terminal_reward": 1.0
}
```

日志要求：

- 每轮都记录输入 observation。
- 同时保留模型原始输出和结构化结果。
- 工具失败、解析失败、timeout 不能被丢弃。
- 使用临时文件加原子 rename，避免中途写出损坏 JSON。
- 不记录 API key 等秘密信息。

## 11. Token 所有权与训练 mask

这是后续 Agent RL 数据处理最重要的边界。

### 模型 action token

由模型采样产生、可计算 policy log-prob 的 token：

- assistant message 中的 `thought` / reasoning token。
- assistant message 中的 `action.name`。
- assistant message 中的 `action.arguments`。
- `final` action 和最终 answer。
- 失败后的 reflection，仅当它也是模型生成且要训练 reflection policy 时才算 action token。

### 环境 observation token

不是当前 policy 采样产生、不能算作 action log-prob：

- system prompt 和工具 schema。
- 用户 task。
- 初始 observation。
- calculator、Python、文件搜索的输出。
- tool error、timeout、invalid-action feedback。
- verifier feedback 和 terminal reward。
- 从 `memory.jsonl` 读取并注入的历史 reflection。

注意：模型上一轮生成的 action 在下一轮会成为上下文，但它仍然是上一轮的模型 action。token 身份由来源决定，不因后来进入上下文而改变。

### SFT loss mask

默认只对 assistant token 计算 loss：

```text
system/user/tool/environment token → label = -100，mask
assistant thought/action/final token → 保留 label，不 mask
padding token → label = -100，mask
```

如果不希望训练可见 chain-of-thought，可以采用：

```text
assistant thought token → mask
assistant action JSON 和 final answer token → 不 mask
```

两种配置必须在数据集 metadata 中明确记录，不能混用后无法区分。

### Policy gradient / Agent RL mask

- reward 来自 terminal verifier。
- log-prob 和 policy loss 只覆盖本次 rollout 中模型采样的 assistant token。
- environment observation、工具输出和 verifier token 全部 mask。
- terminal reward 可以分配给整条 action-token 序列；后续再加入 advantage、折扣或 step reward。
- 训练时应保存 rollout policy/model version，避免把外部生成的 assistant token误当成当前 policy 的 on-policy action。

建议每条 tokenized trajectory 额外保存：

```json
{
  "input_ids": [101, 102, 103],
  "attention_mask": [1, 1, 1],
  "action_mask": [0, 1, 0],
  "labels": [-100, 102, -100]
}
```

## 12. Reflexion memory

当 verifier 失败时，额外调用一次模型：

```text
输入：task、失败 trajectory 摘要、verifier feedback
输出：简短 reflection，说明失败原因和下次策略
```

然后追加到 `memory.jsonl`。下一次执行同类 task 时，把最近若干条 reflection 作为 system context 注入。

限制：

- reflection 不改变当前 episode 的 terminal reward。
- reflection 与原 trajectory 分开标记来源。
- 设置 memory 数量和长度上限。
- 不把未经验证的 reflection 当成事实。
- 首版可以通过配置开关关闭 Reflexion，以便比较有无 memory 的结果。

## 13. 实现步骤

### 阶段 A：协议与数据结构

- 定义 `Message`、`Action`、`ToolResult`、`Turn` 和 `Trajectory`。
- 定义严格 JSON action schema。
- 实现 JSON parser、字段检查和错误 observation。

### 阶段 B：工具

- 实现安全受限的 AST calculator。
- 实现带 timeout 的 Python subprocess。
- 实现 workspace 范围内的文件搜索。
- 统一异常、耗时和输出截断逻辑。

### 阶段 C：ReAct loop

- 注入 system prompt、task、工具说明和可选 memory。
- 调用可替换的 `llm(messages) -> str` 接口。
- parse → validate → execute → observe → append history。
- 实现 `final`、最大轮数和连续无效 action停止条件。

### 阶段 D：验证与日志

- 实现 exact-match 和固定 command verifier。
- 生成 terminal reward。
- 将完整 trajectory 原子写入 `runs/`。
- 失败后可选生成 reflection 并追加到 memory。

### 阶段 E：测试

- calculator 正常输入和危险 AST。
- Python 正常输出、异常、超时和输出截断。
- 文件搜索正常匹配和路径逃逸。
- malformed JSON、未知工具和错误参数。
- 达到最大轮数。
- verifier 成功、失败和超时。
- trajectory schema 完整性。
- action mask 不覆盖 observation token。

## 14. 验收任务

至少运行以下场景：

1. 纯计算任务：调用 calculator 后正确提交 final。
2. 文件任务：搜索 workspace 并回答文件数量。
3. Python 任务：执行短代码并使用 stdout 完成答案。
4. 工具异常：模型收到 error observation 后改用其他 action。
5. 无效 JSON：模型收到格式反馈后修正。
6. 超时：Python 工具被终止，trajectory 保留 timeout。
7. 最大轮数：Agent 可控停止且 verifier 返回 0。
8. verifier 失败：生成 reflection 并写入 memory。

## 15. 完成标准

- 核心 ReAct 实现约 100–300 行 Python，不依赖 LangChain。
- 具备 history、Thought/Action/Observation、三个工具和严格 action parser。
- tool error、timeout、invalid action 都会成为可见 observation。
- 最大交互轮数可靠生效。
- terminal verifier 独立于模型并产生 `terminal_reward`。
- 每次运行保存完整、可重放的 trajectory。
- trajectory 明确区分模型 action 和环境 observation。
- 文档与数据代码明确实现 SFT/RL token mask。
- 自动测试覆盖正常路径与主要失败路径。
- 能稳定保存以下逻辑结构：

```text
task
  turn_1: observation, reasoning, action, tool_result
  turn_2: observation, reasoning, action, tool_result
  ...
  terminal_reward
```

## 16. 后续扩展

完成最小版本后，再按顺序扩展：

1. 多次采样 trajectory，并按 terminal reward 筛选。
2. Reflexion 重试与 memory ablation。
3. 为每步增加 process reward。
4. Tree of Thoughts：branch、evaluate、backtrack。
5. trajectory replay 与离线训练数据导出。
6. SFT action mask 和 policy-gradient action mask 的自动构造。
