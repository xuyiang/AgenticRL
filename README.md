# AgenticRL

从零实现一个最小、可观测、可训练的 ReAct Agent，并逐步扩展到 trajectory search 和 Agent RL。

> 当前实现进度：**约 4%**
>
> 已完成基础数据结构和模型输出协议，正在实现 action parser。尚未进入工具和 Agent loop 阶段。

## 项目目标

第一阶段不使用 LangChain，用 Python 实现一个约 100–300 行的最小 ReAct loop：

```text
Task / Observation
        ↓
Thought → Action → Tool Result
        ↓
更新 History
        ↓
继续交互或提交 Final Answer
        ↓
Terminal Verifier → Reward
```

需要完整保存可用于分析和训练的 trajectory：

```text
task
  turn_1: observation, reasoning, action, tool_result
  turn_2: observation, reasoning, action, tool_result
  ...
  terminal_reward
```

详细设计见 [MINIMAL_AGENT_PLAN.md](./MINIMAL_AGENT_PLAN.md)。

## 计划功能

### 阶段 1：协议与数据结构

- [x] 定义 `ParsedStep`、`ToolResult` 和 `Turn` 数据结构
- [ ] 定义 Message/history 数据结构
- [x] 定义 Thought/Action/Observation 协议
- [ ] 定义严格的 JSON action schema
- [ ] 实现 action parser（进行中：Notebook 已验证，源文件尚需修复）
- [ ] 实现参数与工具名校验
- [ ] 区分模型 action 和环境 observation

### 阶段 2：基础工具

- [ ] 实现安全受限的 Calculator
- [ ] 实现带超时的 Python subprocess 工具
- [ ] 实现限制在 workspace 内的文件搜索工具
- [ ] 统一工具返回结构
- [ ] 限制工具输出长度

### 阶段 3：ReAct Loop

- [ ] 实现 LLM 调用接口
- [ ] 实现完整 history 管理
- [ ] 实现 Thought → Action → Observation 循环
- [ ] 将工具结果追加回模型上下文
- [ ] 实现 Final Answer
- [ ] 实现最大交互轮数
- [ ] 实现连续无效 action 停止条件

### 阶段 4：错误处理

- [ ] 捕获 Tool error
- [ ] 处理工具 timeout
- [ ] 处理无效 JSON
- [ ] 处理未知工具
- [ ] 处理错误 arguments
- [ ] 将错误转换为模型可见的 observation

### 阶段 5：验证与日志

- [ ] 实现 exact-match verifier
- [ ] 实现固定命令 terminal verifier
- [ ] 生成 terminal reward
- [ ] 保存完整 trajectory JSON
- [ ] 保存模型原始输出
- [ ] 保存每轮 action、observation 和 tool result
- [ ] 使用原子写入避免日志损坏

### 阶段 6：Reflexion Memory

- [ ] verifier 失败后生成语言反思
- [ ] 将 reflection 写入 `memory.jsonl`
- [ ] 下一次尝试时注入相关 memory
- [ ] 限制 memory 数量和长度
- [ ] 支持关闭 memory 进行对照实验

### 阶段 7：训练数据

- [ ] 生成 SFT labels
- [ ] 对 system、user 和 tool token 设置 mask
- [ ] 对模型 action token保留训练 loss
- [ ] 支持隐藏 reasoning token 的训练配置
- [ ] 生成 policy-gradient action mask
- [ ] 保存 rollout model/policy version

### 阶段 8：测试

- [ ] Calculator 正常和危险输入测试
- [ ] Python 正常、异常和超时测试
- [ ] 文件搜索和路径逃逸测试
- [ ] 无效 action 测试
- [ ] 最大轮数测试
- [ ] verifier 成功、失败和超时测试
- [ ] trajectory schema 完整性测试
- [ ] token mask 边界测试

## Token 边界

后续训练时必须根据 token 来源进行区分。

模型生成的 action token：

- assistant reasoning / thought
- action name
- action arguments
- final answer

环境提供的 observation token：

- system prompt 和工具 schema
- 用户 task
- 工具执行结果
- tool error 和 timeout
- verifier feedback
- terminal reward
- 从 memory 读取的历史 reflection

默认训练规则：

```text
system / user / tool / environment token → mask
assistant thought / action / final token  → 计算 loss
padding token                             → mask
```

如果不训练可见 reasoning，则额外 mask assistant thought，只训练 action JSON 和 final answer。

## 预期项目结构

```text
AgenticRL/
├── README.md
├── MINIMAL_AGENT_PLAN.md
├── minimal_agent.py
├── runbook.ipynb
├── memory.jsonl
├── tasks/
│   └── example.json
├── runs/
└── tests/
    └── test_agent.py
```

当前已创建 `minimal_agent.py`、实验 Notebook、空的 memory 文件和测试文件；工具、loop、verifier 与正式测试尚未实现。

## 后续方向

完成最小 Agent 后，再逐步研究：

- [ ] 多次采样和 trajectory filtering
- [ ] Reflexion 重试与 memory ablation
- [ ] Process reward
- [ ] Tree of Thoughts branching
- [ ] Candidate evaluation
- [ ] Backtracking
- [ ] Offline trajectory replay
- [ ] SFT 与 Agent RL 数据导出

## 必读材料

- [x] ReAct: Synergizing Reasoning and Acting
- [x] Reflexion
- [x] Toolformer
- [x] Tree of Thoughts

## 当前状态

- 设计计划：已记录，不计入实现进度
- 基础数据结构：已完成
- 输出协议：已完成
- Action parser：进行中
- Agent loop：未开始
- 工具：未开始
- Verifier：未开始
- Trajectory logging：未开始
- Reflexion memory：未开始
- 训练 mask：未开始
- 测试：未开始

**实现进度：约 4%**
