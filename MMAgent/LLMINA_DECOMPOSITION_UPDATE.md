# LLMINA问题分解原则重构说明

## 修改概述

针对LLMINA问题的特殊性，重新设计了问题分解原则，从通用数学建模的分解方式转变为灵活的算法设计指导原则。

**设计理念：** 提供关键考虑点和指导性建议，而非强制性的步骤模板，允许LLM根据具体问题特征自主决定最合理的分解方式。

## 核心区别

### LLMINA vs. 传统数学建模问题（如MCM）

| 特征维度 | 传统MCM问题 | LLMINA问题 |
|---------|-----------|-----------|
| **问题性质** | 开放性、探索性 | 确定性、已建模 |
| **用户提供** | 问题背景、数据 | 完整MILP公式、约束 |
| **核心任务** | 建模 + 求解 | 算法设计 |
| **数据分析** | 需要大量探索 | 不需要（无数据文件） |
| **约束类型** | 软约束为主 | 硬约束（必须满足） |
| **输出要求** | 报告、分析 | 可执行代码、接口对齐 |

### 示例对比

**传统MCM问题（2018_C - 能源生产）：**
- 问题："分析美国各州能源生产的变化趋势"
- 任务：探索数据 → 建立指标 → 统计建模 → 预测分析
- 分解：数据清洗、特征工程、模型构建、敏感性分析

**LLMINA问题：**
- 问题："给定MILP模型，设计启发式算法最小化makespan"
- 任务：算法设计 → 实现 → 验证 → 优化
- 分解：初始构造、迭代改进、约束验证、系统集成

## 新分解原则（4个Subtask）

### Subtask 1: 初始解构造 (Initial Solution Construction)

**目标：** 快速生成满足所有硬约束的可行初始解

**方法论：**
- 贪心启发式（基于拓扑中心性、worker连接度）
- 构造性算法（逐步构建解决方案）
- 约束优先（确保预算、容量等硬性限制）

**输出：** 可行的INA部署位置 + 初始路由分配

**关键点：**
- ✅ 可行性是第一优先级
- 利用网络拓扑领域知识
- 为后续优化提供基础

### Subtask 2: 迭代改进与局部搜索 (Iterative Refinement)

**目标：** 在保持可行性下优化解的质量

**方法论：**
- 邻域搜索（swap、move操作）
- 局部优化（hill-climbing、模拟退火）
- 负载均衡调整

**输出：** 改进的INA部署和路由方案

**关键点：**
- 探索与利用的平衡
- 收敛标准（迭代次数、改进阈值）
- 保持可行性的同时提升质量

### Subtask 3: 约束验证与可行性保障 (Constraint Validation)

**目标：** 严格验证所有MILP约束被满足

**验证项：**
1. INA预算约束：`|selected_switches| ≤ K`
2. Worker独占性：每个worker选择唯一聚合点
3. 交换机处理能力：`Σ γ_jws ≤ C_s`
4. 链路带宽限制：`load_e ≤ bandwidth_e`
5. 路径一致性：基于`allPathDict`的流量聚合

**工具：**
- `evaluate_completion_time()`: 黑盒oracle
- 路径字典和带宽映射

**关键点：**
- ❌ 任何约束违反 = 解不可行
- 必须有修复机制
- 验证是迭代过程的一部分

### Subtask 4: 全局整合与接口对齐 (System Integration)

**目标：** 组装完整求解器，对齐评估接口

**任务：**
1. 模块编排（构造 → 优化 → 验证）
2. 数据结构转换
3. 输出格式对齐（ina_placement, jobs_routing）
4. 参数配置与调优
5. 鲁棒性保障（异常处理、边界情况）

**关键点：**
- 严格遵循`llm_solver()`接口
- 确保输出格式正确
- 处理各种极端情况

## 代码修改

### 文件：`MMAgent/agent/llmina_problem_decompse.py`

**主要变更：**

1. **移除对通用分解原则的依赖：**
   ```python
   # 旧代码
   self.decomposed_principles = read_json_file('MMAgent/prompt/decompose_prompt.json')
   decomposed_principle = self.decomposed_principles.get(problem_type, ...)
   
   # 新代码
   self.llmina_decompose_principle = self._get_llmina_principle()
   decomposed_principle = self.llmina_decompose_principle
   ```

2. **新增专用分解原则方法：**
   ```python
   def _get_llmina_principle(self):
       """返回LLMINA专用的4阶段分解原则"""
       return """..."""  # 详细的算法工程导向的分解说明
   ```

3. **保持接口兼容：**
   - `decompose()` 方法签名不变
   - `refine()` 方法不变
   - `decompose_and_refine()` 方法不变

## 设计理念

### 从"数据科学流程"到"算法工程流程"

**旧范式（不适合LLMINA）：**
```
数据准备 → 特征工程 → 模型训练 → 结果分析
```

**新范式（适合LLMINA）：**
```
初始构造 → 迭代优化 → 约束验证 → 系统集成
```

### 关键设计原则

1. **约束驱动：** 可行性是硬性要求，不是优化目标
2. **算法设计：** 强调启发式、邻域搜索等算法技术
3. **工程实现：** 关注接口、数据结构、模块化
4. **领域知识：** 利用网络拓扑、流量特性等领域信息

### 为什么不使用通用分解原则？

通用的decompose_prompt.json包含C/D/A/B/E/F类问题的分解原则，都是针对开放性数学建模问题设计的：

- **C类（4/3/5 subtasks）：** 数据预处理 → 评分系统 → 预测建模 → 敏感性分析
- **D类（3/4/5 subtasks）：** 基础模型 → 优化机制 → 敏感性分析
- **A类（5/3/4 subtasks）：** 建立框架 → 约束分析 → 优化评估 → 敏感性

这些都假设：
- ❌ 需要从数据中发现模式
- ❌ 需要构建数学模型
- ❌ 软约束和权衡

而LLMINA需要：
- ✅ 设计高效算法
- ✅ 满足硬性约束
- ✅ 实现可执行代码

## 使用示例

```python
# 初始化分解器
decomposer = ProblemDecompose(llm)

# 执行分解（自动使用LLMINA专用原则）
subtasks, tasknum = decomposer.decompose_and_refine(
    modeling_problem=problem_str,
    modeling_solution=high_level_solution,
    problem_type='optimization'  # 参数值不再影响分解原则的选择
)

# 输出示例：
# subtasks = [
#     "Subtask 1: Design a greedy heuristic for initial INA placement...",
#     "Subtask 2: Implement iterative local search with swap operations...",
#     "Subtask 3: Validate all MILP constraints using evaluate_completion_time...",
#     "Subtask 4: Integrate modules and align with llm_solver interface..."
# ]
# tasknum = 4
```

## 测试建议

### 验证分解质量

检查生成的subtasks是否：
1. ✅ 强调"算法设计"而非"数据分析"
2. ✅ 提到具体算法技术（贪心、邻域搜索等）
3. ✅ 关注约束验证和可行性
4. ✅ 包含接口对齐和系统集成
5. ❌ 不包含"数据预处理"、"特征工程"等不相关内容

### 对比测试

可以尝试对比使用旧原则（C类4-subtask）和新原则生成的subtask描述，应该能看到明显的风格差异。

## 文档资源

1. **LLMINA_DECOMPOSITION_PRINCIPLE.md**: 详细的设计说明和原理分析
2. **LLMINA_README.md**: LLMINA求解器的整体架构文档
3. **本文档**: 修改说明和使用指南

## 后续工作建议

1. **提示词优化：** 调整`TASK_DECOMPOSE_PROMPT`和`TASK_DESCRIPTION_PROMPT`，使其更适合算法设计任务
2. **示例注入：** 在prompt中加入算法伪代码示例
3. **约束强调：** 在subtask描述中更明确地列出需要满足的约束
4. **评估工具说明：** 在分解时就说明如何使用`evaluate_completion_time`和其他工具函数

## 总结

这次重构将LLMINA问题的分解从"数学建模范式"转变为"算法设计范式"，更符合问题的本质特征。新的分解原则强调：

- 🎯 算法工程而非数据科学
- 🔒 硬约束满足而非软优化权衡  
- 💻 代码实现而非分析报告
- 🏗️ 模块化设计而非探索性分析

这将帮助LLM更准确地理解任务目标，生成更高质量的算法设计方案。
