# LLMINA: Large Language Model-driven In-Network Aggregation Solver System

## 1. Introduction

In-Network Aggregation (INA) 及类似的组合优化问题通常具有高度的复杂性（NP-Hard）和严格的约束条件。传统的求解方法要么依赖通用的精确求解器（如 MILP Solver），在大规模问题上难以扩展；要么依赖专家手工设计的启发式算法，耗时费力且难以适应变化的需求。

**LLMINA (Large Language Model-driven In-Network Aggregation)** 系统提出了一种全新的自动化求解范式。该系统不仅仅利用大语言模型（LLM）生成代码，而是构建了一个**基于蒙特卡洛树搜索（MCTS）的进化多智能体框架**。该框架将算法设计过程建模为搜索空间中的决策过程，通过**关注点分离（Separation of Concerns）**的设计理念，将算法的"架构设计"与"代码实现"解耦，并引入**多层次反馈机制**来确保生成的算法既符合语法规范又满足严格的领域约束。

## 2. System Architecture

LLMINA 系统的核心架构由三个主要子系统组成，形成闭环控制流：

1.  **Evolutionary Architect Agent (进化架构师智能体)**:
    *   作为系统的核心生成引擎，负责理解问题、设计算法架构以及编写具体函数代码。
    *   该智能体具备"自我反思"能力，能够根据评估反馈修正其设计。

2.  **MCTS Engine (MCTS 演化引擎)**:
    *   作为系统的导航器，维护一棵"算法演化树"。
    *   负责管理算法种群，记录每个算法变体的性能（Reward），并利用 Upper Confidence Bound for Trees (UCT) 算法决定下一步的搜索方向（探索新策略或利用旧策略）。

3.  **Evaluation Environment (评估与验证环境)**:
    *   提供沙箱化的运行时环境，负责执行生成的代码。
    *   包含严格的**可行性检查器 (Feasibility Checker)** 和 **性能分析器 (Performance Analyzer)**，为 MCTS 和 Agent 提供定量的奖励信号（Reward）和定性的错误报告。

---

## 3. Workflow & Methodology

LLMINA 的工作流采用 **LangGraph** 进行编排，表现为一个复杂的有向图（StateGraph）。整体流程可以划分为初始化、架构设计、实现、评估与反馈、进化循环五个阶段。

### 3.1 Problem Decomposition & Architectural Design (问题拆解与架构设计)

为了解决 LLM 在生成长代码时容易出现的逻辑混乱和幻觉问题，LLMINA 采用了**关注点分离**策略。

*   **Architect Node (i1 Operator)**:
    在此阶段，LLM 扮演系统架构师的角色。它**不编写**具体的 Python 代码，而是输出一个结构化的 JSON 对象 (`heuristic_architecture`)。
    *   **Problem Analysis**: 分析数学模型的变量、目标和约束。
    *   **Strategy Overview**: 制定宏观求解策略（如"贪心策略"、"松弛-修补策略"）。
    *   **Function Architecture**: 将复杂的求解逻辑拆解为一系列模块化的函数（Steps）。每个函数定义了：
        *   `strategic_role`:该函数的战略目标（What to do）。
        *   `inputs/outputs`: 数据接口。
        *   `dependencies`: 依赖关系。

这种设计将算法的"逻辑骨架"与"实现血肉"分离，使得系统能够处理更复杂的逻辑结构。

### 3.2 Iterative Code Generation (迭代式代码生成)

*   **Function Generator Node**:
    系统根据架构蓝图，**逐个**生成函数的 Python 实现。
    *   **Context-Aware**: 在生成第 `i` 个函数时，LLM 可以看到第 `1` 到 `i-1` 个函数的签名和功能摘要。这确保了生成的代码能够正确调用前序步骤，形成连贯的逻辑链。
    *   **Integration**: 所有生成的函数最终被 `Code Integration Node` 组装进标准的求解器模板 (`ModelSolver`) 中，形成可执行的完整类。

### 3.3 Multi-Level Feedback Loop (多层次反馈闭环)

LLMINA 设计了独特的双重反馈机制，以应对不同类型的错误：

#### Level 1: Syntax & Runtime Correction (Code Fix)
*   **触发条件**: Python解释器抛出异常（如 `SyntaxError`, `AttributeError`, `NameError`）。
*   **处理机制 (`code_fix_node`)**:
    *   系统捕获完整的 Traceback 和源代码。
    *   Agent 定位出错的函数，并进行"原地修复"。
    *   这是一个快速的内循环，旨在确保代码"能跑通"。

#### Level 2: Logical & Feasibility Correction (Reflection)
*   **触发条件**: 代码运行成功，但**可行性检查失败**（如违反带宽约束、资源超限）或**运行超时**（由 `func_timeout` 捕获）。
*   **处理机制 (`heuristic_reflect_node`)**:
    *   系统生成详细的**质量报告 (`quality_report`)**，列出具体的违规项。
    *   Agent 进入反思模式，分析导致违规的**逻辑根源**（例如："贪心策略过于激进，导致 INA 节点过载"）。
    *   Agent 生成**经验教训 (`experience_lessons`)**，并指定需要**重写**的函数列表。
    *   MCTS 将此过程视为一次**变异 (Mutation / e1)** 操作，生成一个新的子节点。

### 3.4 Prompt Engineering & Context Management

为了最大化 LLM 在复杂算法设计任务中的表现，LLMINA 采用了一套精细的提示词工程（Prompt Engineering）策略，核心在于**全上下文注入**与**结构化约束**。

#### 1. Dual-Source Context Injection (双源上下文注入)
在架构设计和代码生成阶段，系统始终向 LLM 同时提供两类关键信息：
*   **Formal Problem Formulation**: 来源于 `problem.json` 的结构化描述，包含数学符号定义的集合、变量、目标函数及约束条件（C1-C12）。这确保 LLM 理解问题的**理论边界**。
*   **Reference Code Base**: 注入带有详细类型注解（Type Hints）和文档字符串（DocStrings）的 `ModelSolver.py` 模板代码及依赖库（如 `networkx`, `pyomo`）。
    *   **目的**：利用 In-Context Learning，让 LLM 模仿现有的代码风格，并复用已有的工具函数（如 `_get_flows_on_link`），减少幻觉并提高代码的一致性。

#### 2. Incremental State Tracking (增量状态追踪)
在 `Function Generator` 的循环生成过程中，Prompt 是动态变化的：
*   **Function Specification**: 当前待生成函数的 `Strategic Role`（战略目标）和输入输出定义。
*   **Completed Function Summary**: 自动生成已完成函数的**摘要列表**，包括函数签名和功能简介。
    *   **机制**：通过正则解析或元数据追踪，即时更新摘要。
    *   **效果**：强制 LLM 在实现后续步骤（如 Step 3）时，显式调用前序步骤（Step 1, Step 2）的结果，确保数据流的连通性。

### 3.5 Constraint-Driven Feedback Details (基于约束的反馈机制详解)

传统的 LLM 往往难以处理隐式的数学约束。LLMINA 通过**显式化（Explicitation）**和**语义转化（Semantic Translation）**机制来解决这一问题。

#### 1. Hard Constraint Verification (硬约束验证)
系统内置了基于规则的验证器 `check_feasibility()`，它独立于 LLM 运行，直接检查解对象是否满足物理限制：
*   **Resource Budget**: 检查使用的 INA 节点数是否超过预算 (`len(distinct_ina_switches) > budget`)。
*   **Routing Logic**: 检查每个 Worker 是否都被分配了合法的汇聚点。
*   **Capacity Limits**: 检查每条物理链路的带宽负载是否溢出。

#### 2. Semantic Translation of Errors (错误的语义转化)
当验证失败时，系统**不会**仅仅返回一个 "False"。而是生成一份结构化的自然语言报告，直接注入到 Reflection Prompt 中：
*   **原始错误**: `Error Code: INA_BUDGET_EXCEEDED, Val: 7, Limit: 5`
*   **语义转化**: 
    > "Feasibility Check Failed: The solution deployed INA on 7 switches, which strictly violates the constraint of maximum 5 switches. This suggests the node selection strategy is too aggressive."
*   **反馈闭环**: Reflection Agent 接收到此语义信息后，会将其映射回代码逻辑，例如修改排序函数 `select_top_k_nodes` 中的 `k` 值设定或是增加剪枝逻辑。

### 3.6 Implementation via LangGraph (基于 LangGraph 的图编排实现)

LLMINA 的动态工作流通过 **LangGraph** 实现为一个状态图（StateGraph）。这种图结构允许系统在确定的线性流程（如代码集成）和动态的循环流程（如进化迭代、代码修复）之间灵活切换。

#### 1. Graph Nodes (核心节点)
图中的节点映射到系统的具体功能模块：
*   **Initialization Nodes**: `load_problem` (加载问题上下文)。
*   **Operator Nodes (MCTS Actions)**:
    *   `heuristic_architect`: 初始架构设计 (i1)。
    *   `reflect`: 逻辑反思与变异 (e1)。
    *   `crossover`: 算法交叉 (e2)。
    *   `exploration`: 探索性架构设计 (e3)。
*   **Worker Nodes**:
    *   `generate_function`: 循环生成单个函数代码。
    *   `integrate`: 代码组装与持久化。
    *   `evaluate`: 代码执行与初步评分。
    *   `fix`: 针对 Python 异常的代码修复。
*   **Decision Node**: `constraint_analyze` (负责 MCTS 反向传播与下一轮选择)。

#### 2. Edges & Routing Logic (流向与路由逻辑)

系统的智能体现在复杂的**条件边 (Conditional Edges)** 设计上：

*   **The Generation Loop (生成循环)**:
    `generate_function` 节点具有指向自身的条件边。每生成一个函数后，系统检查 `current_function_id` 是否达到总数。
    *   *未完成* &rarr; `generate_function` (继续生成下一个)。
    *   *已完成* &rarr; `integrate` (进入集成阶段)。

*   **The Evaluation Router (评估路由)**:
    在 `evaluate` 节点结束后，根据评估结果的状态 (`next_action`) 进行三路分流：
    1.  **Runtime Error** &rarr; `fix`: 进入代码修复快速闭环。
    2.  **Feasibility Failure** &rarr; `reflect`: 降级为逻辑错误，进入反思进化分支（Level 2 Feedback）。
    3.  **Success** &rarr; `constraint_analyze`: 进入性能分析与 MCTS 更新阶段。

*   **The MCTS Router (进化路由)**:
    在 `constraint_analyze` 节点完成 UCT 选择后，根据选定的算子 (`mcts_operator`) 决定下一代的产生方式：
    *   `i1` &rarr; `generate_function` (重置生成)。
    *   `e1` &rarr; `reflect` (反思变异)。
    *   `e2` &rarr; `crossover` (交叉融合)。
    *   `e3` &rarr; `exploration` (随机探索)。

这种图结构设计确保了系统既有严谨的执行顺序，又具备处理突发错误和动态进化的弹性。

---

## 4. MCTS Mechanism for Algorithm Evolution

MCTS 机制是 LLMINA 实现持续优化的关键。搜索树中的每个节点代表一个具体的**算法实例**（包含其架构 JSON 和代码实现）。

### 4.1 State & Action
*   **State (Node)**: 一个完整的 heuristic algorithm 实现。
*   **Action (Edge)**: 用于生成新算法的进化算子。

### 4.2 Evolutionary Operators
系统定义了多种算子来扩展搜索树：

1.  **Genesis (i1)**: 初始化算子。调用 Architect Node 从零设计全新的算法架构。
2.  **Mutation / Reflection (e1)**: 变异算子。基于 Level 2 的反馈，对现有表现不佳但有潜力的算法进行逻辑修正。这也是一种有向的局部搜索。
3.  **Crossover (e2)**: 交叉算子。提取两个优秀父代算法的特征（如架构思路或特定函数实现）进行融合。
4.  **Exploration (e3)**: 探索算子。强制 Agent 尝试与当前最优解截然不同的设计思路，防止陷入局部最优。

### 4.3 Selection & Backpropagation
*   **Selection**: 使用 UCT 公式选择最有潜力的节点进行扩展，平衡 Exploration（尝试访问次数少的节点）和 Exploitation（深挖高分节点）。
*   **Backpropagation**: 当一个新算法在 `Evaluation Environment` 中完成测试并获得 Reward（基于解的质量和可行性）后，该分数会向上传播，更新其父节点的统计信息。

## 5. Summary

LLMINA 通过将 Code Generation 转化为 Algorithm Evolution，解决了传统 LLM 编程 Agent "写完即止"、难以处理复杂约束的痛点。其核心创新在于：
1.  **架构与实现的显式解耦**。
2.  **基于运行时反馈的逻辑反思机制**。
3.  **MCTS 引导的算法空间搜索**。

这使得系统能够从零开始，逐步演化出针对特定领域问题的高质量、可行的启发式求解器。
