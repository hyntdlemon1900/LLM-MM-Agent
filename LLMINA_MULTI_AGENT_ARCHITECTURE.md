# LLMINA Multi-Agent System Architecture

## 系统概述

LLMINA多智能体系统是一个专门用于求解复杂优化问题的AI驱动框架，特别针对**In-Network Aggregation (INA)部署和路由优化问题**。系统采用分层、模块化的设计，通过多个专业智能体的协作，将复杂问题分解为可管理的子任务，并渐进式地生成完整的求解代码。

---

## 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LLMINA Multi-Agent System                             │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Pipeline Orchestrator                             │   │
│  │              (llmina_pipeline_enhanced.py)                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                   │                                          │
│                                   ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │  Stage 0: Initialization & Environment Setup                       │     │
│  │  • Load problem definition (LLMINA.json)                          │     │
│  │  • Load code templates                                             │     │
│  │  • Create output directories                                       │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                   │                                          │
│                                   ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │  Stage 1: Problem Understanding & Clarification (Optional)         │     │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │     │
│  │  │ Problem          │  │ Problem          │  │ Problem         │ │     │
│  │  │ Clarification    │→ │ Judge            │→ │ Clarification   │ │     │
│  │  │ Agent            │  │ Agent            │  │ Summary Agent   │ │     │
│  │  └──────────────────┘  └──────────────────┘  └─────────────────┘ │     │
│  │  • Interactive Q&A with user                                      │     │
│  │  • Ambiguity detection and resolution                             │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                   │                                          │
│                                   ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │  Stage 2: High-Level Solution Design                              │     │
│  │  ┌──────────────────────────────────────────────────────────┐    │     │
│  │  │  Problem Solving Agent                                    │    │     │
│  │  │  (llmina_problem_solving.py)                             │    │     │
│  │  │  • Design overall algorithm strategy                      │    │     │
│  │  │  • Identify key components (INA placement, routing)       │    │     │
│  │  │  • Define optimization objectives                         │    │     │
│  │  └──────────────────────────────────────────────────────────┘    │     │
│  │  Output: modeling_solution (Algorithm blueprint)                  │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                   │                                          │
│                                   ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │  Stage 3: Problem Decomposition                                   │     │
│  │  ┌──────────────────────────────────────────────────────────┐    │     │
│  │  │  Problem Decompose Agent                                  │    │     │
│  │  │  (llmina_problem_decompose.py)                           │    │     │
│  │  │  • Break down into sequential sub-tasks                   │    │     │
│  │  │  • Define task boundaries and objectives                  │    │     │
│  │  │  • Ensure task completeness and coherence                 │    │     │
│  │  └──────────────────────────────────────────────────────────┘    │     │
│  │  Output: task_descriptions[] (N tasks)                            │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                   │                                          │
│                                   ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │  Stage 4: Task Classification & Dependency Analysis               │     │
│  │  ┌─────────────────────┐        ┌──────────────────────────┐    │     │
│  │  │  Task Classifier    │        │  Coordinator Agent       │    │     │
│  │  │  Agent              │        │  (llmina_coordinator.py) │    │     │
│  │  │  (llmina_task_     │        │  • Build DAG             │    │     │
│  │  │   classifier.py)    │        │  • Analyze dependencies  │    │     │
│  │  │  • Categorize tasks │        │  • Determine execution   │    │     │
│  │  │  • Assess complexity│        │    order                 │    │     │
│  │  │  • Plan strategies  │        │  • Track task results    │    │     │
│  │  └─────────────────────┘        └──────────────────────────┘    │     │
│  │  Output: task_classifications[], task_strategies[], DAG, order[] │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                   │                                          │
│                                   ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │  Stage 5: Adaptive Task Solving (Main Loop)                       │     │
│  │  ┌──────────────────────────────────────────────────────────┐    │     │
│  │  │  FOR each task in execution_order:                        │    │     │
│  │  │  ┌────────────────────────────────────────────────────┐  │    │     │
│  │  │  │ Computational Solving Module                        │  │    │     │
│  │  │  │ (llmina_computational_solving_enhanced.py)         │  │    │     │
│  │  │  │  ┌──────────────────────────────────────────┐      │  │    │     │
│  │  │  │  │  Progressive Task Solver                  │      │  │    │     │
│  │  │  │  │  (llmina_task_solver_clean.py)          │      │  │    │     │
│  │  │  │  │  ┌────────────────────────────────┐     │      │  │    │     │
│  │  │  │  │  │ 1. Build Progressive Prompt    │     │      │  │    │     │
│  │  │  │  │  │    • Copy previous tasks       │     │      │  │    │     │
│  │  │  │  │  │    • Focus current task        │     │      │  │    │     │
│  │  │  │  │  │    • Placeholder future tasks  │     │      │  │    │     │
│  │  │  │  │  └────────────────────────────────┘     │      │  │    │     │
│  │  │  │  │           ▼                              │      │  │    │     │
│  │  │  │  │  ┌────────────────────────────────┐     │      │  │    │     │
│  │  │  │  │  │ 2. Generate Complete Solver    │     │      │  │    │     │
│  │  │  │  │  │    • LLM generates llm_solver  │     │      │  │    │     │
│  │  │  │  │  │    • Include all task logic    │     │      │  │    │     │
│  │  │  │  │  └────────────────────────────────┘     │      │  │    │     │
│  │  │  │  │           ▼                              │      │  │    │     │
│  │  │  │  │  ┌────────────────────────────────┐     │      │  │    │     │
│  │  │  │  │  │ 3. Code Validation              │     │      │  │    │     │
│  │  │  │  │  │  ┌──────────────────────────┐  │     │      │  │    │     │
│  │  │  │  │  │  │ Code Validator Agent     │  │     │      │  │    │     │
│  │  │  │  │  │  │ (llmina_code_validator)  │  │     │      │  │    │     │
│  │  │  │  │  │  │ • Syntax check           │  │     │      │  │    │     │
│  │  │  │  │  │  │ • Template compatibility │  │     │      │  │    │     │
│  │  │  │  │  │  │ • Algorithm analysis     │  │     │      │  │    │     │
│  │  │  │  │  │  └──────────────────────────┘  │     │      │  │    │     │
│  │  │  │  │  └────────────────────────────────┘     │      │  │    │     │
│  │  │  │  │           ▼                              │      │  │    │     │
│  │  │  │  │  ┌────────────────────────────────┐     │      │  │    │     │
│  │  │  │  │  │ 4. Test with test_llm_solver   │     │      │  │    │     │
│  │  │  │  │  │    • Run on test instances     │     │      │  │    │     │
│  │  │  │  │  │    • Validate constraints      │     │      │  │    │     │
│  │  │  │  │  │    • Compute makespan          │     │      │  │    │     │
│  │  │  │  │  └────────────────────────────────┘     │      │  │    │     │
│  │  │  │  │           ▼                              │      │  │    │     │
│  │  │  │  │  ┌────────────────────────────────┐     │      │  │    │     │
│  │  │  │  │  │ 5. Fix if Failed (Iterative)   │     │      │  │    │     │
│  │  │  │  │  │    • Error feedback to LLM     │     │      │  │    │     │
│  │  │  │  │  │    • Regenerate fixed code     │     │      │  │    │     │
│  │  │  │  │  │    • Re-test (max iterations)  │     │      │  │    │     │
│  │  │  │  │  └────────────────────────────────┘     │      │  │    │     │
│  │  │  │  │           ▼                              │      │  │    │     │
│  │  │  │  │  ┌────────────────────────────────┐     │      │  │    │     │
│  │  │  │  │  │ 6. Save & Update Task History  │     │      │  │    │     │
│  │  │  │  │  │    • Store in task_history     │     │      │  │    │     │
│  │  │  │  │  │    • Update coordinator.memory │     │      │  │    │     │
│  │  │  │  │  └────────────────────────────────┘     │      │  │    │     │
│  │  │  │  └──────────────────────────────────────────┘      │  │    │     │
│  │  │  └────────────────────────────────────────────────────┘  │    │     │
│  │  └──────────────────────────────────────────────────────────┘    │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                   │                                          │
│                                   ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │  Stage 6: Result Integration & Reporting                          │     │
│  │  • Collect all task results                                       │     │
│  │  • Compute success rate statistics                                │     │
│  │  • Generate final reports                                         │     │
│  │  • Save complete solution JSON                                    │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                   │                                          │
│                                   ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │  Stage 7: Final Solver Generation (Optional)                      │     │
│  │  • Integrate all successful task codes                            │     │
│  │  • Generate unified llm_solver function                           │     │
│  │  • Ready for deployment                                           │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 核心智能体详细说明

### 1. **Problem Solving Agent** (问题建模智能体)
**文件**: `agent/llmina_problem_solving.py`

**职责**:
- 理解LLMINA优化问题的本质
- 设计高层次算法策略（如贪心启发式、分层优化）
- 识别关键组件：INA选址、worker-INA分配、路由决策
- 定义优化目标和约束

**输出**: `modeling_solution` - 算法设计蓝图

**示例输出**:
```
Algorithm Strategy:
1. INA Placement: Use topology-aware greedy selection based on:
   - Switch centrality in network
   - Aggregation potential (distance to workers/PS)
   - Link congestion reduction impact

2. Job Routing: For each job, assign workers to INA or PS:
   - Minimize makespan via LP relaxation
   - Consider INA capacity constraints
   - Respect link bandwidth limits
```

---

### 2. **Problem Decompose Agent** (问题分解智能体)
**文件**: `agent/llmina_problem_decompose.py`

**职责**:
- 将整体算法分解为N个渐进式子任务
- 确保每个任务有明确的输入、输出和目标
- 保持任务间的逻辑连贯性

**输出**: `task_descriptions[]` - 任务描述列表

**任务分解示例** (LLMINA问题):
```
Task 1: INA候选交换机选择与初步筛选
  - 根据拓扑结构筛选候选交换机
  - 计算每个候选的优先级分数
  - 选择top-K作为INA部署位置

Task 2: Worker到聚合点的初步分配策略
  - 为每个worker选择一个聚合点(INA或PS)
  - 使用简单贪心策略(距离/容量)
  - 生成初步路由方案

Task 3: 基于容量的路由优化
  - 考虑INA处理容量约束
  - 优化worker-INA分配
  - 使用LP松弛求解流量分配

Task 4: 链路带宽约束下的路由调整
  - 考虑网络链路容量限制
  - 调整路由以避免拥塞
  - 计算最终makespan

Task 5: 迭代优化与微调
  - 基于前述结果进行全局优化
  - 处理边界情况
  - 确保所有约束满足
```

---

### 3. **Task Classifier Agent** (任务分类智能体)
**文件**: `agent/llmina_task_classifier.py`

**职责**:
- 分析每个子任务的类型和复杂度
- 为不同类型任务制定处理策略

**任务分类维度**:
```python
{
    "category": "algorithm_design",  # 算法设计 / data_processing / testing
    "complexity": "high",              # high / medium / low
    "requires_modeling": True,         # 是否需要建模分析
    "requires_knowledge": False,       # 是否需要知识库检索
    "estimated_lines": 150             # 预估代码行数
}
```

**策略映射**:
```python
{
    "code_generation_strategy": "progressive_with_validation",
    "max_debug_iterations": 5,
    "enable_llm_deep_analysis": True
}
```

---

### 4. **Coordinator Agent** (协调器智能体)
**文件**: `agent/llmina_coordinator.py`

**职责**:
- 分析任务间的依赖关系，构建DAG (Directed Acyclic Graph)
- 确定任务执行顺序 (拓扑排序)
- 管理任务间的数据传递
- 跟踪任务执行状态

**核心数据结构**:
```python
coordinator.DAG = {
    "1": [],           # Task 1 无依赖
    "2": ["1"],        # Task 2 依赖 Task 1
    "3": ["2"],        # Task 3 依赖 Task 2
    "4": ["2", "3"],   # Task 4 依赖 Task 2 和 3
    "5": ["4"]         # Task 5 依赖 Task 4
}

coordinator.memory = {
    "1": {
        "task_code": "...",
        "is_pass": True,
        "execution_result": "..."
    },
    ...
}
```

---

### 5. **Progressive Task Solver** (渐进式任务求解器)
**文件**: `agent/llmina_task_solver_clean.py`

**核心设计理念**: **渐进式完整Solver生成**

每个任务都生成**完整的`llm_solver`函数**，但focus不同：
- **Task N**: 精确实现Task N的算法 + 复制Task 1~N-1的代码 + 占位Task N+1~End

**关键特性**:
1. **Task History持久化**: 跨任务维护`task_history`，避免重复实现
2. **代码复用强调**: 提示词明确要求复用已有代码，避免重复编写
3. **完整可测试**: 每个task输出的solver都能独立运行和测试
4. **性能渐进**: Task 1→2→3→...→N，makespan逐步优化

**工作流程**:
```python
def solve_task(task_id, total_tasks, ...):
    # 1. 构建渐进式提示词
    prompt = _build_progressive_prompt(
        task_id=task_id,
        task_history=self.task_history,  # 包含前面任务的代码
        task_description=current_task,
        modeling_solution=overall_strategy,
        code_template=template,
        helper_functions=tools
    )
    
    # 2. 生成完整solver
    code = llm.generate(prompt)
    
    # 3. 验证代码
    code, is_valid = code_validator.validate_and_fix(code, ...)
    
    # 4. 测试执行
    code, is_pass, result = _test_complete_solver(code, ...)
    
    # 5. 保存到task_history
    self.task_history[task_id] = {
        'description': task_description,
        'code': code
    }
    
    return result
```

**提示词设计要点** (避免重复实现):
```python
# 中间任务提示词
"""
# How to Implement

You are building upon previous work and preparing for future tasks.

1. **Build on existing code**: The code shown above includes Tasks 1-{task_id-1}. 
   **DO NOT rewrite or duplicate existing logic** - treat it as your foundation.

2. **Code modification principles**:
   - **Identify what to change**: Locate the specific section for Task {task_id}
   - **Reuse existing components**: If Tasks 1-{task_id-1} already implement 
     helper logic, variable initialization, or data structures, REUSE them directly
   - **Avoid duplication**: Don't create new variables with similar names 
     (e.g., if `candidates` exists, don't create `candidates_new`)
   - **Incremental modification**: Only modify/add parts directly related to Task {task_id}

**Warning**: Avoid duplicating functionality that already exists in previous tasks. 
This causes inconsistencies and bugs. Think of your job as *extending* the code, 
not *rewriting* it.
"""
```

---

### 6. **Code Validator Agent** (代码验证智能体)
**文件**: `agent/llmina_code_validator.py`

**核心功能**: **算法设计合理性分析 + 代码正确性检查**

**验证层次**:

```
┌─────────────────────────────────────────────────────────┐
│  Level 1: Syntax Validation (Fast)                      │
│  • Python syntax check (AST parsing)                    │
│  • Import statement validation                          │
└─────────────────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Level 2: Template Compatibility (Structural)           │
│  • Function signature matching (llm_solver)             │
│  • Parameter alignment check                            │
│  • Return format validation                             │
└─────────────────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Level 3: Algorithm Design Analysis (LLM Deep Analysis) │
│  ┌───────────────────────────────────────────────────┐ │
│  │ Primary Focus: Algorithm Quality                  │ │
│  │ • Is the algorithm appropriate for the task?      │ │
│  │ • Does it comprehensively address requirements?   │ │
│  │ • Are algorithmic steps logically sound?          │ │
│  │ • Is the strategy aligned with best practices?    │ │
│  │ • Are there missing components in the design?     │ │
│  └───────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────┐ │
│  │ Secondary Focus: Critical Code Issues             │ │
│  │ • Runtime errors (undefined vars, type errors)    │ │
│  │ • Data structure misuse                           │ │
│  │ • Mathematical formula correctness                │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**LLM深度分析提示词**:
```python
"""
You are an expert algorithm designer and code reviewer.

**Analysis Instructions:**
Your primary focus is on **algorithm design quality and correctness**.

**1. Algorithm Design Analysis (Primary Focus):**
   - Is the chosen algorithm appropriate for the task requirements?
   - Does the algorithm comprehensively address all aspects of the problem?
   - Are the core algorithmic steps logically sound and mathematically correct?
   - Does the solution strategy align with best practices?
   - Are there missing components or incomplete logic?

**2. Problem Coverage & Completeness:**
   - Does the implementation handle all required scenarios?
   - Are all constraints and conditions properly addressed?
   - Are edge cases and boundary conditions considered?

**3. Algorithm Correctness:**
   - Are mathematical formulations correct?
   - Do algorithm steps follow correct sequence and logic?
   - Are optimization objectives properly implemented?

**4. Code Quality (Secondary):**
   - Critical runtime errors only
   - Data structure appropriateness
   - Interface compatibility

**Important:** Prioritize algorithm design issues over minor code style issues.
"""
```

---

### 7. **Evaluation Module** (评估模块)
**文件**: `evaluation.py`

**核心功能**: 测试生成的solver并验证约束

**测试流程**:
```python
def test_llm_solver(solver_func, ...):
    # 1. 创建测试环境
    network = FatTree(k=4, ...)
    dataset = generate_dataset(...)
    
    # 2. 调用solver
    for instance in dataset:
        ina_placement, jobs_routing = solver_func(
            instance, network, K, jobs_num, Cs, topo_name
        )
        
        # 3. 验证约束
        validate_solution_constraints(
            ina_placement, jobs_routing, instance, network, K, ...
        )
        # Constraint 1: INA数量 ≤ K
        # Constraint 2: 每个worker恰好选择1个聚合点
        # Constraint 3: worker只能选择已部署INA的交换机
        
        # 4. 计算makespan (使用Gurobi LP)
        makespan = evaluate_completion_time(
            instance, network, ina_placement, jobs_routing, ...
        )
    
    return results
```

**约束验证**:
```python
def validate_solution_constraints(...):
    # 约束1: INA数量不超过K
    if sum(ina_placement) > K:
        raise ValueError("INA数量超过预算")
    
    # 约束2: 每个worker有且仅有1个聚合点
    for j, w in enumerate(workers):
        if sum(y_ina[j][w]) + y_ps[j][w] != 1:
            raise ValueError("Worker聚合点选择错误")
    
    # 约束3: y_j_w_s ≤ x_s (只能选择已部署INA的交换机)
    for j, w, s in ...:
        if y_ina[j][w][s] > 0 and ina_placement[s] == 0:
            raise ValueError("选择了未部署INA的交换机")
```

---

## 数据流图

```
┌─────────────────┐
│ LLMINA.json     │ (Problem Definition)
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  Problem String + Code Template             │
└────────┬─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  modeling_solution                              │
│  "Use greedy INA selection + LP routing..."    │
└────────┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  task_descriptions[]                                │
│  [Task1: "INA selection...",                        │
│   Task2: "Worker assignment...", ...]               │
└────────┬────────────────────────────────────────────┘
         │
         ├──────────────────────────────┬───────────────┐
         ▼                              ▼               ▼
┌──────────────────┐    ┌────────────────────┐  ┌────────────────┐
│ task_classifications│  │ task_strategies[]  │  │ DAG + order[]  │
│ [cat, complexity]   │  │ [strategy params]  │  │ [1,2,3,4,5]    │
└──────────────────┘    └────────────────────┘  └────────┬───────┘
                                                          │
         ┌────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│  Task Solving Loop (Progressive Generation)              │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Task 1: Generate complete llm_solver              │ │
│  │  • Implement Task 1 logic                          │ │
│  │  • Placeholder for Task 2-5                        │ │
│  │  → test_llm_solver() → makespan_1                  │ │
│  └────────────────────────────────────────────────────┘ │
│         ▼                                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Task 2: Generate complete llm_solver              │ │
│  │  • Copy Task 1 exact code                          │ │
│  │  • Implement Task 2 logic                          │ │
│  │  • Placeholder for Task 3-5                        │ │
│  │  → test_llm_solver() → makespan_2 (improved)       │ │
│  └────────────────────────────────────────────────────┘ │
│         ▼                                                │
│  ... (Task 3, 4, 5 similar pattern)                     │
│         ▼                                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Task 5: Generate complete llm_solver (final)      │ │
│  │  • Copy Task 1-4 exact code                        │ │
│  │  • Implement Task 5 optimization                   │ │
│  │  → test_llm_solver() → makespan_5 (best)           │ │
│  └────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│  Final Outputs                                            │
│  • solver_task5.py (最终完整solver)                       │
│  • complete_solution.json (所有任务结果)                  │
│  • Makespan progression: [m1, m2, m3, m4, m5]            │
└──────────────────────────────────────────────────────────┘
```

---

## 关键设计模式

### 1. **渐进式Solver生成模式**

**问题**: 如何避免大模型在生成复杂代码时出现逻辑混乱？

**解决方案**: 
- ✅ 每个任务focus一个明确目标
- ✅ 通过`task_history`精确复制前面任务的代码
- ✅ 每个任务输出完整可运行的solver
- ✅ 实时测试验证，性能渐进提升

**优势**:
```
传统方式: Task1_code + Task2_code + ... + Task5_code → 集成(易出错) → 测试
                                                          ↑
                                                      容易失败

渐进式:   Task1_code → 测试 ✓ → Task2_code(含Task1) → 测试 ✓ → ... → Task5_code → 测试 ✓
                ↓                        ↓                              ↓
          makespan_1              makespan_2 (better)           makespan_5 (best)
```

### 2. **代码复用强化模式**

**问题**: LLM倾向于重新实现已有功能，导致代码冗余和不一致

**解决方案**:
- ✅ 提示词明确强调"Reuse, don't reimplement"
- ✅ 在prompt中提供完整的前序代码
- ✅ 警告重复实现的危害
- ✅ 具体指导如何定位和修改特定部分

**示例**:
```python
# Bad (重复实现)
# Task 2 重新创建了candidates变量
ina_candidates_new = get_ina_candidates(network)  # 重复!

# Good (复用)
# Task 2 直接使用Task 1已定义的candidates
# ina_candidates已在Task 1中定义，直接使用
```

### 3. **算法优先的验证模式**

**问题**: 传统代码验证关注语法和类型，忽略算法正确性

**解决方案**:
- ✅ 验证层次分明：语法 → 接口 → **算法设计**
- ✅ LLM深度分析focus算法合理性
- ✅ 检查问题覆盖度、逻辑完整性、数学正确性
- ✅ 优先报告算法设计问题，而非代码风格问题

### 4. **约束驱动的测试模式**

**问题**: 生成的代码可能产生不可行解

**解决方案**:
- ✅ 在测试阶段强制验证LLMINA核心约束
- ✅ 约束违反时提供详细错误信息
- ✅ 将错误反馈给LLM进行修复
- ✅ 迭代修复直到满足所有约束

---

## 配置文件示例

```yaml
# config.yaml
model_name: "qwen3-30b-a3b"
problem_type: "optimization"

# Stage控制
enable_clarification: false
problem_modeling_round: 2

# 任务求解
max_debug_iterations: 5
enable_code_validation: true
generate_final_solver: false

# 模板路径
code_template_path: "MMAgent/code_template/llmina_solver_template.py"
template_dir: "MMAgent/code_template"

# 测试参数
test_topo_name: "FatTree"
test_ina_num: [3]
test_jobs_num: [6]
test_instances_num: 1
```

---

## 输出文件结构

```
output/LLMINA-Solver/LLMINA_20251024-HHMMSS/
├── modeling_solution.txt                      # 高层算法方案
├── task_descriptions.json                     # 任务分解
├── task_classifications_and_strategies.json   # 任务分类与策略
├── dependency_dag.json                        # 任务依赖图
├── task_dependency_analysis.json              # 依赖分析详情
├── complete_solution.json                     # 完整求解记录
├── code/
│   ├── task_1.py                              # Task 1的完整solver
│   ├── task_2.py                              # Task 2的完整solver (含Task 1)
│   ├── task_3.py                              # Task 3的完整solver (含Task 1-2)
│   ├── task_4.py                              # Task 4的完整solver (含Task 1-3)
│   └── task_5.py                              # Task 5的完整solver (最终版)
├── task_1_result.json                         # Task 1执行结果
├── task_2_result.json
├── task_3_result.json
├── task_4_result.json
├── task_5_result.json
└── usage/
    ├── LLMINA.json                            # LLM token使用统计
    └── runtime.txt                            # 总运行时间
```

---

## 性能指标

### 任务成功率
```
Overall Statistics:
  Total Tasks: 5
  Passed: 5
  Failed: 0
  Success Rate: 100.0%

By Task Category:
  algorithm_design: 5/5 (100.0%)
```

### Makespan渐进改进
```
Task 1 → Makespan: 245.67
Task 2 → Makespan: 198.34  (19.3% improvement)
Task 3 → Makespan: 176.89  (11.1% improvement)
Task 4 → Makespan: 168.45  (4.8% improvement)
Task 5 → Makespan: 163.12  (3.2% improvement)

Total improvement: 33.6% from Task 1 to Task 5
```

---

## 技术栈

### 核心依赖
- **LLM Backend**: OpenAI-compatible API (Qwen3-30B, GPT-4, etc.)
- **Optimization**: Gurobi (LP solving for makespan evaluation)
- **Network Modeling**: NetworkX (topology management)
- **Testing**: Python unittest framework

### 辅助工具
- **AST Parsing**: Python `ast` module (syntax validation)
- **Graph Algorithms**: DAG construction, topological sort
- **JSON Schema**: Task metadata management

---

## 扩展性设计

### 支持新问题类型
1. 定义新的problem JSON schema
2. 创建specialized solver template
3. 实现problem-specific helper functions
4. 配置task decomposition策略

### 支持新的LLM
1. 实现LLM wrapper接口
2. 适配prompt格式
3. 调整token限制和重试策略

### 支持新的验证器
1. 继承`BaseAgent`
2. 实现`validate()`方法
3. 在pipeline中注册

---

## 最佳实践

### 1. Prompt工程
- ✅ 使用叙事风格，讲清楚背景、目标、方法
- ✅ 明确输出格式（JSON、代码块）
- ✅ 提供具体示例
- ✅ 强调关键约束和注意事项

### 2. 任务分解
- ✅ 每个任务focus单一目标
- ✅ 任务粒度适中（100-200行代码）
- ✅ 任务间依赖清晰
- ✅ 渐进式改进（baseline → optimization）

### 3. 代码生成
- ✅ 提供完整的helper functions
- ✅ 明确函数签名和返回格式
- ✅ 强调复用已有代码
- ✅ 避免重复实现

### 4. 验证与测试
- ✅ 快速语法检查在前
- ✅ 算法分析在后
- ✅ 实际测试验证
- ✅ 迭代修复机制

---

## 总结

LLMINA多智能体系统通过**分层架构**、**渐进式生成**和**智能验证**三大核心机制，成功将复杂优化问题的求解自动化。系统的关键创新在于：

1. **渐进式完整Solver生成**：每个任务输出完整可测试的代码，避免集成问题
2. **代码复用强化**：通过task_history和精心设计的提示词，确保代码复用而非重复实现
3. **算法优先验证**：验证重点从代码语法转向算法设计合理性
4. **约束驱动测试**：强制验证问题约束，确保解的可行性

该架构具有良好的**可扩展性**和**可维护性**，可适配不同类型的优化问题和LLM后端。

---

**文档版本**: v1.0  
**最后更新**: 2024-10-24  
**作者**: LLMINA Team
