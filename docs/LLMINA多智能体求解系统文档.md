# LLMINA 多智能体求解系统文档

## 目录

1. [系统概述](#系统概述)
2. [整体架构](#整体架构)
3. [核心组件](#核心组件)
4. [工作流程](#工作流程)
5. [智能体详解](#智能体详解)
6. [状态管理](#状态管理)
7. [实际案例：网络内聚合问题](#实际案例网络内聚合问题)
8. [使用方法](#使用方法)
9. [输出结果](#输出结果)

---

## 系统概述

LLMINA (Large Language Model for Integer Network Aggregation) 是一个基于 LangGraph 框架构建的多智能体协作系统，专门用于求解复杂的混合整数线性规划（MILP）问题。该系统通过多个专业化的智能体协同工作，将复杂问题分解为可管理的子任务，并逐步生成完整的求解代码。

### 核心特点

- **多智能体协作**：7个专业化智能体各司其职，协同完成问题求解
- **渐进式求解**：通过任务分解和逐步求解，降低问题复杂度
- **自动化流程**：从问题理解到代码生成，全程自动化
- **可扩展架构**：基于 LangGraph 的状态图设计，易于扩展和修改

### 适用场景

- 网络优化问题（如本案例的网络内聚合优化）
- 资源分配问题
- 调度优化问题
- 其他可建模为 MILP 的复杂优化问题

---

## 整体架构

### 技术栈

```
┌─────────────────────────────────────────┐
│         LangGraph Framework             │  状态图编排引擎
├─────────────────────────────────────────┤
│         Multi-Agent System              │  多智能体协作层
├─────────────────────────────────────────┤
│         Large Language Model            │  大语言模型（如 Qwen）
├─────────────────────────────────────────┤
│         Python Solver Template          │  代码模板和工具函数
└─────────────────────────────────────────┘
```

### 系统架构图

```
┌──────────────┐
│  用户问题    │
│  (JSON格式)  │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│                    LLMINA 工作流                          │
│                                                            │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐     │
│  │ 问题加载   │───▶│ 问题澄清   │───▶│ 算法设计   │     │
│  └────────────┘    └────────────┘    └────────────┘     │
│                                                            │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐     │
│  │ 任务分解   │───▶│ 依赖分析   │───▶│ 任务求解   │     │
│  └────────────┘    └────────────┘    └─────┬──────┘     │
│                                             │             │
│                                             │ 循环N次     │
│                                             ▼             │
│                                      ┌────────────┐      │
│                                      │ 代码集成   │      │
│                                      └────────────┘      │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│  完整求解器  │
│  (Python代码)│
└──────────────┘
```

---

## 核心组件

### 1. 状态管理（State Management）

系统使用 `LLMINAState` TypedDict 来管理整个工作流的状态，包含以下关键信息：

#### 基础状态
- `problem_path`: 问题文件路径
- `problem_str`: 问题描述字符串
- `config`: 配置信息
- `output_dir`: 输出目录
- `llm`: 大语言模型实例

#### 阶段性状态
- **问题澄清阶段**：`clarification_history`, `clarification_summary`
- **算法设计阶段**：`algorithm_solution`, `algorithm_design_round`
- **任务分解阶段**：`task_descriptions`, `tasknum`, `dependency_dag`
- **任务求解阶段**：`current_task_id`, `task_results`
- **代码生成阶段**：`solver_code`, `solver_code_path`

#### 控制状态
- `current_node`: 当前执行节点
- `next_action`: 下一步动作
- `loop_count`: 循环计数
- `errors`: 错误记录
- `messages`: 消息历史

### 2. 工作流图（Workflow Graph）

基于 LangGraph 的 StateGraph 构建：

```python
workflow = StateGraph(LLMINAState)

# 添加节点
workflow.add_node("start", start_node)
workflow.add_node("load_problem", load_problem_node)
workflow.add_node("clarification", problem_clarification_node)
workflow.add_node("algorithm_design", algorithm_design_node)
workflow.add_node("decompose", task_decompose_node)
workflow.add_node("analyze_dependencies", dependency_analysis_node)
workflow.add_node("solve_task", task_solving_node)
workflow.add_node("integrate", code_integration_node)
workflow.add_node("end", end_node)

# 添加边（定义执行流程）
workflow.add_edge(START, "start")
workflow.add_edge("start", "load_problem")
# ... 更多边的定义
```

### 3. 智能体节点（Agent Nodes）

每个节点封装一个专业化的智能体：

| 节点名称 | 智能体 | 功能描述 |
|---------|--------|---------|
| `load_problem` | 问题加载器 | 读取和解析问题描述 |
| `clarification` | 问题澄清智能体 | 理解问题细节，消除歧义 |
| `algorithm_design` | 算法设计智能体 | 设计高层次求解方案 |
| `decompose` | 任务分解智能体 | 将问题分解为子任务 |
| `analyze_dependencies` | 依赖分析智能体 | 分析任务间依赖关系 |
| `solve_task` | 任务求解智能体 | 实现单个子任务 |
| `integrate` | 代码集成智能体 | 整合所有子任务代码 |

---

## 工作流程

### 完整流程概览

```
1. 启动 (START)
   │
   ├─▶ 2. 问题加载 (Load Problem)
   │     • 读取 JSON 格式的问题描述
   │     • 加载代码模板
   │     • 构建完整的问题字符串
   │
   ├─▶ 3. 问题澄清 (Problem Clarification)
   │     • 分析问题描述的完整性
   │     • 识别潜在的歧义或遗漏
   │     • 生成澄清后的问题描述
   │     • 可选：与用户交互获取额外信息
   │
   ├─▶ 4. 算法方案设计 (Algorithm Design)
   │     • 设计高层次的求解策略
   │     • 选择合适的算法方法论
   │     • 分析可行性和优化方向
   │     • 输出：建模方案文本
   │
   ├─▶ 5. 任务分解 (Task Decomposition)
   │     • 将整体方案分解为N个子任务
   │     • 为每个任务生成详细描述
   │     • 默认分解为3-6个任务
   │     • 输出：任务描述列表
   │
   ├─▶ 6. 依赖分析 (Dependency Analysis)
   │     • 分析任务间的依赖关系
   │     • 构建依赖 DAG（有向无环图）
   │     • 确定任务执行顺序
   │     • 当前版本：简化为顺序执行
   │
   ├─▶ 7. 任务求解 (Task Solving) ★ 核心循环 ★
   │     • 按顺序求解每个任务
   │     • 每个任务：
   │       ├─ 生成任务代码
   │       ├─ 验证代码正确性
   │       ├─ 执行并测试
   │       └─ 保存结果
   │     • 渐进式集成：第N个任务包含1到N的所有代码
   │
   ├─▶ 8. 代码集成 (Code Integration)
   │     • 提取最终任务的完整代码
   │     • 生成完整求解器
   │     • 保存为可执行的 Python 文件
   │     • 生成解决方案摘要和 README
   │
   └─▶ 9. 结束 (END)
         • 输出求解器路径
         • 统计 LLM 使用量
         • 记录执行时间
```

### 数据流动

```
输入 JSON
    ↓
[问题字符串] 
    ↓
[澄清后的问题]
    ↓
[算法方案] ────────────┐
    ↓                  │
[任务列表] ────────┐   │
    ↓              │   │
[执行顺序]         │   │
    ↓              │   │
循环求解：         │   │
  Task 1 ◄────────┤   │
  Task 2 ◄────────┤   ├─▶ 上下文传递
  Task 3 ◄────────┤   │
  Task N ◄────────┴───┘
    ↓
[完整代码]
    ↓
输出 Python 文件
```

---

## 智能体详解

### 1. 问题加载智能体 (Problem Loader)

**职责**：读取并解析问题描述

**输入**：
- 问题文件路径（JSON 格式）
- 代码模板路径

**处理流程**：
```python
1. 读取 JSON 文件
2. 提取关键字段：
   - background: 问题背景
   - problem_requirement: 求解要求
   - variable_description: 变量说明
   - problem_formulation: 数学模型
3. 读取代码模板
4. 格式化为完整的问题字符串
5. 创建输出目录结构
```

**输出**：
- `problem_str`: 格式化的完整问题描述
- `intermediate_results`: 原始问题数据

**示例输出结构**：
```
Problem Background:
[问题背景描述]

Problem Requirement:
[求解要求]

Variable Description:
[变量说明]

Problem Formulation:
[数学模型]

Code Template:
[代码模板]
```

---

### 2. 问题澄清智能体 (Problem Clarification)

**职责**：理解问题细节，消除歧义，确保问题描述完整

**核心组件**：
- `ProblemClarification`: 主澄清智能体
- `ProblemJudge`: 判断智能体，评估澄清是否充分
- `ProblemClarificationSummary`: 总结智能体

**工作模式**：

#### 交互模式（可选）
```python
clarification_interactive: true  # 在 config.yaml 中配置

工作流程：
1. Agent 分析问题
2. Agent 提出问题或建议
3. 打印给用户
4. 等待用户输入
5. 整合用户反馈
6. 重复直到判断充分
```

#### 自动模式（默认）
```python
clarification_interactive: false

工作流程：
1. Agent 分析问题
2. Agent 自动推理缺失信息
3. Agent 做出合理假设
4. 生成澄清总结
```

**处理逻辑**：
```python
def clarification_actor(judge, summary, problem_str, user_input_handler):
    history = []
    max_rounds = 3
    
    for round in range(max_rounds):
        # 1. 分析当前问题
        analysis = analyze_problem(problem_str)
        
        # 2. 生成澄清问题/建议
        clarification = generate_clarification(analysis)
        
        # 3. 获取用户输入（交互模式）或自动推理（自动模式）
        user_feedback = user_input_handler(clarification)
        
        # 4. 更新问题描述
        problem_str = update_problem(problem_str, user_feedback)
        history.append({'clarification': clarification, 'feedback': user_feedback})
        
        # 5. 判断是否充分
        if judge.is_sufficient(problem_str, history):
            break
    
    # 6. 生成总结
    summary_text = summary.summarize(problem_str, history)
    
    return problem_str, history, summary_text
```

**输出**：
- `problem_str`: 澄清后的问题描述
- `clarification_history`: 澄清历史记录
- `clarification_summary`: 澄清总结

---

### 3. 算法设计智能体 (Algorithm Designer)

**职责**：设计高层次的求解方案和算法策略

**设计内容**：

1. **核心方法论**
   - 选择合适的算法类型（贪心、启发式、分解等）
   - 确定优化目标和约束处理策略

2. **算法概览**
   - 分步骤描述求解流程
   - 说明各步骤的输入输出

3. **关键策略**
   - 离散决策变量的处理方法
   - 连续变量的分配策略
   - 约束满足的实现方式

4. **可行性分析**
   - 算法的理论可行性
   - 实际实现的复杂度
   - 预期性能表现

5. **优化考虑**
   - 性能优化点
   - 可扩展性设计
   - 鲁棒性保证

**工作流程**：
```python
def algorithm_design(problem_str, round):
    # 1. 分析问题特征
    features = extract_problem_features(problem_str)
    
    # 2. 选择算法框架
    framework = select_algorithm_framework(features)
    
    # 3. 设计求解步骤
    steps = design_solution_steps(framework, features)
    
    # 4. 细化实现细节
    details = refine_implementation_details(steps)
    
    # 5. 生成方案文档
    solution = format_algorithm_solution(framework, steps, details)
    
    return solution
```

**输出示例**：
```markdown
## 算法方案：渐进式贪心放置与流量分配

### 1. 核心思想
采用两阶段方法：
- 阶段1：基于网络拓扑和流量特征，贪心选择 ina_budget个 INA 交换机
- 阶段2：为每个 worker 分配最优聚合点，最大化吞吐量

### 2. 详细步骤

#### 步骤 1：候选交换机评分
- 为每个交换机计算重要性得分
- 考虑因素：位置中心性、连接度、路径覆盖
- 输出：交换机得分列表

#### 步骤 2：贪心选择 ina_budget个交换机
- 按得分排序
- 选择前 ina_budget个作为 INA 交换机
- 输出：INA 交换机集合

#### 步骤 3：Worker 流量分配
- 对每个 job 的每个 worker
- 计算到各 INA 交换机和 PS 的路径
- 选择瓶颈最小的路径
- 输出：分配方案

#### 步骤 4：容量验证与调整
- 检查交换机处理容量
- 检查链路带宽容量
- 若超限，调整分配
- 输出：可行分配

### 3. 复杂度分析
- 时间复杂度：O(|S|² + |W| × |S|)
- 空间复杂度：O(|S| + |W|)
- 适用规模：中大型网络（千级节点）

### 4. 优化点
- 使用缓存避免重复路径计算
- 并行处理多个 job
- 预计算网络拓扑特征
```

**保存位置**：`output_dir/modeling_solution.txt`

---

### 4. 任务分解智能体 (Task Decomposer)

**职责**：将整体算法方案分解为可独立实现的子任务

**分解原则**：

1. **渐进性**：每个任务在前一任务基础上构建
2. **独立性**：每个任务有明确的输入输出
3. **可测试性**：每个任务可以独立验证
4. **完整性**：所有任务组合形成完整解决方案

**典型分解模式**：

```
Task 1: 数据读取和初始化
├─ 读取网络拓扑
├─ 解析 job 配置
└─ 初始化数据结构

Task 2: 核心算法框架
├─ 实现候选评分函数
├─ 实现贪心选择逻辑
└─ 返回基础可行解

Task 3: 流量分配优化
├─ 实现路径计算
├─ 实现流量分配算法
└─ 更新解的质量

Task 4: 容量约束处理
├─ 实现容量检查
├─ 实现冲突解决
└─ 确保可行性

Task 5: 性能优化
├─ 添加启发式优化
├─ 实现局部搜索
└─ 提升解的质量

Task 6: 完整集成与测试
├─ 整合所有组件
├─ 添加鲁棒性处理
└─ 最终测试验证
```

**工作流程**：
```python
def decompose_and_refine(problem_str, algorithm_solution):
    # 1. 初始分解
    initial_tasks = decompose_algorithm(algorithm_solution)
    
    # 2. Critic 评估
    feedback = critic.evaluate(initial_tasks)
    
    # 3. Improver 改进
    refined_tasks = improver.refine(initial_tasks, feedback)
    
    # 4. 生成详细描述
    task_descriptions = []
    for i, task in enumerate(refined_tasks):
        desc = generate_task_description(
            task_id=i+1,
            task=task,
            problem=problem_str,
            solution=algorithm_solution
        )
        task_descriptions.append(desc)
    
    return task_descriptions, len(task_descriptions)
```

**任务描述格式**：
```json
{
  "task_id": 1,
  "title": "数据读取和初始化",
  "description": "实现读取网络拓扑、job配置的功能...",
  "inputs": ["topo_name", "ina_num", "jobs_num"],
  "outputs": ["topology_data", "jobs_config"],
  "dependencies": [],
  "validation": "检查数据结构完整性"
}
```

**输出**：
- `task_descriptions`: 任务描述列表
- `tasknum`: 任务数量

**保存位置**：`output_dir/task_descriptions.json`

---

### 5. 依赖分析智能体 (Dependency Analyzer)

**职责**：分析任务间的依赖关系，确定执行顺序

**当前实现**：简化版本，采用顺序执行

```python
def dependency_analysis_node(state):
    tasknum = state['tasknum']
    
    # 生成顺序执行顺序：[1, 2, 3, ..., N]
    execution_order = list(range(1, tasknum + 1))
    
    # 简化的 DAG（无依赖）
    dependency_dag = {str(i): [] for i in execution_order}
    
    return {
        'execution_order': execution_order,
        'dependency_dag': dependency_dag
    }
```

**未来扩展**（可选）：

如果需要支持并行执行，可以实现完整的依赖分析：

```python
def advanced_dependency_analysis(tasks):
    # 1. 分析数据依赖
    data_deps = analyze_data_dependencies(tasks)
    
    # 2. 分析逻辑依赖
    logic_deps = analyze_logic_dependencies(tasks)
    
    # 3. 构建 DAG
    dag = build_dependency_dag(data_deps, logic_deps)
    
    # 4. 拓扑排序
    execution_order = topological_sort(dag)
    
    # 5. 识别可并行任务
    parallel_groups = identify_parallel_tasks(dag)
    
    return dag, execution_order, parallel_groups
```

**输出**：
- `execution_order`: 执行顺序列表
- `dependency_dag`: 依赖有向无环图

**保存位置**：`output_dir/dependency_dag.json`

---

### 6. 任务求解智能体 (Task Solver) ★ 核心 ★

**职责**：逐个实现每个子任务，生成可执行代码

**核心特点**：渐进式集成

```
Task 1: solver_task1.py  [包含 Task 1]
Task 2: solver_task2.py  [包含 Task 1 + Task 2]
Task 3: solver_task3.py  [包含 Task 1 + Task 2 + Task 3]
...
Task N: solver_taskN.py  [包含 Task 1 到 Task N 的完整实现]
```

**单任务求解流程**：

```python
def solve_task(llm, task_id, task_description, modeling_solution, 
               past_results, config):
    """
    求解单个任务
    
    Args:
        task_id: 当前任务 ID
        task_description: 任务描述
        modeling_solution: 整体算法方案
        past_results: 之前任务的结果（包含代码）
        config: 配置信息
    
    Returns:
        {
            'task_code': 本次生成的完整代码,
            'is_pass': 是否通过验证,
            'execution_result': 执行结果,
            'validation_info': 验证信息
        }
    """
    
    # 1. 准备上下文
    context = prepare_task_context(
        task_id=task_id,
        task_description=task_description,
        modeling_solution=modeling_solution,
        past_results=past_results
    )
    
    # 2. 生成代码（Actor）
    max_attempts = config.get('max_retries', 2)
    
    for attempt in range(max_attempts):
        # 生成代码
        code = generate_task_code(llm, context, attempt)
        
        # 3. 验证代码（Validator）
        validation = validate_code(code)
        
        if not validation['is_valid']:
            # 记录错误，准备重试
            context['validation_errors'] = validation['errors']
            continue
        
        # 4. 执行代码（Executor）
        execution = execute_code(code, config)
        
        if execution['is_pass']:
            # 成功！
            return {
                'task_code': code,
                'is_pass': True,
                'execution_result': execution['result'],
                'validation_info': validation,
                'attempt': attempt + 1
            }
        else:
            # 执行失败，记录错误
            context['execution_errors'] = execution['errors']
    
    # 所有尝试都失败
    return {
        'task_code': code,
        'is_pass': False,
        'execution_result': execution['result'],
        'validation_info': validation,
        'attempt': max_attempts
    }
```

**代码生成提示词结构**：

```python
prompt = f"""
You are implementing Task {task_id} of {total_tasks}.

## Overall Algorithm Solution:
{modeling_solution}

## Current Task Description:
{task_description}

## Previous Task Results:
{format_past_results(past_results)}

## Code Template:
{code_template}

## Requirements:
1. Implement Task {task_id} functionality
2. Integrate with previous tasks (if any)
3. Ensure the llm_solver function is complete and runnable
4. Follow the template structure
5. Include necessary imports and helper functions

## Output Format:
Provide ONLY the complete Python code for solver_task{task_id}.py
Do not include explanations or markdown formatting.

Generate the code:
"""
```

**验证步骤**：

1. **语法验证**
   ```python
   try:
       ast.parse(code)
       syntax_valid = True
   except SyntaxError as e:
       syntax_valid = False
       errors.append(f"Syntax error: {e}")
   ```

2. **结构验证**
   ```python
   # 检查必须存在的函数
   required_functions = ['llm_solver']
   for func in required_functions:
       if func not in code:
           errors.append(f"Missing function: {func}")
   ```

3. **执行验证**
   ```python
   # 实际运行测试用例
   from solver_task_N import llm_solver
   result = test_llm_solver(llm_solver, topo_name, ina_num, jobs_num)
   
   if result['makespan'] > 0 and result['is_feasible']:
       is_pass = True
   ```

**输出**：
- `task_results[task_id]`: 包含代码、执行结果等
- 代码文件：`output_dir/code/solver_task{task_id}.py`
- 结果文件：`output_dir/results/task_{task_id}_result.json`

---

### 7. 代码集成智能体 (Code Integrator)

**职责**：生成最终的完整求解器和解决方案文档

**工作内容**：

1. **提取最终代码**
   ```python
   # 最后一个任务包含完整实现
   final_task_id = execution_order[-1]
   final_code = task_results[final_task_id]['task_code']
   ```

2. **保存求解器**
   ```python
   solver_path = os.path.join(output_dir, 'llm_solver_final.py')
   with open(solver_path, 'w') as f:
       f.write(final_code)
   ```

3. **生成完整解决方案摘要**
   ```json
   {
     "problem_path": "...",
     "total_tasks": 6,
     "algorithm_solution": "...",
     "task_descriptions": [...],
     "execution_order": [1, 2, 3, 4, 5, 6],
     "task_results": {
       "1": {"is_pass": true, "execution_result": "..."},
       "2": {"is_pass": true, "execution_result": "..."},
       ...
     },
     "final_solver_path": ".../llm_solver_final.py",
     "generation_time": "2025-10-31 11:26:28",
     "llm_usage": {
       "prompt_tokens": 150000,
       "completion_tokens": 50000,
       "total_tokens": 200000
     }
   }
   ```

4. **创建 README**
   ```markdown
   # LLMINA Solution Output
   
   ## Overview
   - Problem: ...
   - Generated: ...
   - Total Tasks: 6
   
   ## Directory Structure
   ...
   
   ## Usage
   ...
   ```

**输出文件**：
- `llm_solver_final.py`: 最终求解器
- `complete_solution.json`: 完整解决方案摘要
- `README.md`: 使用说明
- `workflow.log`: 工作流日志

---

## 状态管理

### 状态设计哲学

LangGraph 使用 **增量状态更新** 机制：

```python
# ❌ 错误：返回完整状态
def node(state):
    return {
        'problem_path': state['problem_path'],  # 不需要
        'config': state['config'],              # 不需要
        'new_field': 'new_value'                # 只需要这个
    }

# ✅ 正确：只返回增量
def node(state):
    return {
        'new_field': 'new_value'  # 只返回改变的字段
    }
```

### 状态合并策略

LangGraph 自动合并状态，支持自定义合并函数：

```python
from typing import Annotated

class LLMINAState(TypedDict):
    # 简单字段：直接覆盖
    current_task_id: int
    
    # 列表字段：追加合并
    messages: Annotated[List[Message], merge_lists]
    errors: Annotated[List[str], merge_lists]
    
    # 字典字段：更新合并
    task_results: Annotated[Dict[int, Dict], merge_dicts]
    intermediate_results: Annotated[Dict[str, Any], merge_dicts]
```

### 状态合并示例

```python
# 初始状态
state = {
    'current_task_id': 0,
    'messages': [msg1],
    'task_results': {1: result1}
}

# 节点返回增量
node_output = {
    'current_task_id': 1,
    'messages': [msg2],
    'task_results': {2: result2}
}

# LangGraph 自动合并后
merged_state = {
    'current_task_id': 1,                      # 覆盖
    'messages': [msg1, msg2],                  # 追加
    'task_results': {1: result1, 2: result2}   # 更新
}
```

### 状态访问

```python
def task_solving_node(state: LLMINAState) -> dict:
    # 读取状态
    task_id = state['current_task_id']
    past_results = state.get('task_results', {})
    
    # 处理逻辑
    result = solve_task(task_id, past_results)
    
    # 返回增量更新
    return {
        'current_task_id': task_id + 1,
        'task_results': {task_id: result},
        'messages': [Message(content=f'Task {task_id} done')]
    }
```

---

## 实际案例：网络内聚合问题

### 问题背景

现代分布式机器学习集群同时运行多个训练任务。在每次迭代中，每个 worker 必须向其任务的参数服务器（PS）发送大型梯度张量。可编程交换机可以启用网络内聚合（INA），使得部分和在流量到达 PS 之前在网络内部计算，减少总传输字节数。

### 优化目标

**最小化迭代完成时间（makespan）**，同时满足：
- INA 只能部署在有限数量的交换机上（预算约束 K）
- 每个交换机有处理吞吐量限制
- 网络链路有带宽容量限制
- 每个 worker 只选择一个聚合点
- 最多一跳 INA（worker → INA → PS 或 worker → PS）

### 数学模型

**决策变量**：
- $x_s \in \{0,1\}$: 交换机 $s$ 是否部署 INA
- $y_{jwv} \in \{0,1\}$: worker $w$ 的 job $j$ 是否路由到聚合点 $v$
- $\gamma_j$: job $j$ 的有效聚合速率
- $\alpha$: 辅助变量，$\alpha = 1/t$（makespan）

**目标函数**：
$$\max \alpha$$

**关键约束**：
1. Makespan: $\frac{\gamma_j}{m_j} \geq \alpha, \forall j$
2. INA 预算: $\sum_{s} x_s \leq K$
3. Worker 聚合选择: $\sum_v y_{jwv} = 1$
4. 交换机容量: $\sum_{j,w} \gamma_{jws} \leq C_s \cdot x_s$
5. 链路带宽: $l_e^{w \to s} + l_e^{w \to PS} + l_e^{s \to PS} \leq C_e^{bw}$

### LLMINA 求解过程

#### 第一步：问题澄清

系统分析问题描述后，识别关键信息：
- 网络拓扑类型：FatTree
- 任务数量：并发多个训练任务
- 约束类型：预算、容量、带宽
- 性能指标：makespan

#### 第二步：算法设计

生成两阶段启发式算法：

```markdown
## 算法方案

### 阶段 1：INA 交换机选择
1. 计算每个交换机的中心性得分
   - 考虑 worker 到 PS 的路径覆盖
   - 考虑交换机的处理容量
   - 考虑网络位置（靠近核心 vs 边缘）

2. 贪心选择前 ina_budget个交换机
   - 按得分排序
   - 选择得分最高的 ina_budget个
   - 确保地理分布合理

### 阶段 2：流量分配
1. 对每个 job 的每个 worker：
   - 计算到各 INA 交换机的路径和瓶颈
   - 计算直接到 PS 的路径和瓶颈
   - 选择瓶颈最大的路径

2. 容量验证和调整：
   - 检查交换机处理容量是否超限
   - 检查链路带宽是否超限
   - 若超限，重新分配部分 worker

### 复杂度
- 时间：O(|S|² + |W| × |S|)
- 空间：O(|S| + |W|)
```

#### 第三步：任务分解

系统将算法分解为 6 个任务：

1. **Task 1: 数据读取和初始化**
   - 读取网络拓扑（节点、链路、容量）
   - 解析 job 配置（workers、PS、数据量）
   - 初始化数据结构

2. **Task 2: 网络分析和路径计算**
   - 实现最短路径算法（Dijkstra/Floyd）
   - 计算 worker 到交换机的路径
   - 计算交换机到 PS 的路径
   - 计算路径瓶颈带宽

3. **Task 3: 交换机评分和选择**
   - 实现中心性计算函数
   - 实现交换机评分函数
   - 贪心选择前 ina_budget个交换机

4. **Task 4: 流量分配算法**
   - 实现 worker 到聚合点的分配
   - 考虑容量约束
   - 实现瓶颈最大化策略

5. **Task 5: 容量验证和调整**
   - 实现容量检查函数
   - 实现冲突解决算法
   - 确保所有约束满足

6. **Task 6: Makespan 计算和优化**
   - 计算每个 job 的完成时间
   - 计算整体 makespan
   - 集成所有组件
   - 添加最终优化

#### 第四步：逐任务求解

**Task 1 执行**：
```python
# 生成的 solver_task1.py 包含：

def llm_solver(topo_name, ina_num, jobs_num):
    """Task 1: 数据读取和初始化"""
    
    # 1. 读取拓扑
    topology = load_topology(topo_name)
    nodes = topology['nodes']
    links = topology['links']
    
    # 2. 解析 jobs
    jobs = parse_jobs(jobs_num)
    
    # 3. 初始化数据结构
    graph = build_graph(nodes, links)
    
    # 4. 返回简单可行解（所有 worker 直接连 PS）
    x = {s: 0 for s in switches}  # 不使用 INA
    y = {(j, w, ps[j]): 1 for j, w in all_workers}  # 直接连 PS
    makespan = calculate_direct_makespan(jobs, topology)
    
    return {
        'x': x,
        'y': y,
        'makespan': makespan,
        'info': 'Task 1: Basic initialization'
    }
```

**Task 2 执行**：
```python
# solver_task2.py 在 Task 1 基础上添加：

def llm_solver(topo_name, ina_num, jobs_num):
    """Task 2: 增加路径计算"""
    
    # [Task 1 的所有代码]
    topology = load_topology(topo_name)
    ...
    
    # [Task 2 新增代码]
    
    # 计算所有路径
    paths = {}
    for worker in all_workers:
        for switch in switches:
            path = shortest_path(worker, switch, graph)
            paths[(worker, switch)] = path
        
        path_to_ps = shortest_path(worker, ps[job], graph)
        paths[(worker, ps[job])] = path_to_ps
    
    # 计算瓶颈带宽
    bottlenecks = {}
    for (u, v), path in paths.items():
        bottleneck = min(link_capacity[e] for e in path)
        bottlenecks[(u, v)] = bottleneck
    
    # [返回改进的解]
    ...
```

**Task 3-6 类似**，每个任务在前一个基础上增加功能，最终 `solver_task6.py` 包含完整实现。

#### 第五步：结果

最终生成的 `llm_solver_final.py` 可以直接使用：

```python
from output.code.solver_task6 import llm_solver

result = llm_solver(
    topo_name='FatTree',
    ina_num=5,
    jobs_num=10
)

print(f"Makespan: {result['makespan']}")
print(f"INA Switches: {[s for s, v in result['x'].items() if v == 1]}")
```

---

## 使用方法

### 环境准备

1. **安装依赖**
   ```bash
   pip install -r requirements_langgraph.txt
   ```

2. **配置 LLM**
   ```yaml
   # config.yaml
   model_name: "qwen3-30b-a3b"
   api_key: "your-api-key"
   temperature: 0.7
   ```

3. **准备问题文件**
   ```json
   {
     "background": "问题背景描述...",
     "problem_requirement": "求解要求...",
     "variable_description": {...},
     "problem_formulation": {...}
   }
   ```

### 运行求解器

#### 基础用法

```bash
python MMAgent/main_langgraph.py --task LLMINA
```

#### 高级用法

```bash
# 启用 checkpoint（可恢复执行）
python MMAgent/main_langgraph.py --task LLMINA --checkpoint

# 可视化工作流图
python MMAgent/main_langgraph.py --task LLMINA --viz

# 指定配置文件
python MMAgent/main_langgraph.py --task LLMINA --config custom_config.yaml
```

### 配置选项

```yaml
# config.yaml

# LLM 配置
model_name: "qwen3-30b-a3b"
api_key: "your-api-key"
temperature: 0.7
max_tokens: 4096

# LLMINA 配置
llmina:
  # 问题澄清
  clarification_rounds: 3
  clarification_interactive: false  # true: 交互模式, false: 自动模式
  
  # 算法设计
  modeling_rounds: 2
  
  # 任务分解
  default_task_num: 3  # 建议分解任务数
  
  # 任务求解
  agents:
    task_solving:
      temperature: 0.7
      max_retries: 2  # 每个任务最多重试次数
  
  # Checkpoint
  enable_checkpoints: false
  checkpoint_path: "./checkpoints/llmina.db"

# 输出配置
output:
  save_intermediate: true  # 保存中间结果
  save_logs: true          # 保存详细日志
  verbose: true            # 打印详细信息
```

### 测试生成的求解器

```python
# test_solver.py

import sys
from MMAgent.evaluation import test_llm_solver

# 动态导入生成的求解器
task_id = int(sys.argv[1]) if len(sys.argv) > 1 else 6
module = __import__(f'output.code.solver_task{task_id}', fromlist=['llm_solver'])
llm_solver = module.llm_solver

# 测试
results = test_llm_solver(
    solver_func=llm_solver,
    topo_name='FatTree',
    ina_num_list=[3, 5, 7],
    jobs_num_list=[10, 20, 30],
    instances_num=10
)

# 打印结果
for config, result in results.items():
    print(f"{config}: makespan={result['makespan']:.2f}, "
          f"feasible={result['is_feasible']}")
```

运行测试：
```bash
python test_solver.py 6  # 测试 Task 6 的求解器
```

---

## 输出结果

### 输出目录结构

```
MMAgent/output/LLMINA-LangGraph/LLMINA_20251031-111355/
├── README.md                      # 使用说明
├── modeling_solution.txt          # 算法方案
├── task_descriptions.json         # 任务描述
├── dependency_dag.json            # 依赖关系
├── complete_solution.json         # 完整解决方案摘要
├── llm_solver_final.py            # 最终求解器
│
├── code/                          # 代码文件
│   ├── solver_task1.py            # Task 1 求解器
│   ├── solver_task2.py            # Task 2 求解器（包含 Task 1+2）
│   ├── solver_task3.py            # Task 3 求解器（包含 Task 1+2+3）
│   ├── solver_task4.py
│   ├── solver_task5.py
│   └── solver_task6.py            # Task 6 求解器（完整实现）
│
├── results/                       # 执行结果
│   ├── task_1_result.json         # Task 1 结果
│   ├── task_2_result.json
│   ├── task_3_result.json
│   ├── task_4_result.json
│   ├── task_5_result.json
│   └── task_6_result.json
│
├── logs/                          # 日志文件
│   ├── workflow.log               # 工作流日志
│   └── performance.json           # 性能指标
│
└── usage/                         # LLM 使用统计
    ├── LLMINA.json                # Token 使用统计
    ├── LLMINA_result.json         # 执行结果摘要
    └── runtime.txt                # 运行时间
```

### 关键文件说明

#### 1. modeling_solution.txt

高层次算法方案：

```
算法方案：两阶段启发式算法

阶段 1：INA 交换机选择
- 计算交换机中心性
- 贪心选择前 ina_budget个
- 确保覆盖关键路径

阶段 2：流量分配
- 为每个 worker 选择最优聚合点
- 最大化瓶颈带宽
- 满足容量约束

预期性能：
- 时间复杂度：O(|S|² + |W| × |S|)
- 解质量：80-95% 的最优解
- 适用规模：千级节点
```

#### 2. task_descriptions.json

任务分解详情：

```json
[
  {
    "task_id": 1,
    "title": "数据读取和初始化",
    "description": "读取网络拓扑和 job 配置，初始化数据结构，返回简单可行解",
    "dependencies": [],
    "estimated_complexity": "Low"
  },
  {
    "task_id": 2,
    "title": "路径计算和分析",
    "description": "实现最短路径算法，计算所有必要的路径和瓶颈带宽",
    "dependencies": [1],
    "estimated_complexity": "Medium"
  },
  ...
]
```

#### 3. task_N_result.json

单个任务的执行结果：

```json
{
  "task_id": 3,
  "task_description": "交换机评分和选择",
  "is_pass": true,
  "execution_result": {
    "makespan": 0.0245,
    "is_feasible": true,
    "selected_switches": [5, 12, 18],
    "execution_time": 0.15
  },
  "validation_info": {
    "syntax_valid": true,
    "structure_valid": true,
    "test_passed": true
  },
  "attempt": 1,
  "task_code_path": ".../code/solver_task3.py"
}
```

#### 4. complete_solution.json

完整解决方案摘要：

```json
{
  "problem_path": "/path/to/LLMINA.json",
  "generation_time": "2025-10-31 11:26:28",
  "total_tasks": 6,
  "algorithm_solution": "两阶段启发式算法...",
  "execution_order": [1, 2, 3, 4, 5, 6],
  "task_summaries": [
    {
      "task_id": 1,
      "is_pass": true,
      "makespan": 0.0380
    },
    ...
  ],
  "final_performance": {
    "makespan": 0.0198,
    "improvement": "47.4%",
    "is_feasible": true
  },
  "llm_usage": {
    "prompt_tokens": 145230,
    "completion_tokens": 48650,
    "total_tokens": 193880
  },
  "runtime": "342.56s"
}
```

### 性能指标

#### LLM 使用统计

```json
{
  "total_tokens": 193880,
  "prompt_tokens": 145230,
  "completion_tokens": 48650,
  "cost_estimate": "$2.91",
  "breakdown": {
    "clarification": 15200,
    "algorithm_design": 28400,
    "task_decomposition": 12300,
    "task_solving": 136980,
    "integration": 1000
  }
}
```

#### 执行时间分析

```
Total Runtime: 342.56s

Breakdown:
- Problem Loading:        2.3s  (0.7%)
- Clarification:         18.5s  (5.4%)
- Algorithm Design:      45.2s  (13.2%)
- Task Decomposition:    28.7s  (8.4%)
- Dependency Analysis:    1.2s  (0.4%)
- Task Solving:         238.6s  (69.7%)
  - Task 1:  15.3s
  - Task 2:  28.9s
  - Task 3:  42.1s
  - Task 4:  48.2s
  - Task 5:  55.7s
  - Task 6:  48.4s
- Code Integration:       8.1s  (2.4%)
```

---

## 总结

LLMINA 多智能体系统通过以下方式实现复杂问题的自动化求解：

### 核心优势

1. **问题分解**：将复杂 MILP 问题分解为可管理的子任务
2. **渐进求解**：每个任务在前一任务基础上构建，降低单次生成难度
3. **自动验证**：每个任务都经过语法、结构和执行验证
4. **可追溯性**：完整记录从问题到解决方案的全过程
5. **可扩展性**：基于 LangGraph 的模块化设计，易于扩展

### 适用范围

- ✅ 可建模为 MILP 的优化问题
- ✅ 需要启发式算法的大规模问题
- ✅ 有明确数学模型的问题
- ✅ 可分解为子任务的问题

### 局限性

- ❌ 不保证找到全局最优解（启发式算法）
- ❌ 依赖 LLM 的代码生成能力
- ❌ 需要较长的求解时间（5-10分钟）
- ❌ LLM 调用成本较高（$2-5 per run）

### 未来改进方向

1. **并行化**：支持任务并行求解
2. **增量学习**：从历史求解中学习
3. **人机协作**：支持人工介入和指导
4. **模型库**：积累常见问题的解决方案模板
5. **性能优化**：减少 LLM 调用次数和时间

---

## 附录

### A. 配置文件完整示例

```yaml
# config.yaml - 完整配置示例

# ============ LLM 配置 ============
model_name: "qwen3-30b-a3b"
api_key: "your-api-key-here"
temperature: 0.7
max_tokens: 4096
timeout: 300

# ============ 任务配置 ============
task: "LLMINA"

# ============ LLMINA 配置 ============
llmina:
  # 问题澄清
  clarification_enabled: true
  clarification_rounds: 3
  clarification_interactive: false
  
  # 算法设计
  modeling_rounds: 2
  
  # 任务分解
  default_task_num: 3
  
  # Checkpoint
  enable_checkpoints: false
  checkpoint_path: "./checkpoints/llmina.db"
  
  # Agent 配置
  agents:
    problem_clarification:
      max_rounds: 3
      temperature: 0.7
    
    problem_modeling:
      max_rounds: 2
      temperature: 0.7
    
    task_decomposition:
      default_num: 3
      temperature: 0.7
    
    task_solving:
      temperature: 0.7
      max_retries: 2

# ============ 评估配置 ============
evaluation:
  enable: false
  topo_name: "FatTree"
  ina_num_list: [3, 5, 7]
  jobs_num_list: [10, 20, 30]
  instances_num: 10

# ============ 输出配置 ============
output:
  save_intermediate: true
  save_logs: true
  verbose: true

# ============ 路径配置 ============
paths:
  dataset: "MMBench/dataset"
  problem: "MMBench/problem"
  output: "MMAgent/output"
  checkpoints: "./checkpoints"
  logs: "./logs"
  template_dir: "MMAgent/code_template"

# ============ 监控配置 ============
enable_monitoring: false
log_level: "INFO"
```

### B. 常见问题 (FAQ)

**Q1: 生成的代码不能运行怎么办？**

A: 系统有自动验证和重试机制。如果多次重试仍失败，可以：
1. 检查 `results/task_N_result.json` 中的错误信息
2. 手动修改 `code/solver_task_N.py`
3. 调整配置中的 `max_retries` 参数

**Q2: 如何调整任务数量？**

A: 在 `config.yaml` 中设置：
```yaml
llmina:
  default_task_num: 5  # 建议分解为 5 个任务
```

**Q3: 能否中断后恢复执行？**

A: 启用 checkpoint 功能：
```bash
python main_langgraph.py --task LLMINA --checkpoint
```

**Q4: 如何减少 LLM 调用成本？**

A: 
1. 减少任务数量
2. 减少澄清轮次
3. 减少重试次数
4. 使用更便宜的模型

**Q5: 支持哪些问题类型？**

A: 主要支持可建模为 MILP 的优化问题，特别是：
- 网络优化
- 资源分配
- 调度问题
- 组合优化

---

**文档版本**: 1.0  
**最后更新**: 2025-10-31  
**作者**: LLMINA Development Team  
**联系**: [GitHub Repository](https://github.com/usail-hkust/LLM-MM-Agent)
