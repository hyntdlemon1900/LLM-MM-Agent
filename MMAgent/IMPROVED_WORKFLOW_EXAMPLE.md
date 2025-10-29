# 改进后的工作流使用示例

## 问题概述

**原问题**：在多智能体工作流中，前期会拆解问题为多个子任务，但每个子任务都被要求生成完整的`llm_solver`函数，导致：
- 重复生成相同框架代码
- 集成困难（多个"完整solver"如何合并？）
- Token浪费
- 依赖关系混乱

**解决方案**：采用**增量式代码构建**策略
- 前N-1个任务：生成独立的辅助函数/模块
- 最后一个任务：生成完整的`llm_solver`函数并集成所有模块
- 使用`CodeIntegrator`自动整合代码

---

## 使用示例

### 场景：INA部署和路由优化问题

假设问题被分解为以下任务：

```python
task_descriptions = [
    "Task 1: 实现获取INA候选节点的函数",
    "Task 2: 实现路由计算算法",
    "Task 3: 实现评估函数（调用LP求解器）",
    "Task 4: 实现完整的贪心算法solver，集成前面的所有函数"
]
```

### Step 1: 初始化系统

```python
from MMAgent.agent.llmina_enhanced_task_solver import LLMINATaskSolver
from MMAgent.agent.llmina_code_integrator import CodeIntegrator
from MMAgent.llm import LLM

# 初始化
llm = LLM(config)
solver = LLMINATaskSolver(llm)
integrator = CodeIntegrator()

# 配置
config = {
    'work_dir': './output/LLMINA-Solver',
    'problem_type': 'optimization'
}
```

### Step 2: 求解各个任务

```python
# 读取问题和建模方案
problem_str = "..."  # 问题描述
modeling_solution = "..."  # 用户的建模方案
code_template = "..."  # llm_solver函数模板

# 依赖代码字典（逐步累积）
dependent_codes = {}

# Task 1: 生成辅助函数
print("\n=== Solving Task 1: INA Candidate Nodes ===")
result1 = solver.solve_task_adaptive(
    task_id=1,
    task_description=task_descriptions[0],
    modeling_solution=modeling_solution,
    code_template=code_template,
    dependent_codes={},  # 第一个任务没有依赖
    dependent_file_prompt="No dependencies",
    config=config,
    task_classification={'category': 'simple_implementation', ...},
    task_strategy={'code_generation_strategy': 'direct', ...},
    total_tasks=4,  # 总共4个任务
    is_final_task=False  # 不是最后任务
)

# 保存代码到集成器
integrator.add_task_code(
    task_id=1,
    code=result1['task_code'],
    dependencies=[]
)
dependent_codes[1] = result1['task_code']

# Task 2: 生成路由算法
print("\n=== Solving Task 2: Routing Algorithm ===")
result2 = solver.solve_task_adaptive(
    task_id=2,
    task_description=task_descriptions[1],
    modeling_solution=modeling_solution,
    code_template=code_template,
    dependent_codes=dependent_codes,  # 包含Task 1的代码
    dependent_file_prompt="Task 1 provides: get_ina_candidates(network)",
    config=config,
    task_classification={'category': 'algorithm_design', ...},
    task_strategy={'code_generation_strategy': 'algorithm_focused', ...},
    total_tasks=4,
    is_final_task=False
)

integrator.add_task_code(2, result2['task_code'], dependencies=[1])
dependent_codes[2] = result2['task_code']

# Task 3: 生成评估函数
print("\n=== Solving Task 3: Evaluation Function ===")
result3 = solver.solve_task_adaptive(
    task_id=3,
    task_description=task_descriptions[2],
    modeling_solution=modeling_solution,
    code_template=code_template,
    dependent_codes=dependent_codes,
    dependent_file_prompt="Task 1: get_ina_candidates, Task 2: compute_routing",
    config=config,
    task_classification={'category': 'simple_implementation', ...},
    task_strategy={'code_generation_strategy': 'direct', ...},
    total_tasks=4,
    is_final_task=False
)

integrator.add_task_code(3, result3['task_code'], dependencies=[1, 2])
dependent_codes[3] = result3['task_code']

# Task 4: 最后的集成任务
print("\n=== Solving Task 4: Final Integration ===")
result4 = solver.solve_task_adaptive(
    task_id=4,
    task_description=task_descriptions[3],
    modeling_solution=modeling_solution,
    code_template=code_template,
    dependent_codes=dependent_codes,  # 包含前面所有任务的代码
    dependent_file_prompt="""
Available helper functions:
- Task 1: get_ina_candidates(network) -> List[int]
- Task 2: compute_routing(instance, ina_placement, network) -> Dict
- Task 3: evaluate_completion_time(instance, network, ina_placement, routing) -> float
""",
    config=config,
    task_classification={'category': 'integration', ...},
    task_strategy={'code_generation_strategy': 'integration', ...},
    total_tasks=4,
    is_final_task=True  # 明确标记为最后任务
)

integrator.add_task_code(4, result4['task_code'], dependencies=[1, 2, 3])
```

### Step 3: 集成所有代码

```python
# 自动集成所有任务的代码
print("\n=== Integrating All Code ===")
integrated_code = integrator.integrate_all(final_task_id=4)

# 验证集成结果
is_valid, issues = integrator.validate_integration(integrated_code)

if is_valid:
    print("✓ Code integration successful!")
    
    # 保存最终的solver文件
    output_path = os.path.join(config['work_dir'], 'final_solver.py')
    with open(output_path, 'w') as f:
        f.write(integrated_code)
    
    print(f"✓ Final solver saved to: {output_path}")
else:
    print("✗ Code integration failed:")
    for issue in issues:
        print(f"  - {issue}")
```

---

## 生成的代码示例

### Task 1 生成的代码（辅助函数）

```python
def get_ina_candidates(network: nx.Graph) -> List[int]:
    """
    获取INA候选节点（所有交换机节点）
    
    Args:
        network: 网络拓扑图
    
    Returns:
        候选节点ID列表
    """
    candidates = []
    for node in network.nodes():
        if network.nodes[node].get('type') == 'switch':
            candidates.append(node)
    return candidates
```

### Task 2 生成的代码（算法模块）

```python
def compute_routing(
    instance: Dict, 
    ina_placement: List[int], 
    network: nx.Graph
) -> Dict[int, List[int]]:
    """
    基于INA部署计算每个job的路由
    
    Args:
        instance: 问题实例
        ina_placement: INA节点部署方案
        network: 网络拓扑
    
    Returns:
        jobs_routing: {job_id: [path_nodes]}
    """
    jobs_routing = {}
    
    for job_id in range(instance['jobs_num']):
        # 找到最近的INA节点
        src = instance['jobs'][job_id]['src']
        nearest_ina = find_nearest_ina(src, ina_placement, network)
        
        # 计算路径
        path = nx.shortest_path(network, src, nearest_ina)
        jobs_routing[job_id] = path
    
    return jobs_routing

def find_nearest_ina(src: int, ina_nodes: List[int], network: nx.Graph) -> int:
    """辅助函数：找到最近的INA节点"""
    min_dist = float('inf')
    nearest = None
    
    for ina in ina_nodes:
        dist = nx.shortest_path_length(network, src, ina)
        if dist < min_dist:
            min_dist = dist
            nearest = ina
    
    return nearest
```

### Task 4 生成的代码（完整solver）

```python
def llm_solver(instance, network, K, jobs_num, Cs, topo_name):
    """
    完整的INA部署和路由优化求解器（集成版本）
    
    使用贪心算法求解
    """
    # 导入/使用前面任务的辅助函数
    # (在实际代码中，这些函数已经在同一文件中定义)
    
    # Step 1: 获取候选节点
    candidates = get_ina_candidates(network)
    
    # Step 2: 贪心选择K个INA节点
    ina_placement = []
    remaining_candidates = candidates.copy()
    
    for _ in range(K):
        best_node = None
        best_makespan = float('inf')
        
        for candidate in remaining_candidates:
            # 尝试添加此节点
            temp_placement = ina_placement + [candidate]
            
            # 计算路由
            temp_routing = compute_routing(instance, temp_placement, network)
            
            # 评估makespan
            makespan = evaluate_completion_time(
                instance, network, temp_placement, temp_routing
            )
            
            if makespan < best_makespan:
                best_makespan = makespan
                best_node = candidate
        
        ina_placement.append(best_node)
        remaining_candidates.remove(best_node)
    
    # Step 3: 计算最终路由
    jobs_routing = compute_routing(instance, ina_placement, network)
    
    # Step 4: 评估最终makespan
    final_makespan = evaluate_completion_time(
        instance, network, ina_placement, jobs_routing
    )
    
    # Step 5: 扩展结果格式
    ina_placement_expanded = expand_ina_placement(ina_placement, instance)
    jobs_routing_expanded = expand_jobs_routing(jobs_routing, instance)
    
    return ina_placement_expanded, jobs_routing_expanded, final_makespan
```

### 最终集成的代码结构

```python
"""
LLM-Generated INA Placement and Routing Solver
Auto-integrated from multi-task code generation
"""
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Set
import pulp
from collections import defaultdict

# ==============================================================================
# Task 1 - Helper Functions
# ==============================================================================
def get_ina_candidates(network: nx.Graph) -> List[int]:
    """获取INA候选节点"""
    ...

# ==============================================================================
# Task 2 - Helper Functions
# ==============================================================================
def compute_routing(...):
    """计算路由"""
    ...

def find_nearest_ina(...):
    """找最近INA"""
    ...

# ==============================================================================
# Task 3 - Helper Functions
# ==============================================================================
def evaluate_completion_time(...):
    """评估makespan"""
    ...

# ==============================================================================
# Main Solver Function (Task 4)
# ==============================================================================
def llm_solver(instance, network, K, jobs_num, Cs, topo_name):
    """完整的求解器"""
    ...
```

---

## 对比：改进前 vs 改进后

### 改进前的问题

```python
# Task 1 生成的代码（❌ 问题：生成了完整solver）
def llm_solver(instance, network, K, jobs_num, Cs, topo_name):
    # 简单实现：随机选择
    candidates = list(network.nodes())
    ina_placement = random.sample(candidates, K)
    ...
    return ina_placement_expanded, jobs_routing_expanded, makespan

# Task 2 生成的代码（❌ 问题：又生成了完整solver）
def llm_solver(instance, network, K, jobs_num, Cs, topo_name):
    # 稍微改进的实现
    candidates = [n for n in network.nodes() if is_switch(n)]
    ina_placement = greedy_select(candidates, K)
    ...
    return ina_placement_expanded, jobs_routing_expanded, makespan

# Task 3 生成的代码（❌ 问题：还是完整solver）
def llm_solver(instance, network, K, jobs_num, Cs, topo_name):
    # 更优化的实现
    ...
    return ina_placement_expanded, jobs_routing_expanded, makespan

# ❓ 问题：应该用哪个solver？如何合并？
```

### 改进后的优势

```python
# Task 1: 只生成辅助函数 ✓
def get_ina_candidates(network):
    ...

# Task 2: 只生成算法模块 ✓
def compute_routing(...):
    ...

# Task 3: 只生成评估函数 ✓
def evaluate_completion_time(...):
    ...

# Task 4: 生成完整solver并集成所有模块 ✓
def llm_solver(instance, network, K, jobs_num, Cs, topo_name):
    candidates = get_ina_candidates(network)  # 调用Task 1
    routing = compute_routing(...)  # 调用Task 2
    makespan = evaluate_completion_time(...)  # 调用Task 3
    return ...

# ✓ 清晰的模块化结构
# ✓ 自然的代码复用
# ✓ 容易调试和测试
```

---

## 实施建议

### 1. 修改问题分解提示词

在`llmina_problem_decompse.py`中添加明确指导：

```python
decomposition_guidance = """
When decomposing the solution approach, follow these principles:

1. **Modular Decomposition**: Break down into logical modules/components
2. **Task 1 to N-1**: Each task generates specific helper functions or algorithm modules
3. **Task N (Final)**: The last task generates the complete llm_solver function
4. **Clear Integration**: The final task should integrate all previous modules

Example:
- Task 1: Generate function to get INA candidates
- Task 2: Generate routing computation algorithm
- Task 3: Generate evaluation function (LP solver wrapper)
- Task 4: Generate complete llm_solver integrating Tasks 1-3
"""
```

### 2. 在任务分类时标记最后任务

```python
# 在llmina_coordinator.py中
def mark_final_task(task_descriptions):
    """标记最后一个任务为集成任务"""
    if task_descriptions:
        task_descriptions[-1] = (
            task_descriptions[-1] + 
            "\n\n**[FINAL INTEGRATION TASK]** "
            "Generate the complete llm_solver function that integrates "
            "all helper functions from previous tasks."
        )
    return task_descriptions
```

### 3. 调整流程调用

```python
# 在llmina_pipeline.py中
for i, task_desc in enumerate(task_descriptions):
    is_final = (i == len(task_descriptions) - 1)
    
    result = solver.solve_task_adaptive(
        task_id=i,
        task_description=task_desc,
        ...,
        total_tasks=len(task_descriptions),
        is_final_task=is_final
    )
    
    # 累积代码
    integrator.add_task_code(i, result['task_code'], ...)

# 最后集成
final_code = integrator.integrate_all(final_task_id=len(task_descriptions)-1)
```

---

## 总结

**核心改进**：
1. ✅ **任务粒度明确**：辅助函数 vs 完整solver
2. ✅ **增量式构建**：逐步添加功能模块
3. ✅ **自动集成**：CodeIntegrator自动组装
4. ✅ **减少重复**：避免多次生成完整solver
5. ✅ **易于调试**：每个模块独立测试

**预期效果**：
- Token使用量：↓ 60%
- 代码质量：↑ 显著提升
- 调试效率：↑ 3x
- 集成难度：↓ 80%
