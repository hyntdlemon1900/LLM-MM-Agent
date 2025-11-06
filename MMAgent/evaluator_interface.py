"""
评估器接口 - 用于分离系统代码和用户评估代码

核心思想：
- 智能体系统不应该依赖具体的评估实现
- 用户通过实现标准接口提供评估函数
- 系统通过动态加载机制调用用户代码

使用方式：
1. 用户在问题文件夹中提供 evaluation.py
2. evaluation.py 需要实现 test_solver 函数和相关辅助函数
3. 系统通过 load_evaluator() 动态加载用户的评估模块
"""

import os
import sys
import importlib.util
from typing import Callable, Dict, Any, List


def load_evaluator(problem_dir: str) -> Dict[str, Callable]:
    """
    动态加载用户提供的评估模块
    
    Args:
        problem_dir: 问题文件夹路径（例如：MMBench/problem/LLMINA）
        
    Returns:
        包含评估函数的字典：
        {
            'test_solver': 测试solver的函数,
            'helper_functions': 辅助函数字典（如 get_ina_candidates, evaluate_completion_time等）,
            'runtime_dir': runtime目录路径（用于导入）
        }
        
    Raises:
        FileNotFoundError: 如果找不到 evaluation.py
        ImportError: 如果导入失败
        ValueError: 如果缺少必需的函数
    """
    # 首先检查 runtime 子目录
    runtime_dir = os.path.join(problem_dir, 'runtime')
    evaluation_path = os.path.join(runtime_dir, 'evaluation.py')
    
    # 如果 runtime 目录不存在，尝试直接在 problem_dir 查找
    if not os.path.exists(evaluation_path):
        evaluation_path = os.path.join(problem_dir, 'evaluation.py')
        runtime_dir = problem_dir  # 没有 runtime 子目录，使用问题目录本身
    
    if not os.path.exists(evaluation_path):
        raise FileNotFoundError(
            f"User evaluation module not found: {evaluation_path}\n"
            f"Please provide an evaluation.py file in the problem directory or runtime subdirectory."
        )
    
    # 动态加载模块
    module_name = f"user_evaluation_{os.path.basename(problem_dir)}"
    spec = importlib.util.spec_from_file_location(module_name, evaluation_path)
    evaluation_module = importlib.util.module_from_spec(spec)
    
    # 添加 runtime 目录到 sys.path，以便 evaluation.py 可以导入同目录的其他模块
    if runtime_dir not in sys.path:
        sys.path.insert(0, runtime_dir)
    
    # 添加问题目录到 sys.path
    if problem_dir not in sys.path:
        sys.path.insert(0, problem_dir)
    
    # 添加 MMAgent 到 sys.path，以便 evaluation.py 可以导入 MMAgent 的模块
    mmagent_dir = os.path.dirname(os.path.dirname(evaluation_path))
    mmragent_parent = os.path.dirname(mmagent_dir)
    if mmragent_parent not in sys.path:
        sys.path.insert(0, mmragent_parent)
    
    try:
        spec.loader.exec_module(evaluation_module)
    except Exception as e:
        raise ImportError(f"Failed to load evaluation module: {e}")
    
    # 验证必需的函数
    required_test_func = 'test_llm_solver'  # 测试函数名
    if not hasattr(evaluation_module, required_test_func):
        raise ValueError(
            f"Evaluation module must provide '{required_test_func}' function.\n"
            f"Expected signature: {required_test_func}(solver_func, topo_name, ina_num_list, jobs_num_list, instances_num)"
        )
    
    # 收集辅助函数（用于代码生成时的参考）
    helper_function_names = [
        'get_ina_candidates',
        'evaluate_completion_time', 
        'links_set',
        'path_to_links_dict',
        'l_i',
        'link_capacity_limitation'
    ]
    
    helper_functions = {}
    for func_name in helper_function_names:
        if hasattr(evaluation_module, func_name):
            helper_functions[func_name] = getattr(evaluation_module, func_name)
    
    print(f"✓ Loaded user evaluation module from: {evaluation_path}")
    print(f"  - Test function: {required_test_func}")
    print(f"  - Helper functions: {list(helper_functions.keys())}")
    print(f"  - Runtime directory: {runtime_dir}")
    
    return {
        'test_solver': getattr(evaluation_module, required_test_func),
        'helper_functions': helper_functions,
        'module': evaluation_module,  # 保留模块引用，以便访问其他内容
        'runtime_dir': runtime_dir  # 返回 runtime 目录路径
    }


def get_helper_functions_code(problem_dir: str) -> str:
    """
    读取辅助函数的源代码（用于prompt生成）
    
    Args:
        problem_dir: 问题文件夹路径
        
    Returns:
        辅助函数的源代码字符串
    """
    # 首先检查 runtime 子目录
    runtime_dir = os.path.join(problem_dir, 'runtime')
    evaluation_path = os.path.join(runtime_dir, 'evaluation.py')
    
    # 如果 runtime 目录不存在，尝试直接在 problem_dir 查找
    if not os.path.exists(evaluation_path):
        evaluation_path = os.path.join(problem_dir, 'evaluation.py')
    
    if not os.path.exists(evaluation_path):
        return "# Helper functions not available"
    
    try:
        with open(evaluation_path, 'r', encoding='utf-8') as f:
            full_code = f.read()
        
        # 提取辅助函数部分（可以通过标记或函数名识别）
        # 这里简单返回整个文件，实际使用时可以更精细地提取
        lines = full_code.split('\n')
        
        # 查找辅助函数的开始位置
        helper_start = -1
        for i, line in enumerate(lines):
            if 'def get_ina_candidates' in line or 'def links_set' in line:
                helper_start = i
                break
        
        if helper_start >= 0:
            # 提取从第一个辅助函数到文件末尾的所有函数
            # 但排除 test_llm_solver 和类定义
            helper_lines = []
            in_helper = False
            for i in range(helper_start, len(lines)):
                line = lines[i]
                # 跳过类定义和测试函数
                if 'class ' in line or 'def test_llm_solver' in line:
                    in_helper = False
                    continue
                if line.strip().startswith('def ') and any(
                    func in line for func in [
                        'get_ina_candidates', 'links_set', 'path_to_links_dict',
                        'l_i', 'link_capacity_limitation', 'evaluate_completion_time',
                        'validate_solution_constraints'
                    ]
                ):
                    in_helper = True
                
                if in_helper:
                    helper_lines.append(line)
            
            return '\n'.join(helper_lines) if helper_lines else full_code
        else:
            return full_code
            
    except Exception as e:
        print(f"Warning: Failed to read helper functions code: {e}")
        return "# Helper functions code unavailable"


def wrap_solver_with_imports(llm_solver_code: str, problem_dir: str) -> str:
    """
    将生成的 llm_solver 函数包装上必要的导入语句
    
    Args:
        llm_solver_code: LLM生成的 solver 函数代码
        problem_dir: 问题文件夹路径（绝对路径）
        
    Returns:
        完整的可执行代码（包含导入）
    """
    # 检查是否有 runtime 子目录
    abs_problem_dir = os.path.abspath(problem_dir)
    runtime_dir = os.path.join(abs_problem_dir, 'runtime')
    
    # 如果 runtime 目录存在且包含 evaluation.py，使用 runtime 目录
    if os.path.exists(os.path.join(runtime_dir, 'evaluation.py')):
        import_dir = runtime_dir
    else:
        import_dir = abs_problem_dir
    
    imports = f"""# Auto-generated solver with helper function imports
import sys
from pathlib import Path
import copy

# Add problem directory to path (using absolute path)
problem_dir = r'{import_dir}'
if problem_dir not in sys.path:
    sys.path.insert(0, problem_dir)

# Import helper functions from user's evaluation module
from evaluation import (
    get_ina_candidates,
    evaluate_completion_time,
    links_set,
    path_to_links_dict,
    l_i,
    link_capacity_limitation
)

"""
    
    # 组合：导入 + LLM生成的solver函数
    complete_code = imports + "\n" + llm_solver_code
    return complete_code


# ============================================================================
# 标准接口定义（文档化）
# ============================================================================

"""
用户评估模块需要实现以下接口：

1. test_llm_solver(solver_func, topo_name, ina_num_list, jobs_num_list, instances_num)
   - 输入: solver_func (可调用对象), 测试配置参数
   - 输出: Dict[Tuple, List[float]] - 测试结果字典
   - 功能: 在多个测试实例上评估 solver 函数，返回性能指标

2. get_ina_candidates(network) -> List[int]
   - 输入: network 对象
   - 输出: INA候选交换机ID列表（有序）
   - 功能: 根据拓扑类型返回可部署INA的交换机列表

3. evaluate_completion_time(instance, network, ina_placement, jobs_routing, verbose, ina_capacity, ps_bandwidth) -> float
   - 输入: 实例数据、网络、INA部署、作业路由、参数
   - 输出: makespan（完成时间）
   - 功能: 评估给定方案的makespan

4. 其他辅助函数：links_set, path_to_links_dict, l_i, link_capacity_limitation
   - 这些函数用于路径和链路容量计算
   
示例 evaluation.py 文件结构：

```python
# evaluation.py (用户提供)

from MMAgent.topo import FatTree, SpineLeaf
from MMAgent.dataset import generate_dataset_job_random_fattree

def test_llm_solver(solver_func, topo_name='FatTree', ina_num_list=[3], jobs_num_list=[10], instances_num=10):
    # 实现测试逻辑
    ...
    return results_dict

def get_ina_candidates(network):
    # 实现候选列表生成
    ...
    return sorted_candidates

def evaluate_completion_time(instance, network, ina_placement, jobs_routing, ...):
    # 实现makespan计算
    ...
    return makespan

# 其他辅助函数...
```
"""
