"""
输出管理模块

负责将工作流的中间结果和最终结果保存到指定目录，
包括任务描述、代码、执行结果、建模方案等。

目录结构：
output_dir/
├── modeling_solution.txt          # 建模方案
├── task_descriptions.json         # 任务描述列表
├── dependency_dag.json            # 依赖DAG
├── task_dependency_analysis.json  # 依赖分析
├── complete_solution.json         # 完整解决方案
├── code/                          # 代码文件
│   ├── solver_task1.py
│   ├── solver_task2.py
│   └── ...
├── results/                       # 执行结果
│   ├── task_1_result.json
│   ├── task_2_result.json
│   └── ...
└── logs/                          # 日志文件
    ├── workflow.log
    └── performance.json
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


def ensure_output_dirs(output_dir: str) -> Dict[str, str]:
    """
    确保输出目录结构存在
    
    Args:
        output_dir: 输出根目录
        
    Returns:
        目录路径字典
    """
    output_path = Path(output_dir)
    
    # 创建子目录
    dirs = {
        'root': str(output_path),
        'code': str(output_path / 'code'),
        'results': str(output_path / 'results'),
        'logs': str(output_path / 'logs'),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def save_modeling_solution(output_dir: str, modeling_solution: str) -> str:
    """
    保存建模方案
    
    Args:
        output_dir: 输出目录
        modeling_solution: 建模方案文本
        
    Returns:
        保存的文件路径
    """
    file_path = Path(output_dir) / 'modeling_solution.txt'
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modeling_solution)
    
    return str(file_path)


def save_task_descriptions(output_dir: str, task_descriptions: List[str]) -> str:
    """
    保存任务描述列表
    
    Args:
        output_dir: 输出目录
        task_descriptions: 任务描述列表
        
    Returns:
        保存的文件路径
    """
    file_path = Path(output_dir) / 'task_descriptions.json'
    
    data = {
        'total_tasks': len(task_descriptions),
        'tasks': [
            {
                'task_id': i + 1,
                'description': desc
            }
            for i, desc in enumerate(task_descriptions)
        ]
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return str(file_path)


def save_dependency_info(
    output_dir: str,
    dependency_dag: Dict[str, List[str]],
    dependency_analysis: List[str],
    execution_order: List[int]
) -> Dict[str, str]:
    """
    保存依赖信息
    
    Args:
        output_dir: 输出目录
        dependency_dag: 依赖DAG
        dependency_analysis: 依赖分析文本列表
        execution_order: 执行顺序
        
    Returns:
        保存的文件路径字典
    """
    paths = {}
    
    # 保存DAG
    dag_path = Path(output_dir) / 'dependency_dag.json'
    with open(dag_path, 'w', encoding='utf-8') as f:
        json.dump({
            'dag': dependency_dag,
            'execution_order': execution_order
        }, f, indent=2, ensure_ascii=False)
    paths['dag'] = str(dag_path)
    
    # 保存依赖分析
    if dependency_analysis:
        analysis_path = Path(output_dir) / 'task_dependency_analysis.json'
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump({
                'analyses': [
                    {
                        'task_id': i + 1,
                        'analysis': analysis
                    }
                    for i, analysis in enumerate(dependency_analysis)
                ]
            }, f, indent=2, ensure_ascii=False)
        paths['analysis'] = str(analysis_path)
    
    return paths


def save_task_code(output_dir: str, task_id: int, code: str) -> str:
    """
    保存任务代码
    
    Args:
        output_dir: 输出目录
        task_id: 任务ID
        code: 代码内容
        
    Returns:
        保存的文件路径
    """
    code_dir = Path(output_dir) / 'code'
    os.makedirs(code_dir, exist_ok=True)
    
    file_path = code_dir / f'solver_task{task_id}.py'
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code)
    
    return str(file_path)


def save_task_result(
    output_dir: str,
    task_id: int,
    task_description: str,
    result: Dict[str, Any]
) -> str:
    """
    保存任务执行结果
    
    Args:
        output_dir: 输出目录
        task_id: 任务ID
        task_description: 任务描述
        result: 任务结果（包含code, is_pass, execution_result等）
        
    Returns:
        保存的文件路径
    """
    results_dir = Path(output_dir) / 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    file_path = results_dir / f'task_{task_id}_result.json'
    
    # 准备保存的数据（避免保存过大的代码内容到JSON）
    data = {
        'task_id': task_id,
        'task_description': task_description,
        'is_pass': result.get('is_pass', False),
        'execution_result': result.get('execution_result', ''),
        'validation_result': result.get('validation_result', {}),
        'code_path': result.get('task_code_path', ''),
        'timestamp': datetime.now().isoformat()
    }
    
    # 如果执行结果太长，截断
    if isinstance(data['execution_result'], str) and len(data['execution_result']) > 1000:
        data['execution_result'] = data['execution_result'][:1000] + '\n... (truncated)'
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return str(file_path)


def save_complete_solution(
    output_dir: str,
    state: Dict[str, Any]
) -> str:
    """
    保存完整解决方案摘要
    
    Args:
        output_dir: 输出目录
        state: 完整的state字典
        
    Returns:
        保存的文件路径
    """
    file_path = Path(output_dir) / 'complete_solution.json'
    
    # 提取关键信息
    data = {
        'timestamp': datetime.now().isoformat(),
        'problem_path': state.get('problem_path', ''),
        'modeling_solution': state.get('modeling_solution', ''),
        'total_tasks': state.get('total_tasks', 0),
        'execution_order': state.get('execution_order', []),
        'task_summary': [],
        'solver_code_path': state.get('solver_code_path', ''),
        'evaluation_metrics': state.get('evaluation_metrics', {}),
    }
    
    # 添加任务摘要
    task_results = state.get('task_results', {})
    for task_id in sorted(task_results.keys()):
        result = task_results[task_id]
        data['task_summary'].append({
            'task_id': task_id,
            'description': result.get('task_description', ''),
            'is_pass': result.get('is_pass', False),
            'code_path': f'code/solver_task{task_id}.py'
        })
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return str(file_path)


def save_workflow_log(
    output_dir: str,
    event: str,
    data: Dict[str, Any],
    level: str = 'INFO'
) -> None:
    """
    记录工作流日志
    
    Args:
        output_dir: 输出目录
        event: 事件名称
        data: 事件数据
        level: 日志级别 (INFO/WARNING/ERROR)
    """
    logs_dir = Path(output_dir) / 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    
    log_file = logs_dir / 'workflow.log'
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'level': level,
        'event': event,
        'data': data
    }
    
    # 追加写入日志
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')


def save_performance_metrics(
    output_dir: str,
    metrics: Dict[str, Any]
) -> str:
    """
    保存性能指标
    
    Args:
        output_dir: 输出目录
        metrics: 性能指标字典
        
    Returns:
        保存的文件路径
    """
    logs_dir = Path(output_dir) / 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    
    file_path = logs_dir / 'performance.json'
    
    # 如果文件存在，加载并追加
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data['history'] = data.get('history', [])
    else:
        data = {'history': []}
    
    # 添加时间戳
    metrics['timestamp'] = datetime.now().isoformat()
    data['history'].append(metrics)
    data['latest'] = metrics
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return str(file_path)


def create_readme(output_dir: str, state: Dict[str, Any]) -> str:
    """
    创建README文件，说明输出目录结构
    
    Args:
        output_dir: 输出目录
        state: state字典
        
    Returns:
        README文件路径
    """
    readme_path = Path(output_dir) / 'README.md'
    
    total_tasks = state.get('total_tasks', 0)
    
    content = f"""# LLMINA Solution Output

## Overview

This directory contains the complete solution generated by the LLMINA agent workflow.

- **Problem**: {state.get('problem_path', 'N/A')}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Tasks**: {total_tasks}

## Directory Structure

```
.
├── README.md                      # This file
├── modeling_solution.txt          # High-level modeling solution
├── task_descriptions.json         # Detailed task descriptions
├── dependency_dag.json            # Task dependency graph
├── complete_solution.json         # Complete solution summary
├── code/                          # Generated solver code
│   ├── solver_task1.py            # Task 1 solver
│   ├── solver_task2.py            # Task 2 solver (builds on Task 1)
│   └── ...
├── results/                       # Task execution results
│   ├── task_1_result.json         # Task 1 result (pass/fail, metrics)
│   ├── task_2_result.json
│   └── ...
└── logs/                          # Workflow logs
    ├── workflow.log               # Detailed workflow events
    └── performance.json           # Performance metrics
```

## Files

### Core Files

- **modeling_solution.txt**: The high-level algorithmic approach designed by the modeling agent
- **task_descriptions.json**: Breakdown of the problem into {total_tasks} sequential tasks
- **dependency_dag.json**: Task dependencies and execution order
- **complete_solution.json**: Summary of the entire solution

### Code Files

Each `solver_task<N>.py` file contains:
- Complete `llm_solver` function that can be directly evaluated
- Implementation of tasks 1 through N (progressive refinement)
- Helper function imports from evaluation module

The final solver (`solver_task{total_tasks}.py`) contains the complete solution.

### Result Files

Each `task_<N>_result.json` file contains:
- Task description
- Pass/fail status
- Execution results (makespan, constraints validation)
- Code file path

## Usage

### Test a Specific Task

```python
from output.code.solver_task3 import llm_solver
from MMAgent.evaluation import test_llm_solver

results = test_llm_solver(
    solver_func=llm_solver,
    topo_name='FatTree',
    ina_num_list=[3],
    jobs_num_list=[6],
    instances_num=1
)
```

### Use the Test Script

```bash
# Test Task 3
python test_solver.py --task 3

# Run detailed tests
python test_solver.py --task 3 --detailed
```

## Notes

- Each task builds upon previous tasks (progressive refinement approach)
- The final task ({total_tasks}) contains the complete, optimized solution
- All solvers follow the LLMINA interface specification
- Code is validated against problem constraints before being saved

"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return str(readme_path)
