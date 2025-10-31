"""
增强的LLMINA计算求解模块 - 渐进式完整Solver生成
每个任务都生成完整solver，用标准测试框架验证
"""
import os
from utils.utils import save_solution
from agent.llmina_task_solver_clean import solve_task  # 使用通用的渐进式求解器
import json


def computational_solving_enhanced(
    llm, coordinator, problem, task_id, task_descriptions,
    modeling_solution, config, output_dir, task_classification=None, task_strategy=None
):
    """
    增强的计算求解流程（针对分解后的子任务）
    
    流程设计理念：
    - 输入：经过精心分解的子任务（已明确定义、范围清晰）
    - 无需知识库检索（任务已充分明确，不需要模糊匹配）
    - 直接基于任务描述、高层方案和依赖关系进行求解
    
    Args:
        llm: 语言模型
        coordinator: 协调器（管理任务依赖和结果）
        problem: 问题描述
        task_id: 当前任务ID
        task_descriptions: 所有任务描述列表
        modeling_solution: 高层次建模方案
        config: 配置字典
        output_dir: 输出目录
        task_classification: 预分类结果（任务类型和复杂度）
        task_strategy: 预计算策略（求解参数配置）
        
    Returns:
        任务求解结果字典
    """
    print(f"\n{'='*80}")
    print(f"Enhanced Computational Solving: Task {task_id}")
    print(f"{'='*80}")

    # 获取任务上下文
    task_description = task_descriptions[task_id - 1]
    classification = task_classification
    strategy = task_strategy
    
    print(f"  Task Type: {classification['category']}")
    print(f"  Complexity: {classification['complexity']}")
    print(f"  Strategy: {strategy['code_generation_strategy']}")
    
    # 准备求解配置
    solving_config = {
        'work_dir': os.path.join(output_dir, 'code'),
        'max_debug_iterations': strategy.get('max_debug_iterations', 5),
        **config
    }
    
    # 执行求解 - 调用新的solve_task方法
    try:
        # 兼容无 Coordinator 的执行：从 coordinator 提取依赖信息，若无则使用顺序执行默认值
        dependency_dag = getattr(coordinator, 'DAG', {}) if coordinator is not None else {}
        dependency_analysis = getattr(coordinator, 'task_dependency_analysis', []) if coordinator is not None else []
        past_results = getattr(coordinator, 'memory', {}) if coordinator is not None else {}

        result = solve_task(
            llm=llm,
            task_id=task_id,
            total_tasks=len(task_descriptions),
            task_description=task_description,
            modeling_solution=modeling_solution,
            dependency_dag=dependency_dag,
            dependency_analysis=dependency_analysis,
            past_results=past_results,
            config=solving_config
        )
                        
        # ========================================================================
        # Stage 2: 结果验证和保存
        # ========================================================================
        print("\n[Stage 2] Validation and Result Saving")
        code_file_path = _save_and_update_results(
            coordinator, task_id, task_description, result, output_dir
        )
        
        # 将代码文件路径添加到结果中
        result['task_code_path'] = code_file_path
        
        # 打印结果摘要
        _print_result_summary(task_id, result)
        
        return result
        
    except Exception as e:
        print(f"\n[Error] Task solving failed: {e}")
        import traceback
        traceback.print_exc()
        
        # 返回失败结果（只包含必要字段）
        return {
            'task_code': f"# Error during code generation\n# {str(e)}",
            'is_pass': False,
            'execution_result': str(e),
            'validation_result': {'is_valid': False, 'severity': 'critical', 'error': str(e)}
        }


def _save_and_update_results(coordinator, task_id, task_description, result, output_dir):
    """
    保存求解结果并更新协调器
    
    在渐进式求解模式下：
    - 保存完整的 llm_solver 代码（包含所有前置任务的实现）
    - 记录任务描述和执行状态
    - 不提取代码结构（因为每个任务都只是同一个 llm_solver 函数的演进版本）
    
    Returns:
        code_file: 保存的代码文件路径
    """
    # 保存代码文件
    code_dir = os.path.join(output_dir, 'code')
    os.makedirs(code_dir, exist_ok=True)
    
    code_file = os.path.join(code_dir, f'task_{task_id}.py')
    with open(code_file, 'w', encoding='utf-8') as f:
        f.write(result['task_code'])
    print(f"  Code saved to: {code_file}")
    
    # 构建任务结果字典（简化版，只保存关键信息）
    task_dict = {
        'task_id': task_id,
        'task_description': task_description,
        'task_code': result['task_code'],
        'code_file_path': code_file,
        'is_pass': result['is_pass'],
        'execution_result': result.get('execution_result', ''),
        'validation_result': result.get('validation_result', {})
    }
    
    # 更新协调器内存（供后续任务查看依赖状态）
    coordinator.memory[str(task_id)] = task_dict
    
    # 保存任务结果到JSON（便于分析和调试）
    result_file = os.path.join(output_dir, f'task_{task_id}_result.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(task_dict, f, indent=2, ensure_ascii=False)
    print(f"  Result metadata saved to: {result_file}")
    
    print(f"  Task {task_id} stored in coordinator.memory (for dependency reference)")
    
    return code_file

def _print_result_summary(task_id, result):
    """打印任务求解结果摘要"""
    print(f"\n{'='*80}")
    print(f"Task {task_id} Completion Summary")
    print(f"{'='*80}")
    
    # 执行状态
    print(f"Execution: {'✓ PASSED' if result['is_pass'] else '✗ FAILED'}")
    
    # 验证结果
    validation = result.get('validation_result', {})
    if validation and isinstance(validation, dict):
        severity = validation.get('severity', 'unknown').upper()
        print(f"Validation: {severity}")
        
        # 只显示存在的验证问题
        if validation.get('syntax_errors'):
            print(f"  - Syntax Errors: {len(validation['syntax_errors'])}")
        if validation.get('compatibility_issues'):
            print(f"  - Compatibility Issues: {len(validation['compatibility_issues'])}")
        if validation.get('potential_runtime_errors'):
            print(f"  - Potential Runtime Errors: {len(validation['potential_runtime_errors'])}")
        if validation.get('error'):
            print(f"  - Error: {validation['error']}")
    
    # 执行结果摘要
    execution_result = result.get('execution_result', '')
    if execution_result and len(execution_result) < 200:
        print(f"Result: {execution_result}")
    
    print(f"{'='*80}\n")

