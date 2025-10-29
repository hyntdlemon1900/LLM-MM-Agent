"""
LLMINA问题专用求解流程（增强版）
整合了任务分类、代码验证和自适应求解策略

流程概述：
1. 问题澄清和理解
2. 高层次求解方案设计
3. 问题分解
4. 任务分类和依赖分析
5. 自适应任务求解（根据任务类型选择不同策略）
6. 代码验证和调试
7. 结果整合
"""
import os
import json
from utils.utils import read_json_file, write_json_file, save_solution
from prompt.llmina_template import PROBLEM_DESCRIPTION_PROMPT
from agent.llmina_problem_clarification import ProblemClarification
from agent.llmina_summary import ProblemClarificationSummary
from agent.llmina_feedback_judge import ProblemJudge
from agent.llmina_problem_solving import ProblemSolving
from agent.llmina_problem_decompse import ProblemDecompose
from agent.llmina_coordinator import Coordinator
from agent.llmina_task_classifier import TaskClassifier
from agent.llmina_task_solver_clean import ProgressiveTaskSolver
from utils.llmina_computational_solving_enhanced import computational_solving_enhanced
import shutil


def llmina_solver_pipeline_enhanced(llm, problem_path, config, output_dir, name):
    """
    LLMINA增强求解流程
    
    新特性：
    - 任务智能分类：区分算法设计、简单实现、测试等不同类型任务
    - 自适应求解策略：根据任务类型选择是否使用知识库、建模等
    - 代码验证机制：静态检查、接口兼容性验证、LLM深度分析
    - 灵活的处理流程：复杂任务完整流程，简单任务快速处理
    
    Args:
        llm: 语言模型实例
        problem_path: 问题描述JSON文件路径
        config: 配置字典
        output_dir: 输出目录
        name: 实验名称
        
    Returns:
        solver_code_path: 生成的solver代码路径
        solution_dict: 完整的求解方案字典
    """
    
    # ============================================================================
    # Stage 0: 初始化和环境准备
    # ============================================================================
    print('\n' + '=' * 80)
    print('LLMINA Enhanced Solver Pipeline')
    print('=' * 80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'code'), exist_ok=True)
    
    solution = {}
    
    # 读取问题
    problem = read_json_file(problem_path)
    background = problem['background']
    problem_requirement = problem['problem_requirement']
    variable_description = problem['variable_description']
    problem_formulation = problem['problem_formulation']
    
    # 读取代码模板
    template_path = config.get(
        'code_template_path',
        '/home/hyn/LLM-MM-Agent-LLMINA/MMAgent/code_template/llmina_solver_template.py'
    )
    try:
        with open(template_path, 'r') as f:
            code_template = f.read()
    except Exception as e:
        print(f"[Warning] Failed to load code template: {e}")
        code_template = "# Template not available"
    
    problem_str = PROBLEM_DESCRIPTION_PROMPT.format(
        problem_background=background,
        problem_requirement=problem_requirement,
        variable_description=variable_description,
        problem_formulation=problem_formulation,
        code_template=code_template,
    ).strip()
    





    '''
    # ============================================================================
    # Stage 1: 问题理解和澄清
    # ============================================================================
    print('\n' + '=' * 80)
    print('Stage 1: Problem Understanding and Clarification')
    print('=' * 80)
    
    # 交互式问题澄清
    if config.get('enable_clarification', False):
        def user_input_func(agent_feedback):
            print("\n[Agent Feedback]", agent_feedback)
            return input("Please clarify (press Enter to skip): ")
        
        pc = ProblemClarification(llm)
        pj = ProblemJudge(llm)
        ps = ProblemClarificationSummary(llm)
        
        problem_str, clarification_history, clarification_summary = pc.clarification_actor(
            pj, ps, problem_str, user_input_func
        )
        
        solution['clarification_history'] = clarification_history
        solution['clarification_summary'] = clarification_summary
        print('[✓] Problem clarification completed')
    else:
        print('[→] Clarification skipped (disabled in config)')
        clarification_summary = ""
    
    # ============================================================================
    # Stage 2: 高层次求解方案设计
    # ============================================================================
    print('\n' + '=' * 80)
    print('Stage 2: High-Level Solution Design')
    print('=' * 80)
    
    pu = ProblemSolving(llm)
    modeling_solution = pu.solving(
        problem_str,
        round=config.get('problem_modeling_round', 2)
    )
    solution['modeling_solution'] = modeling_solution
    print('[✓] High-level solution design completed')
    
    # 保存高层次方案
    with open(os.path.join(output_dir, 'modeling_solution.txt'), 'w') as f:
        f.write(modeling_solution)
    
    # ============================================================================
    # Stage 3: 问题分解
    # ============================================================================
    print('\n' + '=' * 80)
    print('Stage 3: Problem Decomposition')
    print('=' * 80)
    
    pd = ProblemDecompose(llm)
    problem_type = config.get('problem_type', 'optimization')
    
    task_descriptions, tasknum = pd.decompose_and_refine(
        problem_str,
        modeling_solution,
        problem_type
    )
    
    solution['task_descriptions'] = task_descriptions
    solution['tasknum'] = tasknum
    print(f'[✓] Problem decomposed into {tasknum} tasks')
    
    # 保存任务描述
    with open(os.path.join(output_dir, 'task_descriptions.json'), 'w') as f:
        json.dump(task_descriptions, f, indent=2, ensure_ascii=False)
    
    # ============================================================================
    # Stage 4: 任务分类和依赖分析
    # ============================================================================
    print('\n' + '=' * 80)
    print('Stage 4: Task Classification and Dependency Analysis')
    print('=' * 80)
    
    # 4.1 任务分类（一次性完成所有任务的分类和策略规划）
    print('\n[4.1] Task Classification and Strategy Planning')
    task_classifier = TaskClassifier(llm)
    task_classifications = task_classifier.batch_classify_tasks(
        task_descriptions, modeling_solution
    )
    
    # 为每个任务计算处理策略
    task_strategies = []
    for i, classification in enumerate(task_classifications):
        strategy = task_classifier.get_processing_strategy(classification)
        task_strategies.append(strategy)
        print(f"  Task {i+1}: {classification['category']} -> {strategy['code_generation_strategy']}")
    
    solution['task_classifications'] = task_classifications
    solution['task_strategies'] = task_strategies
    
    # 保存分类和策略结果
    classification_and_strategy = [
        {
            'task_id': i + 1,
            'classification': task_classifications[i],
            'strategy': task_strategies[i]
        }
        for i in range(len(task_classifications))
    ]
    with open(os.path.join(output_dir, 'task_classifications_and_strategies.json'), 'w') as f:
        json.dump(classification_and_strategy, f, indent=2, ensure_ascii=False)
    
    # 4.2 依赖分析
    print('\n[4.2] Dependency Analysis')
    coordinator = Coordinator(llm)
    order = coordinator.analyze_dependencies(
        problem_str, modeling_solution, task_descriptions
    )
    order = [int(i) for i in order]
    
    solution['execution_order'] = order
    print(f'[✓] Task execution order: {order}')
    
    # 保存依赖图
    with open(os.path.join(output_dir, 'dependency_dag.json'), 'w') as f:
        json.dump(coordinator.DAG, f, indent=2)
    
    # 保存任务依赖分析
    if hasattr(coordinator, 'task_dependency_analysis') and coordinator.task_dependency_analysis:
        with open(os.path.join(output_dir, 'task_dependency_analysis.json'), 'w') as f:
            json.dump(coordinator.task_dependency_analysis, f, indent=2, ensure_ascii=False)
        print(f'[✓] Saved task_dependency_analysis')
    '''
    




    # ============================================================================
    # 从预计算文件夹加载变量
    # ============================================================================
    print('\n' + '=' * 80)
    print('Loading Variables from Precomputed Files')
    print('=' * 80)
    
    precomputed_dir = '/home/hyn/LLM-MM-Agent-LLMINA/MMAgent/output/LLMINA-Solver/LLMINA_20251023-171459'
    
    # 1. 加载 modeling_solution
    modeling_solution_path = os.path.join(precomputed_dir, 'modeling_solution.txt')
    with open(modeling_solution_path, 'r', encoding='utf-8') as f:
        modeling_solution = f.read()
    print(f'[✓] Loaded modeling_solution from {modeling_solution_path}')
    solution['modeling_solution'] = modeling_solution
    
    # 2. 加载 task_descriptions
    task_descriptions_path = os.path.join(precomputed_dir, 'task_descriptions.json')
    with open(task_descriptions_path, 'r', encoding='utf-8') as f:
        task_descriptions = json.load(f)
    tasknum = len(task_descriptions)
    print(f'[✓] Loaded {tasknum} task_descriptions from {task_descriptions_path}')
    solution['task_descriptions'] = task_descriptions
    solution['tasknum'] = tasknum
    
    # 3. 加载 task_classifications 和 task_strategies
    classification_and_strategy_path = os.path.join(precomputed_dir, 'task_classifications_and_strategies.json')
    with open(classification_and_strategy_path, 'r', encoding='utf-8') as f:
        classification_and_strategy = json.load(f)
    
    task_classifications = [item['classification'] for item in classification_and_strategy]
    task_strategies = [item['strategy'] for item in classification_and_strategy]
    print(f'[✓] Loaded task_classifications and task_strategies from {classification_and_strategy_path}')
    solution['task_classifications'] = task_classifications
    solution['task_strategies'] = task_strategies
    
    # 4. 加载 coordinator.DAG 和执行顺序
    dependency_dag_path = os.path.join(precomputed_dir, 'dependency_dag.json')
    coordinator = Coordinator(llm)
    with open(dependency_dag_path, 'r', encoding='utf-8') as f:
        coordinator.DAG = json.load(f)
    
    # 初始化 task_dependency_analysis（兼容性处理）
    # 如果存在保存的分析文件则加载，否则设为空列表
    task_dependency_analysis_path = os.path.join(precomputed_dir, 'task_dependency_analysis.json')
    if os.path.exists(task_dependency_analysis_path):
        with open(task_dependency_analysis_path, 'r', encoding='utf-8') as f:
            coordinator.task_dependency_analysis = json.load(f)
        print(f'[✓] Loaded task_dependency_analysis from {task_dependency_analysis_path}')
    else:
        # 如果文件不存在，初始化为空列表以避免AttributeError
        coordinator.task_dependency_analysis = []
        print(f'[!] task_dependency_analysis not found, initialized to empty list')
    
    # 从DAG中提取执行顺序
    order = [int(task_id) for task_id in sorted(coordinator.DAG.keys(), key=int)]
    print(f'[✓] Loaded dependency_dag from {dependency_dag_path}')
    print(f'[✓] Task execution order: {order}')
    solution['execution_order'] = order
    
    print('\n' + '=' * 80)
    print('All Variables Loaded Successfully')
    print('=' * 80)
    









    # ============================================================================
    # Stage 5: 自适应任务求解
    # ============================================================================
    print('\n' + '=' * 80)
    print('Stage 5: Adaptive Task Solving')
    print('=' * 80)
    
    # 创建共享的ProgressiveTaskSolver实例，确保task_history在所有任务间持久化
    task_solver = ProgressiveTaskSolver(llm)
    print(f'[✓] Created shared ProgressiveTaskSolver for maintaining task_history across all tasks')
    
    task_results = []
    
    for idx, task_id in enumerate(order):
        print(f'\n{"="*80}')
        print(f'Solving Task {task_id} ({idx + 1}/{len(order)})')
        print(f'Category: {task_classifications[task_id - 1]["category"]}')
        print(f'Complexity: {task_classifications[task_id - 1]["complexity"]}')
        print(f'Strategy: {task_strategies[task_id - 1]["code_generation_strategy"]}')
        print(f'{"="*80}')
        
        try:
            # 使用增强的计算求解模块，传入预分类结果和策略，避免重复分类
            # 传入共享的task_solver实例以保持task_history状态
            task_result = computational_solving_enhanced(
                llm=llm,
                coordinator=coordinator,
                problem=problem,
                task_id=task_id,
                task_descriptions=task_descriptions,
                modeling_solution=modeling_solution,
                config=config,
                output_dir=output_dir,
                task_classification=task_classifications[task_id - 1],  # 传入预分类结果
                task_strategy=task_strategies[task_id - 1],  # 传入预计算的策略
                task_solver=task_solver  # 传入共享的task_solver实例
            )
            
            task_results.append(task_result)
            
            # 打印任务状态
            status = '✓ PASSED' if task_result['is_pass'] else '✗ FAILED'
            print(f'\n[Task {task_id}] {status}')
            
        except Exception as e:
            print(f'\n[Error] Task {task_id} failed: {e}')
            import traceback
            traceback.print_exc()
            
            # 记录失败信息
            task_result = {
                'task_id': task_id,
                'task_description': task_descriptions[task_id - 1],
                'is_pass': False,
                'error': str(e),
                'execution_result': 'Task solving failed'
            }
            task_results.append(task_result)
    
    solution['task_results'] = task_results
    
    # ============================================================================
    # Stage 6: 结果整合和报告
    # ============================================================================
    print('\n' + '=' * 80)
    print('Stage 6: Result Integration and Reporting')
    print('=' * 80)
    
    # 统计结果
    passed_tasks = sum(1 for r in task_results if r.get('is_pass', False))
    total_tasks = len(task_results)
    success_rate = (passed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    
    print(f'\nOverall Statistics:')
    print(f'  Total Tasks: {total_tasks}')
    print(f'  Passed: {passed_tasks}')
    print(f'  Failed: {total_tasks - passed_tasks}')
    print(f'  Success Rate: {success_rate:.1f}%')
    
    # 按任务类型统计
    print(f'\nBy Task Category:')
    category_stats = {}
    for i, result in enumerate(task_results):
        category = task_classifications[i].get('category', 'unknown')
        if category not in category_stats:
            category_stats[category] = {'total': 0, 'passed': 0}
        category_stats[category]['total'] += 1
        if result.get('is_pass', False):
            category_stats[category]['passed'] += 1
    
    for category, stats in category_stats.items():
        rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f'  {category}: {stats["passed"]}/{stats["total"]} ({rate:.1f}%)')
    
    # 保存完整方案
    final_solution_path = os.path.join(output_dir, 'complete_solution.json')
    with open(final_solution_path, 'w') as f:
        json.dump(solution, f, indent=2, ensure_ascii=False)
    
    print(f'\n[✓] Complete solution saved to {final_solution_path}')
    
    # ============================================================================
    # Stage 7: 生成最终Solver代码（如果需要）
    # ============================================================================
    if config.get('generate_final_solver', False):
        print('\n' + '=' * 80)
        print('Stage 7: Final Solver Code Generation')
        print('=' * 80)
        
        try:
            final_solver_path = _generate_final_solver(
                coordinator, task_results, code_template, output_dir
            )
            solution['final_solver_path'] = final_solver_path
            print(f'[✓] Final solver generated: {final_solver_path}')
        except Exception as e:
            print(f'[Warning] Final solver generation failed: {e}')
    
    print('\n' + '=' * 80)
    print('Pipeline Completed Successfully')
    print('=' * 80)
    
    # 找到最后一个任务生成的solver代码路径
    solver_code_path = None
    if task_results:
        # 从最后一个成功的任务获取solver路径
        for result in reversed(task_results):
            if result.get('is_pass', False) and 'task_code_path' in result:
                solver_code_path = result['task_code_path']
                break
        
        # 如果没有找到task_code_path，尝试从code目录获取最新的solver文件
        if solver_code_path is None:
            code_dir = os.path.join(output_dir, 'code')
            if os.path.exists(code_dir):
                solver_files = [f for f in os.listdir(code_dir) if f.startswith('solver_task') and f.endswith('.py')]
                if solver_files:
                    # 获取编号最大的solver文件
                    solver_files.sort(key=lambda x: int(x.replace('solver_task', '').replace('.py', '')))
                    solver_code_path = os.path.join(code_dir, solver_files[-1])
    
    # 如果仍然没有找到，使用默认路径
    if solver_code_path is None:
        solver_code_path = os.path.join(output_dir, 'code', f'solver_task{len(task_results)}.py')
        print(f'[Warning] No solver code path found in task results, using default: {solver_code_path}')
    
    return solver_code_path, solution


def _generate_final_solver(coordinator, task_results, template, output_dir):
    """
    整合所有子任务代码，生成最终的solver
    """
    final_solver_path = os.path.join(output_dir, 'final_solver.py')
    
    # 收集所有成功的子任务代码
    task_codes = []
    for task_id in sorted(coordinator.memory.keys(), key=int):
        task_data = coordinator.memory[task_id]
        if task_data.get('is_pass', False):
            task_codes.append(f"# Task {task_id}\n{task_data['task_code']}\n")
    
    # 生成最终代码
    final_code = f"""\"\"\"
LLMINA Final Solver
Auto-generated by enhanced pipeline
\"\"\"

{chr(10).join(task_codes)}

# Main solver function
def llm_solver(instance, network, K, jobs_num, Cs, topo_name):
    \"\"\"
    Integrated LLMINA solver combining all sub-tasks
    \"\"\"
    # TODO: Integrate task results
    pass
"""
    
    with open(final_solver_path, 'w') as f:
        f.write(final_code)
    
    return final_solver_path


# 为了向后兼容，保留原函数名
def llmina_solver_pipeline(llm, problem_path, config, output_dir, name):
    """向后兼容的入口函数"""
    return llmina_solver_pipeline_enhanced(llm, problem_path, config, output_dir, name)
