"""
简化的 LLMINA Agent 节点
只返回增量更新，让 LangGraph 自动合并状态
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langgraph_core.state import AgentState, Message
from MMAgent.agent.problem_clarification import ProblemClarification
from MMAgent.agent.summary import ProblemClarificationSummary
from MMAgent.agent.feedback_judge import ProblemJudge
from MMAgent.agent.problem_solving import ProblemSolving
from MMAgent.agent.problem_decompse import ProblemDecompose
from MMAgent.agent.task_solver import solve_task
from prompt.llmina_template import PROBLEM_DESCRIPTION_PROMPT
from utils.output_manager import (
    ensure_output_dirs,
    save_modeling_solution,
    save_task_descriptions,
    save_dependency_info,
    save_task_code,
    save_task_result,
    save_complete_solution,
    save_workflow_log,
    create_readme
)

# ============ 问题加载 ============

def load_problem_node(state: AgentState) -> AgentState:    
    problem_path = state['problem_path']
    config = state['config']
    output_dir = state['output_dir']
    
    # 确保输出目录结构存在
    ensure_output_dirs(output_dir)
    
    # 读取问题
    with open(problem_path, 'r', encoding='utf-8') as f:
        problem = json.load(f)
    
    # 读取代码模板 - 从任务目录获取
    task_dir = state['task_dir']
    template_dir = os.path.join(task_dir, 'code_template')
    template_path = os.path.join(template_dir, 'llmina_solver_template.py')
    with open(template_path, 'r', encoding='utf-8') as f:
        code_template = f.read()
    
    # 构建问题描述
    problem_str = PROBLEM_DESCRIPTION_PROMPT.format(
        problem_background=problem['background'],
        problem_requirement=problem['problem_requirement'],
        variable_description=problem['variable_description'],
        problem_formulation=problem['problem_formulation'],
        code_template=code_template
    ).strip()
    
    print(f"[Load Problem] {len(problem_str)} chars")
    
    # 记录工作流日志
    save_workflow_log(output_dir, 'problem_loaded', {
        'problem_path': problem_path,
        'problem_length': len(problem_str)
    })
    
    # 只返回增量更新，LangGraph 会自动合并
    return {
        'problem_str': problem_str,
        'intermediate_results': {'problem_data': problem},
        'messages': [Message(
            role='system',
            content=f'Problem loaded from {problem_path}',
            agent_name='problem_loader'
        )]
    }


# ============ 问题澄清 ============

def problem_clarification_node(state: AgentState) -> AgentState:
    """问题澄清""" 
    config = state['config']
    
    # 检查是否启用
    if not config.get('clarification_enabled', True):
        print(f"[Clarification] Skipped (disabled)")
        return {
            'clarification_summary': "Clarification skipped",
            'next_action': 'algorithm_design'
        }
    
    llm = state['llm']
    problem_str = state['problem_str']
    
    # 创建 Agent
    pc = ProblemClarification(llm)
    pj = ProblemJudge(llm)
    ps = ProblemClarificationSummary(llm)
    
    # 用户输入函数 - 支持交互模式和自动模式
    def user_input_handler(agent_feedback: str) -> str:
        """
        处理用户输入
        
        在交互模式下，打印 agent 反馈并获取用户输入
        在自动模式下，返回空字符串让 agent 自己判断
        """
        # 检查是否启用交互模式
        interactive = config.get('llmina', {}).get('clarification_interactive', False)
        
        if interactive:
            # 交互模式：打印反馈并获取用户输入
            print(f"\n{'='*60}")
            print(f"[Agent Feedback]")
            print(agent_feedback)
            print(f"{'='*60}")
            print("Please provide clarification (or press Enter to skip):")
            try:
                user_input = input("> ").strip()
                return user_input
            except (EOFError, KeyboardInterrupt):
                print("\n[Auto mode] No user input")
                return ""
        else:
            # 自动模式：返回空字符串，让 agent 基于当前信息判断
            return ""
    
    print(f"[Clarification] Starting...")
    clarified_problem, history, summary = pc.clarification_actor(
        pj, ps, problem_str, user_input_handler
    )
    
    print(f"[Clarification] Done ({len(history)} rounds)")
    
    # 只返回增量更新
    return {
        'problem_str': clarified_problem,
        'clarification_history': history,
        'clarification_summary': summary,
        'clarification_round': state.get('clarification_round', 0) + 1,
        'next_action': 'algorithm_design',
        'messages': [Message(
            role='assistant',
            content=f'Clarification: {summary[:100]}...',
            agent_name='problem_clarification'
        )]
    }


# ============ 算法方案设计 ============

def algorithm_design_node(state: AgentState) -> AgentState:
    """
    算法方案设计节点
    
    生成高层次的算法求解方案，包括：
    - 核心方法论（启发式算法、分解策略等）
    - 算法概览（分步骤描述）
    - 关键策略（离散决策、连续分配、约束满足等）
    - 可行性和优化性讨论
    - 可扩展性和鲁棒性考虑
    
    通过多轮迭代（Actor-Critic-Improver）不断完善方案
    """
    llm = state['llm']
    problem_str = state['problem_str']
    algorithm_design_round = state.get('algorithm_design_round', 0)
    output_dir = state['output_dir']
    
    # 创建 ProblemSolving Agent（负责算法方案设计）
    ps = ProblemSolving(llm)
    
    print(f"[Algorithm Design] Round {algorithm_design_round + 1}")
    algorithm_solution = ps.solving(problem_str, round=algorithm_design_round)
    
    print(f"[Algorithm Design] Done ({len(algorithm_solution)} chars)")
    
    # 保存算法方案
    saved_path = save_modeling_solution(output_dir, algorithm_solution)
    print(f"[Algorithm Design] Saved to {saved_path}")
    
    # 记录日志
    save_workflow_log(output_dir, 'algorithm_design_completed', {
        'round': algorithm_design_round + 1,
        'solution_length': len(algorithm_solution)
    })
    
    # 只返回增量更新
    return {
        'algorithm_solution': algorithm_solution,
        'algorithm_design_round': algorithm_design_round + 1,
        'next_action': 'decompose',
        'messages': [Message(
            role='assistant',
            content=f'Algorithm Design: {algorithm_solution[:100]}...',
            agent_name='algorithm_designer'
        )]
    }


# ============ 任务分解 ============

def task_decompose_node(state: AgentState) -> AgentState:
    """任务分解"""
    llm = state['llm']
    problem_str = state['problem_str']
    algorithm_solution = state.get('algorithm_solution') or state.get('modeling_solution', '')
    output_dir = state['output_dir']
    
    # 创建 Agent
    pd = ProblemDecompose(llm)
    
    print(f"[Decompose] Starting...")
    task_descriptions, tasknum = pd.decompose_and_refine(
        problem_str, algorithm_solution
    )
    
    print(f"[Decompose] {tasknum} tasks")
    
    # 保存任务描述
    saved_path = save_task_descriptions(output_dir, task_descriptions)
    print(f"[Decompose] Saved to {saved_path}")
    
    # 记录日志
    save_workflow_log(output_dir, 'decomposition_completed', {
        'total_tasks': tasknum,
        'task_descriptions': [desc[:100] + '...' for desc in task_descriptions]
    })
    
    # 只返回增量更新
    return {
        'task_descriptions': task_descriptions,
        'tasknum': tasknum,
        'total_tasks': tasknum,
        'current_task_id': 0,
        'next_action': 'analyze_dependencies',
        'messages': [Message(
            role='assistant',
            content=f'Decomposed into {len(task_descriptions)} tasks',
            agent_name='task_decompose'
        )]
    }


# ============ 依赖分析 ============

def dependency_analysis_node(state: AgentState) -> AgentState:
    """
    依赖分析（简化版：顺序执行）
    
    不使用 Coordinator，直接生成顺序执行顺序 [1, 2, 3, ..., n]
    """
    tasknum = state['tasknum']
    output_dir = state['output_dir']
    
    # 生成顺序执行顺序
    execution_order = list(range(1, tasknum + 1))
    
    print(f"[Dependency] Sequential execution: {execution_order}")
    
    # 保存依赖信息（简化版：空DAG，顺序执行）
    dependency_dag = {str(i): [] for i in execution_order}
    saved_paths = save_dependency_info(
        output_dir,
        dependency_dag=dependency_dag,
        dependency_analysis=[],
        execution_order=execution_order
    )
    print(f"[Dependency] Saved to {saved_paths['dag']}")
    
    # 记录日志
    save_workflow_log(output_dir, 'dependency_analysis_completed', {
        'execution_order': execution_order,
        'mode': 'sequential'
    })
    
    # 只返回增量更新
    # 不需要 DAG 和 task_dependency_analysis，因为是顺序执行
    return {
        'execution_order': execution_order,
        'dependency_dag': dependency_dag,
        'next_action': 'solve_tasks',
        'messages': [Message(
            role='assistant',
            content=f'Sequential execution order: {execution_order}',
            agent_name='dependency_analysis'
        )]
    }


# ============ 任务求解 ============

def task_solving_node(state: AgentState) -> AgentState:
    """求解单个任务（顺序执行，无依赖分析）"""
    llm = state['llm']
    execution_order = state['execution_order']
    current_task_id = state['current_task_id']
    task_descriptions = state['task_descriptions']
    algorithm_solution = state.get('algorithm_solution') or state.get('modeling_solution', '')
    config = state['config']
    output_dir = state['output_dir']
    task_results = state.get('task_results', {})
    dependency_dag = state.get('dependency_dag', {})
    dependency_analysis = state.get('task_dependency_analysis', [])
    
    task_id = execution_order[current_task_id]
    total_tasks = len(execution_order)
    
    print(f"[Task {task_id}/{total_tasks}] Solving...")
    
    # 获取当前任务描述
    task_description = task_descriptions[task_id - 1] if task_id <= len(task_descriptions) else ""
    
    # 将 output_dir 添加到 config 中以便 task_solver 使用正确的路径
    task_config = {**config, 'output_dir': output_dir}
    
    # 求解任务
    result = solve_task(
        llm=llm,
        task_id=task_id,
        total_tasks=total_tasks,
        task_description=task_description,
        modeling_solution=algorithm_solution,
        dependency_dag=dependency_dag,
        dependency_analysis=dependency_analysis,
        past_results=task_results,
        config=task_config
    )
    
    print(f"[Task {task_id}] {'Pass' if result['is_pass'] else 'Fail'}")
    
    # 保存任务代码
    task_code = result.get('task_code', '')
    if task_code:
        code_path = save_task_code(output_dir, task_id, task_code)
        result['task_code_path'] = code_path
        print(f"[Task {task_id}] Code saved to {code_path}")
    
    # 保存任务结果
    result_with_desc = {**result, 'task_description': task_description}
    result_path = save_task_result(output_dir, task_id, task_description, result_with_desc)
    print(f"[Task {task_id}] Result saved to {result_path}")
    
    # 记录日志
    save_workflow_log(output_dir, f'task_{task_id}_completed', {
        'task_id': task_id,
        'is_pass': result['is_pass'],
        'execution_result': str(result.get('execution_result', ''))[:200]
    })
    
    # 检查是否还有更多任务
    next_action = 'solve_next' if (current_task_id + 1) < total_tasks else 'integrate'
    
    # 只返回增量更新
    return {
        'current_task_id': current_task_id + 1,
        'task_results': {task_id: result_with_desc},
        'next_action': next_action,
        'messages': [Message(
            role='assistant',
            content=f'Task {task_id} solved',
            agent_name='task_solver'
        )]
    }


# ============ 代码集成 ============

def code_integration_node(state: AgentState) -> AgentState:
    """集成所有任务代码"""
    task_results = state['task_results']
    execution_order = state['execution_order']
    output_dir = state['output_dir']
    
    # 获取最后一个任务的代码（包含所有集成）
    final_task_id = execution_order[-1]
    final_code = task_results.get(final_task_id, {}).get('task_code', "# No code generated")
    
    # 保存最终代码
    solver_code_path = os.path.join(output_dir, 'llm_solver_final.py')
    os.makedirs(os.path.dirname(solver_code_path), exist_ok=True)
    
    with open(solver_code_path, 'w', encoding='utf-8') as f:
        f.write(final_code)
    
    print(f"[Integration] Saved to {solver_code_path}")
    
    # 保存完整解决方案摘要
    complete_solution_path = save_complete_solution(output_dir, state)
    print(f"[Integration] Complete solution saved to {complete_solution_path}")
    
    # 创建README
    readme_path = create_readme(output_dir, state)
    print(f"[Integration] README created at {readme_path}")
    
    # 记录日志
    save_workflow_log(output_dir, 'integration_completed', {
        'final_task_id': final_task_id,
        'total_tasks': len(execution_order),
        'solver_path': solver_code_path
    })
    
    print(f"\n{'='*80}")
    print(f"✓ Solution Complete!")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Final solver: {solver_code_path}")
    print(f"See README.md for details")
    print(f"{'='*80}\n")
    
    # 只返回增量更新
    return {
        'solver_code': final_code,
        'solver_code_path': solver_code_path,
        'next_action': 'end',
        'messages': [Message(
            role='assistant',
            content=f'Code saved to {solver_code_path}',
            agent_name='code_integrator'
        )]
    }
