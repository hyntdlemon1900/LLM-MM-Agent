"""
简化的 LLMINA Agent 节点
只返回增量更新，让 LangGraph 自动合并状态
"""

"""
简化的 LLMINA Agent 节点
只返回增量更新，让 LangGraph 自动合并状态
"""

import json
import os
import re
from pathlib import Path

# 包导入 - 使用相对导入
from .state import AgentState, Message

# 使用绝对导入从 MMAgent 包
from MMAgent.agent import (
    ProblemSolving,
    ProblemDecompose,
    generate_task_code,
)
from MMAgent.prompt import (
    PROBLEM_DESCRIPTION_PROMPT,
    HEURISTIC_ARCHITECT_PROMPT,
    HEURISTIC_FUNCTION_CODE_GENERATION_PROMPT,
)
from MMAgent.utils import (
    ensure_output_dirs,
    save_modeling_solution,
    save_task_descriptions,
    save_dependency_info,
    save_task_code,
    save_task_result,
    save_complete_solution,
    save_workflow_log,
    create_readme,
)

# ============ 问题加载 ============

def load_problem_node(state: AgentState) -> AgentState:    
    problem_path = state['problem_path']
    output_dir = state['output_dir']
    
    # 确保输出目录结构存在
    ensure_output_dirs(output_dir)
    
    # 读取问题
    with open(problem_path, 'r', encoding='utf-8') as f:
        problem = json.load(f)
    
    # 读取 Pyomo 建模代码作为参考 - 从任务的 runtime 目录获取
    # task_dir 已经指向 runtime 目录，无需再添加 /runtime/
    task_dir = state['task_dir']
    pyomo_path = os.path.join(task_dir, 'Pyomo.py')
    with open(pyomo_path, 'r', encoding='utf-8') as f:
        pyomo_code = f.read()
    
    # 构建问题描述
    problem_str = PROBLEM_DESCRIPTION_PROMPT.format(
        problem_background=problem['background'],
        problem_requirement=problem['problem_requirement'],
        variable_description=problem['variable_description'],
        problem_formulation=problem['problem_formulation'],
        pyomo_reference_code=pyomo_code
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
        'pyomo_reference_code': pyomo_code,  # 保存Pyomo代码供后续使用
        'intermediate_results': {'problem_data': problem},
        'next_action': 'generate_heuristic_template',
        'messages': [Message(
            role='system',
            content=f'Problem loaded from {problem_path}',
            agent_name='problem_loader'
        )]
    }


# ============ 生成启发式求解器模板 ============

def generate_heuristic_template_node(state: AgentState) -> AgentState:
    """
    生成启发式求解器模板节点
    
    基于 PyomoINASolver 的建模代码，生成一个 HeuristicSolver 类框架：
    1. 接收与 PyomoINASolver 相同的输入参数
    2. 不继承 PyomoTemplateSolver（独立实现）
    3. 包含空的 solve() 方法供后续填充
    
    重要：模板生成到 output 目录，不修改问题目录（关注点分离）
    
    设计原则：
    - 通过 LLM 动态生成模板，避免硬编码问题特定逻辑
    - 智能体系统不应该知道具体问题的参数细节
    """
    llm = state['llm']
    output_dir = state['output_dir']
    pyomo_code = state.get('pyomo_reference_code', '')
    
    print(f"[Generate Heuristic Template] Creating HeuristicSolver template...")
    
    # 使用 LLM 生成模板代码（而不是硬编码）
    heuristic_template = generate_heuristic_solver_template(llm, pyomo_code)
    
    # 保存到 output 目录（而不是问题的 runtime 目录）
    template_path = os.path.join(output_dir, 'HeuristicSolver_template.py')
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(heuristic_template)
    
    print(f"✓ HeuristicSolver template created at: {template_path}")
    
    # 记录工作流日志
    save_workflow_log(output_dir, 'heuristic_template_generated', {
        'template_path': template_path,
        'template_length': len(heuristic_template)
    })
    
    return {
        'heuristic_template_code': heuristic_template,
        'heuristic_template_path': template_path,
        'next_action': 'heuristic_architect',
        'messages': [Message(
            role='system',
            content=f'HeuristicSolver template generated at {template_path}',
            agent_name='template_generator'
        )]
    }


def generate_heuristic_solver_template(llm, pyomo_code: str) -> str:
    """
    使用 LLM 生成启发式求解器模板代码
    
    设计原则（关注点分离）：
    - 智能体系统不硬编码问题特定逻辑
    - 通过 LLM 理解 Pyomo 代码结构，动态生成接口一致的模板
    - 模板只提供框架，不包含具体实现
    
    Args:
        llm: 语言模型实例
        pyomo_code: PyomoINASolver 的参考代码
    
    Returns:
        HeuristicSolver 模板代码字符串
    """
    from MMAgent.prompt import HEURISTIC_TEMPLATE_GENERATION_PROMPT
    
    # 构建提示词
    prompt = HEURISTIC_TEMPLATE_GENERATION_PROMPT.format(
        pyomo_solver_code=pyomo_code
    )
    
    # 调用 LLM 生成模板
    response = llm.generate(prompt)
    template_code = response.content if hasattr(response, 'content') else str(response)
    
    # 清理可能的 markdown 标记
    if '```python' in template_code:
        template_code = template_code.split('```python')[1].split('```')[0].strip()
    elif '```' in template_code:
        template_code = template_code.split('```')[1].split('```')[0].strip()
    
    return template_code


# ============ 启发式架构设计 ============

def heuristic_architect_node(state: AgentState) -> AgentState:
    """
    启发式架构设计节点
    
    不编写实现代码，只负责软件架构设计。
    分析问题、Pyomo代码和HeuristicSolver模板，输出结构化的函数签名列表（JSON格式），
    定义启发式算法的模块化组件。
    
    关键：读取 HeuristicSolver 模板，提取可用的成员变量信息，
    传递给 LLM，以便设计出正确的成员方法（而非独立函数）。
    
    输出示例:
    [
        {"name": "determine_placement_decisions", "description": "..."},
        {"name": "optimize_flow_allocation", "description": "..."},
        {"name": "validate_constraints", "description": "..."}
    ]
    """
    llm = state['llm']
    problem_str = state['problem_str']
    output_dir = state['output_dir']
    heuristic_template_code = state.get('heuristic_template_code', '')
    
    print(f"[Heuristic Architect] Designing function architecture...")
    
    # 构建提示词（传入问题描述和模板代码）
    prompt = HEURISTIC_ARCHITECT_PROMPT.format(
        modeling_problem=problem_str,
        heuristic_template_code=heuristic_template_code
    )
    
    # 调用LLM生成函数架构设计
    try:
        response = llm.generate(prompt)
        
        # 解析响应 - 严格要求JSON格式
        import json
        import re
        
        # 提取内容
        content = response.content if hasattr(response, 'content') else str(response)
        
        # 清理可能的markdown代码块
        # 移除 ```json 或 ``` 标记
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            # 尝试提取代码块内容
            code_blocks = re.findall(r'```(?:\w+)?\s*(.*?)```', content, re.DOTALL)
            if code_blocks:
                content = code_blocks[0].strip()
        
        # 尝试找到JSON数组的边界
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        
        if json_start == -1 or json_end == 0:
            raise ValueError(
                f"No JSON array found in LLM response. "
                f"Response starts with: {content[:200]}..."
            )
        
        # 提取JSON字符串
        json_str = content[json_start:json_end].strip()
        
        # 解析JSON
        try:
            heuristic_functions = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON syntax in LLM response: {str(e)}\n"
                f"JSON string (first 500 chars): {json_str[:500]}..."
            )
        
        # 验证格式 - 详细检查
        if not isinstance(heuristic_functions, list):
            raise ValueError(
                f"Expected a list of function definitions, got {type(heuristic_functions).__name__}. "
                f"Content: {str(heuristic_functions)[:200]}..."
            )
        
        if len(heuristic_functions) == 0:
            raise ValueError("Function list is empty. At least one function is required.")
        
        # 验证每个函数的必需字段
        for i, func in enumerate(heuristic_functions):
            if not isinstance(func, dict):
                raise ValueError(
                    f"Function {i+1} is not a dictionary, got {type(func).__name__}"
                )
            
            # 检查必需字段
            required_fields = ['name', 'description']
            missing_fields = [field for field in required_fields if field not in func]
            if missing_fields:
                raise ValueError(
                    f"Function {i+1} missing required fields: {missing_fields}. "
                    f"Function data: {func}"
                )
            
            # 验证字段类型
            if not isinstance(func['name'], str) or not func['name'].strip():
                raise ValueError(f"Function {i+1} has invalid 'name': {func.get('name')}")
            
            if not isinstance(func['description'], str) or not func['description'].strip():
                raise ValueError(f"Function {i+1} has invalid 'description': {func.get('description')}")
        
        print(f"[Heuristic Architect] ✓ Successfully designed {len(heuristic_functions)} functions:")
        for func in heuristic_functions:
            print(f"  - {func['name']}")
        
        # 保存架构设计
        architecture_path = os.path.join(output_dir, 'heuristic_architecture.json')
        os.makedirs(output_dir, exist_ok=True)
        with open(architecture_path, 'w', encoding='utf-8') as f:
            json.dump(heuristic_functions, f, indent=2, ensure_ascii=False)
        print(f"[Heuristic Architect] Saved to {architecture_path}")
        
        # 记录日志
        save_workflow_log(output_dir, 'heuristic_architect_completed', {
            'function_count': len(heuristic_functions),
            'functions': [f['name'] for f in heuristic_functions]
        })
        
        # 返回增量更新 - 如果有函数则触发函数生成循环
        next_action = 'generate_next_function' if len(heuristic_functions) > 0 else 'integrate'
        
        return {
            'heuristic_functions': heuristic_functions,
            'current_function_id': 0,  # 重置函数ID计数器
            'next_action': next_action,
            'messages': [Message(
                role='assistant',
                content=f'Designed {len(heuristic_functions)} heuristic functions',
                agent_name='heuristic_architect',
                metadata={'functions': [f['name'] for f in heuristic_functions]}
            )]
        }
    
    except Exception as e:
        error_msg = f"Failed to parse heuristic architecture from LLM response: {str(e)}"
        print(f"\n[Heuristic Architect] ✗ ERROR: {error_msg}")
        
        # 打印原始响应用于调试
        if 'response' in locals():
            raw_content = response.content if hasattr(response, 'content') else str(response)
            print(f"[Heuristic Architect] Raw LLM response (first 1000 chars):\n{raw_content[:1000]}\n")
        
        # 记录错误日志
        save_workflow_log(output_dir, 'heuristic_architect_failed', {
            'error': error_msg,
            'error_type': type(e).__name__
        })
        
        # 返回错误状态
        return {
            'heuristic_functions': [],
            'next_action': 'integrate',  # 跳过函数生成，直接集成
            'errors': [error_msg],
            'messages': [Message(
                role='system',
                content=error_msg,
                agent_name='heuristic_architect'
            )]
        }


# ============ 启发式函数代码生成 ============

def function_code_generator_node(state: AgentState) -> AgentState:
    """
    启发式函数代码生成节点
    
    根据 heuristic_architect_node 输出的函数列表，逐个生成函数实现。
    类似于 task_solving_node 的迭代逻辑。
    
    工作流程：
    1. 获取当前要生成的函数（根据 current_function_id）
    2. 构建提示词，包含函数规范和已完成函数的上下文
    3. 调用 LLM 生成函数代码
    4. 保存函数代码
    5. 判断是否还有更多函数要生成
    """
    llm = state['llm']
    heuristic_functions = state.get('heuristic_functions', [])
    current_function_id = state.get('current_function_id', 0)
    function_codes = state.get('function_codes', {})
    problem_str = state['problem_str']
    output_dir = state['output_dir']
    heuristic_template_code = state.get('heuristic_template_code', '')
    
    if not heuristic_functions:
        print("[Function Code Generator] No heuristic functions defined, skipping...")
        return {
            'next_action': 'integrate',
            'messages': [Message(
                role='system',
                content='No heuristic functions to generate',
                agent_name='function_code_generator'
            )]
        }
    
    total_functions = len(heuristic_functions)
    
    # 检查是否已完成所有函数
    if current_function_id >= total_functions:
        print(f"[Function Code Generator] All {total_functions} functions completed")
        
        # 合并所有函数代码到一个文件
        combined_code = "# Heuristic Functions - Auto-generated\n\n"
        combined_code += "from typing import Dict, List, Any, Tuple\n\n"
        
        for func_id in range(total_functions):
            func_info = heuristic_functions[func_id]
            func_code = function_codes.get(func_id, "# Function not generated")
            combined_code += f"# ============ Function {func_id + 1}: {func_info['name']} ============\n\n"
            combined_code += func_code + "\n\n"
        
        # 保存合并后的代码
        combined_path = os.path.join(output_dir, 'heuristic_functions.py')
        os.makedirs(output_dir, exist_ok=True)
        with open(combined_path, 'w', encoding='utf-8') as f:
            f.write(combined_code)
        print(f"[Function Code Generator] All functions saved to {combined_path}")
        
        return {
            'next_action': 'integrate',
            'messages': [Message(
                role='assistant',
                content=f'Generated all {total_functions} heuristic functions',
                agent_name='function_code_generator'
            )]
        }
    
    # 生成当前函数
    func_info = heuristic_functions[current_function_id]
    func_name = func_info['name']
    
    print(f"\n{'='*80}")
    print(f"[Function {current_function_id + 1}/{total_functions}] {func_name}")
    print(f"{'='*80}")
    
    # 构建输入输出规范字符串
    inputs_spec = ""
    for inp in func_info.get('inputs', []):
        inputs_spec += f"- {inp['name']} ({inp['type']}): {inp['description']}\n"
    
    outputs_spec = ""
    for out in func_info.get('outputs', []):
        outputs_spec += f"- {out['name']} ({out['type']}): {out['description']}\n"
    
    # 构建已完成函数摘要
    completed_summary = ""
    if current_function_id > 0:
        completed_summary = f"The following {current_function_id} functions have been completed:\n\n"
        for i in range(current_function_id):
            prev_func = heuristic_functions[i]
            completed_summary += f"{i+1}. {prev_func['name']}: {prev_func['description'][:100]}...\n"
    else:
        completed_summary = "This is the first function. No previous functions have been completed yet."
    
    # 提取输入输出参数名
    member_variables_read = ", ".join([inp for inp in func_info.get('member_variables_read', [])])
    member_variables_written = ", ".join([out for out in func_info.get('member_variables_written', [])])
    
    # 构建提示词
    prompt = HEURISTIC_FUNCTION_CODE_GENERATION_PROMPT.format(
        problem_str=problem_str,
        heuristic_template_code=heuristic_template_code,
        function_id=current_function_id + 1,
        total_functions=total_functions,
        function_name=func_name,
        function_description=func_info['description'],
        inputs_spec=inputs_spec if inputs_spec else "None",
        outputs_spec=outputs_spec if outputs_spec else "None",
        dependencies=", ".join(func_info.get('dependencies', [])) if func_info.get('dependencies') else "None",
        completed_functions_summary=completed_summary,
        member_variables_read=member_variables_read,
        member_variables_written=member_variables_written
    )
    
    # 调用 LLM 生成代码
    try:
        print(f"  [Code Generation] Generating code for {func_name}...")
        response = llm.generate(prompt)
        func_code = response.content if hasattr(response, 'content') else str(response)
        
        # 清理代码（移除可能的 markdown 标记）
        if '```python' in func_code:
            func_code = func_code.split('```python')[1].split('```')[0].strip()
        elif '```' in func_code:
            func_code = func_code.split('```')[1].split('```')[0].strip()
        
        print(f"  [Code Generation] Done ({len(func_code)} chars)")
        
        # 保存单个函数代码
        func_path = os.path.join(output_dir, 'functions', f'{current_function_id + 1:02d}_{func_name}.py')
        os.makedirs(os.path.dirname(func_path), exist_ok=True)
        with open(func_path, 'w', encoding='utf-8') as f:
            f.write(func_code)
        print(f"  [Code] Saved to {func_path}")
        
        # 记录日志
        save_workflow_log(output_dir, f'function_{current_function_id + 1}_completed', {
            'function_id': current_function_id + 1,
            'function_name': func_name,
            'code_length': len(func_code)
        })
        
        # 更新函数代码字典
        updated_function_codes = {current_function_id: func_code}
        
        # 返回增量更新
        return {
            'current_function_id': current_function_id + 1,
            'function_codes': updated_function_codes,
            'next_action': 'generate_next_function',
            'messages': [Message(
                role='assistant',
                content=f'Generated function {current_function_id + 1}/{total_functions}: {func_name}',
                agent_name='function_code_generator',
                metadata={
                    'function_id': current_function_id + 1,
                    'function_name': func_name
                }
            )]
        }
    
    except Exception as e:
        error_msg = f"Failed to generate function {func_name}: {str(e)}"
        print(f"  [Error] {error_msg}")
        
        # 记录错误但继续下一个函数
        return {
            'current_function_id': current_function_id + 1,
            'function_codes': {current_function_id: f"# Error generating function\n# {error_msg}"},
            'next_action': 'generate_next_function',
            'errors': [error_msg],
            'messages': [Message(
                role='system',
                content=error_msg,
                agent_name='function_code_generator'
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
    algorithm_design_max_rounds = state.get('algorithm_design_max_rounds', 2)
    output_dir = state['output_dir']
    
    # 创建 ProblemSolving Agent（负责算法方案设计）
    ps = ProblemSolving(llm)
    
    print(f"[Algorithm Design] Iteration with {algorithm_design_max_rounds} rounds")
    algorithm_solution = ps.solving(problem_str, round=algorithm_design_max_rounds)
    
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
    algorithm_solution = state.get('algorithm_solution', '')
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
    """
    求解单个任务（渐进式开发，只在最后测试）
    
    核心思路：
    1. Task 1: 基于模板生成初始实现
    2. Task 2-N: 基于上一个任务的代码继续开发，累积功能
    3. 只在最后一个任务（Task N）完成后进行完整测试
    
    任务衔接信息：
    - previous_code: 上一个任务生成的完整代码
    - algorithm_solution: 整体算法方案（所有任务共享）
    - task_description: 当前任务的具体要求
    - completed_tasks: 已完成的任务描述列表（用于上下文）
    """
    llm = state['llm']
    execution_order = state['execution_order']
    current_task_id = state['current_task_id']
    task_descriptions = state['task_descriptions']
    algorithm_solution = state.get('algorithm_solution', '')
    output_dir = state['output_dir']
    task_results = state.get('task_results', {})
    
    task_id = execution_order[current_task_id]
    total_tasks = len(execution_order)
    is_final_task = (task_id == total_tasks)
    
    print(f"\n{'='*80}")
    print(f"[Task {task_id}/{total_tasks}] {'(Final Task)' if is_final_task else ''}")
    print(f"{'='*80}")
    
    # 获取当前任务描述
    task_description = task_descriptions[task_id - 1] if task_id <= len(task_descriptions) else ""
    
    # ============================================================
    # 读取上一个任务的代码（Task 2 及以后）
    # ============================================================
    previous_code = None
    previous_task_description = None
    completed_task_summaries = []
    
    if task_id > 1:
        # 获取上一个任务的结果
        prev_task_id = task_id - 1
        prev_result = task_results.get(prev_task_id, {})
        previous_code = prev_result.get('task_code', '')
        previous_task_description = task_descriptions[prev_task_id - 1] if prev_task_id <= len(task_descriptions) else ""
        
        if not previous_code:
            print(f"  [Warning] No code found from Task {prev_task_id}, starting fresh")
            previous_code = None
        else:
            print(f"  [Context] Loading code from Task {prev_task_id} ({len(previous_code)} chars)")
        
        # 构建已完成任务的摘要（用于上下文）
        for tid in range(1, task_id):
            task_desc = task_descriptions[tid - 1] if tid <= len(task_descriptions) else f"Task {tid}"
            completed_task_summaries.append(f"Task {tid}: {task_desc[:100]}...")
    
    # ============================================================
    # 读取模板（Task 1）
    # ============================================================
    template_code = None
    if task_id == 1:
        # 第一个任务尝试从模板目录读取基础模板
        template_dir = state.get('template_dir', '')
        template_path = os.path.join(template_dir, 'solver_template.py')
        if os.path.exists(template_path):
            with open(template_path, 'r', encoding='utf-8') as f:
                template_code = f.read()
            print(f"  [Template] Loaded from {template_path} ({len(template_code)} chars)")
        else:
            print(f"  [Warning] No template found, agent will start from scratch")
    
    # ============================================================
    # 生成当前任务的代码
    # ============================================================
    print(f"  [Code Generation] Generating code for Task {task_id}...")
    
    # 构建 prompt 的上下文信息
    code_generation_context = {
        'task_id': task_id,
        'total_tasks': total_tasks,
        'is_final_task': is_final_task,
        'task_description': task_description,
        'algorithm_solution': algorithm_solution,
        'template_code': template_code,  # Task 1 使用
        'previous_code': previous_code,  # Task 2+ 使用
        'previous_task_description': previous_task_description,
        'completed_task_summaries': completed_task_summaries,
        'problem_dir': state.get('problem_dir', ''),
        'output_dir': output_dir,
        'template_dir': state.get('template_dir', ''),
        'runtime_dir': state.get('runtime_dir', ''),
    }
    
    # 调用代码生成函数
    task_code, generation_info = generate_task_code(llm, code_generation_context)
    
    print(f"  [Code Generation] Done ({len(task_code)} chars)")
    
    # ============================================================
    # 保存代码（所有任务都保存，用于调试和追踪）
    # ============================================================
    code_path = save_task_code(output_dir, task_id, task_code)
    print(f"  [Code] Saved to {code_path}")
    
    # ============================================================
    # 测试代码（只在最后一个任务执行）
    # ============================================================
    is_pass = False
    execution_result = "Not tested yet (intermediate task)"
    validation_result = {}
    
    # ============================================================
    # 保存任务结果
    # ============================================================
    result_with_desc = {
        'task_code': task_code,
        'task_code_path': code_path,
        'task_description': task_description,
        'is_pass': is_pass,
        'execution_result': execution_result,
        'validation_result': validation_result,
        'generation_info': generation_info,
        'is_final_task': is_final_task,
        'previous_task_id': task_id - 1 if task_id > 1 else None,
    }
    
    result_path = save_task_result(output_dir, task_id, task_description, result_with_desc)
    print(f"  [Result] Saved to {result_path}")
    
    # 记录日志
    save_workflow_log(output_dir, f'task_{task_id}_completed', {
        'task_id': task_id,
        'is_final_task': is_final_task,
        'is_pass': is_pass if is_final_task else None,
        'code_length': len(task_code),
        'execution_result': str(execution_result)[:200] if is_final_task else 'skipped'
    })
    
    # ============================================================
    # 决定下一步动作
    # ============================================================
    if is_final_task:
        # 最后一个任务完成
        if is_pass:
            next_action = 'integrate'  # 测试通过，进入集成阶段
            print(f"\n✓ All tasks completed successfully!")
        else:
            # 测试失败，可以选择重试或直接集成
            max_validation_attempts = state.get('max_validation_attempts', 2)
            current_retry = state.get('current_retry', 0)
            
            if current_retry < max_validation_attempts - 1:
                next_action = 'retry_final_task'  # 重试最后一个任务
                print(f"\n✗ Final task failed (retry {current_retry + 1}/{max_validation_attempts})")
            else:
                next_action = 'integrate'  # 超过重试次数，强制进入集成
                print(f"\n✗ Final task failed after {max_validation_attempts} attempts, proceeding to integration")
    else:
        # 中间任务完成，继续下一个
        next_action = 'solve_next'
        print(f"\n→ Task {task_id} completed, moving to Task {task_id + 1}")
    
    # ============================================================
    # 返回增量更新
    # ============================================================
    return {
        'current_task_id': current_task_id + 1,
        'task_results': {task_id: result_with_desc},
        'next_action': next_action,
        'current_retry': state.get('current_retry', 0) + (1 if not is_pass and is_final_task else 0),
        'messages': [Message(
            role='assistant',
            content=f'Task {task_id} {"completed and tested" if is_final_task else "completed"}',
            agent_name='task_solver',
            metadata={
                'task_id': task_id,
                'is_final': is_final_task,
                'is_pass': is_pass if is_final_task else None
            }
        )]
    }



# ============ 代码集成 ============

def code_integration_node(state: AgentState) -> AgentState:
    """
    代码集成节点 - 将启发式函数集成到 HeuristicSolver 模板中
    
    工作流程：
    1. 获取之前生成的 HeuristicSolver 模板（空 solve() 方法）
    2. 获取所有已生成的启发式函数代码
    3. 使用 LLM 将函数集成到模板中，填充 solve() 方法
    4. 保存完整的 HeuristicSolver 实现
    
    设计原则：
    - 通过 LLM 理解函数功能和模板结构，智能集成
    - 不硬编码集成逻辑，保持灵活性
    """
    llm = state['llm']
    output_dir = state['output_dir']
    
    # 获取模板和函数代码
    heuristic_template = state.get('heuristic_template_code', '')
    heuristic_functions = state.get('heuristic_functions', [])
    function_codes = state.get('function_codes', {})
    
    if not heuristic_template:
        print("[Integration] Warning: No HeuristicSolver template found")
        return {
            'next_action': 'end',
            'errors': ['No template available for integration'],
            'messages': [Message(
                role='system',
                content='Integration failed: No template',
                agent_name='code_integrator'
            )]
        }
    
    if not heuristic_functions:
        print("[Integration] No heuristic functions to integrate, using template as-is")
        # 直接保存模板
        final_code = heuristic_template
    else:
        print(f"[Integration] Integrating {len(heuristic_functions)} functions into HeuristicSolver...")
        
        # 收集所有函数代码
        all_function_codes = []
        for i, func_info in enumerate(heuristic_functions):
            func_code = function_codes.get(i, f"# Function {func_info['name']} not generated")
            all_function_codes.append({
                'name': func_info['name'],
                'description': func_info['description'],
                'code': func_code
            })
        
        # 使用 LLM 集成代码
        final_code = integrate_functions_into_template(
            llm,
            heuristic_template,
            all_function_codes,
            heuristic_functions
        )
        
        print(f"[Integration] Integration complete ({len(final_code)} chars)")
    
    # 保存最终代码
    solver_code_path = os.path.join(output_dir, 'HeuristicSolver_final.py')
    os.makedirs(os.path.dirname(solver_code_path), exist_ok=True)
    
    with open(solver_code_path, 'w', encoding='utf-8') as f:
        f.write(final_code)
    
    print(f"[Integration] Saved to {solver_code_path}")
    
    # 记录日志
    save_workflow_log(output_dir, 'integration_completed', {
        'function_count': len(heuristic_functions),
        'solver_path': solver_code_path,
        'code_length': len(final_code)
    })
    
    print(f"\n{'='*80}")
    print(f"✓ HeuristicSolver Complete!")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Final solver: {solver_code_path}")
    print(f"Functions integrated: {len(heuristic_functions)}")
    print(f"{'='*80}\n")
    
    # 返回增量更新
    return {
        'solver_code': final_code,
        'solver_code_path': solver_code_path,
        'next_action': 'evaluate',
        'messages': [Message(
            role='assistant',
            content=f'HeuristicSolver integrated and saved to {solver_code_path}',
            agent_name='code_integrator'
        )]
    }


def integrate_functions_into_template(
    llm,
    template_code: str,
    function_codes: list,
    function_specs: list
) -> str:
    """
    使用 LLM 将启发式函数集成到 HeuristicSolver 模板中
    
    设计原则（关注点分离）：
    - LLM 理解模板结构和函数功能，智能填充 solve() 方法
    - 不硬编码集成逻辑，保持对不同问题的通用性
    
    Args:
        llm: 语言模型实例
        template_code: HeuristicSolver 模板代码（空 solve() 方法）
        function_codes: 函数代码列表 [{'name': ..., 'description': ..., 'code': ...}, ...]
        function_specs: 函数规范列表（来自架构设计）
    
    Returns:
        完整的 HeuristicSolver 代码字符串
    """
    from MMAgent.prompt import HEURISTIC_CODE_INTEGRATION_PROMPT
    
    # 构建函数代码字符串
    functions_code_str = ""
    for func in function_codes:
        functions_code_str += f"# ============ Function: {func['name']} ============\n"
        functions_code_str += f"# Description: {func['description']}\n\n"
        functions_code_str += func['code'] + "\n\n"
    
    # 构建函数依赖关系描述
    function_dependencies_str = ""
    for i, spec in enumerate(function_specs):
        deps = spec.get('dependencies', [])
        deps_str = ", ".join(deps) if deps else "None"
        function_dependencies_str += f"{i+1}. {spec['name']}: depends on [{deps_str}]\n"
    
    # 构建提示词
    prompt = HEURISTIC_CODE_INTEGRATION_PROMPT.format(
        template_code=template_code,
        functions_code=functions_code_str,
        function_count=len(function_codes),
        function_dependencies=function_dependencies_str
    )
    
    # 调用 LLM 生成集成代码
    response = llm.generate(prompt)
    integrated_code = response.content if hasattr(response, 'content') else str(response)
    
    # 清理可能的 markdown 标记
    if '```python' in integrated_code:
        integrated_code = integrated_code.split('```python')[1].split('```')[0].strip()
    elif '```' in integrated_code:
        integrated_code = integrated_code.split('```')[1].split('```')[0].strip()
    
    return integrated_code


# ============ 启发式求解器评估 ============
def heuristic_evaluation_node(state: AgentState) -> AgentState:
    """
    启发式求解器评估节点
    
    工作流程：
    1. 获取生成的 HeuristicSolver_final.py 路径
    2. 从 task_dir 导入 evaluate_pyomo_solver 模块
    3. 使用 evaluate_solver_from_file 函数评估求解器
    4. 将评估结果保存到 state
    
    设计原则：
    - 使用 __init__.py 统一管理包导入
    - 从 state 获取评估配置（拓扑、INA预算、任务数量等）
    - 将评估结果存储在 state['evaluation_results'] 中
    """

    
    output_dir = state['output_dir']
    solver_code_path = state.get('solver_code_path', '')
    task_dir = state.get('task_dir', '')
    evaluation_config = state.get('evaluation_config', {})
    
    print(f"\n{'='*80}")
    print(f"[Evaluation] Starting HeuristicSolver evaluation...")
    print(f"{'='*80}")
    print(f"Solver path: {solver_code_path}")
    print(f"Task directory: {task_dir}")
    
    '''
    output_dir = "./output"
    solver_code_path = "/home/hyn/LLM-MM-Agent-LLMINA-clean/output/LLMINA_20251114-140534/HeuristicSolver_final.py"
    task_dir = "/home/hyn/LLM-MM-Agent-LLMINA-clean/MMBench/problem/LLMINA/runtime"
    evaluation_config = state.get('evaluation_config', {})
    '''


    # 检查求解器文件是否存在
    if not solver_code_path or not os.path.exists(solver_code_path):
        error_msg = f"HeuristicSolver file not found: {solver_code_path}"
        print(f"✗ {error_msg}")
        return {
            'evaluation_results': {
                'evaluation_success': False,
                'error': error_msg
            },
            'next_action': 'end',
            'errors': [error_msg],
            'messages': [Message(
                role='system',
                content=error_msg,
                agent_name='heuristic_evaluator'
            )]
        }
    
    try:
        # 动态导入 evaluate_pyomo_solver 模块
        # 使用 importlib 从 task_dir 导入
        import sys
        import importlib.util
        
        evaluate_module_path = os.path.join(task_dir, 'evaluate_pyomo_solver.py')
        
        if not os.path.exists(evaluate_module_path):
            error_msg = f"evaluate_pyomo_solver.py not found in {task_dir}"
            print(f"✗ {error_msg}")
            return {
                'evaluation_results': {
                    'evaluation_success': False,
                    'error': error_msg
                },
                'next_action': 'end',
                'errors': [error_msg],
                'messages': [Message(
                    role='system',
                    content=error_msg,
                    agent_name='heuristic_evaluator'
                )]
            }
        
        # 临时添加 task_dir 到 sys.path 以确保依赖模块可以被导入
        if task_dir not in sys.path:
            sys.path.insert(0, task_dir)
        
        # 动态加载模块
        spec = importlib.util.spec_from_file_location("evaluate_pyomo_solver", evaluate_module_path)
        evaluate_module = importlib.util.module_from_spec(spec)
        sys.modules["evaluate_pyomo_solver"] = evaluate_module
        evaluate_module.__package__ = "MMBench.problem.LLMINA.runtime"

        spec.loader.exec_module(evaluate_module)
        
        # 获取评估函数
        evaluate_solver_from_file = evaluate_module.evaluate_solver_from_file
        
        print(f"✓ Successfully imported evaluate_pyomo_solver module")
        
        # 从 evaluation_config 获取评估参数（带默认值）
        topo_name = evaluation_config.get('topo_name', 'FatTree')
        ina_num_list = evaluation_config.get('ina_num_list', [3])
        jobs_num_list = evaluation_config.get('jobs_num_list', [6])
        instances_num = evaluation_config.get('instances_num', 2)
        solver_name = evaluation_config.get('solver_name', 'gurobi')
        time_limit = evaluation_config.get('time_limit', 300)
        mip_gap = evaluation_config.get('mip_gap', 0.01)
        verbose = evaluation_config.get('verbose', True)
        
        print(f"\n[Evaluation] Configuration:")
        print(f"  - Topology: {topo_name}")
        print(f"  - INA budgets: {ina_num_list}")
        print(f"  - Job counts: {jobs_num_list}")
        print(f"  - Instances per config: {instances_num}")
        print(f"  - Solver: {solver_name}")
        print(f"  - Time limit: {time_limit}s")
        print(f"  - MIP gap: {mip_gap}")
        print(f"\n[Evaluation] Running evaluation...\n")
        
        # 调用评估函数
        evaluation_results = evaluate_solver_from_file(
            solver_file_path=solver_code_path,
            class_name="HeuristicSolver",
            topo_name=topo_name,
            ina_num_list=ina_num_list,
            jobs_num_list=jobs_num_list,
            instances_num=instances_num,
            solver_name=solver_name,
            time_limit=time_limit,
            mip_gap=mip_gap,
            verbose=verbose,
        )
        
        # 清理 sys.path
        if task_dir in sys.path:
            sys.path.remove(task_dir)
        
        print(f"\n{'='*80}")
        print(f"✓ Evaluation completed successfully!")
        print(f"{'='*80}\n")
        
        # 提取性能摘要
        performance_summary = {}
        for key, value in evaluation_results.items():
            if isinstance(value, dict) and 'average_makespan' in value:
                performance_summary[key] = {
                    'average_makespan': value['average_makespan'],
                    'average_solve_time': value['average_solve_time'],
                    'success_count': value['success_count']
                }
        
        # 记录日志
        save_workflow_log(output_dir, 'evaluation_completed', {
            'solver_path': solver_code_path,
            'evaluation_config': evaluation_config,
            'performance_summary': performance_summary
        })
        
        # 返回增量更新
        return {
            'evaluation_results': {
                'evaluation_success': True,
                'raw_results': evaluation_results,
                'performance_summary': performance_summary
            },
            'next_action': 'end',
            'messages': [Message(
                role='assistant',
                content=f'HeuristicSolver evaluation completed with {len(performance_summary)} configurations',
                agent_name='heuristic_evaluator'
            )]
        }
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_msg = f"Evaluation failed: {str(e)}"
        
        print(f"\n{'='*80}")
        print(f"✗ {error_msg}")
        print(f"{'='*80}")
        print(f"Traceback:\n{error_traceback}")
        
        # 清理 sys.path（即使出错也要清理）
        if task_dir in sys.path:
            sys.path.remove(task_dir)
        
        # 记录错误日志
        save_workflow_log(output_dir, 'evaluation_failed', {
            'solver_path': solver_code_path,
            'error': error_msg,
            'traceback': error_traceback
        })
        
        # 返回错误状态
        return {
            'evaluation_results': {
                'evaluation_success': False,
                'error': error_msg,
                'traceback': error_traceback
            },
            'next_action': 'end',
            'errors': [error_msg],
            'messages': [Message(
                role='system',
                content=error_msg,
                agent_name='heuristic_evaluator'
            )]
        }
