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
import sys
import importlib.util
import traceback
import inspect
import textwrap
# 包导入 - 使用相对导入
from .state import AgentState, Message
from typing import Dict, List, Any, Type, Optional
from collections import defaultdict

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
    pyomo_path = os.path.join(task_dir, 'ModelSolver.py')
    with open(pyomo_path, 'r', encoding='utf-8') as f:
        pyomo_code = f.read()
    
    # 构建问题描述
    problem_str = PROBLEM_DESCRIPTION_PROMPT.format(
        problem_background=problem['background'],
        problem_requirement=problem['problem_requirement'],
        variable_description=problem['variable_description'],
        problem_formulation=problem['problem_formulation'],
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

# ============ 启发式架构设计 ============

def heuristic_architect_node(state: AgentState) -> AgentState:
    """
    启发式架构设计节点
    
    不编写实现代码，只负责软件架构设计。
    分析问题、Pyomo代码和ModelSolver模板，输出结构化的函数签名列表（JSON格式），
    定义启发式算法的模块化组件。
    
    此版本适配的JSON结构:
    {
        "problem_analysis": "...",
        "strategy_overview": "...",
        "function_architecture": [ {...}, {...} ]
    }
    """
    llm = state['llm']
    problem_str = state['problem_str']
    output_dir = state['output_dir']
    solver_class_code = state.get('pyomo_reference_code', '')
    
    print(f"[Heuristic Architect] Designing function architecture...")
    
    # 构建提示词（传入问题描述和模板代码）
    prompt = HEURISTIC_ARCHITECT_PROMPT.format(
        modeling_problem=problem_str,
        solver_class_code=solver_class_code
    )
    
    # 用于在失败时引用的原始响应内容
    raw_content = ""
    
    try:
        response = llm.generate(prompt)
        
        # 提取内容
        content = response.content if hasattr(response, 'content') else str(response)
        raw_content = content # 保存原始响应以备调试
        
        # 清理可能的markdown代码块
        # 移除 ```json 或 ``` 标记
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            # 尝试提取代码块内容
            code_blocks = re.findall(r'```(?:\w+)?\s*(.*?)```', content, re.DOTALL)
            if code_blocks:
                content = code_blocks[0].strip()
        
        # --- 修改点：查找JSON对象的边界 ---
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            raise ValueError(
                f"No JSON object found in LLM response. "
                f"Response starts with: {content[:200]}..."
            )
        
        # 提取JSON字符串
        json_str = content[json_start:json_end].strip()
        
        # 解析JSON
        try:
            parsed_json = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON syntax in LLM response: {str(e)}\n"
                f"JSON string (first 500 chars): {json_str[:500]}..."
            )
        
        # --- 修改点：验证新的顶级结构 ---
        if not isinstance(parsed_json, dict):
            raise ValueError(
                f"Expected a top-level JSON object, got {type(parsed_json).__name__}. "
                f"Content: {str(parsed_json)[:200]}..."
            )
            
        required_top_keys = ['problem_analysis', 'strategy_overview', 'function_architecture']
        missing_top_keys = [key for key in required_top_keys if key not in parsed_json]
        if missing_top_keys:
            raise ValueError(f"JSON object missing required top-level keys: {missing_top_keys}")

        problem_analysis = parsed_json['problem_analysis']
        strategy_overview = parsed_json['strategy_overview']
        heuristic_functions = parsed_json['function_architecture']

        # 验证 function_architecture 列表
        if not isinstance(heuristic_functions, list):
            raise ValueError(
                f"Expected 'function_architecture' to be a list, got {type(heuristic_functions).__name__}."
            )
        
        if len(heuristic_functions) == 0:
            raise ValueError("Function list 'function_architecture' is empty. At least one function is required.")
        
        # 验证每个函数的必需字段
        for i, func in enumerate(heuristic_functions):
            if not isinstance(func, dict):
                raise ValueError(
                    f"Function {i+1} is not a dictionary, got {type(func).__name__}"
                )
            
            # --- 修改点：更新必需字段 ---
            required_fields = ['name', 'strategic_role', 'description']
            missing_fields = [field for field in required_fields if field not in func]
            if missing_fields:
                raise ValueError(
                    f"Function {i+1} missing required fields: {missing_fields}. "
                    f"Function data: {func}"
                )
            
            # 验证字段类型
            if not isinstance(func['name'], str) or not func['name'].strip():
                raise ValueError(f"Function {i+1} has invalid 'name': {func.get('name')}")
            
            if not isinstance(func['strategic_role'], str) or not func['strategic_role'].strip():
                raise ValueError(f"Function {i+1} has invalid 'strategic_role': {func.get('strategic_role')}")
                
            if not isinstance(func['description'], str) or not func['description'].strip():
                raise ValueError(f"Function {i+1} has invalid 'description': {func.get('description')}")
        
        print(f"[Heuristic Architect] ✓ Successfully designed strategy and {len(heuristic_functions)} functions:")
        print(f"[Heuristic Architect] Strategy: {strategy_overview}")
        for func in heuristic_functions:
            print(f"  - {func['name']} (Role: {func['strategic_role']})")
        
        # 保存架构设计
        architecture_path = os.path.join(output_dir, 'heuristic_architecture.json')
        os.makedirs(output_dir, exist_ok=True)
        # --- 修改点：保存完整的JSON对象 ---
        with open(architecture_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_json, f, indent=2, ensure_ascii=False)
        print(f"[Heuristic Architect] Saved to {architecture_path}")
        
        # 记录日志
        save_workflow_log(output_dir, 'heuristic_architect_completed', {
            'problem_analysis': problem_analysis,
            'strategy_overview': strategy_overview,
            'function_count': len(heuristic_functions),
            'functions': [f['name'] for f in heuristic_functions]
        })
        
        # 返回增量更新 - 如果有函数则触发函数生成循环
        next_action = 'generate_next_function' if len(heuristic_functions) > 0 else 'integrate'
        
        # --- 修改点：规划新的返回字典以更新State ---
        return {
            'heuristic_strategy': {
                'problem_analysis': problem_analysis,
                'strategy_overview': strategy_overview
            },
            'heuristic_functions': heuristic_functions,
            'current_function_id': 0,  # 重置函数ID计数器
            'next_action': next_action,
            'messages': [Message(
                role='assistant',
                content=f'Designed heuristic strategy and {len(heuristic_functions)} functions.',
                agent_name='heuristic_architect',
                metadata={
                    'problem_analysis': problem_analysis,
                    'strategy_overview': strategy_overview,
                    'functions': [f['name'] for f in heuristic_functions]
                }
            )]
        }
    
    except Exception as e:
        error_msg = f"Failed to parse heuristic architecture from LLM response: {str(e)}"
        print(f"\n[Heuristic Architect] ✗ ERROR: {error_msg}")
        
        # 打印原始响应用于调试
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
    heuristic_template_code = state.get('pyomo_reference_code', '')
    
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
            completed_summary += json.dumps(prev_func) + "\n\n"
    else:
        completed_summary = "This is the first function. No previous functions have been completed yet."
    
    # 提取输入输出参数名
    member_variables_read = ", ".join(
        f"{mv['name']} ({mv['type']}): {mv['description']}"
        for mv in func_info.get('member_variables_read', [])
    )
    member_variables_written = ", ".join(
        f"{mv['name']} ({mv['type']}): {mv['description']}"
        for mv in func_info.get('member_variables_written', [])
    )
    
    # 构建提示词
    prompt = HEURISTIC_FUNCTION_CODE_GENERATION_PROMPT.format(
        modeling_problem=problem_str,
        solver_class_code=heuristic_template_code,
        function_id=current_function_id + 1,
        total_functions=total_functions,
        function_name=func_name,
        function_strategic_role=func_info['strategic_role'],
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

# ============ 代码集成 ============

def code_integration_node(state: AgentState) -> AgentState:
    """
    代码集成节点 - 将启发式函数直接集成到 ModelSolver 模板中（工具节点，不使用 LLM）
    
    工作流程：
    1. 获取 ModelSolver 模板代码
    2. 获取所有已生成的启发式函数代码
    3. 直接通过代码逻辑将函数添加为类方法
    4. 在 solve() 方法中按依赖顺序调用这些函数
    5. 保存完整的 ModelSolver 实现
    
    设计原则：
    - 不使用 LLM，避免长时间等待
    - 直接代码操作，确保集成的确定性和可靠性
    """
    output_dir = state['output_dir']
    
    # 获取模板和函数代码
    heuristic_template_code = state.get('pyomo_reference_code', '')
    heuristic_functions = state.get('heuristic_functions', [])
    function_codes = state.get('function_codes', {})
    
    if not heuristic_template_code:
        print("[Integration] Warning: No ModelSolver template found")
        return {
            'next_action': 'end',
            'errors': ['No template available for integration'],
            'messages': [Message(
                role='system',
                content='Integration failed: No template',
                agent_name='code_integrator'
            )]
        }
    
    print(f"[Integration] Integrating {len(heuristic_functions)} functions into ModelSolver...")
    
    # 直接集成代码（不使用 LLM）
    final_code = integrate_functions_directly(
        heuristic_template_code,
        heuristic_functions,
        function_codes
    )
    
    print(f"[Integration] Integration complete ({len(final_code)} chars)")
    
    # 保存最终代码
    solver_code_path = os.path.join(output_dir, 'ModelSolver_final.py')
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
    print(f"✓ ModelSolver Complete!")
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
            content=f'ModelSolver integrated and saved to {solver_code_path}',
            agent_name='code_integrator'
        )]
    }

def integrate_functions_directly(
    template_code: str,
    function_specs: list,
    function_codes: dict
) -> str:
    """
    直接将启发式函数集成到 ModelSolver 模板中（不使用 LLM）
    
    集成策略：
    1. 提取模板中的 import 语句和类定义部分
    2. 将生成的函数作为类的方法插入到类定义中
    3. 在 solve() 方法中按依赖顺序依次调用这些函数
    4. 返回最终的 best_solution
    
    Args:
        template_code: ModelSolver 模板代码
        function_specs: 函数规范列表（来自架构设计）
        function_codes: 函数代码字典 {index: code_string}
    
    Returns:
        完整的 ModelSolver 代码字符串
    """
    # 先保留模板原样
    new_code_lines = [template_code.rstrip(), ""]
    
    # 4. 添加生成的启发式函数作为类方法
    new_code_lines.append('    # ============ Heuristic Functions ============')
    new_code_lines.append('')
    
    for i, func_spec in enumerate(function_specs):
        func_code = function_codes.get(i, f"# Function {func_spec['name']} not generated")
        func_name = func_spec['name']
        
        new_code_lines.append(f"    # Function {i+1}: {func_name}")
        new_code_lines.append(f"    # Role: {func_spec['strategic_role']}")
        new_code_lines.append(f"    # Description: {func_spec['description']}")
        
        # 处理函数代码的缩进（添加一级缩进使其成为类方法）
        func_lines = func_code.split('\n')
        for func_line in func_lines:
            if func_line.strip():
                new_code_lines.append('    ' + func_line)
            else:
                new_code_lines.append('')
        
        new_code_lines.append('')

    return '\n'.join(new_code_lines)


# ============ 启发式求解器评估 ============
def heuristic_evaluation_node(state: AgentState) -> AgentState:
    """
    启发式求解器评估节点
    
    工作流程：
    1. 获取生成的 ModelSolver_final.py 路径
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
    print(f"[Evaluation] Starting ModelSolver evaluation...")
    print(f"{'='*80}")
    print(f"Solver path: {solver_code_path}")
    print(f"Task directory: {task_dir}")
    
    # 检查求解器文件是否存在
    if not solver_code_path or not os.path.exists(solver_code_path):
        error_msg = f"ModelSolver file not found: {solver_code_path}"
        print(f"✗ {error_msg}")
        return {
            'evaluation_results': {
                "evaluation_success": False,
                "error": error_msg,
                "traceback": None,
                "raw_results": None,
                "performance_summary": {},
            },
            'next_action': 'end',
            'messages': [Message(
                role='system',
                content=error_msg,
                agent_name='heuristic_evaluator'
            )]
        }

    # 动态导入 evaluate_pyomo_solver 模块
    # 使用 importlib 从 task_dir 导入

    evaluate_module_path = os.path.join(task_dir, 'evaluate_pyomo_solver.py')
    
    if not os.path.exists(evaluate_module_path):
        error_msg = f"evaluate_pyomo_solver.py not found in {task_dir}"
        print(f"✗ {error_msg}")
        return {
            'evaluation_results': {
                "evaluation_success": False,
                "error": error_msg,
                "traceback": None,
                "raw_results": None,
                "performance_summary": {},
            },
            'next_action': 'end',
            'messages': [Message(
                role='system',
                content=error_msg,
                agent_name='heuristic_evaluator'
            )]
        }
    
    task_dir_added = False

    # 临时添加 task_dir 到 sys.path 以确保依赖模块可以被导入
    if task_dir not in sys.path:
        sys.path.insert(0, task_dir)
        task_dir_added = True
    
    try:
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
        instances_num = evaluation_config.get('instances_num', 1)
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
        
        # 4. 调用评估函数，并捕获异常信息到字符串
        evaluation_results = None
        error_info = None

        # 调用评估函数
        try:
            evaluation_results = evaluate_solver_from_file(
                solver_file_path=solver_code_path,
                class_name="ModelSolver",
                topo_name=topo_name,
                ina_num_list=ina_num_list,
                jobs_num_list=jobs_num_list,
                instances_num=instances_num,
                solver_name=solver_name,
                time_limit=time_limit,
                mip_gap=mip_gap,
                verbose=verbose,
            )
        except Exception:
            # 把完整 traceback 写成字符串
            error_info = traceback.format_exc()
            print("✗ evaluate_solver_from_file raised an exception:")
            print(error_info)

        # 5. 判断评估是否成功
        if error_info is not None:
            # evaluate_solver_from_file 抛异常的情况
            error_msg = "Evaluation failed inside evaluate_solver_from_file"

            save_workflow_log(
                output_dir,
                "evaluation_failed",
                {
                    "solver_path": solver_code_path,
                    "evaluation_config": evaluation_config,
                    "error": error_msg,
                    "traceback": error_info,
                },
            )

            return {
                "evaluation_results": {
                    "evaluation_success": False,
                    "error": error_msg,
                    "traceback": error_info,
                    "raw_results": None,
                    "performance_summary": {},
                },
                "next_action": "fix",
                "errors": [error_msg],
                "messages": [
                    Message(
                        role="system",
                        content=error_msg,
                        agent_name="heuristic_evaluator",
                    )
                ],
            }

        if evaluation_results is None:
            # 没有抛异常，但返回 None，也视为评估失败
            error_msg = (
                "Evaluation returned None (no results). "
                "Solver may have failed without raising an exception."
            )

            print(f"✗ {error_msg}")

            save_workflow_log(
                output_dir,
                "evaluation_failed",
                {
                    "solver_path": solver_code_path,
                    "evaluation_config": evaluation_config,
                    "error": error_msg,
                    "traceback": None,
                },
            )

            return {
                "evaluation_results": {
                    "evaluation_success": False,
                    "error": error_msg,
                    "traceback": None,
                    "raw_results": None,
                    "performance_summary": {},
                },
                "next_action": "end",
                "errors": [error_msg],
                "messages": [
                    Message(
                        role="system",
                        content=error_msg,
                        agent_name="heuristic_evaluator",
                    )
                ],
            }

        # 走到这里说明评估成功拿到了结果
        print(f"\n{'=' * 80}")
        print(f"✓ Evaluation completed successfully!")
        print(f"{'=' * 80}\n")

        # 6. 提取性能摘要（只挑几个核心指标存到 state）
        performance_summary = {}
        if isinstance(evaluation_results, dict):
            for key, value in evaluation_results.items():
                if (
                    isinstance(value, dict)
                    and "average_makespan" in value
                    and "average_solve_time" in value
                ):
                    performance_summary[key] = {
                        "average_makespan": value["average_makespan"],
                        "average_solve_time": value["average_solve_time"],
                        "success_count": value.get("success_count", None),
                    }

        # 7. 记录日志
        save_workflow_log(
            output_dir,
            "evaluation_completed",
            {
                "solver_path": solver_code_path,
                "evaluation_config": evaluation_config,
                "performance_summary": performance_summary,
            },
        )

        # 8. 返回增量 state（LangGraph 会 merge 到全局 AgentState）
        return {
            "evaluation_results": {
                "evaluation_success": True,
                "error": None,
                "traceback": None,
                "raw_results": evaluation_results,
                "performance_summary": performance_summary,
            },
            "next_action": "end",
            "messages": [
                Message(
                    role="assistant",
                    content=(
                        f"ModelSolver evaluation completed with "
                        f"{len(performance_summary)} configurations"
                    ),
                    agent_name="heuristic_evaluator",
                )
            ],
        }
    finally:
            # 无论成功失败都清理 sys.path
            if task_dir_added and task_dir in sys.path:
                try:
                    sys.path.remove(task_dir)
                except ValueError:
                    pass


# ============ 代码修复节点 ============

def extract_error_function_from_traceback(traceback_str: str, solver_code: str = None) -> list:
    """
    从 traceback 中提取出错的函数名
    
    优化策略：
    1. 只提取在 ModelSolver_final.py 文件中出现的函数
    2. 过滤掉 Python 内部函数和其他模块的函数
    3. 优先返回最内层的用户函数
    
    Args:
        traceback_str: 完整的 traceback 字符串
        solver_code: 求解器代码（用于验证函数是否在用户代码中）
    
    Returns:
        出错函数名列表（按调用栈顺序）
    """
    function_names = []
    
    # 匹配 traceback 中的函数调用，格式：
    # File "xxx.py", line 123, in function_name
    pattern = r'File\s+"([^"]+)",\s+line\s+\d+,\s+in\s+(\w+)'
    matches = re.findall(pattern, traceback_str)
    
    # 提取所有在 ModelSolver_final.py 中的函数
    for file_path, func_name in matches:
        # 只关注 ModelSolver_final.py 文件中的函数
        if 'ModelSolver_final.py' in file_path:
            # 过滤掉一些非用户函数
            if func_name not in ['<module>', '__init__', 'solve'] and func_name not in function_names:
                # 如果提供了求解器代码，验证函数是否存在
                if solver_code:
                    func_pattern = rf'def\s+{re.escape(func_name)}\s*\('
                    if re.search(func_pattern, solver_code):
                        function_names.append(func_name)
                else:
                    function_names.append(func_name)
    
    # 如果没找到任何函数，尝试从所有 .py 文件中提取（回退策略）
    if not function_names:
        common_internal_funcs = [
            '<module>', '__init__', 'solve', 'exec_module', 'get_code', 
            'source_to_code', '_call_with_frames_removed', 'load_module',
            '__import__', 'import_module', '_find_and_load', '_load'
        ]
        
        for file_path, func_name in matches:
            if func_name not in common_internal_funcs and func_name not in function_names:
                # 验证函数是否在用户代码中
                if solver_code:
                    func_pattern = rf'def\s+{re.escape(func_name)}\s*\('
                    if re.search(func_pattern, solver_code):
                        function_names.append(func_name)
    
    return function_names


def extract_function_from_code(code: str, function_name: str) -> tuple:
    """
    从完整代码中提取指定函数的代码
    
    Args:
        code: 完整的代码字符串
        function_name: 要提取的函数名
    
    Returns:
        (函数代码, 起始行号, 结束行号, 缩进字符串) 或 (None, -1, -1, "") 如果未找到
    """
    lines = code.split('\n')
    
    # 查找函数定义行
    func_pattern = rf'^\s*def\s+{re.escape(function_name)}\s*\('
    start_line = -1
    indent_level = -1
    indent_prefix = ''
    
    for i, line in enumerate(lines):
        if re.match(func_pattern, line):
            start_line = i
            indent_level = len(line) - len(line.lstrip())
            indent_prefix = line[:indent_level]
            break
    
    if start_line == -1:
        return None, -1, -1, ''
    
    end_line = len(lines)
    for i in range(start_line + 1, len(lines)):
        line = lines[i]
        if line.strip():
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level:
                end_line = i
                break
    
    function_code = '\n'.join(lines[start_line:end_line])
    return function_code, start_line, end_line, indent_prefix


def replace_function_in_code(original_code: str, function_name: str, new_function_code: str) -> str:
    """
    替换代码中的指定函数并保持缩进一致
    
    Args:
        original_code: 原始完整代码
        function_name: 要替换的函数名
        new_function_code: 新的函数代码
    
    Returns:
        替换后的完整代码
    """
    lines = original_code.split('\n')
    old_func_code, start_line, end_line, indent_prefix = extract_function_from_code(original_code, function_name)
    
    if old_func_code is None:
        print(f"[Warning] Function {function_name} not found in code, appending...")
        formatted_code = textwrap.dedent(new_function_code.strip('\n'))
        return original_code + '\n\n' + formatted_code

    formatted_code = format_function_code_with_indent(new_function_code, indent_prefix)
    new_lines = lines[:start_line] + formatted_code.split('\n') + lines[end_line:]
    return '\n'.join(new_lines)


def format_function_code_with_indent(new_function_code: str, indent_prefix: str) -> str:
    """
    将新的函数代码去除多余缩进并应用目标缩进
    """
    stripped = new_function_code.strip('\n')
    if not stripped:
        return ''
    dedented = textwrap.dedent(stripped)
    lines = dedented.split('\n')
    if not indent_prefix:
        return '\n'.join(lines)
    formatted_lines = [f"{indent_prefix}{line}" if line.strip() else '' for line in lines]
    return '\n'.join(formatted_lines)




# ============ 代码修复辅助工具 ============

def load_solver_class_from_file(file_path: str, class_name: str = "HeuristicSolver") -> Type:
    """
    从 Python 文件动态加载 Solver 类
    
    Args:
        file_path: Python 文件的路径（绝对路径或相对路径）
        class_name: 要加载的类名（默认为 "HeuristicSolver"）
    
    Returns:
        加载的类对象
    
    Raises:
        FileNotFoundError: 如果文件不存在
        AttributeError: 如果类不存在
        Exception: 其他加载错误
    """
    file_path = Path(file_path).resolve()
    
    if not file_path.exists():
        raise FileNotFoundError(f"Solver 文件不存在: {file_path}")
    
    # 动态加载模块
    spec = importlib.util.spec_from_file_location("dynamic_solver_module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules["dynamic_solver_module"] = module
    spec.loader.exec_module(module)
    
    # 获取类
    if not hasattr(module, class_name):
        raise AttributeError(f"模块中不存在类 '{class_name}': {file_path}")
    
    solver_class = getattr(module, class_name)
    
    return solver_class


def extract_method_source_from_class(solver_class, method_name: str):
    """Extract method source using inspect for precise function repair."""
    if not solver_class or not hasattr(solver_class, method_name):
        return None, -1, -1
    try:
        method_obj = getattr(solver_class, method_name)
        source_lines, start_line = inspect.getsourcelines(method_obj)
    except (OSError, TypeError, AttributeError) as exc:
        print(f"[Code Fix] Warning: Unable to inspect {method_name}: {exc}")
        return None, -1, -1
    code = ''.join(source_lines)
    start_idx = max(start_line - 1, 0)
    end_idx = start_idx + len(source_lines)
    return code, start_idx, end_idx


def code_fix_node(state: AgentState) -> AgentState:
    """
    代码修复节点（优化版）
    
    策略：
    1. 从 traceback 中识别出错的函数
    2. 从 ModelSolver_final.py 中提取该函数的代码
    3. 只修复该函数（而非整个文件）
    4. 将修复后的函数替换回文件中
    
    优势：
    - 减少 LLM 输入长度，加快响应速度
    - 提高修复准确性（LLM 只需关注单个函数）
    - 降低成本
    """
    import re
    import traceback

    llm = state["llm"]
    output_dir = state.get("output_dir", "./output")
    solver_code_path = state.get("solver_code_path", "")
    evaluation_results = state.get("evaluation_results", {})
    fix_attempt_count = state.get("fix_attempt_count", 0)
    max_fix_attempts = state.get("max_fix_attempts", 3)
    problem_str = state["problem_str"]
    # heuristic_functions = state.get("heuristic_functions", [])
    
    with open(state['output_dir']+'/heuristic_architecture.json', "r", encoding="utf-8") as f:
        data = json.load(f)
    heuristic_functions = data.get("function_architecture", [])

    solver_class_name = state.get("solver_class_name", "ModelSolver")

    print(f"\n{'='*80}")
    print(f"[Code Fix] Starting code repair (Attempt {fix_attempt_count + 1}/{max_fix_attempts})")
    print(f"{'='*80}")

    # 1. 最大修复次数检查
    if fix_attempt_count >= max_fix_attempts:
        error_msg = f"Maximum fix attempts ({max_fix_attempts}) reached. Stopping."
        print(f"✗ {error_msg}")
        return {
            "fix_attempt_count": fix_attempt_count,
            "next_action": "end",
            "messages": [
                Message(
                    role="system",
                    content=error_msg,
                    agent_name="code_fixer",
                )
            ],
        }

    # 2. 提取错误信息
    if evaluation_results.get("evaluation_success", True):
        print(f"[Code Fix] No errors detected. Code is working correctly.")
        return {
            "next_action": "end",
            "messages": [
                Message(
                    role="assistant",
                    content="Code validation successful, no fixes needed.",
                    agent_name="code_fixer",
                )
            ],
        }

    error_info = evaluation_results.get("error", "Unknown error")
    traceback_info = evaluation_results.get("traceback", "")

    print(f"[Code Fix] Error detected:")
    print(f"  Error: {error_info}")
    if traceback_info:
        print(f"  Traceback (first 500 chars): {traceback_info[:500]}...")

    # 3. 读取当前求解器代码
    try:
        with open(solver_code_path, "r", encoding="utf-8") as f:
            full_code = f.read()
    except Exception as e:
        error_msg = f"Failed to read solver code: {str(e)}"
        print(f"✗ {error_msg}")
        return {
            "fix_attempt_count": fix_attempt_count,
            "next_action": "end",
            "messages": [
                Message(
                    role="system",
                    content=error_msg,
                    agent_name="code_fixer",
                )
            ],
        }

    solver_class = load_solver_class_from_file(solver_code_path, solver_class_name)

    # 4. 从 traceback 中提取出错的函数名（传入当前代码以验证）
    error_functions = extract_error_function_from_traceback(traceback_info, full_code)
    
    print(f"[Code Fix] Identified {len(error_functions)} potential error function(s): {error_functions}")
    
    # 5. 如果无法定位具体函数，回退到修复整个文件
    if not error_functions:
        print(f"[Code Fix] Cannot identify specific function, falling back to full file repair...")
        target_function_name = None
        target_function_code = full_code
        fix_scope = "entire file"
    else:
        # 优先修复最内层（最后）的函数
        target_function_name = error_functions[-1]
        target_function_code = None
        start_line = -1
        end_line = -1
        source_label = "source scan"

        if solver_class and hasattr(solver_class, target_function_name):

            inspected_code, inspected_start, inspected_end = extract_method_source_from_class(
                solver_class, target_function_name
            )
            if inspected_code:
                target_function_code = inspected_code
                start_line = inspected_start
                end_line = inspected_end
                source_label = "class inspection"

        if target_function_code is None:
            target_function_code, start_line, end_line, _ = extract_function_from_code(
                full_code, target_function_name
            )

        if target_function_code is None:
            print(f"[Code Fix] Function '{target_function_name}' not found in code, falling back to full file repair...")
            target_function_name = None
            target_function_code = full_code
            fix_scope = "entire file"
        else:
            print(f"[Code Fix] Targeting function: {target_function_name} (lines {start_line+1}-{end_line}, via {source_label})")
            fix_scope = f"function '{target_function_name}'"
        
        function_snippet = target_function_code.strip("\n")
        fullcode_snippet = full_code.strip("\n")

        fix_prompt = f"""You are a Python bug-fix assistant.

You are given:
1. The full source code of a Python file.
2. A traceback from running this file.

Your task:
- Fix ONLY the minimal amount of code necessary to eliminate the specific error(s) shown in the traceback.
- DO NOT redesign algorithms, DO NOT refactor, DO NOT reorder large code blocks.
- Prefer local changes near the lines mentioned in the traceback.
- Preserve the existing control flow and high-level logic as much as possible.

For example:
- If you see KeyError: 'foo', only change the specific dict access that causes this error,
  e.g., use `d.get("foo", default)` or add a local existence check.
- If you see AttributeError: 'NoneType' object has no attribute 'bar',
  add a local None-check near that line instead of rewriting the whole function.

Very important:
- Avoid deleting logical branches or loops unless they are clearly unreachable or directly cause the error.
- DO NOT remove entire blocks that compute intermediate values (e.g., load contributions, path enumerations) unless the traceback explicitly indicates they are wrong.

## Problem Description
{problem_str}

## Current ModelSolver Code 
```python
{fullcode_snippet}
```

## Error Information
{traceback_info}

## Task
Fix the function '{target_function_name}' to resolve the error. Focus on:
1. Syntax errors: Fix any Python syntax issues with minimal changes
2. Runtime errors: Fix attribute errors, type errors, undefined variables with local checks
3. Type mismatches: Add defensive checks or type conversions only where needed
4. Method calls: Ensure 'self.' is used for accessing instance attributes/methods

Guidelines:
- Make the SMALLEST possible change to fix the error
- Preserve all existing logic, loops, and computational blocks
- Only modify lines directly related to the error in the traceback
- Do NOT restructure or optimize the code
- Do NOT remove intermediate computation steps

## Expected Output
Provide ONLY the complete fixed function code wrapped in ```python ``` markers.
Do NOT include any explanations outside the code block.
"""

    print(f"[Code Fix] Requesting LLM to fix {fix_scope}...")

    try:
        response = llm.generate(fix_prompt)
        content = response.content if hasattr(response, "content") else str(response)

        # 7. 提取修复后的代码
        fixed_code = ""
        if "```python" in content:
            code_blocks = content.split("```python")[1:]
            if code_blocks:
                fixed_code = code_blocks[0].split("```")[0].strip()
        elif "```" in content:
            code_blocks = re.findall(r"```(?:\w+)?\s*(.*?)```", content, re.DOTALL)
            if code_blocks:
                fixed_code = code_blocks[0].strip()
        else:
            fixed_code = content.strip()

        if not fixed_code:
            raise ValueError("Extracted code is empty")

        print(f"[Code Fix] ✓ Successfully generated fixed code ({len(fixed_code)} chars)")

        # 8. 整合修复后的代码
        if target_function_name:
            # 只替换该函数
            final_code = replace_function_in_code(full_code, target_function_name, fixed_code)
            print(f"[Code Fix] Replaced function '{target_function_name}' in file")
        else:
            # 替换整个文件
            final_code = fixed_code
            print(f"[Code Fix] Replaced entire file")

        # 9. 保存修复后的代码
        fixed_code_path = solver_code_path.replace(
            ".py", f"_fix{fix_attempt_count + 1}.py"
        )
        with open(fixed_code_path, "w", encoding="utf-8") as f:
            f.write(final_code)

        # 第一次修复时备份原代码
        if fix_attempt_count == 0:
            backup_path = solver_code_path.replace(".py", "_original.py")
            with open(backup_path, "w", encoding="utf-8") as f:
                f.write(full_code)
            print(f"[Code Fix] Original code backed up to: {backup_path}")

        # 覆盖原文件
        with open(solver_code_path, "w", encoding="utf-8") as f:
            f.write(final_code)

        print(f"[Code Fix] Fixed code saved to: {solver_code_path}")
        print(f"[Code Fix] Also saved as: {fixed_code_path}")

        # 10. 记录日志
        save_workflow_log(
            output_dir,
            f"code_fix_attempt_{fix_attempt_count + 1}",
            {
                "fix_attempt": fix_attempt_count + 1,
                "error_info": error_info,
                "target_function": target_function_name or "entire_file",
                "fix_scope": fix_scope,
                "fixed_code_path": fixed_code_path,
                "solver_code_path": solver_code_path,
            },
        )

        # 11. 返回状态更新，触发重新评估
        return {
            "fix_attempt_count": fix_attempt_count + 1,
            "previous_errors": [error_info],
            "next_action": "re_evaluate",
            "messages": [
                Message(
                    role="assistant",
                    content=(
                        f"Fixed {fix_scope} (attempt {fix_attempt_count + 1}). "
                        f"Re-evaluating..."
                    ),
                    agent_name="code_fixer",
                    metadata={
                        "fix_attempt": fix_attempt_count + 1,
                        "target_function": target_function_name,
                        "fixed_code_path": fixed_code_path,
                    },
                )
            ],
        }

    except Exception as e:
        error_traceback = traceback.format_exc()
        error_msg = f"Code fix failed: {str(e)}"

        print(f"\n{'='*80}")
        print(f"✗ {error_msg}")
        print(f"{'='*80}")
        print(f"Traceback:\n{error_traceback}")

        save_workflow_log(
            output_dir,
            f"code_fix_failed_attempt_{fix_attempt_count + 1}",
            {
                "fix_attempt": fix_attempt_count + 1,
                "error": error_msg,
                "traceback": error_traceback,
            },
        )

        return {
            "fix_attempt_count": fix_attempt_count + 1,
            "next_action": "end",
            "messages": [
                Message(
                    role="system",
                    content=error_msg,
                    agent_name="code_fixer",
                )
            ],
        }


# ============ 约束分析节点 ============

def constraint_analyzer_node(state: AgentState) -> AgentState:
    """
    约束分析节点 - 调用 runtime 模块运行多实例 Pyomo 求解并使用 ConstraintAnalyzer
    生成紧约束/松约束统计。
    """
    output_dir = state['output_dir']
    solver_code_path = state.get('solver_code_path', '')
    task_dir = state.get('task_dir', '')
    evaluation_config = state.get('evaluation_config', {}) or {}
    constraint_config = state.get('constraint_analysis_config', {}) or {}
    solver_class_name = state.get('solver_class_name', 'ModelSolver')

    print(f"\n{'='*80}")
    print("[Constraint Analysis] Running multi-instance MILP analysis...")
    print(f"{'='*80}")

    def _build_error_state(message: str, extra: Optional[Dict[str, Any]] = None) -> AgentState:
        print(f"✗ {message}")
        payload = {
            'constraint_analysis': {
                'success': False,
                'error': message,
            },
            'next_action': 'end',
            'messages': [Message(
                role='system',
                content=message,
                agent_name='constraint_analyzer'
            )]
        }
        if extra:
            payload['constraint_analysis'].update(extra)
        return payload

    def _ensure_list(value, default):
        if value is None:
            return list(default)
        if isinstance(value, list):
            return value
        return [value]

    def _sanitize_name(name: str) -> str:
        return re.sub(r'[^0-9A-Za-z_.-]+', '_', name)

    def _hydrate_solver_fields(solver_obj):
        problem_data = getattr(solver_obj, 'problem_data', {}) or {}
        instance_data = problem_data.get('instance', {}) or {}
        solver_obj.instance = instance_data
        solver_obj.network = problem_data.get('network')
        solver_obj.K = problem_data.get('K')
        solver_obj.jobs_num = problem_data.get('jobs_num', len(instance_data.get('workers_num', [])))
        solver_obj.Cs = problem_data.get('Cs')
        solver_obj.Ps = problem_data.get('Ps')
        solver_obj.workers_id = instance_data.get('workers_id', [])
        solver_obj.ps_id = instance_data.get('ps_id', [])
        solver_obj.workers_num = instance_data.get('workers_num', [])
        solver_obj.jobs_size = instance_data.get('jobs_size', [])
        solver_obj.workers_num_align = max(solver_obj.workers_num) if solver_obj.workers_num else 0



    def _has_violations(result: Dict[str, Any]) -> bool:
        if not isinstance(result, dict):
            return False
        if isinstance(result.get('violations'), list) and result['violations']:
            return True
        for key in ('violations_ina', 'violations_ps'):
            if isinstance(result.get(key), list) and result[key]:
                return True
        return False

    def _is_constraint_tight(name: str, result: Dict[str, Any]) -> bool:
        if not isinstance(result, dict):
            return False
        status = result.get('status')
        if status == 'not_available':
            return False
        if name == 'ina_budget':
            slack = result.get('slack')
            return isinstance(slack, (int, float)) and abs(slack) < epsilon
        if name == 'worker_choice':
            return status == 'ok' and not result.get('violations')
        if name == 'ina_deployment':
            return status == 'ok' and not result.get('violations')
        if name == 'makespan':
            tight_jobs = result.get('tight_jobs')
            return isinstance(tight_jobs, list) and len(tight_jobs) > 0
        if 'is_tight' in result:
            return bool(result['is_tight'])
        if 'all_tight' in result:
            return bool(result['all_tight'])
        if 'satisfied' in result:
            return bool(result['satisfied'])
        if name in {'worker_rate'}:
            total = result.get('total_count')
            tight = result.get('tight_count')
            return isinstance(total, int) and total > 0 and tight == total
        if name in {'ina_capacity', 'link_capacity'}:
            return result.get('num_tight', 0) > 0
        return False

    def _extract_slack(result: Dict[str, Any]):
        slack = result.get('slack') if isinstance(result, dict) else None
        if isinstance(slack, (int, float)):
            return float(slack)
        return None

    def _update_constraint_stats(stats_map, analysis: Dict[str, Any]):
        for name, data in analysis.items():
            if isinstance(data, dict) and data.get('status') == 'not_available':
                continue
            entry = stats_map[name]
            entry['instances'] += 1
            if _is_constraint_tight(name, data):
                entry['tight_instances'] += 1
            if _has_violations(data):
                entry['violation_instances'] += 1
            slack_value = _extract_slack(data)
            if slack_value is not None:
                entry['slack_samples'].append(slack_value)

    if not solver_code_path or not os.path.exists(solver_code_path):
        return _build_error_state(f"ModelSolver file not found: {solver_code_path}")

    if not task_dir or not os.path.isdir(task_dir):
        return _build_error_state('Task directory missing for constraint analysis')

    evaluate_module_path = os.path.join(task_dir, 'evaluate_pyomo_solver.py')
    constraint_module_path = os.path.join(task_dir, 'constraint_analyzer.py')

    if not os.path.exists(evaluate_module_path):
        return _build_error_state(f"evaluate_pyomo_solver.py not found in {task_dir}")
    if not os.path.exists(constraint_module_path):
        return _build_error_state(f"constraint_analyzer.py not found in {task_dir}")

    combined_config = dict(evaluation_config)
    combined_config.update(constraint_config)
    topo_name = combined_config.get('topo_name', 'FatTree')
    ina_num_list = _ensure_list(combined_config.get('ina_num_list'), [3]) or [3]
    jobs_num_list = _ensure_list(combined_config.get('jobs_num_list'), [6]) or [6]
    instances_num = max(1, int(combined_config.get('instances_num', 1)))
    solver_name = combined_config.get('solver_name', 'gurobi')
    time_limit = combined_config.get('time_limit', 300)
    mip_gap = combined_config.get('mip_gap', 0.01)
    verbose = combined_config.get('verbose', False)
    epsilon = combined_config.get('epsilon', 1e-4)

    analysis_dir = os.path.join(output_dir, 'constraint_analysis')
    instances_dir = os.path.join(analysis_dir, 'instances')
    os.makedirs(instances_dir, exist_ok=True)

    task_dir_added = False
    try:
        if task_dir not in sys.path:
            sys.path.insert(0, task_dir)
            task_dir_added = True

        eval_spec = importlib.util.spec_from_file_location('constraint_runtime_evaluator', evaluate_module_path)
        if not eval_spec or not eval_spec.loader:
            raise ImportError('Failed to load evaluate_pyomo_solver module')
        evaluate_module = importlib.util.module_from_spec(eval_spec)
        sys.modules['constraint_runtime_evaluator'] = evaluate_module
        evaluate_module.__package__ = 'MMBench.problem.LLMINA.runtime'
        eval_spec.loader.exec_module(evaluate_module)

        analyzer_spec = importlib.util.spec_from_file_location('constraint_runtime_analyzer', constraint_module_path)
        if not analyzer_spec or not analyzer_spec.loader:
            raise ImportError('Failed to load constraint_analyzer module')
        analyzer_module = importlib.util.module_from_spec(analyzer_spec)
        sys.modules['constraint_runtime_analyzer'] = analyzer_module
        analyzer_module.__package__ = 'MMBench.problem.LLMINA.runtime'
        analyzer_spec.loader.exec_module(analyzer_module)

        ConstraintAnalyzer = getattr(analyzer_module, 'ConstraintAnalyzer', None)
        SolverEvaluation = getattr(evaluate_module, 'SolverEvaluation', None)
        load_solver_class_from_file = getattr(evaluate_module, 'load_solver_class_from_file', None)

        if ConstraintAnalyzer is None or SolverEvaluation is None or load_solver_class_from_file is None:
            raise AttributeError('Runtime evaluation modules are missing required exports')

        solver_class = load_solver_class_from_file(solver_code_path, solver_class_name)
        evaluator = SolverEvaluation(
            solver_class=solver_class,
            topo_name=topo_name,
            ina_num_list=ina_num_list,
            jobs_num_list=jobs_num_list,
            instances_num=instances_num,
            solver_name=solver_name,
            time_limit=time_limit,
            mip_gap=mip_gap,
            verbose=verbose,
        )
        evaluator.setup_network()

        constraint_stats = defaultdict(lambda: {
            'instances': 0,
            'tight_instances': 0,
            'violation_instances': 0,
            'slack_samples': [],
        })
        analysis_records: List[Dict[str, Any]] = []
        failure_records: List[Dict[str, Any]] = []
        attempted_instances = 0

        for ina_num in ina_num_list:
            for jobs_num in jobs_num_list:
                print(f"[Constraint Analysis] Config topo={topo_name}, K={ina_num}, jobs={jobs_num}")
                dataset = evaluator.generate_instances(jobs_num)
                for instance_idx, (instance_name, instance_data) in enumerate(dataset.items(), 1):
                    attempted_instances += 1
                    try:
                        solver = solver_class(
                            instance=instance_data,
                            network=evaluator.network,
                            K=ina_num,
                            jobs_num=jobs_num,
                            Cs=evaluator.Cs,
                            Ps=evaluator.Ps,
                            topo_name=topo_name,
                        )
                        _hydrate_solver_fields(solver)


                        solution = solver.solve(
                            solver_name=solver_name,
                            time_limit=time_limit,
                            mip_gap=mip_gap,
                            verbose=False,
                        )
                        if solution is None:
                            raise RuntimeError('Solver returned no solution for constraint analysis')
                        solver.solution = solution


                        analyzer = ConstraintAnalyzer(solver, epsilon=epsilon)
                        analysis = analyzer.analyze_all_constraints()

                        base_filename = _sanitize_name(f"{topo_name}_K{ina_num}_J{jobs_num}_{instance_name}")
                        report_path = os.path.join(instances_dir, base_filename + '_report.txt')
                        analyzer.export_report(report_path)
                        json_path = os.path.join(instances_dir, base_filename + '_analysis.json')
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(analysis, f, indent=2, ensure_ascii=False)

                        analyzed_constraints = {
                            name: data
                            for name, data in analysis.items()
                            if not (isinstance(data, dict) and data.get('status') == 'not_available')
                        }
                        skipped_constraints = [
                            name for name, data in analysis.items()
                            if isinstance(data, dict) and data.get('status') == 'not_available'
                        ]

                        per_record = {
                            'config': {
                                'topology': topo_name,
                                'ina_budget': ina_num,
                                'jobs_num': jobs_num,
                                'instance_name': instance_name,
                                'instance_index': instance_idx,
                            },
                            'makespan': solver.solution.get('makespan'),
                            'tight_constraints': [
                                name for name, data in analyzed_constraints.items() if _is_constraint_tight(name, data)
                            ],
                            'violating_constraints': [
                                name for name, data in analyzed_constraints.items() if _has_violations(data)
                            ],
                            'skipped_constraints': skipped_constraints,
                            'report_path': report_path,
                            'json_path': json_path,
                        }
                        analysis_records.append(per_record)
                        _update_constraint_stats(constraint_stats, analysis)

                    except Exception as instance_exc:
                        traceback_str = traceback.format_exc()
                        print(f"{traceback_str}")

                        failure_records.append({
                            'config': {'ina_budget': ina_num, 'jobs_num': jobs_num},
                            'instance_name': instance_name,
                            'reason': str(instance_exc)
                        })
                        print(f"✗ Constraint analysis failed for instance {instance_name}: {instance_exc}")

        if not analysis_records:
            save_workflow_log(
                output_dir,
                'constraint_analysis_failed',
                {
                    'solver_path': solver_code_path,
                    'failures': failure_records,
                }
            )
            return _build_error_state('Constraint analysis failed: no successful instances', {
                'failures': failure_records
            })

        constraint_overview = {}
        for name, stats in constraint_stats.items():
            entry = {
                'instances': stats['instances'],
                'tight_instances': stats['tight_instances'],
                'violation_instances': stats['violation_instances'],
            }
            if stats['slack_samples']:
                entry['avg_slack'] = sum(stats['slack_samples']) / len(stats['slack_samples'])
            constraint_overview[name] = entry

        summary_data = {
            'config': {
                'topology': topo_name,
                'ina_num_list': ina_num_list,
                'jobs_num_list': jobs_num_list,
                'instances_per_config': instances_num,
                'solver_name': solver_name,
                'time_limit': time_limit,
                'mip_gap': mip_gap,
                'epsilon': epsilon,
            },
            'attempted_instances': attempted_instances,
            'successful_instances': len(analysis_records),
            'failed_instances': len(failure_records),
            'constraint_overview': constraint_overview,
            'instances': analysis_records,
            'failures': failure_records,
        }

        summary_lines = [
            '=' * 80,
            'Constraint Analysis Summary',
            '=' * 80,
            f"Topology: {topo_name}",
            f"INA budgets: {ina_num_list}",
            f"Job counts: {jobs_num_list}",
            f"Instances per config: {instances_num}",
            f"Instances attempted: {attempted_instances}",
            f"Instances analyzed: {len(analysis_records)}",
            f"Failures: {len(failure_records)}",
            '',
            'Constraint overview:',
        ]

        for name in sorted(constraint_overview.keys()):
            stats = constraint_overview[name]
            line = (
                f"- {name}: tight {stats['tight_instances']}/{stats['instances']}"
                f", violations {stats['violation_instances']}"
            )
            if 'avg_slack' in stats:
                line += f", avg_slack={stats['avg_slack']:.4f}"
            summary_lines.append(line)

        summary_report_path = os.path.join(analysis_dir, 'constraint_analysis_summary.txt')
        with open(summary_report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(summary_lines))

        summary_json_path = os.path.join(analysis_dir, 'constraint_analysis_summary.json')
        with open(summary_json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        save_workflow_log(
            output_dir,
            'constraint_analysis_completed',
            {
                'solver_path': solver_code_path,
                'summary_report_path': summary_report_path,
                'summary_json_path': summary_json_path,
                'successful_instances': len(analysis_records),
                'failed_instances': len(failure_records),
            }
        )

        message_content = (
            f"Constraint analysis complete: {len(analysis_records)} instances analyzed"
            f" ({len(failure_records)} failures)."
        )
        return {
            'constraint_analysis': {
                'success': True,
                'summary': summary_data,
                'report_path': summary_report_path,
                'json_path': summary_json_path,
            },
            'next_action': 'end',
            'messages': [Message(
                role='assistant',
                content=message_content,
                agent_name='constraint_analyzer',
            )]
        }

    except Exception as exc:
        error_traceback = traceback.format_exc()
        print(f"\n{'='*80}")
        print(f"✗ Constraint analysis failed: {exc}")
        print(f"{'='*80}")
        save_workflow_log(
            output_dir,
            'constraint_analysis_failed',
            {
                'solver_path': solver_code_path,
                'error': str(exc),
                'traceback': error_traceback,
            }
        )
        return {
            'constraint_analysis': {
                'success': False,
                'error': str(exc),
                'traceback': error_traceback,
            },
            'next_action': 'end',
            'messages': [Message(
                role='system',
                content=str(exc),
                agent_name='constraint_analyzer'
            )]
        }

    finally:
        if task_dir_added and task_dir in sys.path:
            try:
                sys.path.remove(task_dir)
            except ValueError:
                pass