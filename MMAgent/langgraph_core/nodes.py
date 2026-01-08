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
import ast
from pathlib import Path
import sys
import importlib.util
import traceback
import inspect
import textwrap
# 包导入 - 使用相对导入
from .state import AgentState
from typing import Dict, List, Any, Type, Optional
from collections import defaultdict
import statistics
from collections import Counter
from MMAgent.prompt import (
    PROBLEM_DESCRIPTION_PROMPT,
    HEURISTIC_ARCHITECT_PROMPT,
    HEURISTIC_FUNCTION_CODE_GENERATION_PROMPT,
    HEURISTIC_FUNCTION_CODE_FIX_PROMPT,
    HEURISTIC_REFLECT_PROMPT
)
from MMAgent.utils import *
from MMBench.problem.LLMINA.runtime.solutionanalyzer import SolutionAnalyzer
from MMBench.problem.LLMINA.runtime.evaluate_solver import SolverEvaluation

# ============ 问题加载 ============
def load_problem_node(state: AgentState) -> AgentState:   
    # 构建问题描述 
    problem_path = state['problem_path']
    with open(problem_path, 'r', encoding='utf-8') as f:
        problem = json.load(f)
    problem_str = PROBLEM_DESCRIPTION_PROMPT.format(
        problem_background=problem['background'],
        problem_requirement=problem['problem_requirement'],
        problem_formulation=problem['problem_formulation'],
    ).strip()

    # 读取代码模板 
    runtime_dir = state['runtime_dir']
    heuristic_reference_code_path = os.path.join(runtime_dir, 'ModelSolver.py')
    with open(heuristic_reference_code_path, 'r', encoding='utf-8') as f:
        heuristic_reference_code = f.read()
    
    output_dir = state['output_dir']
    ensure_output_dirs(output_dir)
    
    # 记录工作流日志
    save_workflow_log(output_dir, 'problem_loaded', {
        'problem_path': problem_path,
    })
    
    # 只返回增量更新，LangGraph 会自动合并
    return {
        'problem_str': problem_str,
        'heuristic_reference_code': heuristic_reference_code,
        'next_action': 'heuristic_architect'
    }

# ============ 启发式架构设计 ============
def heuristic_architect_node(state: AgentState) -> AgentState:
    """
    启发式架构设计节点
    
    不编写实现代码，只负责软件架构设计。
    定义启发式算法的模块化组件。
    
    此版本适配的JSON结构
    """
    llm = state['llm']
    problem_str = state['problem_str']
    output_dir = state['output_dir']
    heuristic_reference_code = state.get('heuristic_reference_code', '')
    
    print(f"[Heuristic Architect] Designing function architecture...")
    
    # 构建提示词（传入问题描述和模板代码）
    prompt = HEURISTIC_ARCHITECT_PROMPT.format(
        problem_str=problem_str,
        heuristic_reference_code=heuristic_reference_code
    )
    
    try:
        response = llm.generate(prompt)
        
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
        # 查找JSON对象的边界 ---
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
            )
        





    # with open(state['output_dir']+'/heuristic_architecture.json', "r", encoding="utf-8") as f:
    #     parsed_json = json.load(f)
    # try:





        # 验证顶层字段    
        required_top_keys = ['problem_analysis', 'strategy_overview', 'function_architecture']
        missing_top_keys = [key for key in required_top_keys if key not in parsed_json]
        if missing_top_keys:
            raise ValueError(f"JSON object missing required top-level keys: {missing_top_keys}")

        problem_analysis = parsed_json['problem_analysis']
        strategy_overview = parsed_json['strategy_overview']
        function_architecture = parsed_json['function_architecture']

        # 验证 function_architecture 列表
        if not isinstance(function_architecture, list):
            raise ValueError(
                f"Expected 'function_architecture' to be a list, got {type(function_architecture).__name__}."
            )
        
        if len(function_architecture) == 0:
            raise ValueError("Function list 'function_architecture' is empty. At least one function is required.")
        
        # 验证每个函数的必需字段
        for i, func in enumerate(function_architecture):
            if not isinstance(func, dict):
                raise ValueError(
                    f"Function {i+1} is not a dictionary, got {type(func).__name__}"
                )
            
            required_fields = ['name', 'strategic_role', 'inputs', 'outputs', 'member_variables_read', 'member_variables_written', 'dependencies']
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
            for list_field in ['inputs', 'outputs', 'member_variables_read', 'member_variables_written', 'dependencies']:
                if not isinstance(func[list_field], list):
                    raise ValueError(f"Function {i+1} field '{list_field}' expected to be a list, got {type(func[list_field]).__name__}")
       
        # 保存架构设计
        architecture_path = os.path.join(output_dir, 'heuristic_architecture.json')
        
        with open(architecture_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_json, f, indent=2, ensure_ascii=False)
        print(f"[Heuristic Architect] Saved to {architecture_path}")
        
        # 记录日志
        save_workflow_log(output_dir, 'heuristic_architect_completed', {
            'problem_analysis': problem_analysis,
            'strategy_overview': strategy_overview,
            'function_count': len(function_architecture),
            'functions': [f['name'] for f in function_architecture]
        })
        
        # 返回增量更新 - 如果有函数则触发函数生成循环
        next_action = 'generate_next_function' if len(function_architecture) > 0 else 'end'
        
        return {
            'problem_analysis': problem_analysis,
            'strategy_overview': strategy_overview,
            'function_architecture': function_architecture,

            'functions_to_generate': [1 for _ in range(len(function_architecture))],  # 标记所有函数待生成
            'current_function_id': 0,  # 重置函数ID计数器
            'next_action': next_action,
        }
    
    except Exception as e:
        error_msg = f"Failed to parse heuristic architecture from LLM response: {str(e)}"
        print(f"\n[Heuristic Architect] ✗ ERROR: {error_msg}")

        # 返回错误状态
        return {
            'function_architecture': [],
            'next_action': 'end',  # 跳过函数生成，直接集成
            'errors': [error_msg],
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
    function_architecture = state.get('function_architecture', [])
    functions_to_generate = state.get('functions_to_generate', [])
    current_function_id = state.get('current_function_id', 0)
    problem_str = state['problem_str']
    output_dir = state['output_dir']
    heuristic_reference_code = state.get('heuristic_reference_code', '')
    experiences = state.get('experiences', [])
    
    total_functions = len(function_architecture)
    
    # 检查是否已完成所有函数
    if current_function_id >= total_functions:
        return {'next_action': 'integrate'}
    elif functions_to_generate[current_function_id] == 0:
        # 跳过已生成的函数
        print(f"[Function {current_function_id + 1}/{total_functions}] Skipping already generated function")
        return {
            'current_function_id': current_function_id + 1,
            'next_action': 'generate_next_function',
        }
    else:
        # 生成当前函数
        func_info = function_architecture[current_function_id]
        func_name = func_info['name']
        
        print(f"[Function {current_function_id + 1}/{total_functions}] {func_name}")
        
        # 构建输入输出规范字符串
        inputs_spec = ""
        for func_inp in func_info.get('inputs', []):
            # 建议增加 default 值的处理，如果 JSON 里有的话
            default_val = f" = {func_inp['default']}" if 'default' in func_inp else ""
            inputs_spec += f"- {func_inp['name']} ({func_inp['type']}){default_val}: {func_inp['description']}\n"
        
        outputs_spec = ""
        for func_out in func_info.get('outputs', []):
            outputs_spec += f"- {func_out['name']} ({func_out['type']}): {func_out['description']}\n"
        
        # 已完成函数摘要
        completed_summary = ""
        if current_function_id > 0:
            completed_summary = f"The following {current_function_id} functions have been completed and are available as methods of `self`. You can call them to reuse their logic:\n"
            for i in range(current_function_id):
                prev_func = function_architecture[i]
                prev_func_name = prev_func['name']
                
                # Construct clear signature
                input_params = [f"{inp['name']}" for inp in prev_func.get('inputs', [])] # Just names for signature
                signature_args = ", ".join(input_params)
                
                # Construct return hint
                outputs_types = [out['type'] for out in prev_func.get('outputs', [])]
                if not outputs_types:
                    return_hint = "None"
                elif len(outputs_types) == 1:
                    return_hint = outputs_types[0]
                else:
                    return_hint = f"Tuple[{', '.join(outputs_types)}]"

                completed_summary += f"\n### Method: `{prev_func_name}`\n"
                completed_summary += f"- **Signature**: `self.{prev_func_name}({signature_args}) -> {return_hint}`\n"
                completed_summary += f"- **Role**: {prev_func['strategic_role']}\n"
                
                # Detailed Input Specs
                if prev_func.get('inputs', []):
                    input_details = ", ".join([f"{inp['name']} ({inp['type']})" for inp in prev_func['inputs']])
                    completed_summary += f"- **Arguments**: {input_details}\n"

                # Side effects
                written_vars = [mv['name'] for mv in prev_func.get('member_variables_written', [])]
                if written_vars:
                    completed_summary += f"- **Updates Member Variables**: {', '.join(written_vars)} (You can access these via `self.*` after calling)\n"
        else:
            completed_summary = "This is the first function. No previous functions have been completed yet."
        
        # 读写成员变量 如果列表为空，显式显示 "None"
        read_vars_list = [
            f"- {mv['name']} ({mv['type']})"
            for mv in func_info.get('member_variables_read', [])
        ]
        member_variables_read = "\n".join(read_vars_list) if read_vars_list else "None"

        written_vars_list = [
            f"- {mv['name']} ({mv['type']}): {mv['description']}"
            for mv in func_info.get('member_variables_written', [])
        ]
        member_variables_written = "\n".join(written_vars_list) if written_vars_list else "None"
        
        # 构建提示词
        prompt = HEURISTIC_FUNCTION_CODE_GENERATION_PROMPT.format(
            problem_str=problem_str,
            heuristic_reference_code=heuristic_reference_code,
            function_id=current_function_id + 1,
            total_functions=total_functions,
            function_name=func_name,
            function_strategic_role=func_info['strategic_role'],
            inputs_spec=inputs_spec if inputs_spec else "None",
            outputs_spec=outputs_spec if outputs_spec else "None",
            dependencies=", ".join(func_info.get('dependencies', [])) if func_info.get('dependencies') else "None",
            completed_functions_summary=completed_summary,
            member_variables_read=member_variables_read,
            member_variables_written=member_variables_written,
            experiences="\n".join(experiences) if experiences else "None",
        )
        
        # 调用 LLM 生成代码
        try:
            print(f"  [Code Generation] Generating code for {func_name}...")
            response = llm.generate(prompt)
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
            # 查找JSON对象的边界 ---
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
                )
            
            func_discription = parsed_json['func_discription']
            func_code = parsed_json['func_code']
            if '```python' in func_code:
                func_code = func_code.split('```python')[1].split('```')[0].strip()
            elif '```' in func_code:
                func_code = func_code.split('```')[1].split('```')[0].strip()
            
            print(f"  [Code Generation] Done ({len(func_code)} chars)")
            
            # 记录日志
            save_workflow_log(output_dir, f'function_{current_function_id + 1}_completed', {
                'function_id': current_function_id + 1,
                'function_name': func_name,
                'code_length': len(func_code)
            })

            
            # 返回增量更新
            return {
                'current_function_id': current_function_id + 1,
                'function_descriptions': {func_name: func_discription},
                'function_codes': {func_name: func_code},
                'next_action': 'generate_next_function',
            }
        
        except Exception as e:
            error_msg = f"Failed to generate function {func_name}: {str(e)}"
            print(f"  [Error] {error_msg}")
            
            # 记录错误但继续下一个函数
            return {
                'current_function_id': current_function_id + 1,
                'function_codes': {func_name: f"# Error generating function\n# {error_msg}"},
                'next_action': 'end',
                'errors': [error_msg],
            }

# ============ 代码集成 ============
# 追加+替换
# def code_integration_node(state: AgentState) -> AgentState:
#     """
#     代码集成节点 - 将启发式函数直接集成到 ModelSolver 模板中（工具节点，不使用 LLM）
    
#     工作流程：
#     1. 获取 ModelSolver 模板代码
#     2. 获取所有已生成的启发式函数代码
#     3. 直接通过代码逻辑将函数添加为类方法
#     4. 在 solve() 方法中按依赖顺序调用这些函数
#     5. 保存完整的 ModelSolver 实现
    
#     设计原则：
#     - 不使用 LLM，避免长时间等待
#     - 直接代码操作，确保集成的确定性和可靠性
#     """
#     output_dir = state['output_dir']
#     current_step = state.get('current_step', 1)

#     if current_step == 1: # 基于heuristic_reference_code修改
#         heuristic_reference_code = state.get('heuristic_reference_code', '')
#         function_architecture = state.get('function_architecture', [])
#         function_codes = state.get('function_codes', {})
        
#         if not heuristic_reference_code:
#             print("[Integration] Warning: No ModelSolver template found")
#             return {
#                 'next_action': 'end',
#                 'errors': ['No template available for integration'],
#             }
        
#         print(f"[Integration] Integrating {len(function_architecture)} functions into ModelSolver...")
        
#         # 在模板代码基础上追加
#         final_code = integrate_functions_directly(
#             heuristic_reference_code,
#             function_architecture,
#             function_codes
#         )
        
#         print(f"[Integration] Integration complete ({len(final_code)} chars)")
        
#         # 保存最终代码
#         algorithm_dir = ensure_algorithm_dirs(output_dir, current_step)
#         solver_code_path = os.path.join(algorithm_dir, 'ModelSolver_final.py')
#         os.makedirs(os.path.dirname(solver_code_path), exist_ok=True)
        
#         with open(solver_code_path, 'w', encoding='utf-8') as f:
#             f.write(final_code)
        
#         print(f"[Integration] Saved to {solver_code_path}")
        
#         # 记录日志
#         save_workflow_log(output_dir, 'integration_completed', {
#             'function_count': len(function_architecture),
#             'solver_path': solver_code_path,
#             'code_length': len(final_code)
#         })
        
#         # 返回增量更新
#         return {
#             'solver_code': final_code,
#             'solver_code_path': solver_code_path,
#             'next_action': 'evaluate',
#         }
#     else: # 基于上一步的ModelSolver_final.py修改
#         base_code = state.get('base_code', '')
#         function_codes = state.get('function_codes', {})

#         with open(output_dir + '/algorithm/' + str(current_step-1)+'/ModelSolver_final.py', "r", encoding="utf-8") as f:
#             base_code = f.read()
#         for func_code in function_codes:
#             target_function_name = func_code
#             new_function_code = function_codes[func_code]
#             final_code = replace_function_in_code(base_code, target_function_name, new_function_code)
#             base_code = final_code
#         # 保存最终代码
#         algorithm_dir = ensure_algorithm_dirs(output_dir, current_step)
#         solver_code_path = os.path.join(algorithm_dir, 'ModelSolver_final.py')
#         os.makedirs(os.path.dirname(solver_code_path), exist_ok=True)
#         with open(solver_code_path, 'w', encoding='utf-8') as f:
#             f.write(base_code)
#         print(f"[Integration] Saved to {solver_code_path}")
#         # 记录日志
#         save_workflow_log(output_dir, 'integration_completed', {
#             'function_count': len(function_codes),
#             'solver_path': solver_code_path,
#             'code_length': len(base_code)
#         })
#         # 返回增量更新
#         return {
#             'solver_code': base_code,
#             'solver_code_path': solver_code_path,
#             'next_action': 'evaluate',
#         }

# ============ 代码集成 ============
# 只追加    
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
    current_step = state.get('current_step', 1)

    heuristic_reference_code = state.get('heuristic_reference_code', '')
    function_architecture = state.get('function_architecture', [])
    function_codes = state.get('function_codes', {})
    
    if not heuristic_reference_code:
        print("[Integration] Warning: No ModelSolver template found")
        return {
            'next_action': 'end',
            'errors': ['No template available for integration'],
        }
    
    print(f"[Integration] Integrating {len(function_architecture)} functions into ModelSolver...")
    
    # 在模板代码基础上追加
    final_code = integrate_functions_directly(
        heuristic_reference_code,
        function_architecture,
        function_codes
    )
    
    print(f"[Integration] Integration complete ({len(final_code)} chars)")
    
    # 保存最终代码
    algorithm_dir = ensure_algorithm_dirs(output_dir, current_step)
    solver_code_path = os.path.join(algorithm_dir, 'ModelSolver_final.py')
    os.makedirs(os.path.dirname(solver_code_path), exist_ok=True)
    
    with open(solver_code_path, 'w', encoding='utf-8') as f:
        f.write(final_code)
    
    print(f"[Integration] Saved to {solver_code_path}")
    
    # 记录日志
    save_workflow_log(output_dir, 'integration_completed', {
        'function_count': len(function_architecture),
        'solver_path': solver_code_path,
        'code_length': len(final_code)
    })
    
    # 返回增量更新
    return {
        'solver_code': final_code,
        'solver_code_path': solver_code_path,
        'next_action': 'evaluate',
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
        func_name = func_spec['name']
        func_code = function_codes.get(func_name, f"# Function {func_spec['name']} not generated")
        
        new_code_lines.append(f"    # Function {i+1}: {func_name}")
        new_code_lines.append(f"    # Role: {func_spec['strategic_role']}")
        
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
    2. 从 runtime_dir 导入 evaluate_solver 模块
    3. 使用 evaluate_solver_from_file 函数评估求解器
    4. 将评估结果保存到 state
    
    设计原则：
    - 使用 __init__.py 统一管理包导入
    - 从 state 获取评估配置（拓扑、INA预算、任务数量等）
    - 将评估结果存储在 state['evaluation_results'] 中
    """
    output_dir = state['output_dir']
    solver_code_path = state.get('solver_code_path', '')
    runtime_dir = state.get('runtime_dir', '')

    # solver_code_path = '/home/hyn/LLM-MM-Agent-LLMINA-clean/output/LLMINA_20260108-092809/algorithm/1/ModelSolver_final.py'


    # 临时添加 runtime_dir 到 sys.path 以确保依赖模块可以被导入
    if runtime_dir not in sys.path:
        sys.path.insert(0, runtime_dir)

    # 4. 调用评估函数，并捕获异常信息到字符串
    error_info = None
    feasibility_failed_message = None

    # 调用评估函数
    try:
        solver_class = load_solver_class_from_file(solver_code_path, "ModelSolver")    
        evaluator = SolverEvaluation(solver_class=solver_class)    
        success, message = evaluator.evaluate()
        if not success:
            # 如果 evaluate 返回 False，说明有不可行性错误
            feasibility_failed_message = message

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
                "error": error_msg,
                "traceback": error_info,
            },
        )

        return {
            "evaluation_results": {
                "evaluation_success": False,
                "error": error_msg,
                "traceback": error_info,
            },
            "next_action": "fix",
            "errors": [error_msg],
        }

    # 情况 B: 可行性检查失败 -> Reflect Node
    if feasibility_failed_message is not None:
        print(f"✗ Feasibility Check Failed. Transitioning to Reflection.")
        
        # 构造一个特殊的质量报告供 reflect 节点使用
        infeasible_report = {
            "status": "Infeasible Solution",
            "description": "The heuristic algorithm produced a solution that violates basic problem constraints.",
            "feasibility_check_errors": feasibility_failed_message,
            "suggestion": "Review the heuristic logic to ensure it respects In-Network Aggregation resource budgets and routing rules."
        }
        
        save_workflow_log(
            output_dir,
            "evaluation_failed_feasibility",
            infeasible_report,
        )

        return {
            "quality_report": infeasible_report,
            "next_action": "reflect",
        }

    print(f"✓ Evaluation completed successfully!")

    # 7. 记录日志
    save_workflow_log(
        output_dir,
        "evaluation_completed",
        {"solver_path": solver_code_path,},
    )

    # 8. 返回增量 state（LangGraph 会 merge 到全局 AgentState）
    return {
        "evaluation_results": {
            "evaluation_success": True,
            "error": None,
            "traceback": None,
        },
        "next_action": "constraint_analyze",
    }

# ============ 代码修复辅助工具 ============
# tools for dynamic loading and evaluation                    
def load_solver_class_from_file(file_path: str, class_name: str = "ModelSolver") -> Type:
    file_path = Path(file_path).resolve()
    
    if not file_path.exists():
        raise FileNotFoundError(f"Solver 文件不存在: {file_path}")
    
    # 动态加载模块
    spec = importlib.util.spec_from_file_location("dynamic_solver_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    solver_class = getattr(module, class_name, None)
    if solver_class is None:
        raise AttributeError(f"模块中不存在类 '{class_name}': {file_path}")

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
    代码修复节点（非JSON版 - 直接代码生成模式）
    
    策略：
    1. 提供 Traceback 和 Full Code。
    2. 要求 LLM 识别错误函数并直接输出修复后的 Python 函数代码。
    3. 通过正则解析返回的代码，提取函数名。
    4. 验证函数名是否在 function_architecture 列表中。
    5. 替换代码并保存。
    """

    llm = state["llm"]
    output_dir = state.get("output_dir", "")
    solver_code_path = state.get("solver_code_path", "")
    evaluation_results = state.get("evaluation_results", {})
    fix_attempt_count = state.get("fix_attempt_count", 0)
    max_fix_attempts = state.get("max_fix_attempts", 3)
    function_architecture = state.get("function_architecture", [])
    current_step = state.get("current_step", 0)
    function_codes = state.get("function_codes", {})
    heuristic_reference_code = state.get("heuristic_reference_code", "")

    functions_list= []
    for func in function_architecture:
        functions_list.append(func["name"])

    print(f"[Code Fix] Starting code repair (Attempt {fix_attempt_count + 1}/{max_fix_attempts})")

    # 1. 最大修复次数检查
    if fix_attempt_count >= max_fix_attempts:
        error_msg = f"Maximum fix attempts ({max_fix_attempts}) reached. Stopping."
        print(f"✗ {error_msg}")
        return {
            "fix_attempt_count": fix_attempt_count,
            "next_action": "end",
        }

    # 2. 提取错误信息
    if evaluation_results.get("evaluation_success", True):
        print(f"[Code Fix] No errors detected. Code is working correctly.")
        return {
            "next_action": "end",
        }

    error_info = evaluation_results.get("error", "Unknown error")
    traceback_info = evaluation_results.get("traceback", "")

    print(f"[Code Fix]  Error: {error_info}")
    
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
        }

    fullcode_snippet = full_code.strip("\n")

    fix_prompt = HEURISTIC_FUNCTION_CODE_FIX_PROMPT.format(
        fullcode_snippet=fullcode_snippet,
        functions_list = functions_list,
        traceback_info = traceback_info,
    )

    print(f"[Code Fix] Requesting LLM to fix function...")
    try:
        response = llm.generate(fix_prompt)
        fixed_function_code = response.content if hasattr(response, 'content') else str(response)
        
        # 清理代码（移除可能的 markdown 标记）
        if '```python' in fixed_function_code:
            fixed_function_code = fixed_function_code.split('```python')[1].split('```')[0].strip()
        elif '```' in fixed_function_code:
            fixed_function_code = fixed_function_code.split('```')[1].split('```')[0].strip()
        
        func_name_match = re.search(
            r"^\s*def\s+([A-Za-z_]\w*)\s*\(",
            fixed_function_code,
            flags=re.MULTILINE,
        )
        if not func_name_match:
            raise ValueError(
                "Could not find a function definition in LLM response. "
                f"Response starts with: {fixed_function_code[:200]}..."
            )
        target_function_name = func_name_match.group(1)

        # 3) 校验函数名是否在 function_architecture 中
        if target_function_name not in functions_list:
            raise ValueError(
                f"LLM returned function '{target_function_name}', "
                f"which is not in functions_list: {functions_list}"
            )

        print(f"[Code Fix] Target function to replace: {target_function_name}")

        if not target_function_name or not fixed_function_code:
            raise ValueError("Invalid response format: missing target_function_name or fixed_function_code.")

        function_codes[target_function_name] = fixed_function_code

        final_code = integrate_functions_directly(
            heuristic_reference_code,
            function_architecture,
            function_codes
        )
        print(f"[Code Fix] Re-integrated functions into ModelSolver (Clean Build)")

        with open(solver_code_path, "w", encoding="utf-8") as f:
            f.write(final_code)

        # 10. 记录日志（如果 save_workflow_log 可用）
        try:
            save_workflow_log(
                output_dir,
                f"code_fix_attempt_{fix_attempt_count + 1}",
                {
                    "fix_attempt": fix_attempt_count + 1,
                    "error_info": error_info,
                    "target_function": target_function_name,
                    "solver_code_path": solver_code_path,
                },
            )
        except Exception:
            # 日志失败不影响主流程
            pass

        # 11. 返回状态更新，触发重新评估
        return {
            "fix_attempt_count": fix_attempt_count + 1,
            "previous_errors": [error_info],
            "next_action": "re_evaluate",
            "function_codes": function_codes,
        }

    except Exception as e:
        error_traceback = traceback.format_exc()
        error_msg = f"Code fix failed: {str(e)}"

        print(f"✗ {error_msg}")
        print(f"Traceback:\n{error_traceback}")

        # 失败也记录一下日志（如果可用）
        try:
            save_workflow_log(
                output_dir,
                f"code_fix_failed_attempt_{fix_attempt_count + 1}",
                {
                    "fix_attempt": fix_attempt_count + 1,
                    "error": error_msg,
                    "traceback": error_traceback,
                },
            )
        except Exception:
            pass

        return {
            "fix_attempt_count": fix_attempt_count + 1,
            "next_action": "end",
        }

# ============ 约束分析节点 ============
def constraint_analyzer_node(state: AgentState) -> AgentState:
    """
    约束分析节点 - 运行多实例 Pyomo 求解并使用 ConstraintAnalyzer 生成统计报告。
    
    功能：
    1. 加载最新的 Solver 代码。
    2. 运行多个测试实例（SolverEvaluation）。
    3. 分析每个实例的约束松紧度（SolutionAnalyzer）。
    4. 汇总报告并更新状态。
    """
    output_dir = state.get('output_dir', './output')
    solver_code_path = state.get('solver_code_path', '')
    runtime_dir = state.get('runtime_dir', '')
    solver_class_name = state.get('solver_class_name', 'ModelSolver')
    current_step = state.get('current_step', 1)
    print("[Constraint Analysis] Running multi-instance MILP analysis...")

    # solver_code_path = '/home/hyn/LLM-MM-Agent-LLMINA-clean/output/LLMINA_20260107-190332/algorithm/1/ModelSolver_final.py'

    try:
        # 1. 环境准备：临时添加 runtime_dir 到 sys.path
        if runtime_dir and runtime_dir not in sys.path:
            sys.path.insert(0, runtime_dir)
            print(f"[Constraint Analysis] Added {runtime_dir} to sys.path")

        # 2. 加载 Solver 类
        try:
            solver_class = load_solver_class_from_file(solver_code_path, solver_class_name)
            print(f"[Constraint Analysis] Successfully loaded class '{solver_class_name}' from {solver_code_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load solver class: {e}")

        analyzer = SolutionAnalyzer(solver_class=solver_class)
        quality_report = analyzer.run_batch_evaluation() # 获取质量报告

        # 保存报告
        algorithm_dir = ensure_algorithm_dirs(output_dir, current_step)
        report_path = os.path.join(algorithm_dir, 'quality_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, indent=2, ensure_ascii=False)

        # 6. 记录工作流日志
        try:
            save_workflow_log(
                output_dir,
                "constraint_analysis_completed",
                {
                    "solver_code": solver_code_path,
                    "report_path": report_path
                }
            )
        except Exception:
            pass
        
        return {
            "quality_report": quality_report, # 保存结构化数据
            "next_action": "reflect",          # 下一步动作（示例）
            'current_step': current_step + 1,
        }

    except Exception as e:
        error_traceback = traceback.format_exc()
        error_msg = f"Constraint analysis failed: {str(e)}"

        print(f"✗ {error_msg}")
        print(f"Traceback:\n{error_traceback}")

        # 记录失败日志
        try:
            save_workflow_log(
                output_dir,
                "constraint_analysis_failed",
                {
                    "error": error_msg,
                    "traceback": error_traceback
                }
            )
        except Exception:
            pass

        return {
            "next_action": "end", # 或者 error_recovery
        }

def heuristic_reflect_node(state: AgentState) -> AgentState:
    """
    启发式反思节点
    
    功能：
    1. 分析 constraint_analyzer 产生的 quality_report。
    2. 结合 fullcode_snippet 分析瓶颈的各种代码成因。
    3. 生成经验教训 (experience_lessons)。
    4. 确定需要重写的函数 (function_update)。
    5. 将经验教训追加到 state，并重置相关函数的生成状态。
    """
    
    llm = state['llm']
    output_dir = state['output_dir']
    solver_code_path = state.get('solver_code_path', '')
    
    # 1. 获取输入数据
    problem_str = state.get('problem_str', '')
    current_step = state.get('current_step', 1)
    function_architecture = state.get('function_architecture', [])
    
    # 构建 func_set 供参考
    func_set = [f['name'] for f in function_architecture]

    # 读取当前代码
    fullcode_snippet = ""
    try:
        if os.path.exists(solver_code_path):
            with open(solver_code_path, 'r', encoding='utf-8') as f:
                fullcode_snippet = f.read()
    except Exception as e:
        print(f"[Heuristic Reflect] Warning: Could not read solver code from {solver_code_path}: {e}")

    # 获取质量报告
    quality_report= state.get('quality_report', {})

    # 3. 构建提示词
    # 将 JSON 数据转为格式化字符串
    report_str = json.dumps(quality_report, indent=2) if isinstance(quality_report, (dict, list)) else str(quality_report)
    
    prompt = HEURISTIC_REFLECT_PROMPT.format(
        problem_str=problem_str,
        fullcode_snippet=fullcode_snippet,
        quality_report=report_str,
        func_set=func_set
    )

    # 4. 调用 LLM
    try:
        print("[Heuristic Reflect] Analyzing bottlenecks and evolving strategy...")
        response = llm.generate(prompt)
        content = response.content if hasattr(response, 'content') else str(response)

        # 5. 解析 JSON
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
        
        evolution_plan = json.loads(content)

        # 提取关键字段
        experience_lessons = evolution_plan.get("experience_lessons", "No experience provided.")
        function_update_list = evolution_plan.get("function_update", [])        

        # Compute functions_to_generate list (0 for keep, 1 for rewrite)
        # Initialize all to 0 (keep)
        functions_to_generate = []
        for f in function_architecture:
            if f['name'] in function_update_list:
                functions_to_generate.append(1)
            else:
                functions_to_generate.append(0)
        
        # 保存架构设计 (虽然架构没变，但为了保持一致性或记录，可以考虑保存，或者不保存)
        # 这里为了流程完整性，我们可以不覆盖 architectural json，除非我们改了 description
        # 本次修改不包含 description 更新，所以不在此处写入 heuristic_architecture.json

        save_workflow_log(
                output_dir,   
                "heuristic_reflection_completed",
                {
                "experience_lessons": experience_lessons,
                "functions_to_update": function_update_list
                }
            )

        return {
            "functions_to_generate": functions_to_generate,
            "experiences": [experience_lessons], # Append to experiences list
            
            # 重置指针
            "current_function_id": 0,
            "fix_attempt_count": 0,
            
            # 下一步进入代码生成
            "next_action": "generate_next_function",
        }

    except Exception as e:
        error_msg = f"Reflection failed: {str(e)}"
        print(f"✗ {error_msg}")
        print(traceback.format_exc())
        return {
            "next_action": "end",
            "errors": [error_msg],
        }

