import json
import os
import re
import sys
import random
import traceback
import copy
import networkx as nx
# 包导入 - 使用相对导入
from .state import AgentState
from MMAgent.prompt import (
    PROBLEM_DESCRIPTION_PROMPT,
    HEURISTIC_FUNCTION_CODE_GENERATION_PROMPT,
    HEURISTIC_FUNCTION_CODE_FIX_PROMPT,
    HEURISTIC_REFLECT_PROMPT,
    HEURISTIC_INITIALIZATION_PROMPT,
    FEEDBACK_GUIDED_OPTIMIZATION_PROMPT,
    EXPLORATORY_REFACTORING_PROMPT
)
from MMAgent.utils import *
from .MCTS.mcts_core import MCTS, MCTSNode
from MMBench.problem.LLMINA.runtime.solutionanalyzer import SolutionAnalyzer
from MMBench.problem.LLMINA.runtime.evaluate_solver import SolverEvaluation
from MMBench.problem.LLMINA.runtime.TemplateSolver import OUTPUT_FUNCTION_TEMPLATE

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
    
    try:
        # Initialize MCTS First Turn (Genesis)
        print("[Load Problem] Initializing MCTS First Turn (i1)...")

        mcts_updates = mcts_turn_step(state, reward=None)
        
        return {
            'problem_str': problem_str,
            'heuristic_reference_code': heuristic_reference_code,
            **mcts_updates
            # next_action_node will be set by mcts_updates (likely 'initialization')
        }
    except Exception as e:
        print(f"[Load Problem] MCTS Init Failed: {e}")
        return {'next_action': 'end', 'errors': [str(e)]}

# ============ 启发式初始化节点 ============
def heuristic_initialization_node(state: AgentState) -> AgentState:
    """
    Heuristic Initialization Node (i1 Operator)
    
    Generates the FIRST complete Function Architecture.
    Acts as the 'Architect' for the genesis phase of the algorithm.
    """
    llm = state['llm']
    
    # Inputs
    problem_str = state.get('problem_str', '')
    heuristic_reference_code = state.get('heuristic_reference_code', '')
    experiences = state.get('experiences', [])
    
    print("[Initialization] Designing new function architecture (i1)...")
    
    # Using HEURISTIC_INITIALIZATION_PROMPT which is aliased to ARCHITECT_PROMPT
    prompt = HEURISTIC_INITIALIZATION_PROMPT.format(
        problem_str=problem_str,
        heuristic_reference_code=heuristic_reference_code,
        experiences="\n".join(experiences) if experiences else "None",
        output_function_template = OUTPUT_FUNCTION_TEMPLATE
    )

    try:
        response = llm.generate(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Robust JSON extraction
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
             match = re.search(r'```(?:json)?(.*?)```', content, re.DOTALL)
             if match: content = match.group(1).strip()
             else: content = content.split('```')[1].strip()
        
        parsed_json = json.loads(content)
        
        # Validation
        function_architecture = parsed_json['function_architecture']
        problem_analysis = parsed_json.get('problem_analysis', '')
        strategy_overview = parsed_json.get('strategy_overview', '')
        
        if not isinstance(function_architecture, list):
             raise ValueError("function_architecture must be a list containing function dictionaries.")

        # ============ Topological Sort based on Dependencies ============
        print(f"[Initialization] Designing dependency graph for {len(function_architecture)} functions...")
        
        # 1. Start Building Helper Structures
        func_map = {f['name']: f for f in function_architecture}
        func_names = set(func_map.keys())
        
        dep_graph_dict = {} # State storage field
        G = nx.DiGraph()
        
        # 2. Add Nodes
        for f in function_architecture:
            G.add_node(f['name'])
            
        # 3. Add Edges (Dependency -> Function)
        # Requirement: "The depended-upon function must appear BEFORE the function that depends on it."
        # If A depends on B (A calls B), then B must come first.
        # Edge B -> A ensures B comes before A in topological sort.
        for f in function_architecture:
            f_name = f['name']
            dependencies = f.get('dependencies', [])
            
            # Filter: Only internal dependencies
            valid_deps = [dep for dep in dependencies if dep in func_names]
            
            # Store for State
            dep_graph_dict[f_name] = valid_deps
            
            for dep in valid_deps:
                G.add_edge(dep, f_name) # dep comes before f_name
        
        # 4. Sort
        try:
            sorted_names = list(nx.topological_sort(G))
            print(f"[Initialization] Topological Sort: {sorted_names}")
        except nx.NetworkXUnfeasible:
            print("[Initialization] Warning: Cycle detected in function dependencies! Falling back to original order.")
            sorted_names = [f['name'] for f in function_architecture]
            
        # 5. Reconstruct sorted architecture list
        sorted_architecture = [func_map[name] for name in sorted_names]

        print(f"[Initialization] Designed and Sorted {len(function_architecture)} functions.")
        
        return {
            'problem_analysis': problem_analysis,
            'strategy_overview': strategy_overview,
            'function_architecture': sorted_architecture,
            'function_dependency_graph': dep_graph_dict, 
            'functions_to_generate': [1 for _ in range(len(sorted_architecture))], # Use sorted len
            'current_function_id': 0,
            'next_action': 'generate_next_function'
        }
        
    except Exception as e:
        print(f"[Initialization] Failed: {traceback.format_exc()}")
        return {"next_action": "end", "errors": [str(e)]}

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
    function_codes = state.get('function_codes', {})
    experiences = state.get('experiences', [])
    
    total_functions = len(function_architecture)
    
    # 检查是否已完成所有函数 (Safety check, though loop condition usually handles this)
    if current_function_id >= total_functions:
         return {'next_action': 'integrate'}

    # Main Logic Block inside the loop
    if functions_to_generate[current_function_id] == 0:
        # PURE SKIP logic
        print(f"[Function {current_function_id + 1}/{total_functions}] Skipping already generated function")
        
        # Calculate next ID
        next_id = current_function_id + 1
        
        # Check if we are done AFTER this skip
        if next_id >= total_functions:
             return {
                 'current_function_id': next_id,
                 'next_action': 'integrate'
             }
        else:
             return {
                 'current_function_id': next_id,
                 'next_action': 'generate_next_function'
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
        
        # 将已完成的函数代码拼接到 reference code 中
        if current_function_id > 0:
            additional_code = ""
            for i in range(current_function_id):
                prev_func = function_architecture[i]
                prev_func_name = prev_func['name']
                if prev_func_name in function_codes:
                    additional_code += f"\n\n    # Completed function: {prev_func_name}\n"
                    additional_code += function_codes[prev_func_name] + "\n"
            
            # 更新 heuristic_reference_code，让 LLM 能看到已实现的函数代码上下文
            heuristic_reference_code = heuristic_reference_code + "\n" + additional_code
        
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
            function_descriptions=func_info['function_descriptions'],
            inputs_spec=inputs_spec,
            outputs_spec=outputs_spec if outputs_spec else "None",
            dependencies=", ".join(func_info.get('dependencies', [])) if func_info.get('dependencies') else "None",
            member_variables_read=member_variables_read,
            member_variables_written=member_variables_written,
            experiences="\n".join(experiences) if experiences else "None",
        )
        
        # 调用 LLM 生成代码
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    print(f"  [Code Generation] Generating code for {func_name}...")
                else:
                    print(f"  [Code Generation] Attempt {attempt + 1}/{max_retries}: Retrying code generation for {func_name}...")

                response = llm.generate(prompt)
                 # 提取内容
                content = response.content if hasattr(response, 'content') else str(response)
                
                # 直接解析代码块
                func_code = ""
                if '```python' in content:
                    func_code = content.split('```python')[1].split('```')[0].strip()
                elif '```' in content:
                    # 尝试匹配generic block
                    func_code = content.split('```')[1].split('```')[0].strip()
                else:
                    # 尝试直接匹配 def 
                    if "def " in content:
                         # 简单的启发式：假设从 def 开始到最后
                         start_idx = content.find("def ")
                         func_code = content[start_idx:].strip()
                    else:
                         raise ValueError(
                            f"No valid Python code block (```python) found in LLM response. "
                            f"Response starts with: {content[:200]}..."
                        )

                # 移除可能的 JSON包装残留 (如果模型不听话)
                if func_code.strip().startswith('{') and '"func_code"' in func_code:
                     try:
                        parsed = json.loads(func_code)
                        if 'func_code' in parsed:
                            func_code = parsed['func_code']
                     except:
                        pass

                
                print(f"  [Code Generation] Done ({len(func_code)} chars)")
                
                # 记录日志
                save_workflow_log(output_dir, f'function_{current_function_id + 1}_completed', {
                    'function_id': current_function_id + 1,
                    'function_name': func_name,
                    'code_length': len(func_code)
                })

                
                # 返回增量更新
                # Determine next step
                next_id = current_function_id + 1
                next_action = 'integrate' if next_id >= len(function_architecture) else 'generate_next_function'
                
                return {
                    'current_function_id': next_id, # Increment ID
                    'function_codes': {func_name: func_code},
                    'next_action': next_action,
                }
            
            except Exception as e:
                last_error = e
                print(f"  [Error] Generation attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    continue
        
        error_msg = f"Failed to generate function {func_name} after {max_retries} attempts: {str(last_error)}"
        print(f"  [Error] {error_msg}")
        
        # Determine next step (Skip this failure or abort? continuing for now)
        next_id = current_function_id + 1
        next_action = 'integrate' if next_id >= len(function_architecture) else 'generate_next_function'
        
        # 记录错误但继续下一个函数
        return {
            'current_function_id': next_id,
            'function_codes': {func_name: f"# Error generating function\n# {error_msg}"},
            'next_action': next_action,
            'errors': [error_msg],
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
    
    # [关键变更] 保存基因信息(Codes & Architecture)到本次迭代的目录，供MCTS后续加载
    # 这样确保了 evolutionary lineage 的数据持久化
    with open(os.path.join(algorithm_dir, 'function_codes.json'), 'w', encoding='utf-8') as f:
        json.dump(function_codes, f, indent=2, ensure_ascii=False)
        
    with open(os.path.join(algorithm_dir, 'function_architecture.json'), 'w', encoding='utf-8') as f:
        json.dump(function_architecture, f, indent=2, ensure_ascii=False)

    print(f"[Integration] Saved solver and genetics to {algorithm_dir}")
    
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

    # solver_code_path = '/home/hyn/LLM-MM-Agent-LLMINA-clean/output/LLMINA_20260117-011108/algorithm/1/ModelSolver_final.py'


    # 临时添加 runtime_dir 到 sys.path 以确保依赖模块可以被导入
    if runtime_dir not in sys.path:
        sys.path.insert(0, runtime_dir)

    # 4. 调用评估函数，并捕获异常信息到字符串
    error_info = None
    feasibility_failed_message = None

    # 调用评估函数

    solver_class = load_solver_class_from_file(solver_code_path, "ModelSolver")    
    evaluator = SolverEvaluation(solver_class=solver_class, time_limit=state.get('time_limit', 100))    
    success, message = evaluator.evaluate()

    if not success and error_info is None:
        # 如果 evaluate 返回 False 或 抛出可行性异常   
        if "Timeout Error" in message:
            # 1. Infeasible -> Reflect
            # 2. Timeout -> Reflect
            feasibility_failed_message = message
        elif "Feasibility Check Failed" in message:
            feasibility_failed_message = message
        elif "Runtime Error" in message:
            # 3. Runtime Error -> Fix Exception
            print(f"⚠ Evaluator reported Runtime Error.")
            error_info = message
        else:
            feasibility_failed_message = message
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
            "description": "The heuristic algorithm produced a solution that violates basic problem constraints or could not complete within the time limit.",
            "feasibility_check_errors": feasibility_failed_message,
        }
        

        save_workflow_log(
            output_dir,
            "evaluation_failed_feasibility",
            infeasible_report,
        )

        return {
            "fix_attempt_count": 0,
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

    # solver_code_path = '/home/hyn/LLM-MM-Agent-LLMINA-clean/output/LLMINA_20260108-155345/algorithm/1/ModelSolver_final.py'

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
        quality_report, reward = analyzer.run_batch_evaluation() # 获取质量报告

        # 保存报告
        algorithm_dir = ensure_algorithm_dirs(output_dir, current_step)
        report_path = os.path.join(algorithm_dir, 'quality_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, indent=2, ensure_ascii=False)

        print(f"[Constraint Analysis] Computed Reward: {reward}")

        mcts_updates = mcts_turn_step(state, reward=reward)
        
        return {
            "quality_report": quality_report,
            "evaluation_results": {"evaluation_success": True}, # 标记成功
            'current_step': current_step + 1,
            # Merge MCTS decision
            **mcts_updates
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
    Heuristic Reflection Node (e1 Operator)
    
    Analyzes execution feedback and decides whether to:
    1. Update specific function implementations (DESCRIPTION_UPDATE).
    2. Re-Architect the solution (ARCHITECTURE_UPDATE).
    """
    llm = state['llm']
    problem_str = state.get('problem_str', '')
    function_architecture = state.get('function_architecture', [])
    quality_report = state.get('quality_report', {})
    experiences = state.get('experiences', [])
    
    # Read Result Code from Disk
    solver_code_path = state.get('solver_code_path', '')
    solver_code_snippet = ""
    if solver_code_path and os.path.exists(solver_code_path):
        try:
             with open(solver_code_path, 'r', encoding='utf-8') as f:
                 solver_code_snippet = f.read()
        except Exception as e:
             solver_code_snippet = f"Error reading code: {e}"

    # Pass Architecture for Strategic Review
    func_arch_str = json.dumps(function_architecture, indent=2)
    
    prompt = HEURISTIC_REFLECT_PROMPT.format(
        problem_str=problem_str,
        function_architecture=func_arch_str,
        solver_code_snippet=solver_code_snippet,
        quality_report=json.dumps(quality_report, indent=2),
        experiences="\n".join(experiences) if experiences else "None"
    )

    try:
        print("[Reflect] Analyzing and Updating strategy...")
        response = llm.generate(prompt)
        content = response.content if hasattr(response, 'content') else str(response)

        # JSON Extraction
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
             match = re.search(r'```(?:json)?(.*?)```', content, re.DOTALL)
             if match: content = match.group(1).strip()
             else: content = content.split('```')[1].strip()
        
        evolution_plan = json.loads(content)
        
        # New Protocol: architecture_updates (MODIFY/ADD/DELETE actions)
        diagnosis = evolution_plan.get("diagnosis", "Optimization based on feedback.")
        architecture_updates = evolution_plan.get("architecture_updates", [])
        
        if not architecture_updates:
            print("[Reflect] Warning: No architecture_updates found in response. Skipping reflection.")
            return {"next_action": "end", "errors": ["No updates provided"]}
        
        # Apply Architecture Patches
        current_architecture = copy.deepcopy(function_architecture)
        functions_to_update_names = []
        
        # Determine dependency graph from previous state or rebuild
        # Since we might add/remove nodes, we can rebuild it dynamically or load it
        # For simplicity and correctness, let's load what we have and update it
        # state.get('function_dependency_graph', {}) could be used, but let's rebuild local representation
        
        for update in architecture_updates:
            action = update.get("action", "").upper()
            func_name = update.get("function_name")
            new_def = update.get("new_definition")
            
            # Robustness: If action is missing but we have new_definition, infer MODIFY/ADD
            if not action and new_def:
                action = "MODIFY" # Attempt modify, fallback to add inside logic
            
            if not func_name:
                print(f"[Reflect] Warning: Update entry missing 'function_name'. Skipping.")
                continue

            if action == "MODIFY":
                if not new_def:
                     print(f"[Reflect] Warning: MODIFY action for '{func_name}' missing 'new_definition'.")
                     continue
                
                found = False
                for i, existing_func in enumerate(current_architecture):
                    if existing_func['name'] == func_name:
                        current_architecture[i] = new_def
                        # Track the NEW name (in case of rename) to ensure generation
                        functions_to_update_names.append(new_def['name'])
                        found = True
                        print(f"[Reflect] MODIFY: {func_name} -> {new_def['name']}")
                        break
                
                if not found:
                    print(f"[Reflect] Warning: Target '{func_name}' for MODIFY not found. Treating as ADD.")
                    # Fallback to ADD logic
                    current_architecture.append(new_def)
                    functions_to_update_names.append(new_def['name'])
                    print(f"[Reflect] ADD (Fallback): {new_def['name']}")

            elif action == "ADD":
                if not new_def:
                     print(f"[Reflect] Warning: ADD action for '{func_name}' missing 'new_definition'.")
                     continue
                
                # Append first, will sort later
                current_architecture.append(new_def)
                functions_to_update_names.append(new_def['name'])
                print(f"[Reflect] ADD: {new_def['name']}")

            elif action == "REMOVE":
                found = False
                for i, existing_func in enumerate(current_architecture):
                    if existing_func['name'] == func_name:
                        current_architecture.pop(i)
                        found = True
                        print(f"[Reflect] REMOVE: {func_name}")
                        break
                
                if not found:
                     print(f"[Reflect] Warning: Target '{func_name}' for REMOVE not found.")

            else:
                 print(f"[Reflect] Warning: Unknown or invalid action '{action}' for function '{func_name}'.")
        
        # ============ Dynamic Topological Re-Sort ============
        print(f"[Reflect] Re-calculating dependency graph for {len(current_architecture)} functions...")
        
        func_map = {f['name']: f for f in current_architecture}
        func_names = set(func_map.keys())
        
        dep_graph_dict = {} 
        G = nx.DiGraph()
        
        # Add Nodes
        for f in current_architecture:
            G.add_node(f['name'])
            
        # Add Edges
        for f in current_architecture:
            f_name = f['name']
            dependencies = f.get('dependencies', [])
            
            # Filter valid
            valid_deps = [dep for dep in dependencies if dep in func_names]
            dep_graph_dict[f_name] = valid_deps
            
            for dep in valid_deps:
                G.add_edge(dep, f_name)
        
        # Sort
        try:
            sorted_names = list(nx.topological_sort(G))
            print(f"[Reflect] Topological Sort: {sorted_names}")
        except nx.NetworkXUnfeasible:
            print("[Reflect] Warning: Cycle detected! Falling back to listed order.")
            sorted_names = [f['name'] for f in current_architecture]
            
        new_architecture = [func_map[name] for name in sorted_names]
        
        if not isinstance(new_architecture, list):
             raise ValueError("Reflection result must contain 'function_architecture' list.")

        print(f"[Reflect] Diagnosis: {diagnosis}...")
        print(f"[Reflect] Targeted Updates: {functions_to_update_names}")


        # Targeted Generation Logic
        funcs_to_gen = []
        # We need to map the new architecture to generation flags
        for func in new_architecture:
            if func['name'] in functions_to_update_names:
                funcs_to_gen.append(1) # Regenerate
            else:
                funcs_to_gen.append(0) # Keep existing (skip)

        if sum(funcs_to_gen) == 0 and len(functions_to_update_names) == 0:
             print("[Reflect] Warning: No functions marked for update. Forcing full regeneration for safety.")
             funcs_to_gen = [1 for _ in range(len(new_architecture))]

        # Prepare updates
        updates = {
            "infeasible_experiences": [diagnosis], # Map diagnosis to experience log
            "function_architecture": new_architecture,
            "function_dependency_graph": dep_graph_dict,
            "functions_to_generate": funcs_to_gen,
            "current_function_id": 0,
            "next_action": "generate_next_function",
            "fix_attempt_count": 0
        }
            
        return updates

    except Exception as e:
        print(f"[Reflect] Failed: {e}")
        return {"next_action": "end", "errors": [str(e)]}

def feedback_guided_optimization_node(state: AgentState) -> AgentState:
    """
    Feedback-Guided Optimization Node (Advanced e1 Operator)
    Uses performance feedback to optimize specific algorithms (Mutation).
    """
    llm = state['llm']
    problem_str = state.get('problem_str', '')
    # heuristic_reference_code is NOT used in the prompt template for this node
    experiences = state.get('experiences', []) # Not used in prompt template
    function_architecture = state.get('function_architecture', [])
    quality_report = state.get('quality_report', {})
    
    # Read Result Code from Disk (Crucial for context)
    solver_code_path = state.get('solver_code_path', '')
    solver_code_snippet = ""
    if solver_code_path and os.path.exists(solver_code_path):
        try:
             with open(solver_code_path, 'r', encoding='utf-8') as f:
                 solver_code_snippet = f.read()
        except Exception as e:
             solver_code_snippet = f"Error reading code: {e}"

    # Pass Architecture for Strategic Review
    func_arch_str = json.dumps(function_architecture, indent=2)
    
    # NOTE: Prompt arguments must match FEEDBACK_GUIDED_OPTIMIZATION_PROMPT in prompt_template.py
    prompt = FEEDBACK_GUIDED_OPTIMIZATION_PROMPT.format(
        problem_str=problem_str,
        solver_code_snippet=solver_code_snippet, # Changed from heuristic_reference_code
        function_architecture=func_arch_str,
        quality_report=json.dumps(quality_report, indent=2)
    )

    try:
        print("[Feedback Optimization] Analyzing performance feedback for optimization...")
        response = llm.generate(prompt)
        content = response.content if hasattr(response, 'content') else str(response)

        # JSON Extraction
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
             match = re.search(r'```(?:json)?(.*?)```', content, re.DOTALL)
             if match: content = match.group(1).strip()
             else: content = content.split('```')[1].strip()
        
        evolution_plan = json.loads(content)
        
        diagnosis = evolution_plan.get("diagnosis", "Optimization based on feedback.")
        architecture_updates = evolution_plan.get("architecture_updates", [])
        
        if not architecture_updates:
            print("[Feedback Optimization] Warning: No architecture_updates found.")
            return {"next_action": "end", "errors": ["No updates provided"]}
        
        # Apply Architecture Patches
        current_architecture = copy.deepcopy(function_architecture)
        functions_to_update_names = []
        
        for update in architecture_updates:
            action = update.get("action", "").upper()
            func_name = update.get("function_name")
            new_def = update.get("new_definition")
            
            if not action and new_def: action = "MODIFY"
            
            if not func_name:
                continue

            if action == "MODIFY":
                if not new_def: continue
                found = False
                for i, existing_func in enumerate(current_architecture):
                    if existing_func['name'] == func_name:
                        current_architecture[i] = new_def
                        functions_to_update_names.append(new_def['name'])
                        found = True
                        print(f"[Opt] MODIFY: {func_name} -> {new_def['name']}")
                        break
                if not found:
                    # Fallback ADD
                    current_architecture.append(new_def)
                    functions_to_update_names.append(new_def['name'])
                    print(f"[Opt] ADD (Fallback): {new_def['name']}")

            elif action == "ADD":
                if not new_def: continue
                # Simply append, topological sort will handle ordering later
                current_architecture.append(new_def)
                functions_to_update_names.append(new_def['name'])
                print(f"[Opt] ADD: {new_def['name']}")

            elif action == "REMOVE":
                found = False
                for i, existing_func in enumerate(current_architecture):
                    if existing_func['name'] == func_name:
                        current_architecture.pop(i)
                        found = True
                        print(f"[Opt] REMOVE: {func_name}")
                        break
                if not found:
                     print(f"[Opt] Warning: REMOVE target '{func_name}' not found.")

        # ============ Dynamic Topological Re-Sort ============
        print(f"[Opt] Re-calculating dependency graph for {len(current_architecture)} functions...")
        
        func_map = {f['name']: f for f in current_architecture}
        func_names = set(func_map.keys())
        
        dep_graph_dict = {} 
        G = nx.DiGraph()
        
        # Add Nodes
        for f in current_architecture:
            G.add_node(f['name'])
            
        # Add Edges
        for f in current_architecture:
            f_name = f['name']
            dependencies = f.get('dependencies', [])
            
            # Filter valid
            valid_deps = [dep for dep in dependencies if dep in func_names]
            dep_graph_dict[f_name] = valid_deps
            
            for dep in valid_deps:
                G.add_edge(dep, f_name)
        
        # Sort
        try:
            sorted_names = list(nx.topological_sort(G))
            print(f"[Opt] Topological Sort: {sorted_names}")
        except nx.NetworkXUnfeasible:
            print("[Opt] Warning: Cycle detected! Falling back to listed order.")
            sorted_names = [f['name'] for f in current_architecture]
            
        new_architecture = [func_map[name] for name in sorted_names]

        # Generation Flags
        funcs_to_gen = []
        for func in new_architecture:
            if func['name'] in functions_to_update_names:
                funcs_to_gen.append(1)
            else:
                funcs_to_gen.append(0)

        if sum(funcs_to_gen) == 0 and len(functions_to_update_names) == 0:
             print("[Opt] Warning: No updates? Forcing full regen safely.")
             funcs_to_gen = [1 for _ in range(len(new_architecture))]

        return {
            "infeasible_experiences": [diagnosis],
            "function_architecture": new_architecture,
            "function_dependency_graph": dep_graph_dict, # Update State
            "functions_to_generate": funcs_to_gen,
            "current_function_id": 0,
            "next_action": "generate_next_function",
            "fix_attempt_count": 0
        }

    except Exception as e:
        print(f"[Opt] Failed: {e}")
        return {"next_action": "end", "errors": [str(e)]}

def exploratory_refactoring_node(state: AgentState) -> AgentState:
    """
    Exploratory Refactoring Node (Advanced e3 Operator)
    Proposes paradigm shifts for stagnant architectures (Exploration).
    """
    llm = state['llm']
    problem_str = state.get('problem_str', '')
    heuristic_reference_code = state.get('heuristic_reference_code', '')
    experiences = state.get('experiences', [])
    function_architecture = state.get('function_architecture', [])
    
    # Prompt matches template arguments
    prompt = EXPLORATORY_REFACTORING_PROMPT.format(
        problem_str=problem_str,
        heuristic_reference_code=heuristic_reference_code,
        output_function_template=OUTPUT_FUNCTION_TEMPLATE,
        experiences="\n".join(experiences) if experiences else "None",
        function_architecture=json.dumps(function_architecture, indent=2)
    )
    
    print("[Exploratory Refactoring] Designing RADICALLY NEW strategy...")
    try:
        response = llm.generate(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
             match = re.search(r'```(?:json)?(.*?)```', content, re.DOTALL)
             if match: content = match.group(1).strip()
             else: content = content.split('```')[1].strip()
        
        parsed_json = json.loads(content)
        temp_architecture = parsed_json['function_architecture']
        strategy_overview = parsed_json.get('strategy_overview', 'Exploratory Strategy')
        problem_analysis = parsed_json.get('problem_analysis', '')
        
        # ============ Topological Sort ============
        print(f"[Exploratory Refactoring] Designing dependency graph for {len(temp_architecture)} functions...")
        
        func_map = {f['name']: f for f in temp_architecture}
        func_names = set(func_map.keys())
        
        dep_graph_dict = {} 
        G = nx.DiGraph()
        
        for f in temp_architecture:
            G.add_node(f['name'])
            
        for f in temp_architecture:
            f_name = f['name']
            dependencies = f.get('dependencies', [])
            valid_deps = [dep for dep in dependencies if dep in func_names]
            dep_graph_dict[f_name] = valid_deps
            for dep in valid_deps:
                G.add_edge(dep, f_name)
        
        try:
            sorted_names = list(nx.topological_sort(G))
            print(f"[Exploratory Refactoring] Topological Sort: {sorted_names}")
        except nx.NetworkXUnfeasible:
            print("[Exploratory Refactoring] Warning: Cycle detected! Falling back.")
            sorted_names = [f['name'] for f in temp_architecture]
            
        new_architecture = [func_map[name] for name in sorted_names]
        
        return {
            'problem_analysis': problem_analysis,
            'strategy_overview': strategy_overview,
            'function_architecture': new_architecture,
            'function_dependency_graph': dep_graph_dict, # Update State
            'functions_to_generate': [1 for _ in range(len(new_architecture))],
            'current_function_id': 0,
            'next_action': 'generate_next_function',
            'function_codes': {} # Full reset for new paradigm
        }
        
    except Exception as e:
        print(f"[Exploratory Refactoring] Failed: {e}")
        return {"next_action": "end", "errors": [str(e)]}


def mcts_turn_step(state: AgentState, reward: float = None) -> AgentState:
    """
    Executes one MCTS step: Backpropagation (optional) + Selection + Expansion.
    Replaces the standalone mcts_management_node.
    """
    mcts = state.get('mcts')
    updates = {}
    
    # === 1. Backpropagation ===
    if reward is not None:
        last_node_id = state.get('current_mcts_node_id')
        solver_code_path = state.get('solver_code_path')
        
        if last_node_id and last_node_id in mcts.nodes:
            print(f"[MCTS] Backpropagating result for Node {last_node_id} (Reward: {reward:.4f})")
            node = mcts.nodes[last_node_id]
            node.score = reward
            
            # Persist Reference (Disk-based MCTS)
            algorithm_dir = os.path.dirname(solver_code_path) if solver_code_path else ""
            node.config = {
                'algorithm_dir': algorithm_dir, 
                'score': reward
            }
            mcts.backpropagate(node)
            # Log update is printed by backprop usually, or we can print here
            print(f"[MCTS] Node {node.node_id} Updated. Q: {node.Q:.4f}, Visits: {node.visits}")
    
    # === 2. Selection & Expansion ===
    init_pop_size = state.get('init_pop_size', 2)
    valid_root_children = [c for c in mcts.root.children if c.visits > 0]
    current_pop_size = len(valid_root_children)
    
    print(f"[MCTS] Population Size: {current_pop_size}/{init_pop_size}")
    
    operator = "i1"
    parent_a = None
    parent_b = None
    selected_node = None

    if current_pop_size < init_pop_size:
        # Phase: Initialization
            operator = "i1"
            selected_node = mcts.root
            print(f"[MCTS] Phase: Init ({current_pop_size+1}/{init_pop_size}) - Operator: i1 (Genesis)")
    else:
        # Phase: Evolution
        selected_node = mcts.select()
        if selected_node.description == "Root":
            operator = "i1"
            print("[MCTS] Phase: Evolution - Root Selected (Reset)")
        else:
            choice = random.random()
            if choice < 0.50:
                operator = "e1"
                print(f"[MCTS] Operator: e1 (Mutation) on {selected_node.node_id}")
            elif choice < 0.90:
                operator = "e3"
                print(f"[MCTS] Operator: e3 (Exploration) on {selected_node.node_id}")
            else:
                 operator = "i1"
                 print("[MCTS] Operator: i1 (Exploration Restart)")

    if parent_a is None:
        parent_a = selected_node

    # 3. Create Child Node (for the NEXT step)
    current_step = state.get('current_step', 1)
    new_node_id = str(current_step + 1) # Next Step ID
    
    new_node = MCTSNode(
        description=f"Pending Eval ({operator})",
        operator=operator,
        node_id=new_node_id
    )
    mcts.register_node(new_node)
    
    new_node.parent = parent_a
    parent_a.children.append(new_node) # Tree Link

    # 4. Determine Next Route
    # We map operators to the specific graph node names
    next_action_map = {
        "i1": "initialization",
        "e1": "optimization",
        "e3": "refactoring",
    }
    next_node = next_action_map.get(operator, "initialization")
    
    # Helper to load config
    def load_node_config(n):
        if not n or not n.config: return {}
        if 'function_codes' in n.config: return n.config # Memory fallback
        algo_dir = n.config.get('algorithm_dir')
        if not algo_dir or not os.path.exists(algo_dir): return {}
        
        cfg = {}
        try:
            # Load Arch
            p_arch = os.path.join(algo_dir, 'function_architecture.json')
            if os.path.exists(p_arch):
                with open(p_arch, 'r', encoding='utf-8') as f:
                    cfg['function_architecture'] = json.load(f)
            # Load Codes
            p_code = os.path.join(algo_dir, 'function_codes.json')
            if os.path.exists(p_code):
                with open(p_code, 'r', encoding='utf-8') as f:
                    cfg['function_codes'] = json.load(f)
            
            # Load Solver Code Path
            p_solver = os.path.join(algo_dir, 'ModelSolver_final.py')
            if os.path.exists(p_solver):
                cfg['solver_code_path'] = p_solver

            # Load Quality Report
            p_report = os.path.join(algo_dir, 'quality_report.json')
            if os.path.exists(p_report):
                with open(p_report, 'r', encoding='utf-8') as f:
                    cfg['quality_report'] = json.load(f)

            return cfg
        except Exception:
            return {}
    
    parent_a_config = load_node_config(parent_a)
    parent_b_config = load_node_config(parent_b)

    # Prepare Updates
    updates.update({
        'mcts': mcts,
        'current_mcts_node_id': new_node_id,
        'mcts_operator': operator,
        'next_action_node': next_node,
        'parent_a_config': parent_a_config,
        'parent_b_config': parent_b_config
    })
    
    # State Inheritance Logic
    if operator in ['e1', 'e3']:
        updates['function_architecture'] = parent_a_config.get('function_architecture')
        f_codes = parent_a_config.get('function_codes')
        if f_codes:
            updates['function_codes'] = f_codes
        
        # Inherit Solver Code Path for Context
        solver_path = parent_a_config.get('solver_code_path')
        if solver_path:
            updates['solver_code_path'] = solver_path
        
        # Inherit Quality Report for Context
        q_report = parent_a_config.get('quality_report')
        if q_report:
            updates['quality_report'] = q_report
            
    elif operator == 'i1':
        # Clear codes for fresh start.  
        # Architecture will be designed in heuristic_initialization_node.
        updates['function_codes'] = {} 
        updates['current_function_id'] = 0
        updates['functions_to_generate'] = [] # Reset, will be filled by initialization node

    return updates
