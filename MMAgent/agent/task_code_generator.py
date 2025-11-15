"""
任务代码生成模块 - 渐进式任务求解

核心思想：
1. Task 1: 基于模板生成初始实现
2. Task 2-N: 基于上一个任务的代码继续开发，累积功能
3. 每个任务生成完整的solver代码（不是代码片段）
"""

from typing import Dict, Tuple, Any
from MMAgent.prompt.prompt_template import TASK_CODE_GENERATION_PROMPT


def generate_task_code(llm, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    生成任务代码（渐进式）
    
    Args:
        llm: 语言模型实例
        context: 代码生成上下文，包含：
            - task_id: 当前任务ID
            - total_tasks: 总任务数
            - is_final_task: 是否最后一个任务
            - task_description: 当前任务描述
            - algorithm_solution: 整体算法方案
            - template_code: 模板代码（Task 1 使用）
            - previous_code: 上一个任务的代码（Task 2+ 使用）
            - previous_task_description: 上一个任务的描述
            - completed_task_summaries: 已完成任务的摘要列表
            - problem_dir: 问题目录
            
    Returns:
        (task_code, generation_info): 生成的代码和生成信息
    """
    task_id = context['task_id']
    total_tasks = context['total_tasks']
    is_final_task = context['is_final_task']
    task_description = context['task_description']
    algorithm_solution = context['algorithm_solution']
    
    # ============================================================
    # 构建 prompt
    # ============================================================
    if task_id == 1:
        # Task 1: 基于模板
        template_code = context.get('template_code', '')
        
        prompt = TASK_CODE_GENERATION_PROMPT.format(
            task_id=task_id,
            total_tasks=total_tasks,
            task_description=task_description,
            algorithm_solution=algorithm_solution,
            base_code_type="template",
            base_code=template_code,
            previous_task_description="N/A (first task)",
            completed_tasks_summary="None (first task)",
            focus_instruction=_get_task1_focus_instruction()
        )
    else:
        # Task 2+: 基于上一个任务的代码
        previous_code = context.get('previous_code', '')
        previous_task_description = context.get('previous_task_description', '')
        completed_task_summaries = context.get('completed_task_summaries', [])
        completed_tasks_str = '\n'.join(completed_task_summaries) if completed_task_summaries else "None"
        
        prompt = TASK_CODE_GENERATION_PROMPT.format(
            task_id=task_id,
            total_tasks=total_tasks,
            task_description=task_description,
            algorithm_solution=algorithm_solution,
            base_code_type="previous task's code",
            base_code=previous_code,
            previous_task_description=previous_task_description,
            completed_tasks_summary=completed_tasks_str,
            focus_instruction=_get_incremental_focus_instruction(is_final_task)
        )
    
    # ============================================================
    # 调用 LLM 生成代码
    # ============================================================
    messages = [{"role": "user", "content": prompt}]
    
    response = llm.generate(messages)
    generated_code = response.content.strip()
    
    # ============================================================
    # 提取代码（处理 markdown 格式）
    # ============================================================
    task_code = _extract_code_from_response(generated_code)
    
    # ============================================================
    # 生成信息
    # ============================================================
    generation_info = {
        'task_id': task_id,
        'total_tasks': total_tasks,
        'is_final_task': is_final_task,
        'base_code_type': 'template' if task_id == 1 else 'previous_code',
        'code_length': len(task_code),
        'prompt_length': len(prompt),
        'response_length': len(generated_code),
    }
    
    return task_code, generation_info


def _get_task1_focus_instruction() -> str:
    """获取 Task 1 的重点指导"""
    return """
## Focus for Task 1:

You are implementing the **first task**. Your focus should be on:

1. **Foundation Setup**: Establish the basic structure of the solver function
2. **Data Extraction**: Properly extract and organize input data
3. **Core Algorithm for Task 1**: Implement the specific algorithm described in Task 1 description
4. **Placeholder for Future Tasks**: For Tasks 2-N (not yet implemented), use simple placeholder logic

**Important Guidelines**:
- Start from the provided template
- Implement ONLY Task 1's specific algorithm
- For future tasks (2-N), use simple baseline logic (e.g., greedy, random) as placeholders
- Ensure the function returns valid output format
- Add clear comments marking "Task 1 implementation" vs "Task 2+ placeholder"

Your code should be a complete, runnable solver function.
"""


def _get_incremental_focus_instruction(is_final_task: bool) -> str:
    """获取增量任务的重点指导"""
    if is_final_task:
        return """
## Focus for Final Task:

You are implementing the **final task**. Your focus should be on:

1. **Preserve Previous Work**: Keep ALL code from previous tasks intact
2. **Complete Final Task**: Implement the specific algorithm described in this task's description
3. **Integration**: Ensure all tasks work together seamlessly
4. **Final Polish**: 
   - Remove any remaining placeholders
   - Optimize for performance
   - Ensure all constraints are satisfied
   - Add final validation checks

**Important Guidelines**:
- DO NOT modify or remove code from previous tasks
- Implement ONLY this task's specific algorithm
- This is the final version that will be tested
- Ensure robustness and correctness

Your code should be a complete, production-ready solver function.
"""
    else:
        return """
## Focus for This Task:

You are implementing an **intermediate task**. Your focus should be on:

1. **Preserve Previous Work**: Keep ALL code from previous tasks intact
2. **Add Current Task Logic**: Implement the specific algorithm described in this task's description
3. **Placeholder for Future Tasks**: For not-yet-implemented tasks, maintain simple placeholder logic

**Important Guidelines**:
- DO NOT modify or remove code from previous tasks (Tasks 1 to {task_id-1})
- Implement ONLY this task's specific algorithm
- For future tasks, keep or update placeholders
- Ensure the function remains complete and runnable
- Add clear comments marking which task each code section belongs to

Your code should be a complete, runnable solver function that includes:
- Previous tasks' implementations (unchanged)
- Current task's implementation (new)
- Future tasks' placeholders (simple baseline)
"""


def _extract_code_from_response(response: str) -> str:
    """
    从 LLM 响应中提取代码
    
    处理可能的 markdown 代码块格式
    """
    code = response.strip()
    
    # 检查是否包含 markdown 代码块
    if "```python" in code:
        # 提取 ```python 和 ``` 之间的代码
        start = code.find("```python") + len("```python")
        end = code.find("```", start)
        if end > start:
            code = code[start:end].strip()
    elif "```" in code:
        # 提取 ``` 和 ``` 之间的代码
        start = code.find("```") + len("```")
        end = code.find("```", start)
        if end > start:
            code = code[start:end].strip()
    
    # 移除可能的前导/尾随空白
    code = code.strip()
    
    return code


def validate_task_code(code: str, task_id: int) -> Tuple[bool, str]:
    """
    验证任务代码的基本正确性
    
    Args:
        code: 生成的代码
        task_id: 任务ID
        
    Returns:
        (is_valid, error_message)
    """
    # 基本检查
    if not code:
        return False, "Code is empty"
    
    # 检查是否包含函数定义
    if "def llm_solver(" not in code and "def solver(" not in code:
        return False, "No solver function definition found"
    
    # 检查是否包含 return 语句
    if "return" not in code:
        return False, "No return statement found"
    
    # 尝试语法检查
    try:
        compile(code, '<string>', 'exec')
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)}"
    
    return True, ""
