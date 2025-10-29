"""
代码集成器（Code Integrator）
用于整合多个任务生成的代码模块，避免重复生成完整solver的问题

核心思想：
- 前N-1个任务生成独立的辅助函数/模块
- 最后一个任务生成主控制流和完整的llm_solver函数
- CodeIntegrator负责将所有模块组装成最终的可运行代码
"""
from typing import Dict, List, Tuple
import re


class CodeIntegrator:
    """代码集成器 - 整合多任务生成的代码"""
    
    def __init__(self):
        self.task_codes = {}  # {task_id: code_module}
        self.task_dependencies = {}  # {task_id: [依赖的task_id列表]}
        
    def add_task_code(self, task_id: int, code: str, dependencies: List[int] = None):
        """
        添加一个任务生成的代码模块
        
        Args:
            task_id: 任务ID
            code: 生成的代码
            dependencies: 此任务依赖的其他任务ID列表
        """
        self.task_codes[task_id] = code
        self.task_dependencies[task_id] = dependencies or []
    
    def integrate_all(self, final_task_id: int) -> str:
        """
        整合所有任务的代码，生成最终的完整solver文件
        
        Args:
            final_task_id: 最后一个任务的ID（应该包含llm_solver主函数）
        
        Returns:
            完整的可运行代码字符串
        """
        integrated_code = []
        
        # 1. 添加文件头注释
        integrated_code.append('"""')
        integrated_code.append('LLM-Generated INA Placement and Routing Solver')
        integrated_code.append('Auto-integrated from multi-task code generation')
        integrated_code.append('"""')
        integrated_code.append('')
        
        # 2. 收集所有导入语句
        all_imports = self._collect_imports()
        integrated_code.extend(all_imports)
        integrated_code.append('')
        
        # 3. 按依赖顺序添加辅助函数
        ordered_tasks = self._get_topological_order(final_task_id)
        
        for task_id in ordered_tasks[:-1]:  # 除了最后一个任务
            if task_id in self.task_codes:
                integrated_code.append(f'# {"=" * 78}')
                integrated_code.append(f'# Task {task_id} - Helper Functions')
                integrated_code.append(f'# {"=" * 78}')
                
                # 提取纯函数定义（去除导入语句和主程序）
                clean_code = self._extract_functions(self.task_codes[task_id])
                integrated_code.append(clean_code)
                integrated_code.append('')
        
        # 4. 添加最终的llm_solver函数
        final_task_code = self.task_codes.get(final_task_id, '')
        integrated_code.append(f'# {"=" * 78}')
        integrated_code.append(f'# Main Solver Function (Task {final_task_id})')
        integrated_code.append(f'# {"=" * 78}')
        integrated_code.append(final_task_code)
        
        return '\n'.join(integrated_code)
    
    def _collect_imports(self) -> List[str]:
        """收集所有代码中的导入语句并去重"""
        imports = set()
        
        # 基础导入（总是需要的）
        base_imports = [
            'import numpy as np',
            'import networkx as nx',
            'from typing import List, Tuple, Dict, Set',
            'import pulp',
            'from collections import defaultdict'
        ]
        imports.update(base_imports)
        
        # 从所有任务代码中提取导入语句
        for code in self.task_codes.values():
            for line in code.split('\n'):
                line = line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    imports.add(line)
        
        # 排序并返回
        sorted_imports = sorted(imports, key=lambda x: (
            0 if x.startswith('import ') else 1,  # import先于from
            x  # 字母顺序
        ))
        
        return sorted_imports
    
    def _get_topological_order(self, final_task_id: int) -> List[int]:
        """
        根据任务依赖关系返回拓扑排序
        确保依赖的任务代码先于使用它的任务代码
        """
        # 简化版：如果没有明确的依赖关系，就按ID顺序
        all_task_ids = sorted(self.task_codes.keys())
        
        # 确保final_task_id在最后
        if final_task_id in all_task_ids:
            all_task_ids.remove(final_task_id)
            all_task_ids.append(final_task_id)
        
        return all_task_ids
    
    def _extract_functions(self, code: str) -> str:
        """
        从代码中提取函数定义，去除导入语句和主程序部分
        """
        lines = code.split('\n')
        cleaned_lines = []
        in_function = False
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            
            # 跳过导入语句
            if stripped.startswith('import ') or stripped.startswith('from '):
                continue
            
            # 跳过主程序部分
            if stripped.startswith('if __name__'):
                break
            
            # 检测函数定义
            if stripped.startswith('def ') or stripped.startswith('class '):
                in_function = True
                indent_level = len(line) - len(line.lstrip())
            
            # 如果在函数内或是函数定义
            if in_function:
                cleaned_lines.append(line)
                
                # 检查是否离开函数
                if stripped and not line.startswith(' ' * (indent_level + 1)) and not stripped.startswith('def ') and not stripped.startswith('class '):
                    if indent_level == 0:
                        in_function = False
            elif stripped.startswith('#'):
                # 保留注释
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def validate_integration(self, integrated_code: str) -> Tuple[bool, List[str]]:
        """
        验证集成后的代码
        
        Returns:
            (is_valid, issues_list)
        """
        issues = []
        
        # 检查1: 是否包含llm_solver函数
        if 'def llm_solver(' not in integrated_code:
            issues.append("Missing llm_solver function definition")
        
        # 检查2: 函数签名是否正确
        if 'def llm_solver(instance, network, K, jobs_num, Cs, topo_name)' not in integrated_code:
            issues.append("llm_solver function signature doesn't match template")
        
        # 检查3: 是否有返回语句 (修改为只返回两个值)
        if 'return ' not in integrated_code or not re.search(
            r'return.*ina_placement.*jobs_routing(?!.*makespan)', 
            integrated_code, 
            re.DOTALL
        ):
            issues.append("llm_solver may not return the required tuple (ina_placement, jobs_routing)")
        
        # 检查4: 基本语法检查（尝试编译）
        try:
            compile(integrated_code, '<integrated>', 'exec')
        except SyntaxError as e:
            issues.append(f"Syntax error in integrated code: {e}")
        
        return len(issues) == 0, issues


class TaskTypeClassifier:
    """任务类型分类器 - 判断任务是否是最后的集成任务"""
    
    @staticmethod
    def is_final_integration_task(task_id: int, total_tasks: int, task_description: str) -> bool:
        """
        判断是否是最后的集成任务
        
        Args:
            task_id: 任务ID
            total_tasks: 总任务数
            task_description: 任务描述
        
        Returns:
            True if this is the final integration task
        """
        # 方法1: 检查是否是最后一个任务
        if task_id == total_tasks - 1 or task_id == total_tasks:
            return True
        
        # 方法2: 检查任务描述中的关键词
        integration_keywords = [
            'integrate', 'final', 'complete solver', 'main function',
            '集成', '最终', '完整', '主函数', 'llm_solver'
        ]
        
        desc_lower = task_description.lower()
        if any(keyword in desc_lower for keyword in integration_keywords):
            return True
        
        return False
    
    @staticmethod
    def get_task_type(task_description: str) -> str:
        """
        根据任务描述判断任务类型
        
        Returns:
            'helper_function' | 'algorithm_module' | 'integration' | 'testing'
        """
        desc_lower = task_description.lower()
        
        # 集成任务
        if any(kw in desc_lower for kw in ['integrate', 'complete solver', '集成', '完整solver']):
            return 'integration'
        
        # 测试任务
        if any(kw in desc_lower for kw in ['test', 'validate', 'verify', '测试', '验证']):
            return 'testing'
        
        # 算法模块
        if any(kw in desc_lower for kw in ['algorithm', 'strategy', 'heuristic', '算法', '策略']):
            return 'algorithm_module'
        
        # 辅助函数
        return 'helper_function'


def create_task_specific_prompt(
    task_id: int,
    total_tasks: int,
    task_description: str,
    task_type: str,
    dependent_codes: Dict[str, str],
    code_template: str
) -> str:
    """
    根据任务类型创建特定的代码生成提示词
    
    Args:
        task_id: 任务ID
        total_tasks: 总任务数
        task_description: 任务描述
        task_type: 任务类型 ('helper_function' | 'algorithm_module' | 'integration' | 'testing')
        dependent_codes: 依赖的代码
        code_template: 代码模板
    """
    if task_type == 'integration':
        # 集成任务：生成完整solver
        return f"""You are implementing the FINAL INTEGRATION task.

**Task Description:**
{task_description}

**Available Helper Functions from Previous Tasks:**
{_format_dependent_codes(dependent_codes)}

**Code Template to Follow:**
```python
{code_template}
```

**YOUR TASK:**
Generate a COMPLETE `llm_solver` function that:
1. **Imports/uses helper functions** from previous tasks
2. **Implements the main algorithm logic** (control flow, optimization loop, etc.)
3. **Follows the template signature exactly**: llm_solver(instance, network, K, jobs_num, Cs, topo_name)
4. **Returns the required tuple**: (ina_placement_expanded, jobs_routing_expanded) - 只返回两个值，不返回makespan

This is the FINAL task - generate the complete solver that integrates everything."""

    elif task_type == 'helper_function':
        # 辅助函数任务：生成单个函数
        return f"""You are implementing a HELPER FUNCTION (not the complete solver).

**Task Description:**
{task_description}

**Dependencies from Previous Tasks:**
{_format_dependent_codes(dependent_codes)}

**YOUR TASK:**
Generate ONE OR MORE helper functions for this specific subtask:
1. **Focus on this subtask only** - don't generate the complete llm_solver
2. **Include clear function signatures** with type hints
3. **Add comprehensive docstrings** explaining inputs/outputs
4. **Make functions reusable** by later tasks

**IMPORTANT:** DO NOT generate the complete `llm_solver` function. That will be done in the final integration task.

Example output:
```python
def helper_function_name(param1: Type1, param2: Type2) -> ReturnType:
    \"\"\"
    Brief description of what this function does.
    
    Args:
        param1: Description
        param2: Description
    
    Returns:
        Description of return value
    \"\"\"
    # Implementation here
    ...
```"""

    elif task_type == 'algorithm_module':
        # 算法模块任务
        return f"""You are implementing an ALGORITHM MODULE.

**Task Description:**
{task_description}

**Dependencies:**
{_format_dependent_codes(dependent_codes)}

**YOUR TASK:**
Implement the algorithm/strategy described in this task:
1. **Create one or more functions** that implement the algorithm
2. **Include algorithm-specific data structures** if needed
3. **Document the algorithm logic** clearly
4. **Make it reusable** by the final integration task

**IMPORTANT:** Generate the algorithm implementation as modular functions, NOT the complete llm_solver."""

    else:  # testing
        return f"""You are implementing TEST CODE.

**Task Description:**
{task_description}

**Code to Test:**
{_format_dependent_codes(dependent_codes)}

**YOUR TASK:**
Generate test code that validates the implementation."""


def _format_dependent_codes(dependent_codes: Dict[str, str]) -> str:
    """格式化依赖代码信息"""
    if not dependent_codes:
        return "No dependencies from previous tasks."
    
    formatted = []
    for task_id, code in dependent_codes.items():
        # 提取函数签名
        functions = re.findall(r'def\s+(\w+)\s*\([^)]*\)', code)
        if functions:
            formatted.append(f"Task {task_id}: {', '.join(functions)}")
    
    return '\n'.join(formatted) if formatted else "Previous task codes available for import."
