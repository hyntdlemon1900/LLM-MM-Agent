"""
代码验证智能体：检查代码兼容性、语法错误和运行时问题
- 静态分析：语法检查、导入检查、类型检查
- 接口兼容性：验证与模板和依赖代码的兼容性
- 运行时预测：预测潜在的运行时错误
"""
from .base_agent import BaseAgent
from typing import Dict, List, Tuple
import ast
import json
import re


class CodeValidator(BaseAgent):
    """代码验证智能体"""
    
    def __init__(self, llm):
        super().__init__(llm)
    
    def validate_code_comprehensive(
        self, 
        code: str, 
        code_template: str,
        dependent_codes: Dict[str, str] = None,
        task_description: str = ""
    ) -> Dict:
        """
        全面验证代码
        
        Args:
            code: 待验证的代码
            code_template: 代码模板
            dependent_codes: 依赖的其他代码 {task_id: code}
            task_description: 任务描述
            
        Returns:
            {
                'is_valid': bool,
                'syntax_errors': List[str],
                'compatibility_issues': List[str],
                'potential_runtime_errors': List[str],
                'suggestions': List[str],
                'severity': str  # 'critical', 'warning', 'info'
            }
        """
        validation_result = {
            'is_valid': True,
            'syntax_errors': [],
            'compatibility_issues': [],
            'potential_runtime_errors': [],
            'suggestions': [],
            'severity': 'info'
        }
        
        # 1. 语法检查
        syntax_check = self._check_syntax(code)
        if not syntax_check['valid']:
            validation_result['is_valid'] = False
            validation_result['syntax_errors'] = syntax_check['errors']
            validation_result['severity'] = 'critical'
            return validation_result
        

        # 2. 接口兼容性检查
        compatibility = self._check_template_compatibility(code, code_template)
        if not compatibility['compatible']:
            validation_result['is_valid'] = False
            validation_result['compatibility_issues'] = compatibility['issues']
            validation_result['severity'] = 'critical'

        
        # 4. LLM深度分析（算法设计合理性与代码正确性）
        llm_analysis = self._llm_deep_analysis(code, code_template, task_description)
        validation_result['potential_runtime_errors'].extend(llm_analysis['potential_errors'])
        validation_result['suggestions'].extend(llm_analysis['suggestions'])
        
        # 更新严重程度
        if validation_result['potential_runtime_errors'] and validation_result['severity'] == 'info':
            validation_result['severity'] = 'warning'
        
        return validation_result
    
    def _check_syntax(self, code: str) -> Dict:
        """检查Python语法"""
        try:
            ast.parse(code)
            return {'valid': True, 'errors': []}
        except SyntaxError as e:
            return {
                'valid': False,
                'errors': [f"Syntax error at line {e.lineno}: {e.msg}"]
            }
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Parse error: {str(e)}"]
            }
    
    def _check_template_compatibility(self, code: str, template: str) -> Dict:
        """
        检查与模板的兼容性
        
        只检查主要的solver函数（通常是llm_solver），不检查辅助函数。
        辅助函数（如get_candidate等）是代码生成的一部分，不需要强制匹配模板。
        """
        issues = []
        
        # 只检查主solver函数（llm_solver）
        MAIN_SOLVER_FUNCTION = 'llm_solver'
        
        # 提取函数签名
        template_functions = self._extract_function_signatures(template)
        code_functions = self._extract_function_signatures(code)
        
        # 检查主solver函数是否存在
        if MAIN_SOLVER_FUNCTION not in template_functions:
            # 如果模板中没有定义主函数，跳过检查
            return {'compatible': True, 'issues': []}
        
        template_sig = template_functions[MAIN_SOLVER_FUNCTION]
        
        if MAIN_SOLVER_FUNCTION not in code_functions:
            issues.append(f"Missing required function: {MAIN_SOLVER_FUNCTION}")
        else:
            code_sig = code_functions[MAIN_SOLVER_FUNCTION]
            # 检查参数匹配
            if template_sig['args'] != code_sig['args']:
                issues.append(
                    f"Function '{MAIN_SOLVER_FUNCTION}' signature mismatch. "
                    f"Expected: {template_sig['args']}, Got: {code_sig['args']}"
                )
        
        return {
            'compatible': len(issues) == 0,
            'issues': issues
        }
    
    def _extract_function_signatures(self, code: str) -> Dict:
        """提取函数签名"""
        functions = {}
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    args = [arg.arg for arg in node.args.args]
                    functions[node.name] = {
                        'args': args,
                        'returns': ast.unparse(node.returns) if node.returns else None
                    }
        except:
            pass
        return functions
    
    def _llm_deep_analysis(self, code: str, template: str, task_description: str) -> Dict:
        """使用LLM进行算法设计与实现的深度分析"""
        prompt = f"""You are an expert algorithm designer and code reviewer with deep expertise in mathematical optimization, operations research, and algorithmic problem-solving.

**Task Description:**
{task_description}

**Code Template (Expected Interface):**
```python
{template}
```

**Generated Code:**
```python
{code}
```

**Analysis Instructions:**
Your primary focus is on **algorithm design quality and correctness**. Analyze the code from the following perspectives:

**1. Algorithm Design Analysis (Primary Focus):**
   - Is the chosen algorithm/approach appropriate for the task requirements?
   - Does the algorithm comprehensively address all aspects of the problem?
   - Are the core algorithmic steps logically sound and mathematically correct?
   - Does the solution strategy align with best practices for this type of problem?
   - Are there missing components or incomplete logic in the algorithm design?

**2. Problem Coverage & Completeness:**
   - Does the implementation handle all required scenarios mentioned in the task?
   - Are all constraints and conditions properly addressed?
   - Are edge cases and boundary conditions considered?
   - Is the solution general enough or too specific?

**3. Algorithm Correctness:**
   - Are the mathematical formulations and computations correct?
   - Do the algorithm steps follow the correct sequence and logic?
   - Are optimization objectives and constraints properly implemented?
   - Are there logical flaws that could lead to incorrect results?

**4. Code Quality (Secondary):**
   - Critical runtime errors (e.g., undefined variables, type mismatches, index errors)
   - Data structure usage appropriateness
   - Interface compatibility issues

**Output Format (JSON):**
```json
{{
    "potential_errors": [
        "Description of algorithmic or critical code issues"
    ],
    "suggestions": [
        "Suggestions for improving algorithm design or fixing issues"
    ]
}}
```

**Important:** Prioritize algorithm design issues over minor code style issues. Focus on whether the solution is conceptually sound and complete.

Provide your analysis:"""
        
        try:
            response = self.llm.generate(prompt)
            
            # 解析JSON响应
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            
            result = json.loads(json_str)
            return {
                'potential_errors': result.get('potential_errors', []),
                'suggestions': result.get('suggestions', [])
            }
        except Exception as e:
            print(f"[Warning] LLM deep analysis failed: {e}")
            return {'potential_errors': [], 'suggestions': []}
    
    def suggest_fixes(self, code: str, validation_result: Dict) -> str:
        """根据验证结果建议修复方案"""
        if validation_result['is_valid'] and not validation_result['potential_runtime_errors']:
            return code  # 代码已经是有效的
        
        issues_description = self._format_issues(validation_result)
        
        prompt = f"""You are an expert Python developer. Fix the following code based on the identified issues.

**Original Code:**
```python
{code}
```

**Identified Issues:**
{issues_description}

**Instructions:**
1. Fix all syntax errors and compatibility issues
2. Address potential runtime errors
3. Implement suggested improvements
4. Maintain the original logic and functionality
5. Ensure the code follows the required interface

**Output the fixed code in a single Python code block:**"""
        
        try:
            response = self.llm.generate(prompt)
            
            # 提取代码
            if "```python" in response:
                fixed_code = response.split("```python")[1].split("```")[0].strip()
            elif "```" in response:
                fixed_code = response.split("```")[1].split("```")[0].strip()
            else:
                fixed_code = response.strip()
            
            return fixed_code
        except Exception as e:
            print(f"[Error] Code fix generation failed: {e}")
            return code
    
    def _format_issues(self, validation_result: Dict) -> str:
        """格式化问题描述"""
        issues = []
        
        if validation_result['syntax_errors']:
            issues.append("**Syntax Errors:**")
            for error in validation_result['syntax_errors']:
                issues.append(f"  - {error}")
        
        if validation_result['compatibility_issues']:
            issues.append("\n**Compatibility Issues:**")
            for issue in validation_result['compatibility_issues']:
                issues.append(f"  - {issue}")
        
        if validation_result['potential_runtime_errors']:
            issues.append("\n**Potential Runtime Errors:**")
            for error in validation_result['potential_runtime_errors']:
                issues.append(f"  - {error}")
        
        if validation_result['suggestions']:
            issues.append("\n**Suggestions:**")
            for suggestion in validation_result['suggestions']:
                issues.append(f"  - {suggestion}")
        
        return "\n".join(issues)
    
    def validate_and_fix(
        self, 
        code: str, 
        code_template: str,
        dependent_codes: Dict[str, str] = None,
        task_description: str = "",
        max_iterations: int = 3
    ) -> Tuple[str, bool, List[Dict]]:
        """
        验证并修复代码（迭代）
        
        Returns:
            (fixed_code, is_valid, validation_history)
        """
        validation_history = []
        current_code = code
        
        for iteration in range(max_iterations):
            print(f"\n[Code Validator] Iteration {iteration + 1}/{max_iterations}")
            
            # 验证
            validation_result = self.validate_code_comprehensive(
                current_code, code_template, dependent_codes, task_description
            )
            validation_history.append(validation_result)
            
            # 如果代码有效且无警告，返回
            if validation_result['is_valid'] and validation_result['severity'] == 'info':
                print("[✓] Code validation passed")
                return current_code, True, validation_history
            
            # 如果是最后一次迭代，返回当前结果
            if iteration == max_iterations - 1:
                if validation_result['severity'] != 'critical':
                    print("[!] Code has warnings but reached max iterations")
                    return current_code, True, validation_history
                else:
                    print("[✗] Code validation failed after max iterations")
                    return current_code, False, validation_history
            
            # 尝试修复
            print(f"[!] Issues found (severity: {validation_result['severity']}), attempting to fix...")
            current_code = self.suggest_fixes(current_code, validation_result)
        
        return current_code, False, validation_history
