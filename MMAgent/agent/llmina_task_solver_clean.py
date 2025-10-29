"""
渐进式任务求解智能体 - 通用版本

核心工作流（适用于任何可分解的优化问题）：
1. 任务拆解：将复杂问题分解为多个子任务
2. 渐进式实现：每个任务都生成完整的solver函数
   - 本任务focus：精确实现当前任务的核心算法
   - 前面任务：复制继承之前任务的精确实现
   - 未来任务：默认占位（使用简单的基线算法）
3. 统一验证：每个任务完成后都用标准验证框架测试完整solver
4. 性能追踪：观察目标函数的渐进改进

优势：
- 每个任务产出完整可运行的solver（无需额外集成）
- 验证框架统一（都用标准测试函数）
- 性能改进可见（Task 1→2→3→N 性能逐步优化）
- 调试友好（每个版本都独立可测）

适用场景：
- 组合优化问题（路由、调度、分配等）
- 机器学习流程（数据处理→特征工程→模型训练→优化）
- 系统设计（架构→模块→集成→优化）
"""
from .base_agent import BaseAgent
from .llmina_code_validator import CodeValidator
import os
import subprocess
import selectors
from typing import Tuple, Dict


class EnvException(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message


def execute_script(script_path, work_dir):
    """执行Python脚本并捕获输出"""
    try:
        device = 0
        python = "python"
        cmd = f"CUDA_VISIBLE_DEVICES={device} {python} -u {script_path}"
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
            text=True, shell=True, cwd=work_dir
        )

        stdout_lines = []
        stderr_lines = []

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        selector.register(process.stderr, selectors.EVENT_READ)

        while process.poll() is None and selector.get_map():
            events = selector.select(timeout=1)
            for key, _ in events:
                line = key.fileobj.readline()
                if key.fileobj == process.stdout:
                    print("STDOUT:", line, end=" ")
                    stdout_lines.append(line)
                else:
                    print("STDERR:", line, end=" ")
                    stderr_lines.append(line)

        for line in process.stdout:
            print("STDOUT:", line, end=" ")
            stdout_lines.append(line)
        for line in process.stderr:
            print("STDERR:", line, end=" ")
            stderr_lines.append(line)

        return_code = process.returncode

        if return_code != 0:
            observation = "".join(stderr_lines)
        else:
            observation = "".join(stdout_lines)
        if observation == "" and return_code == 0:
            observation = "".join(stderr_lines)
        
        return "The script has been executed. Here is the output:\n" + observation
    except Exception as e:
        print("++++", "Wrong!")
        raise EnvException(f"Something went wrong in executing {script_path}: {e}")


class ProgressiveTaskSolver(BaseAgent):
    """
    渐进式任务求解智能体 - 通用版本
    
    适用于任何可分解的优化/求解问题。
    每个任务都生成完整的solver函数，但有明确的focus：
    - Task 1: 精确实现第一部分算法，其余用默认
    - Task 2: 复制Task 1 + 精确实现第二部分算法，其余用默认
    - Task 3: 复制Task 1-2 + 精确实现第三部分算法，其余用默认
    - Task N: 复制Task 1-(N-1) + 精确实现最终优化算法
    
    示例应用：
    - 优化问题：候选生成→初始解→局部优化→全局优化
    - ML流程：数据处理→特征工程→模型训练→超参优化
    - 系统设计：架构设计→核心模块→辅助功能→性能优化
    """
    
    def __init__(self, llm):
        super().__init__(llm)
        self.code_validator = CodeValidator(llm)
        self.task_history = {}  # 存储每个任务的实现，供后续任务继承
    
    def solve_task(
        self,
        task_id: int,
        total_tasks: int,
        task_description: str,
        modeling_solution: str,
        coordinator,
        config: Dict
    ) -> Dict:
        """
        渐进式任务求解 - 每个任务都生成完整的solver（通用版本）
        
        工作流程：
        1. 分析本任务的内容和要求
        2. 设计本任务的算法
        3. 生成完整solver：
           - 复制前面任务的精确实现
           - 精确实现本任务的算法
           - 未来任务用默认占位
        4. 验证完整solver（用标准测试框架）
        
        Args:
            task_id: 任务ID (从1开始)
            total_tasks: 总任务数
            task_description: 任务描述（经过refine的详细描述）
            modeling_solution: 用户建模方案（整体算法设计）
            coordinator: 协调器（管理任务依赖和结果）
            config: 配置（包含work_dir, template_dir等）
            
        Returns:
            {
                'task_code': 完整的solver代码,
                'is_pass': 验证是否通过,
                'execution_result': 执行结果,
                'task_analysis': 任务分析,
                'algorithm_design': 算法设计
            }
        """
        print(f"\n{'='*80}")
        print(f"Task {task_id}/{total_tasks}")
        print(f"{'='*80}")
        
        is_final = (task_id == total_tasks)
        
        # 一步完成：分析、设计、生成（避免幻觉）
        print(f"\n[Generating Complete Solver] Task {task_id}/{total_tasks}")
        code, is_pass, execution_result, validation_result = self._generate_complete_solver(
            task_id, total_tasks, is_final,
            task_description, modeling_solution, coordinator, config
        )
        
        # 保存本任务实现，供后续任务继承
        self.task_history[task_id] = {
            'description': task_description,
            'code': code
        }
        
        return {
            'task_code': code,
            'is_pass': is_pass,
            'execution_result': execution_result,
            'validation_result': validation_result
        }
    
    # ============================================================================
    # 核心方法：一步生成（避免幻觉）
    # ============================================================================
    
    def _generate_complete_solver(
        self,
        task_id: int,
        total_tasks: int,
        is_final: bool,
        task_description: str,
        modeling_solution: str,
        coordinator,
        config: Dict
    ) -> Tuple[str, bool, str, Dict]:
        """
        一步生成完整的solver函数（渐进式）
        
        将分析、设计、实现合并为一步，避免中间步骤的幻觉。
        
        核心逻辑：
        - 复制前面任务(1..task_id-1)的精确实现
        - 精确实现本任务(task_id)的算法
        - 未来任务(task_id+1..total_tasks)用默认占位
        """
        work_dir = config.get('work_dir', './output/code')
        script_name = f'solver_task{task_id}.py'
        
        # 构建一步到位的提示词（包含分析、设计、实现）
        prompt = self._build_progressive_prompt(
            task_id, total_tasks, is_final,
            task_description, modeling_solution, 
            coordinator, config
        )
        
        # 生成代码
        print(f"  Generating code for Task {task_id}...")
        code = self._generate_code_from_prompt(prompt)
        
        # 先进行快速语法验证（不调用LLM修复，只检查）
        print(f"  Quick syntax validation...")
        # 获取代码模板用于验证（如果在prompt构建时已经获取，这里可以复用）
        code_template = self._get_code_template(task_id, config)
        code, has_syntax_error, syntax_issues = self.code_validator.validate_and_fix(
            code, code_template, {}, task_description, max_iterations=0  # 只检查不修复
        )
        
        if has_syntax_error:
            print(f"  [!] Syntax errors found, skipping execution")
            is_pass = False
            execution_result = f"Syntax validation failed: {syntax_issues}"
            validation_result = {'is_valid': False, 'issues': syntax_issues}
            return code, is_pass, execution_result, validation_result
        
        # 执行验证（使用test_llm_solver）- 可能会发现运行时错误
        print(f"  Testing solver with test_llm_solver...")
        code, is_pass, execution_result = self._test_complete_solver(
            code, script_name, work_dir, task_id, total_tasks
        )
        
        # 如果执行失败，用执行错误信息让LLM修复代码
        if not is_pass:
            print(f"  [!] Execution failed, attempting to fix with error feedback...")
            code, is_pass, execution_result = self._fix_with_execution_feedback(
                code, task_description, execution_result,
                script_name, work_dir, task_id, total_tasks,
                coordinator, config,
                max_fix_iterations=2
            )
        
        validation_result = {'is_valid': is_pass, 'execution_result': execution_result}
        return code, is_pass, execution_result, validation_result
    
    def _fix_with_execution_feedback(
        self,
        original_code: str,
        task_description: str,
        execution_error: str,
        script_name: str,
        work_dir: str,
        task_id: int,
        total_tasks: int,
        coordinator,
        config: Dict,
        max_fix_iterations: int = 2
    ) -> Tuple[str, bool, str]:
        """使用执行错误反馈让LLM修复代码"""
        
        current_code = original_code
        
        for iteration in range(max_fix_iterations):
            print(f"    Fix iteration {iteration + 1}/{max_fix_iterations}")
            
            # 获取代码参考（提供helper functions参考）
            code_reference = self._get_code_reference(task_id, config)
            
            # 构建修复提示（叙事风格）
            fix_prompt = f"""The code you generated encountered an error during execution. Let's fix it together.

# What Happened

Your `llm_solver` function for Task {task_id}/{total_tasks} was tested but failed with an execution error. This is attempt {iteration + 1} of {max_fix_iterations} to fix the issue.

# The Task You Were Solving

{task_description}

# Your Code (That Failed)

```python
{current_code}
```

# The Error Message

The test framework reported the following error:

```
{execution_error}
```

# Available Helper Functions (For Reference)

Remember, these helper functions are available and pre-imported in the test environment:

{code_reference['code']}

# How to Fix It

Please analyze the error and fix your code:

1. **Read the error carefully**: What is the error type? Which line failed? What does the error message tell you?

2. **Identify the root cause**: Common issues include:
   - Variable not defined or incorrectly named
   - Wrong function signature or incorrect arguments when calling helper functions
   - Logic errors (indexing, data structure mismatch, etc.)
   - Missing return statement or wrong return format
   - Type errors (list vs dict, etc.)

3. **Fix the issue**: Modify your code to address the root cause. Make sure:
   - All helper functions are called with correct arguments
   - Variable names are consistent
   - Data structures match what's expected
   - Return format is: `(ina_placement_expanded, jobs_routing_expanded)` (只返回两个值)

4. **Keep what works**: Don't change parts of the code that aren't related to the error.

# What to Generate

Generate the COMPLETE FIXED `llm_solver` function:

```python
def llm_solver(instance, network, K, jobs_num, Cs, topo_name):
    # Your fixed implementation here
    # ...
    return ina_placement_expanded, jobs_routing_expanded
```

Now, generate the corrected `llm_solver` function:"""
            
            # 让LLM修复代码
            try:
                fixed_code = self._generate_code_from_prompt(fix_prompt)
                
                # 测试修复后的代码
                fixed_code, is_pass, execution_result = self._test_complete_solver(
                    fixed_code, script_name, work_dir, task_id, total_tasks
                )
                
                if is_pass:
                    print(f"    ✓ Fix successful on iteration {iteration + 1}")
                    return fixed_code, is_pass, execution_result
                else:
                    print(f"    ✗ Fix failed on iteration {iteration + 1}, trying again...")
                    current_code = fixed_code
                    execution_error = execution_result
                    
            except Exception as e:
                print(f"    [!] Fix iteration {iteration + 1} failed with exception: {e}")
                execution_error = f"Fix attempt error: {str(e)}"
        
        print(f"    [!] All fix attempts exhausted")
        return current_code, False, execution_error
    
    def _build_progressive_prompt(
        self,
        task_id: int,
        total_tasks: int,
        is_final: bool,
        task_description: str,
        modeling_solution: str,
        coordinator,
        config: Dict
    ) -> str:
        """
        构建渐进式代码生成提示词（叙事风格）
        
        提示词像讲故事：
        1. 介绍整体任务背景
        2. 说明依赖关系（如果有）
        3. 提供代码基础（template或history）和工具函数
        4. 说明当前任务目标
        5. 指导如何实现
        6. 明确输出格式
        """
        
        # 1. 获取依赖信息
        dependent_info = self._get_dependency_info(coordinator, task_id)
        
        # 2. 获取代码参考（template+tools 或 history+tools）
        code_reference = self._get_code_reference(task_id, config)
        
        # 3. 判断是否为第一个任务
        is_first_task = (task_id == 1)
        
        # ========================================================================
        # 构建叙事式提示词
        # ========================================================================
        
        prompt = f"""You are solving a complex optimization problem through progressive task decomposition. Here's the context and your mission:

# The Big Picture

We are tackling the following problem:

{modeling_solution}

This problem has been decomposed into {total_tasks} sequential tasks. You are now working on Task {task_id}/{total_tasks}.

"""

        # 2. 依赖关系说明
        if dependent_info:
            prompt += f"""# What Has Been Done Before

{dependent_info}

These completed tasks provide the foundation for your work. Their implementations are already integrated into the code you'll see below.

"""

        # 3. 代码基础说明
        prompt += code_reference['introduction']
        prompt += "\n\n"
        prompt += code_reference['code']
        prompt += "\n\n"
        
        # 4. 当前任务说明
        prompt += f"""# Your Mission: Task {task_id}

{task_description}

"""

        # 5. 实现指导
        if is_first_task:
            prompt += """# How to Implement

Since this is the first task, you should:

1. **Start from the template**: Use the `llm_solver` function signature shown above as your starting point.

2. **Implement the core algorithm**: Focus on implementing the key functionality required by Task 1. This is your primary responsibility.

3. **Handle future tasks**: For tasks 2-{}, use simple placeholder implementations (e.g., random selection, greedy baseline). Add comments like `# TODO: Will be optimized in Task X`.

4. **Use the helper functions**: The tool functions shown above (`get_ina_candidates`, `evaluate_completion_time`, etc.) are available in the test environment. Call them directly - do NOT redefine them.

5. **Code organization**: Structure your code in clearly separated sections for each task phase. This will make it easier for future tasks to identify and modify specific parts without affecting others.

6. **Return the required format**: Your function must return a tuple: `(ina_placement_expanded, jobs_routing_expanded)`.

""".format(total_tasks)
        elif is_final:
            prompt += """# How to Implement

This is the final task - time to integrate and optimize!

1. **Build on existing code**: The code shown above already includes all previous tasks (Tasks 1-{}). **DO NOT rewrite or duplicate existing logic** - the previous implementations are tested and working.

2. **Code modification principles**:
   - **Reuse, don't reimplement**: If a function or logic already exists in the previous code, USE IT directly
   - **Extend, don't replace**: Add new functionality on top of existing code, don't start from scratch
   - **Preserve working parts**: Keep all previous task implementations intact unless you have a specific reason to modify them

3. **Your focus for this final task**:
   - Review how all components work together
   - Identify specific optimization opportunities (e.g., better search strategy, refined heuristics)
   - Implement ONLY the final optimization/integration logic
   - Ensure compatibility between all parts

4. **Use the helper functions**: The tool functions shown above are available. Call them as needed.

5. **Maintain compatibility**: Keep the same function signature and return format. Don't break what's already working.

6. **Return the required format**: Your function must return: `(ina_placement_expanded, jobs_routing_expanded)` (只返回两个值，不返回makespan).

**Warning**: Avoid duplicating code or reimplementing features that already exist. This causes bugs and makes the code harder to maintain.

""".format(task_id - 1)
        else:
            prompt += """# How to Implement

You are building upon previous work and preparing for future tasks.

1. **Build on existing code**: The code shown above includes Tasks 1-{}. **DO NOT rewrite or duplicate existing logic** - treat it as your foundation.

2. **Code modification principles**:
   - **Identify what to change**: Locate the specific section that needs modification for Task {}
   - **Reuse existing components**: If Tasks 1-{} already implement helper logic, variable initialization, or data structures, REUSE them directly
   - **Avoid duplication**: Don't create new variables or functions with similar names (e.g., if `candidates` exists, don't create `candidates_new`)
   - **Incremental modification**: Only modify/add the parts directly related to Task {}

3. **Your specific focus for Task {}**:
   - Implement the specific improvement or functionality required by this task
   - Build upon the data structures and intermediate results from previous tasks
   - Maintain consistency with the existing code style and variable naming

4. **Preserve previous work**: Keep the implementations from Tasks 1-{} intact. They've been tested and work correctly.

5. **Prepare for future tasks**: For tasks {}-{}, use simple placeholders. Add comments like `# TODO: Will be optimized in Task X`.

6. **Use the helper functions**: The tool functions shown above are available. Call them directly.

7. **Return the required format**: Your function must return: `(ina_placement_expanded, jobs_routing_expanded)` (只返回两个值，不返回makespan).

**Warning**: Avoid duplicating functionality that already exists in previous tasks. This causes inconsistencies and bugs. Think of your job as *extending* the code, not *rewriting* it.

""".format(task_id - 1, task_id, task_id - 1, task_id, task_id, task_id - 1, task_id + 1, total_tasks)

        # 6. 输出格式说明
        prompt += """# What to Generate

Generate ONLY the complete `llm_solver` function. 

**Do NOT include:**
- Helper function definitions (they already exist in the test environment)
- Import statements (they will be added automatically)
- Any code outside the `llm_solver` function

**Your output should be:**

```python
def llm_solver(instance, network, K, jobs_num, Cs, topo_name):
    # Your complete implementation here
    # ...
    return ina_placement_expanded, jobs_routing_expanded
```

Now, generate the `llm_solver` function:"""

        return prompt
    
    def _format_previous_tasks(self) -> str:
        """
        格式化前面任务的实现（简洁版）
        
        只返回上一个任务的完整 llm_solver 代码
        """
        if not self.task_history:
            return "(No previous tasks - starting from template)"
        
        # 只获取最后一个任务（上一个任务）的代码
        last_task_id = max(self.task_history.keys())
        task = self.task_history[last_task_id]
        
        # 提取纯 llm_solver 函数（去掉导入语句）
        code_lines = task['code'].split('\n')
        
        # 找到 llm_solver 函数的开始
        solver_start = -1
        for i, line in enumerate(code_lines):
            if line.strip().startswith('def llm_solver('):
                solver_start = i
                break
        
        if solver_start >= 0:
            # 提取 llm_solver 函数
            pure_solver = '\n'.join(code_lines[solver_start:])
            
            return f"""Previous implementation from Task {last_task_id}:

```python
{pure_solver}
```

**Note:** This code already includes the implementation of tasks 1-{last_task_id}. Build upon it."""
        else:
            # 如果找不到函数定义，返回全部代码
            return f"""Previous implementation from Task {last_task_id}:

```python
{task['code']}
```

**Note:** This code already includes the implementation of tasks 1-{last_task_id}. Build upon it."""
    
    def _get_copy_hints(self) -> str:
        """获取复制提示"""
        if not self.task_history:
            return "# (No previous tasks to copy)"
        
        hints = []
        for tid in sorted(self.task_history.keys()):
            task = self.task_history[tid]
            desc_short = task['description'][:50].replace('\n', ' ')
            hints.append(f"# Task {tid}: {desc_short}... [copy exact implementation]")
        
        return "\n    ".join(hints)
    
    def _get_dependency_info(self, coordinator, task_id: int) -> str:
        """
        获取任务依赖信息（简洁版）
        
        只返回依赖任务的关键信息：描述和状态
        """
        task_dependency = [int(i) for i in coordinator.DAG.get(str(task_id), [])]
        
        if len(task_dependency) == 0:
            return ""
        
        parts = []
        parts.append(f"Task {task_id} depends on: {task_dependency}\n")
        
        # 添加依赖分析（如果存在）
        if coordinator.task_dependency_analysis and task_id - 1 < len(coordinator.task_dependency_analysis):
            parts.append("**Dependency Analysis:**")
            parts.append(coordinator.task_dependency_analysis[task_id - 1])
            parts.append("")
        
        # 列出依赖任务的状态
        parts.append("**Completed Tasks:**")
        for dep_id in task_dependency:
            dep_id_str = str(dep_id)
            if dep_id_str in coordinator.memory:
                task_info = coordinator.memory[dep_id_str]
                desc = task_info.get('task_description', 'N/A')[:100]
                status = '✓' if task_info.get('is_pass', False) else '✗'
                parts.append(f"- Task {dep_id} [{status}]: {desc}...")
        
        return "\n".join(parts)
    
    def _get_code_reference(self, task_id: int, config: Dict) -> Dict[str, str]:
        """
        获取代码参考（template或history）和工具函数
        
        返回：
        - introduction: 对代码的说明（自然语言）
        - code: 实际的代码内容
        
        第一个任务：返回 template + tools
        后续任务：返回 history (上一个任务的代码) + tools
        """
        template_dir = config.get('template_dir', 'MMAgent/code_template')
        
        # 加载工具函数
        try:
            helper_file = os.path.join(template_dir, 'llmina_helper_functions.py')
            with open(helper_file, 'r', encoding='utf-8') as f:
                helper_functions = f.read()
        except Exception as e:
            print(f"  [Warning] Failed to load helper functions: {e}")
            helper_functions = "# Helper functions not available"
        
        if task_id == 1:
            # 第一个任务：提供模板
            try:
                template_file = os.path.join(template_dir, 'llmina_solver_template.py')
                with open(template_file, 'r', encoding='utf-8') as f:
                    solver_template = f.read()
                print(f"  Template loaded: llmina_solver_template.py + llmina_helper_functions.py")
            except Exception as e:
                print(f"  [Warning] Failed to load template: {e}")
                solver_template = "def llm_solver(instance, network, K, jobs_num, Cs, topo_name):\n    pass"
            
            return {
                'introduction': """# The Code Foundation

You have two resources to help you:

## 1. Reference Template

Below is a reference `llm_solver` function template that shows the expected structure and signature:

```python
{}
```

This template provides a starting point. You should implement the actual algorithm logic.

## 2. Helper Functions (Tools)

The following helper functions are available in the test environment. You can call them directly in your `llm_solver` function:

""".format(solver_template),
                'code': f"```python\n{helper_functions}\n```\n\n**Important**: These helper functions are pre-imported in the test harness. Do NOT redefine them - just call them as needed (e.g., `candidates = get_ina_candidates(...)`, `makespan = evaluate_completion_time(...)`)."
            }
        else:
            # 后续任务：提供上一个任务的代码
            if not self.task_history:
                # 异常情况：应该有历史但没有
                return {
                    'introduction': "# The Code Foundation\n\nNo previous implementation found (this shouldn't happen).",
                    'code': f"```python\n{helper_functions}\n```"
                }
            
            # 获取上一个任务的代码
            last_task_id = max(self.task_history.keys())
            task = self.task_history[last_task_id]
            
            # 提取 llm_solver 函数
            code_lines = task['code'].split('\n')
            solver_start = -1
            for i, line in enumerate(code_lines):
                if line.strip().startswith('def llm_solver('):
                    solver_start = i
                    break
            
            if solver_start >= 0:
                previous_solver = '\n'.join(code_lines[solver_start:])
            else:
                previous_solver = task['code']
            
            print(f"  Using Task {last_task_id}'s implementation + helper functions")
            
            return {
                'introduction': f"""# The Code Foundation

You have two resources to work with:

## 1. Previous Implementation (from Task {last_task_id})

The code below shows the `llm_solver` function from Task {last_task_id}. This implementation already includes the work done in all previous tasks (Tasks 1-{last_task_id}):

```python
{previous_solver}
```

**What this code does**: This implementation has been tested and works correctly for Tasks 1-{last_task_id}. You should use it as your starting point and build upon it.

## 2. Helper Functions (Tools)

The following helper functions are available in the test environment. You can call them directly:

""",
                'code': f"```python\n{helper_functions}\n```\n\n**Important**: These helper functions are pre-imported. Do NOT redefine them - just call them (e.g., `candidates = get_ina_candidates(...)`, `makespan = evaluate_completion_time(...)`)."
            }
    
    def _get_code_template(self, task_id: int, config: Dict) -> str:
        """
        获取代码模板
        
        第一个任务：llmina_solver_template.py + llmina_helper_functions.py
        后续任务：llmina_helper_functions.py（上一个任务的代码在task_history中）
        """
        template_dir = config.get('template_dir', 'MMAgent/code_template')
        
        try:
            # 始终加载辅助函数
            helper_file = os.path.join(template_dir, 'llmina_helper_functions.py')
            with open(helper_file, 'r', encoding='utf-8') as f:
                helper_functions = f.read()
            
            if task_id == 1:
                # 第一个任务：使用完整模板
                template_file = os.path.join(template_dir, 'llmina_solver_template.py')
                with open(template_file, 'r', encoding='utf-8') as f:
                    solver_template = f.read()
                
                # 拼接：工具函数 + 求解器模板
                template_content = f"{helper_functions}\n\n{solver_template}"
                print(f"  Template loaded: llmina_solver_template.py + llmina_helper_functions.py")
            else:
                # 后续任务：使用上一个任务的代码 + 辅助函数
                # 注意：上一个任务的代码会在 _format_previous_tasks() 中提供
                # 这里只返回辅助函数作为参考
                template_content = helper_functions
                print(f"  Template loaded: llmina_helper_functions.py (previous task's llm_solver will be used)")
            
            return template_content
            
        except Exception as e:
            print(f"  [Warning] Failed to load template: {e}")
            return """# Code template not available
# Please implement the solution based on task description and dependencies
"""
    
    def _wrap_solver_with_imports(self, llm_solver_code: str) -> str:
        """将生成的llm_solver函数包装上辅助函数的导入"""
        
        # 导入语句 - 从evaluation模块导入所有辅助函数
        imports = """# Auto-generated solver with helper function imports
import sys
from pathlib import Path
import copy
ss
# Import helper functions from evaluation module
try:
    from MMAgent.evaluation import (
        get_ina_candidates,
        evaluate_completion_time,
        links_set,
        path_to_links_dict,
        l_i,
        link_capacity_limitation
    )
except ModuleNotFoundError:
    # Fallback for different import contexts
    from evaluation import (
        get_ina_candidates,
        evaluate_completion_time,
        links_set,
        path_to_links_dict,
        l_i,
        link_capacity_limitation
    )

"""
        
        # 组合：导入 + LLM生成的solver函数
        complete_code = imports + "\n" + llm_solver_code
        return complete_code
    
    def _test_complete_solver(
        self, 
        code: str, 
        script_name: str, 
        work_dir: str,
        task_id: int,
        total_tasks: int
    ) -> Tuple[str, bool, str]:
        """使用test_llm_solver测试完整solver - 直接调用evaluation函数"""
        
        os.makedirs(work_dir, exist_ok=True)
        
        # 将生成的llm_solver与辅助函数导入组合
        complete_code = self._wrap_solver_with_imports(code)
        
        # 保存完整代码
        solver_path = os.path.join(work_dir, script_name)
        with open(solver_path, 'w') as f:
            f.write(complete_code)
        
        print(f"  Solver saved to: {solver_path}")
        
        try:
            # 导入evaluation模块的test_llm_solver
            # 尝试绝对导入，如果失败则使用相对导入
            try:
                from MMAgent.evaluation import test_llm_solver
            except ModuleNotFoundError:
                # 如果从main_llmina.py启动，使用相对导入
                from evaluation import test_llm_solver
            
            # 动态导入生成的solver函数
            import sys
            import importlib.util
            
            # 添加work_dir到sys.path以便导入solver
            if work_dir not in sys.path:
                sys.path.insert(0, work_dir)
            
            # 动态加载solver模块
            module_name = script_name.replace('.py', '')
            spec = importlib.util.spec_from_file_location(module_name, solver_path)
            solver_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = solver_module
            spec.loader.exec_module(solver_module)
            
            # 获取llm_solver函数
            llm_solver = solver_module.llm_solver
            
            print("="*80)
            print(f"Testing Task {task_id}/{total_tasks} Solver")
            print("="*80)
            
            # 调用test_llm_solver进行测试
            results = test_llm_solver(
                solver_func=llm_solver,
                topo_name='FatTree',
                ina_num_list=[3],
                jobs_num_list=[6],
                instances_num=1  # Quick validation
            )
            
            # 打印结果
            print("\n" + "="*80)
            print("Test Results:")
            observation_lines = [f"Testing Task {task_id}/{total_tasks} Solver"]
            
            for key, jct_list in results.items():
                avg_jct = sum(jct_list) / len(jct_list) if jct_list else float('inf')
                result_line = f"  INA={key[0]}, Jobs={key[1]}: Avg Makespan={avg_jct:.4f}"
                print(result_line)
                observation_lines.append(result_line)
            
            print("="*80)
            
            # 检查是否全部通过
            all_valid = all(
                all(jct < float('inf') for jct in jct_list) 
                for jct_list in results.values()
            )
            
            if all_valid:
                success_msg = f"✓ Task {task_id}/{total_tasks} PASSED"
                print(f"\n{success_msg}")
                observation_lines.append(success_msg)
                is_pass = True
            else:
                fail_msg = f"✗ Task {task_id}/{total_tasks} FAILED - some results are invalid"
                print(f"\n{fail_msg}")
                observation_lines.append(fail_msg)
                is_pass = False
            
            observation = "\n".join(observation_lines)
            
            # 清理sys.modules避免缓存问题
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            return complete_code, is_pass, observation
            
        except Exception as e:
            import traceback
            error_msg = f"✗ Test FAILED with error:\n  {type(e).__name__}: {e}\n{traceback.format_exc()}"
            print(error_msg)
            return complete_code, False, error_msg
    
    def _generate_code_from_prompt(self, prompt: str) -> str:
        """从提示生成代码（只需要llm_solver函数）"""
        max_retry = 5
        for _ in range(max_retry):
            try:
                completion = self.llm.generate(prompt)
                
                # 尝试提取代码
                if "```python" in completion:
                    code = completion.split("```python")[1].split("```")[0].strip()
                elif "```" in completion:
                    code = completion.split("```")[1].split("```")[0].strip()
                else:
                    code = completion.strip()
                
                # 如果代码只包含llm_solver函数定义，说明提取正确
                # 如果包含了完整的代码模板，尝试提取llm_solver函数
                if "def llm_solver" in code:
                    # 检查是否包含了太多内容（例如helper函数）
                    # 如果代码行数很多，可能包含了整个模板，需要提取llm_solver
                    lines = code.split('\n')
                    if len(lines) > 200:  # 如果超过200行，可能包含了完整模板
                        # 尝试只提取llm_solver函数
                        import re
                        # 找到llm_solver函数的开始
                        pattern = r'(def llm_solver\([^)]*\):.*?)(?=\n(?:def |class |if __name__|$))'
                        match = re.search(pattern, code, re.DOTALL)
                        if match:
                            code = match.group(1).strip()
                            print("  [INFO] Extracted only llm_solver function from generated code")
                
                return code
                
            except Exception as e:
                print(f"  [!] Code extraction failed: {e}, retrying...")
                continue
        raise Exception("Failed to generate code after maximum retries")
