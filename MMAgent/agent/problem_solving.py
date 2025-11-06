"""
LLMINA 算法方案设计 Agent
生成高层次的算法求解方案（非数学建模）

注意：
- 这不是在做数学建模（定义变量、约束、目标函数）
- 而是设计"如何用算法求解问题"的策略方案
- 包括：核心方法论、算法步骤、关键策略、可行性分析等
"""
from prompt.llmina_template import PROBLEM_SOLVE_PROMPT, PROBLEM_SOLVE_CRITIQUE_PROMPT, PROBLEM_SOLVE_IMPROVEMENT_PROMPT


class ProblemSolving:
    """
    算法方案设计智能体
    
    通过 Actor-Critic-Improver 模式生成和优化算法求解方案：
    1. Actor: 生成初始方案
    2. Critic: 批判性分析（正确性、完整性、可行性等）
    3. Improver: 根据批评改进方案
    """
    def __init__(self, llm):
        self.llm = llm

    def solving(self, problem_str: str, round: int = 2):
        """
        生成问题的高层次算法求解方案
        
        通过多轮 Actor-Critic-Improver 迭代优化方案质量
        
        Args:
            problem_str: 问题描述（已澄清）
            round: 迭代改进轮次（默认 2 轮）
        
        Returns:
            str: 最终的算法求解方案，包括：
                - 核心方法论（启发式、分解策略等）
                - 算法概览（分步骤描述）
                - 关键策略（离散决策、连续分配、约束满足等）
                - 可行性和优化性讨论
                - 可扩展性和鲁棒性考虑
        """
        # 生成初始求解方案
        print('Generating initial solution...')
        current_solution = self._solve_actor(problem_str)
        
        # 迭代改进
        for i in range(round):
            print(f'  Solution Refinement Round {i+1}/{round}')
            
            # 批评当前方案
            critique = self._solve_critic(problem_str, current_solution)
            
            # 根据批评改进方案
            current_solution = self._solve_improver(
                problem_str, current_solution, critique
            )
        
        return current_solution

    def _solve_actor(self, problem_str: str):
        """
        Actor: 生成初始的算法求解方案
        
        基于问题描述，设计高层次的算法策略，包括方法论、
        算法步骤、关键策略等
        
        Args:
            problem_str: 问题描述
        
        Returns:
            str: 初始算法求解方案
        """
        prompt = PROBLEM_SOLVE_PROMPT.format(
            modeling_problem=problem_str,
        ).strip()
        
        return self.llm.generate(prompt)

    def _solve_critic(self, problem_str: str, modeling_solution: str):
        """
        Critic: 批判性分析当前的算法方案
        
        从多个维度评估方案质量：
        - 正确性与完整性
        - 方法论适当性
        - 可行性
        - 约束处理
        - 优化质量
        - 可扩展性
        - 边界情况和鲁棒性
        - 清晰度和逻辑性
        
        Args:
            problem_str: 问题描述
            modeling_solution: 当前算法方案
        
        Returns:
            str: 批评意见（指出缺陷和不足）
        """
        prompt = PROBLEM_SOLVE_CRITIQUE_PROMPT.format(
            modeling_problem=problem_str,
            modeling_solution=modeling_solution
        ).strip()
        
        return self.llm.generate(prompt)

    def _solve_improver(self, problem_str: str, 
                        modeling_solution: str, critique: str):
        """
        Improver: 根据批评改进算法方案
        
        针对 Critic 提出的问题和不足，优化方案的：
        - 有效性
        - 可行性
        - 鲁棒性
        - 解决所有识别出的缺陷和局限
        
        Args:
            problem_str: 问题描述
            modeling_solution: 当前算法方案
            critique: 批评意见
        
        Returns:
            str: 改进后的算法方案
        """
        prompt = PROBLEM_SOLVE_IMPROVEMENT_PROMPT.format(
            modeling_problem=problem_str,
            modeling_solution=modeling_solution,
            modeling_solution_critique=critique
        ).strip()
        
        return self.llm.generate(prompt)
