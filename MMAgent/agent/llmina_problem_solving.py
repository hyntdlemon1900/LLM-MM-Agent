"""
LLMINA问题求解Agent
生成高层次的问题求解方案
"""
from .base_agent import BaseAgent
from prompt.llmina_template import PROBLEM_SOLVE_PROMPT, PROBLEM_SOLVE_CRITIQUE_PROMPT, PROBLEM_SOLVE_IMPROVEMENT_PROMPT


class ProblemSolving(BaseAgent):
    """
    问题求解智能体：生成高层次的算法求解方案
    """
    def __init__(self, llm):
        super().__init__(llm)

    def solving(self, problem_str: str, round: int = 2):
        """
        生成问题的高层次求解方案
        
        Args:
            problem_str: 问题描述
            problem_analysis: 问题分析结果
            round: 迭代改进轮次
        
        Returns:
            str: 最终的求解方案
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
        生成初始的求解方案
        
        Args:
            problem_str: 问题描述
            problem_analysis: 问题分析
        
        Returns:
            str: 初始求解方案
        """
        prompt = PROBLEM_SOLVE_PROMPT.format(
            modeling_problem=problem_str,
        ).strip()
        
        return self.llm.generate(prompt)

    def _solve_critic(self, problem_str: str, modeling_solution: str):
        """
        批评当前的求解方案
        
        Args:
            problem_str: 问题描述
            problem_analysis: 问题分析
            modeling_solution: 当前求解方案
        
        Returns:
            str: 批评意见
        """
        prompt = PROBLEM_SOLVE_CRITIQUE_PROMPT.format(
            modeling_problem=problem_str,
            modeling_solution=modeling_solution
        ).strip()
        
        return self.llm.generate(prompt)

    def _solve_improver(self, problem_str: str, 
                        modeling_solution: str, critique: str):
        """
        根据批评改进求解方案
        
        Args:
            problem_str: 问题描述
            problem_analysis: 问题分析
            modeling_solution: 当前求解方案
            critique: 批评意见
        
        Returns:
            str: 改进后的求解方案
        """
        prompt = PROBLEM_SOLVE_IMPROVEMENT_PROMPT.format(
            modeling_problem=problem_str,
            modeling_solution=modeling_solution,
            modeling_solution_critique=critique
        ).strip()
        
        return self.llm.generate(prompt)
