from typing import List
from MMAgent.prompt.prompt_template import TASK_DECOMPOSE_PROMPT, TASK_DESCRIPTION_PROMPT


class ProblemDecompose:
    """
    任务分解智能体
    
    负责将高层次的算法求解方案分解为具体的可实施子任务，
    并对每个子任务进行细化描述
    """
    def __init__(self, llm):
        self.llm = llm

    def decompose(self, modeling_problem: str, modeling_solution: str):
        """
        将算法求解方案分解为一系列子任务
        
        Args:
            modeling_problem: 问题描述
            modeling_solution: 算法求解方案（来自 ProblemSolving 节点）
        
        Returns:
            List[str]: 子任务描述列表
        """
        prompt = TASK_DECOMPOSE_PROMPT.format(
            modeling_problem=modeling_problem,
            modeling_solution=modeling_solution
        )
        answer = self.llm.generate(prompt)
        tasks = [task.strip() for task in answer.split('---') if task.strip()]
        return tasks

    def refine(self, modeling_problem: str, modeling_solution: str, 
               decomposed_subtasks: List[str], task_i: int):
        """
        细化指定子任务的描述
        
        Args:
            modeling_problem: 问题描述
            modeling_solution: 算法求解方案
            decomposed_subtasks: 所有子任务列表
            task_i: 要细化的子任务索引（从0开始）
        
        Returns:
            str: 细化后的子任务描述
        """
        decomposed_subtasks_str = '\n---\n'.join(
            [f"Subtask {i+1}:\n{task}" 
             for i, task in enumerate(decomposed_subtasks)]
        )
        
        prompt = TASK_DESCRIPTION_PROMPT.format(
            modeling_problem=modeling_problem,
            modeling_solution=modeling_solution,
            decomposed_subtasks=decomposed_subtasks_str,
            task_i=task_i + 1  # 显示为从1开始的任务编号
        )
        answer = self.llm.generate(prompt)
        return answer

    def decompose_and_refine(self, modeling_problem: str, modeling_solution: str):
        """
        完整的分解和细化流程
        
        1. 将算法方案分解为子任务
        2. 逐个细化每个子任务的描述
        
        Args:
            modeling_problem: 问题描述
            modeling_solution: 算法求解方案
        
        Returns:
            Tuple[List[str], int]: (细化后的子任务列表, 子任务数量)
        """
        # 第一步：分解为子任务
        print('Decomposing solution plan into subtasks...')
        decomposed_subtasks = self.decompose(
            modeling_problem, modeling_solution
        )
        decomposed_subtasks = [t for t in decomposed_subtasks if t.strip()]
        
        # 第二步：细化每个子任务
        print(f'Refining {len(decomposed_subtasks)} subtasks...')
        for task_i in range(len(decomposed_subtasks)):
            print(f'  Refining Subtask {task_i+1}/{len(decomposed_subtasks)}')
            refined_subtask = self.refine(
                modeling_problem, 
                modeling_solution, 
                decomposed_subtasks, 
                task_i
            )
            decomposed_subtasks[task_i] = refined_subtask
        
        tasknum = len(decomposed_subtasks)
        return decomposed_subtasks, tasknum
