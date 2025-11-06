from typing import List
from prompt.llmina_template import TASK_DECOMPOSE_PROMPT, TASK_DESCRIPTION_PROMPT
from utils.utils import read_json_file


class ProblemDecompose:
    def __init__(self, llm):
        self.llm = llm
        # LLMINA问题使用专门的分解原则，不依赖于通用的decompose_prompt.json
        self.llmina_decompose_principle = self._get_llmina_principle()

    def _get_llmina_principle(self):
        """
        LLMINA问题的灵活分解指导原则
        
        提供关键考虑点而非强制步骤，允许模型根据问题特征自主分解
        """
        return """
When decomposing a solution approach for deterministic optimization problems with fully specified mathematical formulations (where decision variables, objective functions, and constraints are explicitly provided), consider the following key characteristics and guidance principles:

**Problem Nature:**
This type of problem differs fundamentally from open-ended mathematical modeling problems. The mathematical model is already given—the core challenge is designing an efficient algorithm to solve it, not building the model itself. Focus on algorithm design and implementation rather than data exploration or hypothesis formulation.

**Key Considerations for Decomposition:**

1. **Algorithmic Thinking:** The decomposition should reflect algorithm design phases rather than data analysis stages. Consider how algorithms for combinatorial optimization or constraint satisfaction problems are typically structured.

2. **Constraint Criticality:** All constraints in the given formulation are hard requirements that must be strictly satisfied. Any subtask dealing with solution construction or modification must ensure feasibility. Consider when and how constraint validation should occur.

3. **Solution Quality vs. Feasibility:** There's a fundamental trade-off between quickly finding a feasible solution and finding a high-quality solution. Consider whether your decomposition separates these concerns or integrates them.

4. **Domain Knowledge Utilization:** The problem may have specific structural properties (e.g., network topology, resource distribution patterns) that can guide algorithm design. Consider whether subtasks should leverage such domain-specific insights.

5. **Modularity and Integration:** Algorithm components need to work together as a cohesive system. Consider how different subtasks will interface with each other and with external evaluation tools.

**Flexibility in Decomposition:**
You are NOT required to follow any fixed number of subtasks or predefined stages. Analyze the specific problem and solution approach, then decompose it in whatever way makes the most logical and practical sense. Some solutions may naturally divide into 2-3 major phases, others into 5-6 distinct components. Let the problem structure guide your decomposition.
"""

    def decompose(self, modeling_problem: str, modeling_solution: str):
        # 对于LLMINA问题，使用专用的分解原则
        decomposed_principle = self.llmina_decompose_principle
        prompt = TASK_DECOMPOSE_PROMPT.format(
            decomposed_principle=decomposed_principle,
            modeling_problem=modeling_problem,
            modeling_solution=modeling_solution
        )
        answer = self.llm.generate(prompt)
        tasks = [task.strip() for task in answer.split('---') if task.strip()]
        return tasks

    def refine(self, modeling_problem: str, problem_analysis: str, modeling_solution: str, decomposed_subtasks: List[str], task_i: int):
        decomposed_subtasks_str = '\n'.join(decomposed_subtasks)
        prompt = TASK_DESCRIPTION_PROMPT.format(
            modeling_problem=modeling_problem,
            problem_analysis=problem_analysis,
            modeling_solution=modeling_solution,
            decomposed_subtasks=decomposed_subtasks_str,
            task_i=task_i+1
        )
        answer = self.llm.generate(prompt)
        return answer

    def decompose_and_refine(self, modeling_problem: str, modeling_solution: str):
        decomposed_subtasks = self.decompose(
            modeling_problem, modeling_solution
        )
        decomposed_subtasks = [t for t in decomposed_subtasks if t.strip()]
        for task_i in range(len(decomposed_subtasks)):
            refined_subtask = self.refine(modeling_problem, '', modeling_solution, decomposed_subtasks, task_i)
            decomposed_subtasks[task_i] = refined_subtask
        tasknum = len(decomposed_subtasks)
        return decomposed_subtasks, tasknum
