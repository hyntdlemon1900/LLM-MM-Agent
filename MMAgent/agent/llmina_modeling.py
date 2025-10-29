"""
LLMINA问题建模Agent
专门用于分析和理解LLMINA的MILP建模
"""
from .base_agent import BaseAgent


class LLMINAModeling(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm)

    def analyze_formulation(self, problem_str: str, problem_analysis: str, problem_formulation: dict):
        """
        分析LLMINA的MILP建模
        """
        # 构建问题建模的字符串表示
        formulation_str = self._format_formulation(problem_formulation)
        
        prompt = f"""You are an expert in optimization and mathematical modeling. 

You are given a problem description, problem analysis, and a Mixed-Integer Linear Programming (MILP) formulation for the LLMINA problem.

## Problem Description
{problem_str}

## Problem Analysis
{problem_analysis}

## MILP Formulation
{formulation_str}

Your task is to analyze this MILP formulation and provide:
1. **Objective Analysis**: Explain the objective function and why maximizing α (or equivalently minimizing makespan t) is appropriate for this problem.
2. **Decision Variables**: Identify and explain the key decision variables (x_s for placement, y_jwv for routing, gamma variables for rates).
3. **Constraint Analysis**: Analyze how each constraint type contributes to ensuring a valid and optimal solution.
4. **Problem Complexity**: Explain why this is an NP-hard problem and why decomposition is needed.
5. **Decomposition Opportunity**: Identify how the problem can be decomposed into placement and routing subproblems.

Provide a comprehensive analysis that will guide the algorithm design process.
"""
        
        return self.llm.generate(prompt.strip())

    def understand_constraints(self, problem_formulation: dict, variable_description: dict):
        """
        深入理解约束条件和变量含义
        """
        formulation_str = self._format_formulation(problem_formulation)
        variables_str = self._format_variables(variable_description)
        
        prompt = f"""You are an expert in optimization problems. 

## Variable Descriptions
{variables_str}

## Problem Formulation
{formulation_str}

Your task is to provide a detailed understanding of:

1. **Variable Relationships**: How do different variables (x_s, y_jwv, gamma_j, gamma_jw, gamma_jws, etc.) relate to each other?

2. **Constraint Groups**:
   - Which constraints control INA placement decisions (x_s)?
   - Which constraints control routing decisions (y_jwv)?
   - Which constraints link placement and routing?
   - Which constraints enforce resource capacity limits?

3. **Coupling Analysis**: Explain how placement decisions (x_s) and routing decisions (y_jwv) are coupled through the constraints.

4. **Optimization Trade-offs**: What are the key trade-offs in this optimization problem?

5. **Practical Implications**: What does each constraint mean in practical terms for the network and DML jobs?

Provide detailed explanations that will help design effective heuristics.
"""
        
        return self.llm.generate(prompt.strip())

    def _format_formulation(self, formulation: dict):
        """格式化MILP建模为字符串"""
        if not formulation:
            return "No formulation provided"
        
        output = []
        
        if 'model_type' in formulation:
            output.append(f"**Model Type**: {formulation['model_type']}")
            output.append("")
        
        if 'objective' in formulation:
            obj = formulation['objective']
            output.append("**Objective**:")
            output.append(f"  - Expression: {obj.get('expression', 'N/A')}")
            output.append(f"  - Meaning: {obj.get('meaning', 'N/A')}")
            output.append("")
        
        if 'key_constraints' in formulation:
            output.append("**Key Constraints**:")
            for i, constraint in enumerate(formulation['key_constraints'], 1):
                output.append(f"{i}. {constraint.get('name', 'Constraint')}")
                output.append(f"   - Expression: {constraint.get('expression', 'N/A')}")
                output.append(f"   - Meaning: {constraint.get('meaning', 'N/A')}")
                output.append("")
        
        return "\n".join(output)

    def _format_variables(self, variables: dict):
        """格式化变量描述为字符串"""
        if not variables:
            return "No variable descriptions provided"
        
        output = []
        for var_name, var_desc in variables.items():
            output.append(f"- **{var_name}**: {var_desc}")
        
        return "\n".join(output)
