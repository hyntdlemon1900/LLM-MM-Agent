"""
LangGraph Core Module
基于 LangGraph 的多智能体系统核心模块
"""

from .state import AgentState, create_agent_state
from .graph import build_workflow
from .nodes import (
    load_problem_node,
    generate_heuristic_template_node,
    heuristic_architect_node,
    function_code_generator_node,
    code_integration_node,
    heuristic_evaluation_node,
    algorithm_design_node,
    task_decompose_node,
    dependency_analysis_node,
    task_solving_node
)

__all__ = [
    # State
    'AgentState',
    'create_agent_state',
    # Workflow
    'build_workflow',
    # Nodes
    'load_problem_node',
    'generate_heuristic_template_node',
    'heuristic_architect_node',
    'function_code_generator_node',
    'code_integration_node',
    'heuristic_evaluation_node',
    'algorithm_design_node',
    'task_decompose_node',
    'dependency_analysis_node',
    'task_solving_node'
]

__version__ = '1.0.0'
