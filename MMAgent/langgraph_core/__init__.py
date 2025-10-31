"""
LangGraph Core Module
基于 LangGraph 的多智能体系统核心模块
"""

from .state import AgentState, LLMINAState, StandardModelingState
from .edges import create_conditional_edge, should_continue
from .config import LangGraphConfig

__all__ = [
    'AgentState',
    'LLMINAState', 
    'StandardModelingState',
    'create_graph',
    'compile_graph',
    'create_node_registry',
    'create_conditional_edge',
    'should_continue',
    'LangGraphConfig'
]

__version__ = '1.0.0'
