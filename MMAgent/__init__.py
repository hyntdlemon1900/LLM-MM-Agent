"""
MMAgent - 基于 LangGraph 的多智能体优化求解器系统
Multi-agent system for optimization problem solving using LangGraph
"""

__version__ = '1.0.0'
__author__ = 'LLM-MM-Agent Team'

# 导出核心组件
from .llm import LLM
from .utils import write_json_file, load_config
from .langgraph_core import (
    AgentState,
    create_agent_state,
    build_workflow,
)

__all__ = [
    'LLM',
    'write_json_file',
    'load_config',
    'AgentState',
    'create_agent_state',
    'build_workflow',
]
