"""
MMAgent - 基于 LangGraph 的多智能体优化求解器系统
Multi-agent system for optimization problem solving using LangGraph
"""

__version__ = '1.0.0'
__author__ = 'LLM-MM-Agent Team'

# 导出核心组件
from .problem.LLMINA.runtime import evaluate_solver

__all__ = [
    'evaluate_solver',
]
