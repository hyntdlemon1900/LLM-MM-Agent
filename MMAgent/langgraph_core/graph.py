"""
简化的图构建模块
直接使用 LangGraph 内置功能，移除不必要的封装
"""

from typing import Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
import os

from .state import AgentState
from .nodes import (
    load_problem_node,
    heuristic_architect_node,
    function_code_generator_node,
    code_integration_node,
    heuristic_evaluation_node,
    code_fix_node,
    constraint_analyzer_node
)

def build_workflow(
    enable_checkpoints: bool = False,
    checkpoint_path: Optional[str] = None,
):
    """
    构建简化的 LLMINA 工作流
    
    直接使用 LangGraph API，无需中间封装层
    
    Args:
        enable_checkpoints: 是否启用检查点功能
        checkpoint_path: 检查点保存路径（SQLite文件）
    """

    # 创建图
    workflow = StateGraph(AgentState)


    
    # 添加节点 - 直接使用 LangGraph API
    workflow.add_node("load_problem", load_problem_node)
    workflow.add_node("heuristic_architect", heuristic_architect_node)
    workflow.add_node("generate_function", function_code_generator_node)
    workflow.add_node("integrate", code_integration_node)    
    workflow.add_node("evaluate", heuristic_evaluation_node)
    workflow.add_node("fix", code_fix_node)
    workflow.add_node("constraint_analyze", constraint_analyzer_node)
    
    # 添加边 - 直接使用 LangGraph API
    workflow.add_edge(START, "load_problem")
    workflow.add_edge("load_problem", "heuristic_architect")
    
    # 条件边 - 函数代码生成循环
    def should_continue_generating(state: AgentState) -> str:
        """判断是否继续生成函数代码"""
        next_action = state.get('next_action', 'integrate')
        if next_action == 'generate_next_function':
            return 'generate_function'
        else:
            return 'integrate'
    
    workflow.add_conditional_edges(
        "heuristic_architect",
        should_continue_generating,
        {
            'generate_function': 'generate_function',  # 开始生成函数
            'integrate': 'integrate'                   # 跳过函数生成（如果没有函数）
        }
    )
    
    workflow.add_conditional_edges(
        "generate_function",
        should_continue_generating,
        {
            'generate_function': 'generate_function',  # 继续下一个函数
            'integrate': 'integrate'                   # 完成所有函数，进入集成
        }
    )
    
    workflow.add_edge("integrate", "evaluate")
    '''


    workflow.add_node("load_problem", load_problem_node)
    workflow.add_node("evaluate", heuristic_evaluation_node)
    workflow.add_node("fix", code_fix_node)
    workflow.add_node("constraint_analyze", constraint_analyzer_node)

    workflow.add_edge(START, "load_problem")
    workflow.add_edge("load_problem","evaluate")
    '''


    # 条件边 - 评估后的修复循环
    def should_fix_code(state: AgentState) -> str:
        """判断是否需要修复代码"""
        evaluation_results = state.get('evaluation_results', {})
        evaluation_success = evaluation_results.get('evaluation_success', False)
        fix_attempt_count = state.get('fix_attempt_count', 0)
        max_fix_attempts = state.get('max_fix_attempts', 3)
        
        # 如果评估成功，直接结束
        if evaluation_success:
            # 检查是否所有配置都成功
            perf_summary = evaluation_results.get('performance_summary', {})
            all_success = all(
                metrics.get('success_count', 0) > 0 
                for metrics in perf_summary.values()
            )
            if all_success:
                return 'constraint_analyze'
            else:
                # 有部分配置失败，但未达到最大修复次数
                if fix_attempt_count < max_fix_attempts:
                    return 'fix'
                else:
                    return 'constraint_analyze'
        else:
            # 评估失败，需要修复（如果未达到最大次数）
            if fix_attempt_count < max_fix_attempts:
                return 'fix'
            else:
                return 'constraint_analyze'
    
    workflow.add_conditional_edges(
        "evaluate",
        should_fix_code,
        {
            'fix': 'fix',      # 需要修复代码
            'constraint_analyze': 'constraint_analyze'         # 评估成功或达到最大修复次数
        }
    )
    
    # 修复后重新评估
    workflow.add_edge("fix", "evaluate")

    # 编译 - 可选启用 checkpoint
    if enable_checkpoints:
        if checkpoint_path is None:
            checkpoint_path = "./checkpoints/llmina.db"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # 使用正确的 SQLite URI 格式
        checkpointer = SqliteSaver.from_conn_string(f"file:{checkpoint_path}")
        return workflow.compile(checkpointer=checkpointer)
    else:
        # 不使用检查点时也要返回编译后的图
        return workflow.compile()
