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
    generate_heuristic_template_node,
    heuristic_architect_node,
    function_code_generator_node,
    code_integration_node,
    heuristic_evaluation_node
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
        enable_evaluation: 是否启用求解器评估节点（默认启用）
    """

    
    # 创建图
    workflow = StateGraph(AgentState)




    '''
    workflow.add_node("evaluate", heuristic_evaluation_node)
    workflow.add_edge(START, "evaluate")
    workflow.add_edge("evaluate", END)
    '''

    # 添加节点 - 直接使用 LangGraph API
    workflow.add_node("load_problem", load_problem_node)
    workflow.add_node("generate_heuristic_template", generate_heuristic_template_node)
    workflow.add_node("heuristic_architect", heuristic_architect_node)
    workflow.add_node("generate_function", function_code_generator_node)
    workflow.add_node("integrate", code_integration_node)    
    workflow.add_node("evaluate", heuristic_evaluation_node)
    
    # 添加边 - 直接使用 LangGraph API
    workflow.add_edge(START, "load_problem")
    workflow.add_edge("load_problem", "generate_heuristic_template")
    workflow.add_edge("generate_heuristic_template", "heuristic_architect")
    
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
    workflow.add_edge("evaluate", END)
    
    


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
