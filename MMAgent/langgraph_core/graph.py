"""
简化的图构建模块
直接使用 LangGraph 内置功能，移除不必要的封装
"""

from typing import Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .nodes import (
    load_problem_node,
    problem_clarification_node,
    algorithm_design_node,
    task_decompose_node,
    dependency_analysis_node,
    task_solving_node,
    code_integration_node
)

def build_workflow(
    enable_checkpoints: bool = False,
    checkpoint_path: Optional[str] = None
):
    """
    构建简化的 LLMINA 工作流
    
    直接使用 LangGraph API，无需中间封装层
    """

    
    # 创建图
    workflow = StateGraph(AgentState)
    
    # 添加节点 - 直接使用 LangGraph API
    workflow.add_node("load_problem", load_problem_node)
    workflow.add_node("clarification", problem_clarification_node)
    workflow.add_node("algorithm_design", algorithm_design_node)
    workflow.add_node("decompose", task_decompose_node)
    workflow.add_node("analyze_dependencies", dependency_analysis_node)
    workflow.add_node("solve_task", task_solving_node)
    workflow.add_node("integrate", code_integration_node)
    
    # 添加边 - 直接使用 LangGraph API
    workflow.add_edge(START, "load_problem")
    workflow.add_edge("load_problem", "clarification")
    workflow.add_edge("clarification", "algorithm_design")
    workflow.add_edge("algorithm_design", "decompose")
    workflow.add_edge("decompose", "analyze_dependencies")
    workflow.add_edge("analyze_dependencies", "solve_task")
    
    # 条件边 - 任务求解循环
    def should_continue_solving(state: AgentState) -> str:
        """判断是否继续求解任务"""
        next_action = state.get('next_action', 'end')
        if next_action == 'solve_next':
            return 'solve_task'
        elif next_action == 'integrate':
            return 'integrate'
        else:
            return END
    
    workflow.add_conditional_edges(
        "solve_task",
        should_continue_solving,
        {
            'solve_task': 'solve_task',  # 继续下一个任务
            'integrate': 'integrate',     # 集成代码
            END: END
        }
    )
    
    workflow.add_edge("integrate", END)
    
    # 编译 - 可选启用 checkpoint
    if enable_checkpoints:
        if checkpoint_path:
            checkpointer = SqliteSaver.from_conn_string(checkpoint_path)
        else:
            checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)
    else:
        return workflow.compile()
