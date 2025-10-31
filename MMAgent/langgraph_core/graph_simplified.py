"""
简化的图构建模块
直接使用 LangGraph 内置功能，移除不必要的封装
"""

from typing import Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

from .state import LLMINAState
from .agent_nodes.llmina_nodes_simplified import (
    load_problem_node,
    problem_clarification_node,
    problem_modeling_node,
    task_decompose_node,
    dependency_analysis_node,
    task_solving_node,
    code_integration_node
)
from .nodes_simplified import start_node, end_node, error_handler_node

def build_simple_llmina_workflow(
    enable_checkpoints: bool = False,
    checkpoint_path: Optional[str] = None
):
    """
    构建简化的 LLMINA 工作流
    
    直接使用 LangGraph API，无需中间封装层
    """

    
    # 创建图
    workflow = StateGraph(LLMINAState)
    
    # 添加节点 - 直接使用 LangGraph API
    workflow.add_node("start", start_node)
    workflow.add_node("load_problem", load_problem_node)
    workflow.add_node("clarification", problem_clarification_node)
    workflow.add_node("modeling", problem_modeling_node)
    workflow.add_node("decompose", task_decompose_node)
    workflow.add_node("analyze_dependencies", dependency_analysis_node)
    workflow.add_node("solve_task", task_solving_node)
    workflow.add_node("integrate", code_integration_node)
    workflow.add_node("error_handler", error_handler_node)
    workflow.add_node("end", end_node)
    
    # 添加边 - 直接使用 LangGraph API
    workflow.add_edge(START, "start")
    workflow.add_edge("start", "load_problem")
    workflow.add_edge("load_problem", "clarification")
    workflow.add_edge("clarification", "modeling")
    workflow.add_edge("modeling", "decompose")
    workflow.add_edge("decompose", "analyze_dependencies")
    workflow.add_edge("analyze_dependencies", "solve_task")
    
    # 条件边 - 任务求解循环
    def should_continue_solving(state: LLMINAState) -> str:
        """判断是否继续求解任务"""
        next_action = state.get('next_action', 'end')
        if next_action == 'solve_next':
            return 'solve_task'
        elif next_action == 'integrate':
            return 'integrate'
        elif next_action == 'error':
            return 'error_handler'
        else:
            return 'end'
    
    workflow.add_conditional_edges(
        "solve_task",
        should_continue_solving,
        {
            'solve_task': 'solve_task',  # 继续下一个任务
            'integrate': 'integrate',     # 集成代码
            'error_handler': 'error_handler',
            'end': 'end'
        }
    )
    
    workflow.add_edge("integrate", "end")
    workflow.add_edge("error_handler", "end")
    workflow.add_edge("end", END)
    
    # 编译 - 可选启用 checkpoint
    if enable_checkpoints:
        if checkpoint_path:
            checkpointer = SqliteSaver.from_conn_string(checkpoint_path)
        else:
            checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)
    else:
        return workflow.compile()


def visualize_workflow(workflow, output_path: str = "workflow.png"):
    """可视化工作流"""
    try:
        import importlib
        ipy_display = importlib.import_module('IPython.display')
        Image = getattr(ipy_display, 'Image')
        display = getattr(ipy_display, 'display')
        display(Image(workflow.get_graph().draw_mermaid_png()))
    except Exception:
        # 保存到文件
        with open(output_path, 'wb') as f:
            f.write(workflow.get_graph().draw_mermaid_png())
        print(f"Workflow diagram saved to {output_path}")
