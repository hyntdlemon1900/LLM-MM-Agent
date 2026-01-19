"""
简化的图构建模块
直接使用 LangGraph 内置功能，移除不必要的封装
"""

from typing import Optional, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import os

from .state import AgentState
from .nodes import (
    load_problem_node,
    heuristic_initialization_node,
    function_code_generator_node,
    code_integration_node,
    heuristic_evaluation_node,
    code_fix_node,
    constraint_analyzer_node,
    heuristic_reflect_node,
    feedback_guided_optimization_node,
    exploratory_refactoring_node,
)

def build_workflow(
    enable_checkpoints: bool = False,
    checkpoint_path: Optional[str] = None,
):
    """
    Builds the MCTS-driven LLMINA workflow.
    """

    # 1. Create Graph
    workflow = StateGraph(AgentState)

    # 2. Add Nodes
    workflow.add_node("load_problem", load_problem_node)
    # workflow.add_node("mcts_manager", mcts_management_node) # Removed
    
    # Operators
    workflow.add_node("reflect", heuristic_reflect_node)              
    workflow.add_node("initialization", heuristic_initialization_node) # i1
    workflow.add_node("optimization", feedback_guided_optimization_node) # e1
    workflow.add_node("refactoring", exploratory_refactoring_node) # e3
    
    # Worker Nodes
    workflow.add_node("generate_function", function_code_generator_node)
    workflow.add_node("integrate", code_integration_node)
    workflow.add_node("evaluate", heuristic_evaluation_node)
    workflow.add_node("fix", code_fix_node)
    workflow.add_node("constraint_analyze", constraint_analyzer_node) # Backprop & Select

    # 3. Edges
    
    # Start -> Load
    workflow.add_edge(START, "load_problem")
    # Load -> Router (MCTS Init happens in load_problem, deciding I1)
    # workflow.add_edge("load_problem","evaluate")  # Temporary direct edge to evaluate for initial testing
    # MCTS Routing Logic
    def route_mcts_decision(state: AgentState) -> str:
        # Routes based on 'next_action_node' set by mcts_turn_step (called in architect or constrain_analyze)
        destination = state.get('next_action_node', 'generate_function')
        print(f"[Graph Router] Routing to: {destination}")
        return destination
        
    mcts_map = {
        "generate_function": "generate_function",
        "reflect": "reflect",
        "initialization": "initialization",
        "optimization": "optimization",
        "refactoring": "refactoring"
    }

    workflow.add_conditional_edges("load_problem", route_mcts_decision, mcts_map)
    # Constraint (End of Loop) -> (Backprop + Selection) -> Next
    workflow.add_conditional_edges("constraint_analyze", route_mcts_decision, mcts_map)
    
    # Generation Loop Logic
    def should_continue_generating(state: AgentState) -> str:
        next_action = state.get('next_action', 'integrate')
        if next_action == 'generate_next_function':
            return 'generate_function'
        else:
            return 'integrate'

    # i1: initialization -> generate_function -> Loop -> Integrate
    workflow.add_conditional_edges("initialization", should_continue_generating, 
                                 {'generate_function': 'generate_function', 'integrate': 'integrate'})
    
    # e1: reflect -> generate_function -> Loop -> Integrate
    workflow.add_conditional_edges("reflect", should_continue_generating, 
                                 {'generate_function': 'generate_function', 'integrate': 'integrate'})
    workflow.add_conditional_edges("optimization", should_continue_generating, 
                                 {'generate_function': 'generate_function', 'integrate': 'integrate'})
    workflow.add_conditional_edges("refactoring", should_continue_generating, 
                                 {'generate_function': 'generate_function', 'integrate': 'integrate'})
    
    # Code Generation Loop (Self-cycle)
    workflow.add_conditional_edges("generate_function", should_continue_generating, 
                                 {'generate_function': 'generate_function', 'integrate': 'integrate'})

    # Integration -> Evaluate
    workflow.add_edge("integrate", "evaluate")
    
    # Evaluation Routing
    def route_evaluation(state: AgentState) -> str:
        next_action = state.get('next_action', 'constraint_analyze')
        
        # Mapping evaluation results to nodes
        if next_action == 'fix':
            return 'fix'
        elif next_action == 'reflect':
            return 'reflect'
        elif next_action == 'constraint_analyze':
            return 'constraint_analyze'
        else:
            return 'constraint_analyze' # Default forward to Analyze
            
    workflow.add_conditional_edges(
        "evaluate",
        route_evaluation,
        {
            "fix": "fix",
            "constraint_analyze": "constraint_analyze",
            "reflect": "reflect"
        }
    )
    
    # Fix Loop
    workflow.add_edge("fix", "evaluate")
    
    if enable_checkpoints:
        if checkpoint_path is None:
            checkpoint_path = "./checkpoints/llmina.db"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpointer = SqliteSaver.from_conn_string(f"file:{checkpoint_path}")
        return workflow.compile(checkpointer=checkpointer)
    else:
        # 不使用检查点时也要返回编译后的图
        return workflow.compile()
