"""
简化的图构建模块
直接使用 LangGraph 内置功能，移除不必要的封装
"""

from typing import Optional, Dict, Any
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
    constraint_analyzer_node,
    heuristic_reflect_node
)

def build_workflow(
    enable_checkpoints: bool = False,
    checkpoint_path: Optional[str] = None,
):
    """
    构建 LLMINA 启发式设计与进化工作流（0→1→2 循环）

    阶段划分：
    - 0→1：从无到有生成启发式代码，并通过可行性评估 + 修复，得到一个“可运行且约束大体满足”的版本。
    - 1→2：在可行性的基础上，基于解质量报告（瓶颈分析）对启发式进行演化，形成多个版本 v1, v2, ...，
            每一版都经历“集成→评估→修复→质量分析”，直到质量达标或达到最大演化轮数。

    状态变量约定（示例）：
    - state["next_action"]: 由 heuristic_architect / generate_function 控制，决定是否继续生成函数。
    - state["evaluation_results"]: 由 evaluate 节点写入，包含运行是否成功、性能指标。
    - state["fix_attempt_count"], state["max_fix_attempts"]: 控制修复次数上限。
    - state["quality_report"]: 由 constraint_analyzer 写入，用于 reflect。
    - state["reflect_round"], state["max_reflect_rounds"]: 控制演化轮数和终止条件。
    """

    # 1. 创建图
    workflow = StateGraph(AgentState)

    # 2. 添加节点
    workflow.add_node("load_problem", load_problem_node)
    workflow.add_node("heuristic_architect", heuristic_architect_node)
    workflow.add_node("generate_function", function_code_generator_node)
    workflow.add_node("integrate", code_integration_node)
    workflow.add_node("evaluate", heuristic_evaluation_node)
    workflow.add_node("fix", code_fix_node)
    workflow.add_node("constraint_analyze", constraint_analyzer_node)
    workflow.add_node("reflect", heuristic_reflect_node)  # 新增：代码演化节点


    '''
    workflow.add_edge(START, "load_problem")
    workflow.add_edge("load_problem", "evaluate")
    workflow.add_edge("evaluate", "constraint_analyze")
    workflow.add_edge("constraint_analyze", END)

    '''
    # 3. 起始边：加载问题 → 生成 plan
    workflow.add_edge(START, "load_problem")
    workflow.add_edge("load_problem", "heuristic_architect")

    # 4. plan → 函数生成循环（0→1：从无到有）
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

    # 5. 集成后进入可行性评估
    workflow.add_edge("integrate", "evaluate")

    # 6. 评估后的修复循环（对每一版代码都适用）
    def should_fix_code(state: AgentState) -> str:
        """判断是否需要修复代码"""
        next_action = state.get('next_action', 'fix')
        fix_attempt_count = state.get('fix_attempt_count', 0)
        max_fix_attempts = state.get('max_fix_attempts', 3)
        
        # 优先使用 node 计算的 next_action
        if next_action == "constraint_analyze":
            return "constraint_analyze"
        elif next_action == "reflect":
            return "reflect"
        
        # 默认 fix 逻辑 (用于 fallback 或 node 返回 'fix') 
        if fix_attempt_count < max_fix_attempts:
            return "fix"
        else:
            return "end"

    workflow.add_conditional_edges(
        "evaluate",
        should_fix_code,
        {
            "fix": "fix",
            "constraint_analyze": "constraint_analyze",
            "reflect": "reflect",
            "end": END
        },
    )

    # 7. 修复后重新评估（仍然在同一版本代码内循环）
    workflow.add_edge("fix", "evaluate")

    # 8. 质量分析后是否进入演化（1→2：从“可行”到“更优”）
    def should_reflect(state: AgentState) -> str:
        """
        决定 constraint_analyze 之后的下一步：

        - 若质量已经“足够好”（例如 gap 足够小 / 没有明显瓶颈） → 'end'
        - 若演化轮数达到上限 → 'end'
        - 否则 → 'reflect'（基于质量报告改进代码）

        依赖状态：
        - state["quality_report"]: constraint_analyzer_node 写入，例如：
            {
              "is_good_enough": bool,
              "global_gap": float,
              "bottlenecks": [...],
              ...
            }
        - state["reflect_round"], state["max_reflect_rounds"]
        """
        quality_report: Dict[str, Any] = state.get("quality_report", {})
        is_good_enough = quality_report.get("is_good_enough", False)

        reflect_round = state.get("reflect_round", 0)
        max_reflect_rounds = state.get("max_reflect_rounds", 3)

        # 1) 质量已满足要求，直接结束
        if is_good_enough:
            return "end"

        # 2) 演化轮数已达上限，也结束
        if reflect_round >= max_reflect_rounds:
            return "end"

        # 3) 否则进入 reflect 节点，生成新版本启发式
        return "reflect"

    workflow.add_conditional_edges(
        "constraint_analyze",
        should_reflect,
        {
            "reflect": "reflect",  # 进入演化节点
            "end": END,          # 结束整个 workflow
        },
    )

    # 9. 演化后重新集成，再进入新一轮可行性评估
    #    注意：在 heuristic_reflect_node 里建议重置：
    #      - state["fix_attempt_count"] = 0
    #      - state["reflect_round"] += 1
    workflow.add_edge("reflect", "generate_function")
    
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
