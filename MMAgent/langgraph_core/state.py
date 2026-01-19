"""
状态定义模块
定义 LangGraph 工作流中的所有状态类型
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from .MCTS.mcts_core import MCTS, MCTSNode

def merge_lists(left: List, right: List) -> List:
    """合并列表，追加新元素（通用列表合并）"""
    return left + right


def merge_dicts(left: Dict, right: Dict) -> Dict:
    """合并字典，更新键值"""
    result = left.copy()
    result.update(right)
    return result

class AgentState(TypedDict, total=False):
    """
    通用 Agent 状态 - 所有工作流的通用状态
    
    使用 TypedDict 实现类型安全的状态管理
    total=False 表示所有字段都是可选的
    """
    # ============ 核心输入 ============
    task_name: str  # 任务名称 (e.g., 'LLMINA')
    task_dir: str  # 任务目录路径 LLMINA
    runtime_dir: str  # 运行时目录
    problem_path: str  # 问题文件路径 LLMINA/problem.json
    problem_str: str  # 问题描述字符串
    output_dir: str  # 输出目录
    algorithm_path: str  # 算法目录路径
    # ============ LLM 相关 ============
    llm: Any  # LLM 实例
    time_limit: int  # 求解时间限制（秒）
    # ============ 工作流控制 ============
    next_action: str  # 下一步动作
    next_action_node: str # Explicit routing for MCTS/Graph branches
    
    # ============ evaluate ============
    errors: Annotated[List[str], merge_lists]  # 错误记录
    evaluation_success: bool  # 评估是否成功
    traceback: Optional[str]  # 错误追踪信息
    evaluation_results: Dict[str, Any]  # 评估结果
    quality_report: Optional[str]  # 质量报告文本
    # 不可行经验
    experiences: Annotated[List[str], merge_lists]  # 不可行经验总结文本

    # ============ 启发式架构生成阶段 ============
    problem_analysis: str  # 问题分析文本
    strategy_overview: str  # 策略概述文本
    function_architecture: List[Dict[str, Any]]  # 启发式函数架构列表 第一部分
    function_dependency_graph: Dict[str, List[str]] # 函数依赖图 {func_name: [dependency_names]}
    functions_to_generate: List[int]  # 待生成函数列表 第三部分
    
    # ============ 启发式函数生成阶段 ============
    current_function_id: int  # 当前正在生成的函数ID
    function_descriptions: Annotated[Dict[int, str], merge_dicts] # 函数ID -> 函数描述
    function_codes: Annotated[Dict[int, str], merge_dicts]  # 函数ID -> 生成的代码
    best_function_codes: Dict[int, str]  # 最佳函数代码
    current_step: int  # 当前步骤数

    # ============ 启发式求解器模板 ============
    heuristic_reference_code: str  # HeuristicSolver模板代码
    
    # ============ 最终结果 ============
    solver_code_path: str  # solver 代码路径
    
    # ============ 代码修复相关 ============
    fix_attempt_count: int  # 代码修复尝试次数
    max_fix_attempts: int  # 最大修复尝试次数
    previous_errors: Annotated[List[str], merge_lists]  # 历史错误记录

    # ============ MCTS State ============
    mcts: Any                      # MCTS Instance Object
    mcts_data: Dict[str, Any]      # Serialized MCTS tree structure
    current_mcts_node_id: str      # Current active node ID
    mcts_operator: str             # Operator chosen for current step
    parent_a_config: Dict[str, Any]
    parent_b_config: Dict[str, Any]
    
    init_pop_size: int             # Target size for initial population

class WorkflowMetadata(TypedDict):
    """工作流元数据"""
    workflow_id: str
    workflow_name: str
    start_time: str
    end_time: Optional[str]
    status: str  # 'running', 'completed', 'failed', 'paused'
    checkpoint_path: Optional[str]
    total_nodes: int
    completed_nodes: int
    current_node: Optional[str]

# ============ 状态工厂函数 ============
def create_agent_state(
    llm: Any,
    config: Dict[str, Any],
    task_name: str,
    output_dir: str,
) -> AgentState:
    """
    创建初始状态
    
    Args:
        llm: LLM 实例
        config: 配置字典（从 config.yaml 加载）
        task_name: 任务名称
        output_dir: 输出目录
    
    Returns:
        初始化的 AgentState
    """
    # 从 config 解析路径
    problem_base = config.get('paths', {}).get('problem', 'MMBench/problem')
    task_dir = Path(problem_base) / task_name
    problem_path = str(task_dir / 'problem.json')
    
    # 验证路径存在
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")
    if not Path(problem_path).exists():
        raise FileNotFoundError(f"Problem file not found: {problem_path}")
    
    # 根据 task_dir 构造路径
    runtime_dir = str(task_dir / 'runtime')
    
    return AgentState(
        # 核心输入
        task_name=task_name,
        task_dir = str(task_dir),
        algorithm_dir = str(task_dir / 'algorithm/0'),
        runtime_dir=runtime_dir,
        problem_path=problem_path,
        problem_str="",
        output_dir=output_dir,
        # output_dir = "./output/LLMINA_20251205-113010", # 临时硬编码，方便调试 --- IGNORE ---
        
        # LLM
        llm=llm,
        time_limit = 100,
        
        # 初始状态
        messages=[],
        next_action="load_problem",  # 第一步是加载问题
        errors=[],
        evaluation_success=False,  # 评估是否成功
        traceback=None,  # 错误追踪信息
        evaluation_results ={},
        
        problem_analysis='',
        strategy_overview='',
        function_architecture=[],
        function_dependency_graph ={},

        function_descriptions={},
        experiences=[],
        
        current_function_id=0,
        functions_to_generate = [],
        function_codes={},
        best_function_codes = {},
        
        current_step=1,
        heuristic_reference_code="",
        solver_code_path="",
        # solver_code_path="./output/LLMINA_20251205-113010/algorithm/0" + "/ModelSolver_final.py", # 临时硬编码，方便调试 --- IGNORE ---
        
        # 代码修复相关
        fix_attempt_count=0,
        max_fix_attempts=3,
        previous_errors=[],

        # MCTS State
        mcts=MCTS(),
        mcts_data={},
        current_mcts_node_id="",
        mcts_operator="",
        parent_a_config={},
        parent_b_config={},        
        # Evolution Parameters
        init_pop_size=2,    )
