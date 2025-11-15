"""
状态定义模块
定义 LangGraph 工作流中的所有状态类型
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


def merge_lists(left: List, right: List) -> List:
    """合并列表，追加新元素（通用列表合并）"""
    return left + right


def merge_dicts(left: Dict, right: Dict) -> Dict:
    """合并字典，更新键值"""
    result = left.copy()
    result.update(right)
    return result


@dataclass
class Message:
    """消息对象，用于 Agent 间通信"""
    role: str  # 'user', 'assistant', 'system', 'agent'
    content: str
    agent_name: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentState(TypedDict, total=False):
    """
    通用 Agent 状态 - 所有工作流的通用状态
    
    使用 TypedDict 实现类型安全的状态管理
    total=False 表示所有字段都是可选的
    """
    # ============ 核心输入 ============
    task_name: str  # 任务名称 (e.g., 'LLMINA')
    task_dir: str  # 任务目录路径
    problem_path: str  # 问题文件路径
    problem_str: str  # 问题描述字符串
    output_dir: str  # 输出目录
    
    # ============ 配置参数（从 config 提取）============
    template_dir: str  # 代码模板目录
    runtime_dir: str  # 运行时目录
    problem_dir: str  # 问题目录
    
    # ============ 工作流配置 ============
    max_loops: int  # 最大循环次数
    max_retries: int  # 最大重试次数
    algorithm_design_max_rounds: int  # 算法设计最大轮次
    modeling_max_rounds: int  # 建模最大轮次
    max_validation_attempts: int  # 最大验证尝试次数
    
    # ============ LLM 相关 ============
    llm: Any  # LLM 实例
    
    # ============ 消息历史 ============
    messages: Annotated[List[Message], merge_lists]  # 所有消息历史
    
    # ============ 工作流控制 ============
    next_action: str  # 下一步动作
    loop_count: int  # 循环计数器
    current_retry: int  # 当前重试次数
    
    # ============ 错误处理 ============
    errors: Annotated[List[str], merge_lists]  # 错误记录
    warnings: Annotated[List[str], merge_lists]  # 警告记录
    
    # ============ 启发式函数生成阶段 ============
    heuristic_functions: List[Dict[str, Any]]  # 启发式函数架构列表
    current_function_id: int  # 当前正在生成的函数ID
    function_codes: Annotated[Dict[int, str], merge_dicts]  # 函数ID -> 生成的代码
    
    # ============ 启发式求解器模板 ============
    pyomo_reference_code: str  # Pyomo建模代码（参考）
    heuristic_template_code: str  # HeuristicSolver模板代码
    heuristic_template_path: str  # HeuristicSolver模板文件路径
    
    # ============ 中间结果 ============
    intermediate_results: Annotated[Dict[str, Any], merge_dicts]  # 中间结果存储
    
    # ============ 最终结果 ============
    solver_code_path: str  # solver 代码路径
    
    # ============ 评估相关 ============
    evaluation_config: Dict[str, Any]  # 评估配置（拓扑、INA预算、任务数量等）
    evaluation_results: Dict[str, Any]  # 评估结果

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
    template_dir = str(task_dir / 'code_template')
    runtime_dir = str(task_dir / 'runtime')
    problem_dir = str(task_dir)
    
    # 从 config 提取工作流配置
    max_loops = config.get('max_loops', 10)
    max_retries = config.get('max_retries', 2)
    algorithm_design_max_rounds = config.get('algorithm_design_max_rounds', 2)
    modeling_max_rounds = config.get('modeling_max_rounds', 2)
    max_validation_attempts = config.get('max_validation_attempts', 2)
    
    return AgentState(
        # 核心输入
        task_name=task_name,
        task_dir=runtime_dir,  # task_dir 指向 runtime 目录
        problem_path=problem_path,
        problem_str="",
        output_dir=output_dir,
        
        # 配置参数（从 task_dir 构造）
        template_dir=template_dir,
        runtime_dir=runtime_dir,
        problem_dir=problem_dir,
        
        # 工作流配置
        max_loops=max_loops,
        max_retries=max_retries,
        algorithm_design_max_rounds=algorithm_design_max_rounds,
        modeling_max_rounds=modeling_max_rounds,
        max_validation_attempts=max_validation_attempts,
        
        # LLM
        llm=llm,
        
        # 初始状态
        messages=[],
        next_action="load_problem",  # 第一步是加载问题
        loop_count=0,
        current_retry=0,
        errors=[],
        warnings=[],
        heuristic_functions=[],
        current_function_id=0,
        function_codes={},
        intermediate_results={},
        pyomo_reference_code="",
        heuristic_template_code="",
        heuristic_template_path="",
        solver_code_path="",
        
        # 评估配置（默认值）
        evaluation_config={
            'topo_name': 'FatTree',
            'ina_num_list': [3],
            'jobs_num_list': [6],
            'instances_num': 2,
            'solver_name': 'gurobi',
            'time_limit': 300,
            'mip_gap': 0.01,
            'verbose': True
        },
        evaluation_results={}
    )

# ============ 状态验证函数 ============

def validate_state(state: AgentState) -> bool:
    """验证状态是否有效"""
    required_fields = ['problem_path', 'output_dir', 'llm']
    return all(field in state for field in required_fields)


def get_state_summary(state: AgentState) -> Dict[str, Any]:
    """获取状态摘要（用于日志和监控）"""
    return {
        'next_action': state.get('next_action', 'unknown'),
        'loop_count': state.get('loop_count', 0),
        'error_count': len(state.get('errors', [])),
        'warning_count': len(state.get('warnings', [])),
        'messages_count': len(state.get('messages', [])),
        'solver_code_path': state.get('solver_code_path', '')
    }
