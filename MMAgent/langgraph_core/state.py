"""
状态定义模块
定义 LangGraph 工作流中的所有状态类型
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from dataclasses import dataclass, field
from datetime import datetime


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
    config: Dict[str, Any]  # 配置信息
    output_dir: str  # 输出目录
    
    # ============ LLM 相关 ============
    llm: Any  # LLM 实例
    
    # ============ 消息历史 ============
    messages: Annotated[List[Message], merge_lists]  # 所有消息历史
    
    # ============ 工作流控制 ============
    next_action: str  # 下一步动作
    loop_count: int  # 循环计数器
    max_loops: int  # 最大循环次数
    
    # ============ 错误处理 ============
    errors: Annotated[List[str], merge_lists]  # 错误记录
    warnings: Annotated[List[str], merge_lists]  # 警告记录
    
    # ============ 问题澄清阶段 ============
    clarification_round: int  # 澄清轮次
    
    # ============ 算法方案设计阶段 ============
    algorithm_solution: str  # 算法求解方案
    algorithm_design_round: int  # 方案设计轮次
    modeling_solution: str  # 建模方案（别名）
    
    # ============ 任务分解阶段 ============
    task_descriptions: List[str]  # 任务描述列表
    tasknum: int  # 任务数量
    task_dependency_analysis: List[str]  # 任务依赖分析
    dependency_dag: Dict[str, List[str]]  # 依赖 DAG
    execution_order: List[int]  # 执行顺序
    
    # ============ 任务求解阶段 ============
    current_task_id: int  # 当前任务 ID
    task_results: Annotated[Dict[int, Dict[str, Any]], merge_dicts]  # 任务结果
    
    # ============ 最终结果 ============
    solver_code_path: str  # solver 代码路径

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
    config: Dict[str, Any],
    llm: Any,
    task_name: str,
    task_dir: str,
    problem_path: str, 
    output_dir: str, 
) -> AgentState:
    """创建初始状态"""
    return AgentState(
        task_name=task_name,
        task_dir=task_dir,
        problem_path=problem_path,
        problem_str="",
        config=config,
        output_dir=output_dir,
        llm=llm,
        messages=[],
        next_action="clarification",
        loop_count=0,
        max_loops=config.get('max_loops', 10),
        errors=[],
        warnings=[],
        clarification_round=0,
        algorithm_solution="",
        algorithm_design_round=0,
        modeling_solution="",
        task_descriptions=[],
        tasknum=0,
        task_dependency_analysis=[],
        dependency_dag={},
        execution_order=[],
        current_task_id=0,
        task_results={},
        solver_code_path=""
    )

# ============ 状态验证函数 ============

def validate_state(state: AgentState) -> bool:
    """验证状态是否有效"""
    required_fields = ['problem_path', 'config', 'output_dir', 'llm']
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
