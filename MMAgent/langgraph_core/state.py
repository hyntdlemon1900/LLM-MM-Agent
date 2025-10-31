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
    基础 Agent 状态 - 所有工作流的通用状态
    
    使用 TypedDict 实现类型安全的状态管理
    total=False 表示所有字段都是可选的
    """
    # ============ 核心输入 ============
    problem_path: str  # 问题文件路径
    problem_str: str  # 问题描述字符串
    config: Dict[str, Any]  # 配置信息
    output_dir: str  # 输出目录
    name: str  # 任务名称
    
    # ============ LLM 相关 ============
    llm: Any  # LLM 实例
    llm_usage: Dict[str, int]  # LLM 使用统计
    
    # ============ 消息历史 ============
    messages: Annotated[List[Message], merge_lists]  # 所有消息历史（使用列表合并）
    
    # ============ 工作流控制 ============
    current_node: str  # 当前节点名称
    next_action: str  # 下一步动作
    loop_count: int  # 循环计数器
    max_loops: int  # 最大循环次数
    
    # ============ 错误处理 ============
    errors: Annotated[List[str], merge_lists]  # 错误记录
    warnings: Annotated[List[str], merge_lists]  # 警告记录
    
    # ============ 执行结果 ============
    result: Dict[str, Any]  # 最终结果
    intermediate_results: Annotated[Dict[str, Any], merge_dicts]  # 中间结果


class LLMINAState(AgentState):
    """
    LLMINA 专用状态 - 用于 LLMINA 问题求解流程
    
    继承基础状态，添加 LLMINA 特定字段
    """
    # ============ 问题澄清阶段 ============
    clarification_history: Annotated[List[Dict[str, str]], merge_lists]  # 澄清历史
    clarification_summary: str  # 澄清总结
    clarification_round: int  # 澄清轮次
    
    # ============ 问题建模阶段 ============
    modeling_solution: str  # 建模方案
    modeling_analysis: str  # 建模分析
    modeling_round: int  # 建模轮次
    
    # ============ 任务分解阶段 ============
    task_descriptions: List[str]  # 任务描述列表
    tasknum: int  # 任务数量（兼容性别名）
    task_dependency_analysis: List[str]  # 任务依赖分析
    dependency_dag: Dict[str, List[str]]  # 依赖 DAG
    execution_order: List[int]  # 执行顺序
    
    # ============ 任务求解阶段 ============
    current_task_id: int  # 当前任务 ID
    total_tasks: int  # 总任务数
    task_results: Annotated[Dict[int, Dict[str, Any]], merge_dicts]  # 任务结果（含代码、执行结果等）
    
    # ============ 代码生成与验证 ============
    solver_code: str  # 完整 solver 代码
    solver_code_path: str  # solver 代码路径
    validation_results: Annotated[List[Dict[str, Any]], merge_lists]  # 验证结果
    
    # ============ 评估阶段 ============
    evaluation_metrics: Dict[str, Any]  # 评估指标
    performance_history: Annotated[List[Dict[str, Any]], merge_lists]  # 性能历史


class StandardModelingState(AgentState):
    """
    标准数学建模状态 - 用于通用数学建模流程
    """
    # ============ 问题分析 ============
    problem_type: str  # 问题类型 (A/B/C/D)
    problem_analysis: str  # 问题分析
    data_description: str  # 数据描述
    
    # ============ 方法检索 ============
    retrieved_methods: List[Dict[str, Any]]  # 检索到的方法
    selected_method: str  # 选择的方法
    
    # ============ 数学建模 ============
    mathematical_model: str  # 数学模型
    assumptions: List[str]  # 假设条件
    variables: Dict[str, str]  # 变量定义
    constraints: List[str]  # 约束条件
    objective: str  # 目标函数
    
    # ============ 任务求解 ============
    task_decomposition: List[str]  # 任务分解
    solution_code: str  # 求解代码
    execution_results: Annotated[List[Any], merge_lists]  # 执行结果
    
    # ============ 结果报告 ============
    charts: List[str]  # 图表路径
    report: str  # 报告内容
    paper_path: str  # 论文路径


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

def create_llmina_state(
    problem_path: str,
    config: Dict[str, Any],
    output_dir: str,
    name: str,
    llm: Any
) -> LLMINAState:
    """创建 LLMINA 初始状态"""
    return LLMINAState(
        problem_path=problem_path,
        problem_str="",
        config=config,
        output_dir=output_dir,
        name=name,
        llm=llm,
        llm_usage={'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
        messages=[],
        current_node="start",
        next_action="clarification",
        loop_count=0,
        max_loops=config.get('max_loops', 10),
        errors=[],
        warnings=[],
        result={},
        intermediate_results={},
        clarification_history=[],
        clarification_summary="",
        clarification_round=0,
        modeling_solution="",
        modeling_analysis="",
        modeling_round=0,
        task_descriptions=[],
        tasknum=0,
        task_dependency_analysis=[],
        dependency_dag={},
        execution_order=[],
        current_task_id=0,
        total_tasks=0,
        task_results={},
        solver_code="",
        solver_code_path="",
        validation_results=[],
        evaluation_metrics={},
        performance_history=[]
    )

def create_standard_state(
    problem_path: str,
    config: Dict[str, Any],
    output_dir: str,
    name: str,
    llm: Any
) -> StandardModelingState:
    """创建标准建模初始状态"""
    return StandardModelingState(
        problem_path=problem_path,
        problem_str="",
        config=config,
        output_dir=output_dir,
        name=name,
        llm=llm,
        llm_usage={'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
        messages=[],
        current_node="start",
        next_action="problem_analysis",
        loop_count=0,
        max_loops=config.get('max_loops', 10),
        errors=[],
        warnings=[],
        result={},
        intermediate_results={},
        problem_type="",
        problem_analysis="",
        data_description="",
        retrieved_methods=[],
        selected_method="",
        mathematical_model="",
        assumptions=[],
        variables={},
        constraints=[],
        objective="",
        task_decomposition=[],
        solution_code="",
        execution_results=[],
        charts=[],
        report="",
        paper_path=""
    )


# ============ 状态验证函数 ============

def validate_state(state: AgentState) -> bool:
    """验证状态是否有效"""
    required_fields = ['problem_path', 'config', 'output_dir', 'name', 'llm']
    return all(field in state for field in required_fields)


def get_state_summary(state: AgentState) -> Dict[str, Any]:
    """获取状态摘要（用于日志和监控）"""
    return {
        'current_node': state.get('current_node', 'unknown'),
        'next_action': state.get('next_action', 'unknown'),
        'loop_count': state.get('loop_count', 0),
        'error_count': len(state.get('errors', [])),
        'warning_count': len(state.get('warnings', [])),
        'messages_count': len(state.get('messages', [])),
        'has_result': bool(state.get('result', {}))
    }
