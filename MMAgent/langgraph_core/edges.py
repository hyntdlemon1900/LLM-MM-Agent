"""
边和条件定义模块
定义 LangGraph 中的边、条件边和路由逻辑
"""

from typing import Callable, Dict, Any, List, Literal, Union
from .state import AgentState


# ============ 条件函数类型 ============

ConditionalEdgeResult = Union[str, Literal["end", "continue", "retry", "error"]]


# ============ 基础条件函数 ============

def should_continue(state: AgentState) -> ConditionalEdgeResult:
    """
    判断是否应该继续执行
    
    Returns:
        'continue': 继续下一步
        'end': 结束流程
        'error': 进入错误处理
    """
    # 检查错误
    if state.get('errors') and len(state['errors']) > 0:
        return 'error'
    
    # 检查循环次数
    loop_count = state.get('loop_count', 0)
    max_loops = state.get('max_loops', 10)
    if loop_count >= max_loops:
        # LangGraph 会自动使用 merge_lists 合并警告列表
        state['warnings'] = [f"Max loops ({max_loops}) reached"]
        return 'end'
    
    # 根据 next_action 决定
    next_action = state.get('next_action', 'continue')
    
    if next_action == 'end':
        return 'end'
    elif next_action == 'error':
        return 'error'
    else:
        return 'continue'


def should_retry(state: AgentState) -> ConditionalEdgeResult:
    """
    判断是否应该重试
    
    Returns:
        'retry': 重试当前节点
        'continue': 继续下一步
        'error': 进入错误处理
    """
    # 获取重试配置
    max_retries = state.get('config', {}).get('max_retries', 3)
    current_retry = state.get('current_retry', 0)
    
    # 检查是否有错误
    has_error = len(state.get('errors', [])) > 0
    
    if has_error and current_retry < max_retries:
        state['current_retry'] = current_retry + 1
        return 'retry'
    elif has_error:
        return 'error'
    else:
        state['current_retry'] = 0
        return 'continue'


# ============ LLMINA 专用条件函数 ============

def should_clarify_more(state: AgentState) -> ConditionalEdgeResult:
    """
    判断是否需要更多澄清
    
    Returns:
        'clarify': 继续澄清
        'modeling': 进入建模阶段
        'error': 错误处理
    """
    # 检查错误
    if state.get('errors'):
        return 'error'
    
    # 检查澄清配置
    config = state.get('config', {})
    clarification_enabled = config.get('clarification_enabled', True)
    
    if not clarification_enabled:
        return 'modeling'
    
    # 检查澄清轮次
    clarification_round = state.get('clarification_round', 0)
    max_rounds = config.get('clarification_max_rounds', 3)
    
    if clarification_round >= max_rounds:
        return 'modeling'
    
    # 检查是否有澄清总结（表示澄清完成）
    if state.get('clarification_summary'):
        return 'modeling'
    
    return 'clarify'


def should_refine_modeling(state: AgentState) -> ConditionalEdgeResult:
    """
    判断是否需要细化建模
    
    Returns:
        'refine': 继续细化建模
        'decompose': 进入任务分解
        'error': 错误处理
    """
    # 检查错误
    if state.get('errors'):
        return 'error'
    
    # 检查建模轮次
    modeling_round = state.get('modeling_round', 0)
    max_rounds = state.get('config', {}).get('modeling_max_rounds', 2)
    
    if modeling_round >= max_rounds:
        return 'decompose'
    
    # 检查建模方案质量（可以添加质量评估逻辑）
    modeling_solution = state.get('modeling_solution', '')
    
    if len(modeling_solution) < 100:  # 简单的质量检查
        if modeling_round < max_rounds - 1:
            return 'refine'
    
    return 'decompose'


def task_solver_router(state: AgentState) -> ConditionalEdgeResult:
    """
    任务求解路由器
    
    Returns:
        'solve_next': 求解下一个任务
        'integrate': 进入代码集成
        'error': 错误处理
    """
    # 检查错误
    if state.get('errors'):
        return 'error'
    
    # 获取任务信息
    current_task_id = state.get('current_task_id', 0)
    total_tasks = state.get('total_tasks', 0)
    
    # 检查是否所有任务都已完成
    if current_task_id >= total_tasks:
        return 'integrate'
    
    # 继续求解下一个任务
    return 'solve_next'


def validation_router(state: AgentState) -> ConditionalEdgeResult:
    """
    验证路由器
    
    Returns:
        'passed': 验证通过，继续
        'failed': 验证失败，重试
        'error': 错误处理
    """
    # 检查错误
    if state.get('errors'):
        return 'error'
    
    # 获取最新的验证结果
    validation_results = state.get('validation_results', [])
    
    if not validation_results:
        return 'error'
    
    latest_result = validation_results[-1]
    is_valid = latest_result.get('is_valid', False)
    
    if is_valid:
        return 'passed'
    
    # 检查重试次数
    max_attempts = state.get('config', {}).get('max_validation_attempts', 2)
    attempt_count = len([r for r in validation_results if not r.get('is_valid', True)])
    
    if attempt_count >= max_attempts:
        return 'error'
    
    return 'failed'


# ============ 标准建模专用条件函数 ============

def problem_type_router(state: AgentState) -> str:
    """
    问题类型路由器
    
    Returns:
        'type_A', 'type_B', 'type_C', 'type_D', 'unknown'
    """
    problem_type = state.get('problem_type', '').upper()
    
    if problem_type in ['A', 'B', 'C', 'D']:
        return f'type_{problem_type}'
    
    return 'unknown'


def task_execution_router(state: AgentState) -> ConditionalEdgeResult:
    """
    任务执行路由器
    
    Returns:
        'execute_next': 执行下一个任务
        'report': 生成报告
        'error': 错误处理
    """
    # 检查错误
    if state.get('errors'):
        return 'error'
    
    # 获取任务信息
    task_decomposition = state.get('task_decomposition', [])
    execution_results = state.get('execution_results', [])
    
    # 检查是否所有任务都已执行
    if len(execution_results) >= len(task_decomposition):
        return 'report'
    
    return 'execute_next'


# ============ 条件边创建函数 ============

def create_conditional_edge(
    condition_func: Callable[[AgentState], ConditionalEdgeResult],
    edge_mapping: Dict[str, str]
) -> Callable:
    """
    创建条件边
    
    Args:
        condition_func: 条件函数
        edge_mapping: 条件结果到节点的映射
        
    Returns:
        条件边函数
    """
    def conditional_edge(state: AgentState) -> str:
        result = condition_func(state)
        next_node = edge_mapping.get(result, edge_mapping.get('default', 'end'))
        
        print(f"[Conditional Edge] Condition: {condition_func.__name__}")
        print(f"  Result: {result} -> Next node: {next_node}")
        
        return next_node
    
    return conditional_edge


def create_loop_edge(
    condition_func: Callable[[AgentState], bool],
    loop_node: str,
    continue_node: str,
    max_iterations: int = 10
) -> Callable:
    """
    创建循环边
    
    Args:
        condition_func: 条件函数（返回 True 表示继续循环）
        loop_node: 循环目标节点
        continue_node: 跳出循环后的节点
        max_iterations: 最大迭代次数
        
    Returns:
        循环边函数
    """
    def loop_edge(state: AgentState) -> str:
        # 增加循环计数
        loop_count = state.get('loop_count', 0) + 1
        state['loop_count'] = loop_count
        
        # 检查最大迭代次数
        if loop_count >= max_iterations:
            print(f"[Loop Edge] Max iterations ({max_iterations}) reached")
            return continue_node
        
        # 检查条件
        should_loop = condition_func(state)
        
        if should_loop:
            print(f"[Loop Edge] Looping to {loop_node} (iteration {loop_count})")
            return loop_node
        else:
            print(f"[Loop Edge] Exiting loop to {continue_node}")
            state['loop_count'] = 0  # 重置计数器
            return continue_node
    
    return loop_edge


# ============ 预定义条件边 ============

# LLMINA 工作流条件边
LLMINA_EDGES = {
    'clarification': create_conditional_edge(
        should_clarify_more,
        {
            'clarify': 'problem_clarification',
            'modeling': 'problem_modeling',
            'error': 'error_handler'
        }
    ),
    'modeling': create_conditional_edge(
        should_refine_modeling,
        {
            'refine': 'problem_modeling',
            'decompose': 'task_decompose',
            'error': 'error_handler'
        }
    ),
    'task_solving': create_conditional_edge(
        task_solver_router,
        {
            'solve_next': 'task_solver',
            'integrate': 'code_integrator',
            'error': 'error_handler'
        }
    ),
    'validation': create_conditional_edge(
        validation_router,
        {
            'passed': 'evaluation',
            'failed': 'code_validator',
            'error': 'error_handler'
        }
    )
}

# 标准建模工作流条件边
STANDARD_EDGES = {
    'problem_type': create_conditional_edge(
        problem_type_router,
        {
            'type_A': 'method_retriever_A',
            'type_B': 'method_retriever_B',
            'type_C': 'method_retriever_C',
            'type_D': 'method_retriever_D',
            'unknown': 'error_handler'
        }
    ),
    'task_execution': create_conditional_edge(
        task_execution_router,
        {
            'execute_next': 'task_executor',
            'report': 'report_generator',
            'error': 'error_handler'
        }
    )
}


# ============ 辅助函数 ============

def get_edge_info(edge_name: str, workflow: str = 'llmina') -> Dict[str, Any]:
    """获取边的信息"""
    edges = LLMINA_EDGES if workflow == 'llmina' else STANDARD_EDGES
    edge = edges.get(edge_name)
    
    if edge:
        return {
            'name': edge_name,
            'type': 'conditional',
            'workflow': workflow
        }
    
    return {}


def list_edges(workflow: str = 'llmina') -> List[str]:
    """列出所有边"""
    edges = LLMINA_EDGES if workflow == 'llmina' else STANDARD_EDGES
    return list(edges.keys())
