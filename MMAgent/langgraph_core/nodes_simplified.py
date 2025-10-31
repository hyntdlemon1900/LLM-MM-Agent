"""
简化的节点定义模块
只返回增量更新，让 LangGraph 自动合并状态
"""

from .state import AgentState, Message


# ============ 基础节点（无需装饰器）============

def start_node(state: AgentState) -> AgentState:
    """起始节点 - 只返回增量"""
    print(f"\n[START] {state.get('name', 'Unknown')}")
    
    # 只返回增量更新
    return {
        'messages': [Message(
            role='system',
            content='Workflow started',
            agent_name='system'
        )]
    }


def end_node(state: AgentState) -> AgentState:
    """结束节点 - 只返回增量"""
    print(f"\n[END] Completed")
    
    # 汇总结果
    result = state.get('result', {})
    if not result:
        result = {
            'status': 'completed',
            'intermediate_results': state.get('intermediate_results', {})
        }
    
    # 只返回增量更新
    return {
        'result': result,
        'next_action': 'end',
        'messages': [Message(
            role='system',
            content='Workflow ended',
            agent_name='system'
        )]
    }


def error_handler_node(state: AgentState) -> AgentState:
    """错误处理节点 - 只返回增量"""
    errors = state.get('errors', [])
    print(f"\n[ERROR] {len(errors)} error(s) occurred")
    for error in errors:
        print(f"  - {error}")
    
    # 只返回增量更新
    return {
        'result': {
            'status': 'failed',
            'errors': errors
        },
        'next_action': 'end'
    }
