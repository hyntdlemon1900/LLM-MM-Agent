"""
工具集成模块
集成 LLMINA 和其他工具函数，使其符合 LangGraph 规范
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain_core.tools import tool
import sys
from pathlib import Path

# 导入原始工具函数
try:
    from evaluation import (
        get_ina_candidates as _get_ina_candidates,
        evaluate_completion_time as _evaluate_completion_time
    )
except ImportError:
    from MMAgent.evaluation import (
        get_ina_candidates as _get_ina_candidates,
        evaluate_completion_time as _evaluate_completion_time
    )


# ============ LangGraph Tool 装饰器包装 ============

@tool
def get_ina_candidates_tool(network: Any) -> List[int]:
    """
    获取候选 INA 交换机列表
    
    根据拓扑类型返回可部署 INA 的交换机 ID 列表：
    - FatTree: ToR + Aggregation + Core switches
    - SpineLeaf: ToR (leaf) + Spine switches
    
    Args:
        network: 网络对象，包含交换机 ID 属性
        
    Returns:
        排序后的候选交换机 ID 列表
    """
    return _get_ina_candidates(network)


@tool
def evaluate_completion_time_tool(
    network: Any,
    ina_num: int,
    jobs_num: int,
    Cs: int,
    ina_placement: List[int],
    jobs_routing: Dict[int, List[int]]
) -> float:
    """
    使用 Gurobi LP 评估任务完成时间（makespan）
    
    Args:
        network: 网络对象
        ina_num: INA 数量
        jobs_num: 任务数量
        Cs: 交换机容量
        ina_placement: INA 放置方案
        jobs_routing: 任务路由方案
        
    Returns:
        makespan（任务完成时间）
    """
    return _evaluate_completion_time(
        network, ina_num, jobs_num, Cs,
        ina_placement, jobs_routing
    )


# ============ 非工具辅助函数（直接调用） ============

def links_set(src: int, dst: int, allPathDict: Dict) -> List[tuple]:
    """
    获取路径上的所有链路
    
    Args:
        src: 源节点 ID
        dst: 目标节点 ID
        allPathDict: 路径字典 [src][dst] -> path
        
    Returns:
        链路列表 [(node_i, node_i+1), ...]
    """
    path = allPathDict[src][dst]
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def path_to_links_dict(
    src_set: List[int],
    dst_set: List[int],
    allPathDict: Dict
) -> Dict[tuple, List[tuple]]:
    """
    创建 (src, dst) 到链路集合的映射
    
    Args:
        src_set: 源节点 ID 列表
        dst_set: 目标节点 ID 列表
        allPathDict: 路径字典
        
    Returns:
        Dict mapping (src, dst) -> list of links
    """
    return {
        (src, dst): links_set(src, dst, allPathDict)
        for src in src_set
        for dst in dst_set
    }


def l_i(
    edge: tuple,
    src_set: List[int],
    dst_set: List[int],
    allPathDict: Dict
) -> List[tuple]:
    """
    查找经过指定边的所有 (src, dst) 对
    
    Args:
        edge: 边 (u, v)
        src_set: 源节点列表
        dst_set: 目标节点列表
        allPathDict: 路径字典
        
    Returns:
        经过该边的 (src, dst) 对列表
    """
    result = []
    for src in src_set:
        for dst in dst_set:
            path = allPathDict[src][dst]
            links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            if edge in links:
                result.append((src, dst))
    return result


def link_capacity_limitation(
    edge: tuple,
    src_set: List[int],
    dst_set: List[int],
    allPathDict: Dict,
    jobs_routing: Dict[int, List[int]],
    jobs_data_size: List[float]
) -> float:
    """
    计算链路容量限制
    
    Args:
        edge: 边 (u, v)
        src_set: 源节点列表
        dst_set: 目标节点列表
        allPathDict: 路径字典
        jobs_routing: 任务路由方案
        jobs_data_size: 任务数据大小列表
        
    Returns:
        链路负载
    """
    pairs = l_i(edge, src_set, dst_set, allPathDict)
    total_load = 0.0
    
    for job_id, route in jobs_routing.items():
        data_size = jobs_data_size[job_id]
        for src, dst in pairs:
            if src in route and dst in route:
                total_load += data_size
    
    return total_load


# ============ 工具注册表 ============

class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self._tools: Dict[str, Any] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """注册默认工具"""
        self.register("get_ina_candidates", get_ina_candidates_tool)
        self.register("evaluate_completion_time", evaluate_completion_time_tool)
        
        # 非 LangChain 工具
        self.register("links_set", links_set, is_langgraph_tool=False)
        self.register("path_to_links_dict", path_to_links_dict, is_langgraph_tool=False)
        self.register("l_i", l_i, is_langgraph_tool=False)
        self.register("link_capacity_limitation", link_capacity_limitation, is_langgraph_tool=False)
    
    def register(self, name: str, func: Any, is_langgraph_tool: bool = True):
        """
        注册工具
        
        Args:
            name: 工具名称
            func: 工具函数
            is_langgraph_tool: 是否为 LangGraph 工具
        """
        self._tools[name] = {
            'func': func,
            'is_langgraph_tool': is_langgraph_tool
        }
    
    def get_tool(self, name: str) -> Optional[Any]:
        """获取工具"""
        tool_info = self._tools.get(name)
        return tool_info['func'] if tool_info else None
    
    def get_all_tools(self) -> List[Any]:
        """获取所有 LangGraph 工具"""
        return [
            info['func'] for info in self._tools.values()
            if info['is_langgraph_tool']
        ]
    
    def get_all_functions(self) -> Dict[str, Any]:
        """获取所有函数（包括非工具）"""
        return {
            name: info['func']
            for name, info in self._tools.items()
        }
    
    def list_tools(self) -> List[str]:
        """列出所有工具名称"""
        return list(self._tools.keys())


# ============ 全局工具注册表 ============

_global_tool_registry = ToolRegistry()


def get_tool(name: str) -> Optional[Any]:
    """从全局注册表获取工具"""
    return _global_tool_registry.get_tool(name)


def get_all_tools() -> List[Any]:
    """获取所有 LangGraph 工具"""
    return _global_tool_registry.get_all_tools()


def get_all_functions() -> Dict[str, Any]:
    """获取所有函数"""
    return _global_tool_registry.get_all_functions()


def register_tool(name: str, func: Any, is_langgraph_tool: bool = True):
    """注册自定义工具"""
    _global_tool_registry.register(name, func, is_langgraph_tool)


# ============ 工具绑定到 LLM ============

def bind_tools_to_llm(llm: Any) -> Any:
    """
    将工具绑定到 LLM
    
    Args:
        llm: LLM 实例
        
    Returns:
        绑定了工具的 LLM
    """
    tools = get_all_tools()
    
    # 检查 LLM 是否支持工具绑定
    if hasattr(llm, 'bind_tools'):
        return llm.bind_tools(tools)
    else:
        print("[Warning] LLM does not support tool binding")
        return llm


# ============ 工具调用包装器 ============

class ToolExecutor:
    """工具执行器"""
    
    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.registry = registry or _global_tool_registry
    
    def execute(self, tool_name: str, **kwargs) -> Any:
        """
        执行工具
        
        Args:
            tool_name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            工具执行结果
        """
        tool = self.registry.get_tool(tool_name)
        
        if tool is None:
            raise ValueError(f"Tool not found: {tool_name}")
        
        print(f"[Tool] Executing: {tool_name}")
        print(f"  Args: {kwargs}")
        
        try:
            result = tool(**kwargs)
            print(f"  Result: {result}")
            return result
        except Exception as e:
            print(f"  Error: {e}")
            raise
    
    def batch_execute(self, tool_calls: List[Dict[str, Any]]) -> List[Any]:
        """
        批量执行工具
        
        Args:
            tool_calls: 工具调用列表，格式为 [{'name': 'tool_name', 'args': {...}}, ...]
            
        Returns:
            执行结果列表
        """
        results = []
        
        for call in tool_calls:
            name = call['name']
            args = call.get('args', {})
            result = self.execute(name, **args)
            results.append(result)
        
        return results


# ============ 便捷函数 ============

def execute_tool(tool_name: str, **kwargs) -> Any:
    """便捷函数：执行工具"""
    executor = ToolExecutor()
    return executor.execute(tool_name, **kwargs)


def create_tool_node(tool_names: List[str]) -> Any:
    """
    创建工具节点（用于 LangGraph）
    
    Args:
        tool_names: 要包含的工具名称列表
        
    Returns:
        工具节点
    """
    from langgraph.prebuilt import ToolNode
    
    tools = [get_tool(name) for name in tool_names if get_tool(name)]
    return ToolNode(tools)
