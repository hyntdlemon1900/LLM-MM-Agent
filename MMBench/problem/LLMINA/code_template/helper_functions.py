"""
LLMINA辅助函数库

这个文件包含所有可供llm_solver使用的辅助函数。
这些函数在测试环境中已经可用，无需在solver代码中重新定义。

主要功能：
1. get_ina_candidates: 获取候选INA交换机列表
2. evaluate_completion_time: 使用Gurobi LP评估makespan
3. links_set, path_to_links_dict, l_i: 路径和链路分析工具
4. link_capacity_limitation: 链路容量约束分析
"""

import sys
from pathlib import Path
from typing import List, Dict
import copy

def get_ina_candidates(network) -> List[int]:
    """
    Get candidate INA switch IDs based on topology type.
    
    According to LLMINA formulation, INA can be deployed on switches (set S).
    For FatTree: ToR (edge) + Aggregation + Core switches
    For SpineLeaf: ToR (leaf) + Spine switches
    
    Args:
        network: Network object with switch ID attributes:
            - FatTree: tors_id, aggrs_id, cores_id, all_switches_id
            - SpineLeaf: tors_id, spines_id, all_switches_id
    
    Returns:
        Sorted list of candidate switch IDs (ascending order)
    """
    # Check topology type by examining available attributes
    has_fattree_attrs = (hasattr(network, 'tors_id') and 
                         hasattr(network, 'aggrs_id') and 
                         hasattr(network, 'cores_id'))
    has_spineleaf_attrs = (hasattr(network, 'tors_id') and 
                           hasattr(network, 'spines_id'))
    
    if has_fattree_attrs:
        # FatTree: ToR + Aggregation + Core
        candidates = list(network.tors_id) + list(network.aggrs_id) + list(network.cores_id)
    elif has_spineleaf_attrs:
        # SpineLeaf: ToR (leaf) + Spine
        candidates = list(network.tors_id) + list(network.spines_id)
    elif hasattr(network, 'all_switches_id'):
        # Fallback: use all switches
        candidates = list(network.all_switches_id)
    
    return sorted(candidates)


def links_set(src: int, dst: int, allPathDict: Dict) -> List[tuple]:
    """
    Get all links (edges) along the path from src to dst.
    
    Args:
        src: Source node ID
        dst: Destination node ID
        allPathDict: Dictionary mapping [src][dst] to path [src, ..., dst]
    
    Returns:
        List of link tuples [(node_i, node_i+1), ...]
    """
    path = allPathDict[src][dst]
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def path_to_links_dict(src_set: List[int], dst_set: List[int], allPathDict: Dict) -> Dict[tuple, List[tuple]]:
    """
    Create a dictionary mapping (src, dst) pairs to their link sets.
    
    Args:
        src_set: List of source node IDs
        dst_set: List of destination node IDs
        allPathDict: Dictionary mapping [src][dst] to path
    
    Returns:
        Dict mapping (src, dst) -> list of links
    """
    return {
        (src, dst): links_set(src, dst, allPathDict)
        for src in src_set
        for dst in dst_set
    }


def l_i(edge: tuple, src_set: List[int], dst_set: List[int], allPathDict: Dict) -> List[tuple]:
    """
    Find all (src, dst) pairs whose paths contain edge e.
    
    Args:
        edge: Edge tuple (u, v)
        src_set: List of source node IDs
        dst_set: List of destination node IDs
        allPathDict: Dictionary mapping [src][dst] to path
    
    Returns:
        List of (row_index, col_index) tuples indicating which (src, dst) pairs use edge e
    """
    path_links_dict = path_to_links_dict(src_set, dst_set, allPathDict)
    edge_reversed = edge[::-1]
    
    index_set = []
    for (src, dst), links in path_links_dict.items():
        if edge in links or edge_reversed in links:
            row_idx = src_set.index(src)
            col_idx = dst_set.index(dst)
            index_set.append((row_idx, col_idx))
    
    return index_set


def link_capacity_limitation(job_idx: int, edge: tuple, workers_id: List[List[int]], 
                            ina_id: List[int], ps_id: List[int], allPathDict: Dict) -> tuple:
    """
    Identify which flows use edge e for job j.
    
    Args:
        job_idx: Job index
        edge: Edge tuple (u, v)
        workers_id: Worker IDs per job [jobs_num][workers_num[j]]
        ina_id: List of selected INA switch IDs
        ps_id: Parameter server IDs per job [jobs_num]
        allPathDict: Dictionary mapping [src][dst] to path
    
    Returns:
        Tuple of (i_j_w_s, i_j_w_d, i_s_d):
        - i_j_w_s: Worker-to-INA flows using edge e
        - i_j_w_d: Worker-to-PS flows using edge e
        - i_s_d: INA-to-PS flows using edge e
    """
    workers_to_ina = l_i(edge, workers_id[job_idx], ina_id, allPathDict)
    workers_to_ps = l_i(edge, workers_id[job_idx], [ps_id[job_idx]], allPathDict)
    ina_to_ps = l_i(edge, ina_id, [ps_id[job_idx]], allPathDict)
    
    return workers_to_ina, workers_to_ps, ina_to_ps

def evaluate_completion_time(
    instance: dict, 
    network, 
    ina_placement: List[int], 
    jobs_routing: List, 
    verbose: bool = False,
    ina_capacity = 750,
    ps_bandwidth = 200,
) -> float:
    """
    Evaluate makespan for given INA placement and routing decisions.
    
    Solves a reduced LP using Gurobi to compute optimal transmission rates.
    
    Args:
        instance: Dict with workers_id, ps_id, workers_num, jobs_size
        network: Object with G, allPathDict, bandwidth_mapping
        ina_placement: Binary vector [len(ina_candidates)], aligned to sorted candidates
        jobs_routing: [y_j_w_s_full, y_j_w_d] where
            - y_j_w_s_full: [jobs_num][max_workers][len(ina_candidates)]
            - y_j_w_d: [jobs_num][max_workers]
        verbose: Print detailed info
    
    Returns:
        Makespan (float), or inf if infeasible
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as e:
        if verbose:
            print(f"Gurobi import error: {e}")
        return float('inf')
    
    try:
        # Setup
        ina_num = sum(ina_placement)
        jobs_num = len(instance['jobs_size'])        
        workers_id = instance['workers_id']
        ps_id = instance['ps_id']
        workers_num = instance['workers_num']
        jobs_size = instance['jobs_size']
        workers_num_align = max(workers_num)
        
        G = copy.deepcopy(network.G)
        mapping = copy.deepcopy(network.mapping) if hasattr(network, 'mapping') else {}
        allPathDict = copy.deepcopy(network.allPathDict)
        sd_id = [list(G.adj._atlas[ps_id[j]].keys())[0] for j in range(jobs_num)]
        
        # Normalize inputs
        ina_candidates = get_ina_candidates(network)
        if len(ina_placement) == len(ina_candidates):
            # Expanded format - convert to compact
            vals = list(ina_placement)
            idx_sorted = sorted(range(len(vals)), key=lambda i: (vals[i], ina_candidates[i]), reverse=True)
            chosen_idx = idx_sorted[:ina_num]
            chosen_idx_sorted_by_id = sorted(chosen_idx, key=lambda i: ina_candidates[i])
            ina_id = [ina_candidates[i] for i in chosen_idx_sorted_by_id]
            
            # Convert routing to compact format
            y_full, y_d = jobs_routing
            max_workers_val = len(y_d[0]) if y_d else 0
            y_compact = [
                [[y_full[j][w][i] for i in chosen_idx_sorted_by_id] for w in range(max_workers_val)]
                for j in range(jobs_num)
            ]
            jobs_routing = [y_compact, y_d]
        else:
            ina_id = sorted(list(ina_placement))
        
        # Create model
        m = gp.Model()
        m.Params.LogToConsole = 0
        
        # Add variables
        gamma_j = m.addMVar(shape=(jobs_num,), vtype=GRB.CONTINUOUS)
        gamma_j_s_d = m.addMVar(shape=(jobs_num, ina_num), vtype=GRB.CONTINUOUS)
        gamma_j_w_s = m.addMVar(shape=(jobs_num, workers_num_align, ina_num), vtype=GRB.CONTINUOUS)
        gamma_j_w_d = m.addMVar(shape=(jobs_num, workers_num_align), vtype=GRB.CONTINUOUS)
        x_s = m.addMVar(shape=(ina_num,), vtype=GRB.BINARY)
        obj = m.addVar(vtype=GRB.CONTINUOUS)
        
        # Set objective
        m.setObjective(obj, GRB.MAXIMIZE)
        
        y_j_w_s = jobs_routing[0]
        y_j_w_d = jobs_routing[1]
        
        # Add constraints
        for j in range(jobs_num):
            for w in range(workers_num_align):
                for s in range(ina_num):
                    m.addConstr(gamma_j_w_s[j, w, s] <= ina_capacity * y_j_w_s[j][w][s])
                m.addConstr(gamma_j_w_d[j, w] <= ps_bandwidth * y_j_w_d[j][w])
        
        for j in range(jobs_num):
            for w in range(workers_num[j], workers_num_align):
                for s in range(ina_num):
                    m.addConstr(gamma_j_w_s[j, w, s] == 0)
                m.addConstr(gamma_j_w_d[j, w] == 0)
        
        m.addConstrs((gamma_j_w_d[j, w] + sum(gamma_j_w_s[j, w, _] for _ in range(ina_num))
                      >= gamma_j[j] for j in range(jobs_num) for w in range(workers_num[j])))
        
        m.addConstrs(obj <= gamma_j[j] / jobs_size[j] for j in range(jobs_num))
        
        m.addConstr(sum(x_s[i] for i in range(ina_num)) <= ina_num)
        m.addConstrs(x_s[i] == 1 for i in range(ina_num))
        
        for j in range(jobs_num):
            for w in range(workers_num_align):
                for s in range(ina_num):
                    m.addConstr(gamma_j_w_s[j, w, s] <= ina_capacity * y_j_w_s[j][w][s] * x_s[s])
        
        m.addConstrs((sum(gamma_j_w_s[j, w, i] for j in range(jobs_num) for w in range(workers_num[j])) 
                      <= ina_capacity * x_s[i] for i in range(ina_num)))
        
        m.addConstrs((gamma_j_w_s[j, w, i] <= gamma_j_s_d[j, i] 
                      for j in range(jobs_num) for w in range(workers_num[j]) for i in range(ina_num)))
        
        # Link capacity constraints
        for link_info in mapping.values():
            edge = link_info[0]
            link_capacity = link_info[1]
            
            link_loads = [0] * jobs_num
            for j in range(jobs_num):
                workers_to_ina, workers_to_ps, ina_to_ps = link_capacity_limitation(
                    j, edge, workers_id, ina_id, ps_id, allPathDict
                )
                
                load_w_to_ina = sum(gamma_j_w_s[j, idx[0], idx[1]] for idx in workers_to_ina)
                load_w_to_ps = sum(gamma_j_w_d[j, idx[0]] for idx in workers_to_ps)
                load_ina_to_ps = sum(gamma_j_s_d[j, idx[0]] for idx in ina_to_ps)
                
                link_loads[j] = load_w_to_ina + load_w_to_ps + load_ina_to_ps
                
                if edge == (sd_id[j], ps_id[j]) or edge == (ps_id[j], sd_id[j]):
                    link_capacity = ps_bandwidth
            
            m.addConstr(sum(link_loads) <= link_capacity)
        
        m.optimize()
        
        if m.status == GRB.OPTIMAL:
            makespan_result = 1 / obj.X
            return makespan_result
        else:
            if verbose:
                print(f"Optimization not optimal. Status: {m.status}")
            return float('inf')
            
    except (gp.GurobiError, ValueError, KeyError) as e:
        if verbose:
            print(f"Optimization error: {e}")
        return float('inf')
