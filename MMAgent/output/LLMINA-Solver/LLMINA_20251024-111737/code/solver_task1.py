# Auto-generated solver with helper function imports
import sys
from pathlib import Path

# Import helper functions from evaluation module
try:
    from MMAgent.evaluation import (
        get_ina_candidates,
        evaluate_completion_time,
        links_set,
        path_to_links_dict,
        l_i,
        link_capacity_limitation
    )
except ModuleNotFoundError:
    # Fallback for different import contexts
    from evaluation import (
        get_ina_candidates,
        evaluate_completion_time,
        links_set,
        path_to_links_dict,
        l_i,
        link_capacity_limitation
    )


def llm_solver(instance, network, K, jobs_num, Cs, topo_name):
    """
    LLMINA solver interface.
    
    Args:
        instance: dict
            - workers_id: List[List[int]] - Worker node IDs, shape [jobs_num][workers_num[j]]
              Example: [[0,1,2], [3,4,5,6]] means job 0 has workers 0,1,2
            - ps_id: List[int] - Parameter Server IDs, shape [jobs_num]
            - workers_num: List[int] - Worker counts per job, shape [jobs_num]
            - jobs_size: List[float] - Gradient volumes (m_j), shape [jobs_num]
            - tor_switch_workers: List[Dict[int,int]] - Worker distribution per tor switch
              Shape [jobs_num][edge_switch_id -> worker_count]
        
        network: object
            - G: networkx.Graph - Network topology
            - allPathDict: Dict[src][dst] -> List[int] - Precomputed paths [src, ..., dst]
              Usage: path = network.allPathDict[src][dst], hops = len(path) - 1
            - bandwidth_mapping: Dict[(u,v)] -> float - Link capacity C_e^bw (Gbps)
            - Switch ID sets:
              * tors_id: Edge switches (connect to workers, edge switches in fattree, leaf switches in spineleaf)
              * aggrs_id: Aggregation switches (FatTree only)
              * cores_id: Core switches (FatTree only)
              * spines_id: Spine switches (SpineLeaf only)
              * all_switches_id: All switches
            - Topology: FatTree (workers->edges->aggrs->cores), SpineLeaf (workers->leafs->spines)
        
        K: int - INA budget (corresponds to K in problem formulation)
        jobs_num: int - Number of concurrent jobs
        Cs: float - INA processing capacity C_s (e.g., 750 Gbps)
        topo_name: str - 'FatTree' or 'SpineLeaf'
    
    Returns:
        (ina_placement_expanded, jobs_routing_expanded):
        
        ina_placement_expanded: List[int]
            Binary vector, length len(ina_candidates), candidates sorted by ID
            Value 1 = selected, 0 = not selected (corresponds to x_s)
            Constraint: sum(ina_placement_expanded) == K
        
        jobs_routing_expanded: [y_j_w_s_full, y_j_w_d]
            y_j_w_s_full: List[List[List[int]]]
                Shape [jobs_num][max_workers][len(ina_candidates)]
                y_j_w_s_full[j][w][s]=1: worker w of job j routes to INA s (y_{jws})
            
            y_j_w_d: List[List[int]]
                Shape [jobs_num][max_workers]
                y_j_w_d[j][w]=1: worker w of job j routes directly to PS (y_{jw d_j})
            
            max_workers = max(workers_num)
            Constraints:
            - Active (w < workers_num[j]): sum(y_j_w_s_full[j][w]) + y_j_w_d[j][w] == 1
            - Inactive (w >= workers_num[j]): all entries = 0 (zero-padding)
    
    """
    
    # Extract instance data
    workers_id = instance['workers_id']
    ps_id = instance['ps_id']
    workers_num = instance['workers_num']
    jobs_size = instance['jobs_size']
    
    # Get sorted candidate INA switches
    ina_candidates = get_ina_candidates(network)
    max_workers = max(workers_num)
    num_candidates = len(ina_candidates)
    
    # Initialize output (default: all zeros - REPLACE WITH YOUR SOLUTION)
    ina_placement_expanded = [0] * num_candidates
    y_j_w_s_full = [[[0] * num_candidates for _ in range(max_workers)] for _ in range(jobs_num)]
    y_j_w_d = [[0] * max_workers for _ in range(jobs_num)]
    jobs_routing_expanded = [y_j_w_s_full, y_j_w_d]
    
    # ================================
    # Task 1: INA Placement Selection (Greedy Heuristic)
    # ================================
    
    # Initialize tracking
    selected = set()
    current_load = {edge: 0.0 for edge in network.bandwidth_mapping}
    total_workers_per_switch = {}
    total_volume_per_switch = {}
    
    # Precompute worker connections to switches
    for j in range(jobs_num):
        for w in range(workers_num[j]):
            worker_node = workers_id[j][w]
            # Find which switch this worker is connected to
            connected_switch = None
            for switch_id in network.tors_id:
                if worker_node in network.G.adj[switch_id]:
                    connected_switch = switch_id
                    break
            # If not found in edge switches, check aggregation/core in FatTree
            if connected_switch is None and hasattr(network, 'aggrs_id'):
                for switch_id in network.aggrs_id:
                    if worker_node in network.G.adj[switch_id]:
                        connected_switch = switch_id
                        break
            if connected_switch is None and hasattr(network, 'cores_id'):
                for switch_id in network.cores_id:
                    if worker_node in network.G.adj[switch_id]:
                        connected_switch = switch_id
                        break
            if connected_switch is None and hasattr(network, 'leafs_id'):
                for switch_id in network.leafs_id:
                    if worker_node in network.G.adj[switch_id]:
                        connected_switch = switch_id
                        break
            if connected_switch is None and hasattr(network, 'spines_id'):
                for switch_id in network.spines_id:
                    if worker_node in network.G.adj[switch_id]:
                        connected_switch = switch_id
                        break
            
            if connected_switch is not None:
                if connected_switch not in total_workers_per_switch:
                    total_workers_per_switch[connected_switch] = 0
                if connected_switch not in total_volume_per_switch:
                    total_volume_per_switch[connected_switch] = 0
                total_workers_per_switch[connected_switch] += 1
                total_volume_per_switch[connected_switch] += jobs_size[j]
    
    # Greedy selection of K switches
    for _ in range(K):
        best_candidate = -1
        best_score = -float('inf')
        
        for idx, switch_id in enumerate(ina_candidates):
            if switch_id in selected:
                continue
            
            # Skip if this switch would exceed processing capacity
            # Worst-case: all workers connected to it send at full rate
            if switch_id in total_volume_per_switch and total_volume_per_switch[switch_id] > Cs:
                continue
            
            # Compute benefit score
            N_s = total_workers_per_switch.get(switch_id, 0)
            M_s = total_volume_per_switch.get(switch_id, 0)
            
            # Compute congestion alleviation: links that would be used
            # Simulate all flows passing through any link connected to switch_id
            link_congestion = 0.0
            for neighbor in network.G.adj[switch_id]:
                edge = tuple(sorted([switch_id, neighbor]))
                capacity = network.bandwidth_mapping.get(edge, 0.0)
                if capacity == 0:
                    continue
                # Current load on this link (excluding the switch being considered)
                # But we use the current global load as proxy
                load = current_load.get(edge, 0.0)
                if capacity > 0:
                    utilization = load / capacity
                    # Penalize high utilization: 1 - utilization gives more benefit when idle
                    link_congestion += (1 - utilization)
            
            # Normalize weights
            alpha, beta, gamma = 1.0, 1.0, 1.0
            score = alpha * N_s + beta * M_s + gamma * link_congestion
            
            if score > best_score:
                best_score = score
                best_candidate = idx
        
        if best_candidate == -1:
            break  # No valid candidate found
        
        # Select best candidate
        switch_id = ina_candidates[best_candidate]
        selected.add(switch_id)
        ina_placement_expanded[best_candidate] = 1
        
        # Update load estimates (simulated: add max possible traffic from this switch)
        for neighbor in network.G.adj[switch_id]:
            edge = tuple(sorted([switch_id, neighbor]))
            if edge not in current_load:
                current_load[edge] = 0.0
            # Add worst-case volume
            volume = total_volume_per_switch.get(switch_id, 0)
            current_load[edge] += volume
    
    # ================================
    # Task 2-5: Placeholder for Future Tasks
    # ================================
    # TODO: Will be optimized in Task 2 - Routing Assignment
    # TODO: Will be optimized in Task 3 - Flow Conservation Enforcement
    # TODO: Will be optimized in Task 4 - Local Search Refinement
    # TODO: Will be optimized in Task 5 - Constraint Validation & Finalization
    
    # For now, assign all workers directly to PS (fallback)
    for j in range(jobs_num):
        for w in range(workers_num[j]):
            y_j_w_d[j][w] = 1
    
    return ina_placement_expanded, jobs_routing_expanded