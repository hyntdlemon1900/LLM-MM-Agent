# Auto-generated solver with helper function imports
import sys
from pathlib import Path
import copy

# Add problem directory to path
problem_dir = Path(__file__).parent.parent.parent / 'MMBench' / 'problem' / 'LLMINA'
sys.path.insert(0, str(problem_dir))

# Import helper functions from user's evaluation module
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
    
    # === TASK 1: Greedy INA Switch Selection Based on Load-Capacity Proxy Scoring and Bottleneck Proximity ===
    
    # Define congestion penalty weight
    eta = 1.0  # Tunable parameter to emphasize bottleneck avoidance
    
    # Compute gradient sending rate per worker
    # Assume uniform rate per worker within a job: jobs_size[j] / workers_num[j]
    gamma = {}
    for j in range(jobs_num):
        rate_per_worker = jobs_size[j] / workers_num[j]
        for w in range(workers_num[j]):
            gamma[(j, w)] = rate_per_worker
    
    # Score each candidate INA switch
    scores = {}
    for idx, s in enumerate(ina_candidates):
        total_incoming_rate = 0.0
        max_link_utilization = 0.0
        
        # Loop over all jobs and workers to compute total incoming traffic to switch s
        for j in range(jobs_num):
            d_j = ps_id[j]
            # Check if path exists from switch s to PS d_j
            if s not in network.allPathDict or d_j not in network.allPathDict[s]:
                continue  # Skip if no path exists
            path = network.allPathDict[s][d_j]
            
            # Estimate total incoming rate from workers whose path to PS goes through s
            for w in range(workers_num[j]):
                worker_id = workers_id[j][w]
                # Check if worker has a valid path to PS
                if worker_id not in network.allPathDict or d_j not in network.allPathDict[worker_id]:
                    continue
                worker_to_ps_path = network.allPathDict[worker_id][d_j]
                # Check if s is in the path from worker to PS
                if s in worker_to_ps_path:
                    total_incoming_rate += gamma[(j, w)]
            
            # Compute max link utilization along path from s to PS
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                link_key = (u, v) if (u, v) in network.bandwidth_mapping else (v, u)
                capacity = network.bandwidth_mapping[link_key]
                # Approximate load: assume all job traffic from this job flows through this link
                # Conservative: assume full job rate goes through this link
                load = jobs_size[j]
                utilization = load / capacity if capacity > 0 else 0
                max_link_utilization = max(max_link_utilization, utilization)
        
        # Compute congestion penalty term
        congestion_penalty = 1 + eta * max_link_utilization
        
        # Compute score: total incoming rate / (C_s * congestion_penalty)
        if Cs > 0:
            score = total_incoming_rate / (Cs * congestion_penalty)
        else:
            score = 0.0
        
        scores[s] = score
    
    # Sort candidates by score in descending order
    sorted_candidates = sorted(ina_candidates, key=lambda s: scores[s], reverse=True)
    
    # Select top K switches
    selected_indices = []
    for idx, s in enumerate(sorted_candidates):
        if len(selected_indices) >= K:
            break
        selected_indices.append(idx)
    
    # Create binary indicator vector
    ina_placement_expanded = [0] * num_candidates
    for idx in selected_indices:
        s = sorted_candidates[idx]
        # Find the position of s in the original ina_candidates list
        pos = ina_candidates.index(s)
        ina_placement_expanded[pos] = 1
    
    # === TASK 2-5: Placeholder Implementations (Will be optimized in future tasks) ===
    # For now, assign all workers to PS (dummy assignment)
    for j in range(jobs_num):
        for w in range(workers_num[j]):
            y_j_w_d[j][w] = 1
        # Zero-pad inactive workers
        for w in range(workers_num[j], max_workers):
            y_j_w_d[j][w] = 0
            for s in range(num_candidates):
                y_j_w_s_full[j][w][s] = 0
    
    # Return the result
    return ina_placement_expanded, jobs_routing_expanded