# Auto-generated solver with helper function imports
import sys
from pathlib import Path
import copy

# Add problem directory to path (using absolute path)
problem_dir = r'/home/hyn/LLM-MM-Agent-LLMINA-clean/MMBench/problem/LLMINA/runtime'
if problem_dir not in sys.path:
    sys.path.insert(0, problem_dir)

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
    tor_switch_workers = instance['tor_switch_workers']
    
    # Get sorted candidate INA switches
    ina_candidates = get_ina_candidates(network)
    max_workers = max(workers_num)
    num_candidates = len(ina_candidates)
    
    # Initialize output (default: all zeros - REPLACE WITH YOUR SOLUTION)
    ina_placement_expanded = [0] * num_candidates
    y_j_w_s_full = [[[0] * num_candidates for _ in range(max_workers)] for _ in range(jobs_num)]
    y_j_w_d = [[0] * max_workers for _ in range(jobs_num)]
    jobs_routing_expanded = [y_j_w_s_full, y_j_w_d]
    
    # --------------------------
    # TASK 1: Candidate Switch Identification and Greedy Placement Score Computation
    # --------------------------
    
    # Step 1: Build worker connectivity per candidate switch
    # For each candidate switch, count total number of workers connected via TORs
    worker_connectivity = {}
    for s in ina_candidates:
        total_workers = 0
        for j in range(jobs_num):
            for tor_switch, count in tor_switch_workers[j].items():
                if tor_switch == s:
                    total_workers += count
        worker_connectivity[s] = total_workers
    
    # Step 2: Compute network centrality (proximity to workers and PSs)
    # Use shortest path distances from switch s to all connected workers and PSs
    centrality_score = {}
    for s in ina_candidates:
        total_distance = 0.0
        count = 0
        # For each job, get workers and PS
        for j in range(jobs_num):
            # Find all workers connected to this switch via TORs
            job_workers = []
            for tor_switch, count in tor_switch_workers[j].items():
                if tor_switch == s:
                    job_workers.extend([w for w in workers_id[j]])
            
            # Add distance to each worker
            for w in job_workers:
                if w in network.allPathDict[s]:
                    path = network.allPathDict[s][w]
                    hops = len(path) - 1
                    total_distance += hops
                    count += 1
            
            # Add distance to PS
            if s in network.allPathDict[ps_id[j]]:
                path = network.allPathDict[s][ps_id[j]]
                hops = len(path) - 1
                total_distance += hops
                count += 1
        
        # Compute average distance (lower = better centrality)
        avg_distance = total_distance / count if count > 0 else float('inf')
        centrality_score[s] = 1 / (1 + avg_distance)  # Inverse to make higher = better
    
    # Step 3: Compute bottleneck reduction potential
    # Sum of m_j * (number of workers in job j connected to switch s)
    bottleneck_potential = {}
    for s in ina_candidates:
        total_volume = 0.0
        for j in range(jobs_num):
            workers_in_job_connected = 0
            for tor_switch, count in tor_switch_workers[j].items():
                if tor_switch == s:
                    workers_in_job_connected += count
            total_volume += jobs_size[j] * workers_in_job_connected
        bottleneck_potential[s] = total_volume
    
    # Step 4: Normalize each component and combine into composite score
    # Normalize worker connectivity
    min_conn = min(worker_connectivity.values())
    max_conn = max(worker_connectivity.values())
    if max_conn == min_conn:
        norm_conn = {s: 0.5 for s in ina_candidates}
    else:
        norm_conn = {s: (worker_connectivity[s] - min_conn) / (max_conn - min_conn) for s in ina_candidates}
    
    # Normalize centrality score
    min_cent = min(centrality_score.values())
    max_cent = max(centrality_score.values())
    if max_cent == min_cent:
        norm_cent = {s: 0.5 for s in ina_candidates}
    else:
        norm_cent = {s: (centrality_score[s] - min_cent) / (max_cent - min_cent) for s in ina_candidates}
    
    # Normalize bottleneck potential
    min_bottleneck = min(bottleneck_potential.values())
    max_bottleneck = max(bottleneck_potential.values())
    if max_bottleneck == min_bottleneck:
        norm_bottleneck = {s: 0.5 for s in ina_candidates}
    else:
        norm_bottleneck = {s: (bottleneck_potential[s] - min_bottleneck) / (max_bottleneck - min_bottleneck) for s in ina_candidates}
    
    # Combine scores with equal weights
    composite_score = {}
    for s in ina_candidates:
        composite_score[s] = 0.33 * norm_conn[s] + 0.33 * norm_cent[s] + 0.34 * norm_bottleneck[s]
    
    # Sort candidates by composite score (descending)
    sorted_candidates = sorted(ina_candidates, key=lambda s: composite_score[s], reverse=True)
    
    # Select top-K switches
    selected_switches = set(sorted_candidates[:K])
    
    # Map selected switches to placement vector (sorted by ID)
    for idx, s in enumerate(ina_candidates):
        if s in selected_switches:
            ina_placement_expanded[idx] = 1
    
    # --------------------------
    # TASK 2-6: Placeholder Implementations (Will be optimized in future tasks)
    # --------------------------
    # Task 2: Simple greedy assignment (to INA if available, else PS)
    for j in range(jobs_num):
        for w in range(workers_num[j]):
            # Try to assign to first INA switch that is selected
            assigned = False
            for s_idx, s in enumerate(ina_candidates):
                if ina_placement_expanded[s_idx] == 1:
                    y_j_w_s_full[j][w][s_idx] = 1
                    assigned = True
                    break
            if not assigned:
                y_j_w_d[j][w] = 1
    
    # Task 3-6: No refinement yet, just return the initial solution
    # These will be implemented in subsequent tasks
    
    return ina_placement_expanded, jobs_routing_expanded