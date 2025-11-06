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
    
    # ------------------------------
    # Task 1: Greedy Initialization of INA Placement Based on Topology-Aware Centrality and Aggregation Potential
    # ------------------------------
    
    # Define weights for the composite score
    w1, w2, w3 = 0.5, 0.3, 0.2  # Centrality, Worker Connectivity, Aggregation Gain
    
    # Dictionary to store score for each candidate switch
    switch_scores = {}
    
    # Iterate over each candidate switch
    for s_idx, s in enumerate(ina_candidates):
        score = 0.0
        
        # 1. Topological Centrality: Number of edge switches connected to s, weighted by inverse of average path length to workers
        centrality = 0.0
        total_path_length = 0.0
        worker_count = 0
        
        # Determine which edge switches are connected to s
        connected_tors = []
        if topo_name == 'FatTree':
            # In FatTree, edge switches are tors_id
            if s in network.tors_id or s in network.aggrs_id or s in network.cores_id:
                # For each edge switch (tor) connected to s
                for tor in network.tors_id:
                    if s in network.G.neighbors(tor):
                        connected_tors.append(tor)
        else:  # SpineLeaf
            # In SpineLeaf, edge switches are tors_id (leaf switches)
            if s in network.tors_id or s in network.spines_id:
                for tor in network.tors_id:
                    if s in network.G.neighbors(tor):
                        connected_tors.append(tor)
        
        # Compute average path length from s to all workers via connected edge switches
        for tor in connected_tors:
            for j in range(jobs_num):
                # For each worker in job j, check if any worker is on this tor
                for w in workers_id[j]:
                    # Assume worker is directly connected to tor
                    if w in tor_switch_workers[j] and tor_switch_workers[j][w] > 0:
                        # Get path from tor to s
                        if tor in network.allPathDict and s in network.allPathDict[tor]:
                            path = network.allPathDict[tor][s]
                            path_length = len(path) - 1
                            total_path_length += path_length
                            worker_count += 1
        
        if worker_count > 0:
            avg_path_length = total_path_length / worker_count
            centrality = len(connected_tors) / (avg_path_length + 1e-6)  # Avoid division by zero
        else:
            centrality = 0.0
        
        # 2. Worker Connectivity: Total number of workers directly connected to s via edge switches
        worker_connectivity = 0
        for j in range(jobs_num):
            for tor in connected_tors:
                if tor in tor_switch_workers[j]:
                    worker_connectivity += tor_switch_workers[j][tor]
        
        # 3. Aggregation Gain Potential: Sum of gradient volumes for jobs with workers on connected edge switches
        aggregation_gain = 0.0
        for j in range(jobs_num):
            for tor in connected_tors:
                if tor in tor_switch_workers[j] and tor_switch_workers[j][tor] > 0:
                    aggregation_gain += jobs_size[j]
        
        # Compute composite score
        score = w1 * centrality + w2 * worker_connectivity + w3 * aggregation_gain
        switch_scores[s] = score
    
    # Sort candidates by score in descending order
    sorted_candidates = sorted(ina_candidates, key=lambda s: switch_scores[s], reverse=True)
    
    # Select top-K switches
    selected_switches = set(sorted_candidates[:K])
    
    # Build binary placement vector in sorted candidate order
    for i, s in enumerate(ina_candidates):
        ina_placement_expanded[i] = 1 if s in selected_switches else 0
    
    # ------------------------------
    # Placeholder for Tasks 2-6 (to be implemented later)
    # ------------------------------
    # Task 2: Greedy Assignment of Workers to INA or PS (Congestion-Aware)
    # TODO: Will be optimized in Task 2
    
    # Task 3: Rate Allocation via LP Relaxation
    # TODO: Will be optimized in Task 3
    
    # Task 4: Iterative Refinement via Local Search (Swap-based)
    # TODO: Will be optimized in Task 4
    
    # Task 5: Feedback Loop for Placement-Routing Co-Optimization
    # TODO: Will be optimized in Task 5
    
    # Task 6: Final Solution Validation and Output Formatting
    # TODO: Will be optimized in Task 6
    
    # For now, assign all workers to PS as a placeholder
    for j in range(jobs_num):
        for w in range(workers_num[j]):
            y_j_w_d[j][w] = 1
    
    return ina_placement_expanded, jobs_routing_expanded