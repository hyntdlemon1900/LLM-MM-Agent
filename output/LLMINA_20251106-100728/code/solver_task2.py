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
    
    # Get sorted candidate INA switches
    ina_candidates = get_ina_candidates(network)
    max_workers = max(workers_num)
    num_candidates = len(ina_candidates)
    
    # Initialize output (default: all zeros - REPLACE WITH YOUR SOLUTION)
    ina_placement_expanded = [0] * num_candidates
    y_j_w_s_full = [[[0] * num_candidates for _ in range(max_workers)] for _ in range(jobs_num)]
    y_j_w_d = [[0] * max_workers for _ in range(jobs_num)]
    jobs_routing_expanded = [y_j_w_s_full, y_j_w_d]
    
    # === TASK 1: Compute Candidate INA Switch Scores Based on Topology-Aware Centrality and Load Potential ===
    # Initialize score dictionary for each candidate switch
    switch_scores = {}
    
    # Define weight factors for composite score (tunable)
    weight_central = 0.5
    weight_volume = 0.3
    weight_proximity = 0.2
    
    # Process each candidate switch
    for switch_id in ina_candidates:
        total_worker_weighted_count = 0.0
        total_hop_count = 0.0
        total_worker_volume = 0.0
        worker_count = 0
        
        # Iterate over all jobs
        for job_idx in range(jobs_num):
            job_workers = workers_id[job_idx]
            job_size = jobs_size[job_idx]
            
            # Check if this switch is connected to any workers in this job
            # Use tor_switch_workers to map workers to switches
            tor_switch_data = instance['tor_switch_workers'][job_idx]
            connected_workers = tor_switch_data.get(switch_id, 0)
            
            if connected_workers > 0:
                # Add weighted worker count: m_j * #workers
                total_worker_weighted_count += job_size * connected_workers
                worker_count += connected_workers
                
                # Compute average hop count from switch to PS
                avg_hop_count = 0.0
                valid_paths = 0
                
                for worker in job_workers:
                    # Find path from worker to PS
                    if switch_id in network.allPathDict.get(worker, {}):
                        path = network.allPathDict[worker][switch_id]
                        switch_to_ps_path = network.allPathDict[switch_id][ps_id[job_idx]]
                        
                        if switch_to_ps_path:
                            total_hops = len(path) + len(switch_to_ps_path) - 2  # Avoid double-counting switch
                            avg_hop_count += total_hops
                            valid_paths += 1
                
                if valid_paths > 0:
                    avg_hop_count /= valid_paths
                    total_hop_count += avg_hop_count
                
                # Estimate maximum possible gradient sending rate (gamma_jw) for each connected worker
                # Assume full rate for worst-case load estimation
                for worker in job_workers:
                    if switch_id in network.allPathDict.get(worker, {}):
                        # Use a proxy rate: job_size / min(1, hop_count) as a simplification
                        hop_count = len(network.allPathDict[worker][switch_id]) - 1
                        rate_proxy = job_size / max(1.0, hop_count + 1)  # Avoid division by zero
                        total_worker_volume += rate_proxy
        
        # Compute centrality score: weighted sum of worker count (weighted by job size) and proximity
        centrality_score = total_worker_weighted_count
        
        # Proximity score: inverse of average hop count (lower is better)
        avg_hop_count = total_hop_count / worker_count if worker_count > 0 else float('inf')
        proximity_score = 1.0 / (avg_hop_count + 1e-6)  # Avoid division by zero
        
        # Volume score: total estimated load volume
        volume_score = total_worker_volume
        
        # Capacity-aware penalty: if estimated load exceeds capacity, penalize
        load_ratio = total_worker_volume / Cs if Cs > 0 else 0
        capacity_penalty = load_ratio * 0.5  # Scale penalty to avoid over-penalization
        
        # Composite score: weighted sum
        composite_score = (
            weight_central * centrality_score +
            weight_volume * volume_score +
            weight_proximity * proximity_score
        )
        
        # Apply penalty for high load
        composite_score -= capacity_penalty
        
        # Store score
        switch_scores[switch_id] = composite_score
    
    # Normalize scores across all candidates (optional but useful for comparison)
    if switch_scores:
        min_score = min(switch_scores.values())
        max_score = max(switch_scores.values())
        if max_score > min_score:
            for switch_id in ina_candidates:
                switch_scores[switch_id] = (switch_scores[switch_id] - min_score) / (max_score - min_score)
        else:
            # All scores are equal
            for switch_id in ina_candidates:
                switch_scores[switch_id] = 1.0
    
    # === TASK 2: Greedy Top-K Selection of INA Switches ===
    # Sort candidates by score in descending order and select top K
    candidate_scores = [(switch_id, switch_scores[switch_id]) for switch_id in ina_candidates]
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select top K switches
    selected_switches = set()
    for i in range(min(K, len(candidate_scores))):
        switch_id = candidate_scores[i][0]
        selected_switches.add(switch_id)
    
    # Build expanded placement vector
    for idx, switch_id in enumerate(ina_candidates):
        if switch_id in selected_switches:
            ina_placement_expanded[idx] = 1
    
    # === TASK 3-7: Placeholder Implementations (to be optimized in future tasks) ===
    # For now, assign all workers to PS (baseline)
    # This will be improved in later tasks
    for j in range(jobs_num):
        for w in range(workers_num[j]):
            y_j_w_d[j][w] = 1
    
    # TODO: Will be optimized in Task 3 (Assignment Phase)
    # TODO: Will be optimized in Task 4 (Rate Balancing)
    # TODO: Will be optimized in Task 5 (Iterative Refinement)
    # TODO: Will be optimized in Task 6 (Constraint Enforcement)
    # TODO: Will be optimized in Task 7 (Final Optimization)
    
    return ina_placement_expanded, jobs_routing_expanded