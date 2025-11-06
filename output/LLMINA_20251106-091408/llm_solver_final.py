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
    
    # --- TASK 1: Candidate INA Switch Identification and Ranking ---
    # Step 1: For each candidate switch, compute traffic centrality and bottleneck potential
    scores = {}
    
    # Weights for composite score
    weight_central = 0.6
    weight_bottleneck = 0.4
    
    for s in ina_candidates:
        traffic_central_score = 0.0
        bottleneck_potential = 0.0
        
        # Iterate over all jobs
        for j in range(jobs_num):
            # Get workers connected to switch s for this job
            # Use tor_switch_workers to get worker count per switch
            worker_count_dict = instance['tor_switch_workers'][j]
            worker_count = worker_count_dict.get(s, 0)
            if worker_count == 0:
                continue  # No workers connected to this switch for job j
            
            # Get PS ID for job j
            ps = ps_id[j]
            
            # Get shortest path from switch s to PS
            path = network.allPathDict.get(s, {}).get(ps, [])
            if not path:
                # No path found, skip
                continue
            path_hops = len(path) - 1
            
            # Traffic centrality: sum over workers connected to s of (m_j / path_hops)
            # Weight by job size and inverse path length (penalize long paths)
            centrality_contrib = jobs_size[j] / (path_hops + 1e-6)  # avoid division by zero
            traffic_central_score += centrality_contrib * worker_count
            
            # Bottleneck potential: sum of link bandwidths along path
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                link = (u, v)
                bandwidth = network.bandwidth_mapping.get(link, 0.0)
                bottleneck_potential += bandwidth
        
        # Normalize scores to avoid bias from scale
        # Compute composite score
        composite_score = weight_central * traffic_central_score + weight_bottleneck * bottleneck_potential
        scores[s] = composite_score
    
    # Step 2: Sort candidates by composite score in descending order
    ranked_candidates = sorted(ina_candidates, key=lambda s: scores[s], reverse=True)
    
    # Step 3: Select top-K switches
    selected_switches = ranked_candidates[:K]
    
    # Step 4: Create expanded placement vector
    for i, s in enumerate(ina_candidates):
        if s in selected_switches:
            ina_placement_expanded[i] = 1
    
    # --- TASK 2: Greedy INA Placement with Budget Enforcement and Initialization of Switch Activation Flags ---
    # The placement is already complete from Task 1.
    # We now ensure that exactly K switches are active (enforced by selection above).
    # No further changes needed here — placement is valid and budget-compliant.
    
    # --- TASK 3: Per-Worker Assignment via Constrained Max-Flow Based on Aggregation Point Feasibility and Rate Maximization ---
    # Initialize residual capacities
    # Residual processing capacity for each INA switch (indexed by candidate index)
    residual_switch_capacity = [Cs if s in selected_switches else 0 for s in ina_candidates]
    
    # Track current load on each link (u, v) — initialize to 0
    # Use a dictionary with tuple (u, v) as key
    link_load = {}
    
    # Precompute all paths to avoid repeated calls
    # We'll use allPathDict directly
    # For each (src, dst), we have the path — we'll use it to get links
    
    # Create list of all (src, dst) pairs for worker-to-aggregation-point routing
    # Workers: each worker id is a node
    # Aggregation points: active INA switches and PSs
    # We'll process workers one by one in a prioritized order
    
    # Priority: sort workers by job size (descending), then by worker ID (ascending)
    # For each job, we'll process its workers in order
    worker_priority = []
    for j in range(jobs_num):
        for w in range(workers_num[j]):
            worker_priority.append((j, w, jobs_size[j]))
    
    # Sort by job size descending, then by worker index ascending
    worker_priority.sort(key=lambda x: (-x[2], x[1]))
    
    # Now assign each worker
    for j, w, job_size in worker_priority:
        # Current worker: job j, worker w
        # Determine all possible aggregation points: active INA switches + PS
        candidates = []
        # Add active INA switches
        for idx, s in enumerate(ina_candidates):
            if ina_placement_expanded[idx] == 1:
                candidates.append(('ina', idx, s))
        # Add PS
        candidates.append(('ps', None, ps_id[j]))
        
        best_rate = 0.0
        best_aggregation = None  # (type, index, target)
        
        # For each candidate aggregation point
        for cand_type, cand_idx, target in candidates:
            # Determine path from worker w to target
            src = workers_id[j][w]
            dst = target
            
            path = network.allPathDict.get(src, {}).get(dst, [])
            if not path:
                continue  # No path exists
            
            # Compute the path links
            path_links = links_set(src, dst, network.allPathDict)
            
            # Find the minimum available rate along the path
            # First, check link capacity constraints
            min_rate = float('inf')
            for u, v in path_links:
                link = (u, v)
                # Get current load on link
                current_load = link_load.get(link, 0.0)
                capacity = network.bandwidth_mapping.get(link, 0.0)
                available_bandwidth = capacity - current_load
                min_rate = min(min_rate, available_bandwidth)
            
            # If candidate is INA switch, check switch capacity
            if cand_type == 'ina':
                switch_capacity = residual_switch_capacity[cand_idx]
                min_rate = min(min_rate, switch_capacity)
            
            # Update best candidate if this one gives higher rate
            # But we want to maximize the resulting job rate, not just worker rate
            # The job rate is limited by the minimum of all workers' rates
            # So we want to maximize the minimum worker rate across the job
            # Therefore: we maximize the rate r we can assign
            # Because higher rate → better chance to improve the bottleneck
            if min_rate > best_rate:
                best_rate = min_rate
                best_aggregation = (cand_type, cand_idx, target)
        
        # Assign the worker to best aggregation point
        if best_aggregation is None:
            # No feasible assignment — assign to PS as fallback
            y_j_w_d[j][w] = 1
            # Update link loads for worker-to-PS path
            src = workers_id[j][w]
            dst = ps_id[j]
            path = network.allPathDict.get(src, {}).get(dst, [])
            if path:
                path_links = links_set(src, dst, network.allPathDict)
                for u, v in path_links:
                    link = (u, v)
                    link_load[link] = link_load.get(link, 0.0) + (best_rate if best_rate > 0 else 0)
        else:
            cand_type, cand_idx, target = best_aggregation
            if cand_type == 'ina':
                # Assign to INA switch
                y_j_w_s_full[j][w][cand_idx] = 1
                # Update switch residual capacity
                residual_switch_capacity[cand_idx] -= best_rate
                # Update link loads
                src = workers_id[j][w]
                path = network.allPathDict.get(src, {}).get(target, [])
                if path:
                    path_links = links_set(src, target, network.allPathDict)
                    for u, v in path_links:
                        link = (u, v)
                        link_load[link] = link_load.get(link, 0.0) + best_rate
            else:  # PS
                y_j_w_d[j][w] = 1
                # Update link loads
                src = workers_id[j][w]
                path = network.allPathDict.get(src, {}).get(target, [])
                if path:
                    path_links = links_set(src, target, network.allPathDict)
                    for u, v in path_links:
                        link = (u, v)
                        link_load[link] = link_load.get(link, 0.0) + best_rate
    
    # --- TASK 4: Dynamic Load Tracking and Capacity Enforcement via Incremental Path-Based Load Updates ---
    # The load tracking is already handled incrementally during assignment in Task 3.
    # Each assignment updates link_load immediately after assignment.
    # No additional changes needed — the current implementation already enforces capacity constraints
    # via residual bandwidth checks in the min_rate computation.
    
    # --- TASK 5: Job Rate Computation and Bottleneck Job Identification for Objective-Driven Optimization ---
    # Compute effective job rates and completion times
    job_rates = []  # γ_j for each job j
    job_completion_times = []  # t_j = m_j / γ_j
    bottleneck_job_idx = -1
    max_completion_time = 0.0
    
    for j in range(jobs_num):
        per_worker_rates = []
        for w in range(workers_num[j]):
            # Sum the rates from all aggregation paths
            rate = 0.0
            # Check worker-to-INA paths
            for s_idx, s in enumerate(ina_candidates):
                if y_j_w_s_full[j][w][s_idx] == 1:
                    # The rate assigned during assignment is stored implicitly in residual_capacity updates
                    # But we need to recover it from the load tracking and path information
                    # Instead, we recompute based on the path and capacity constraints
                    src = workers_id[j][w]
                    dst = s
                    path = network.allPathDict.get(src, {}).get(dst, [])
                    if not path:
                        continue
                    path_links = links_set(src, dst, network.allPathDict)
                    
                    # The actual rate is the minimum capacity along the path
                    min_rate = float('inf')
                    for u, v in path_links:
                        link = (u, v)
                        current_load = link_load.get(link, 0.0)
                        capacity = network.bandwidth_mapping.get(link, 0.0)
                        available = capacity - current_load
                        min_rate = min(min_rate, available)
                    
                    # Also limited by INA switch capacity
                    if ina_placement_expanded[s_idx] == 1:
                        switch_capacity = Cs
                        # But we don't store the exact residual, so use the rate from assignment
                        # Since we assigned the minimum possible rate, we use best_rate from assignment
                        # But we didn't store it — so we must assume the rate used was the one that caused the reduction
                        # We'll use the min_rate as the best estimate
                    rate += min_rate
            # Check worker-to-PS path
            if y_j_w_d[j][w] == 1:
                src = workers_id[j][w]
                dst = ps_id[j]
                path = network.allPathDict.get(src, {}).get(dst, [])
                if not path:
                    continue
                path_links = links_set(src, dst, network.allPathDict)
                
                min_rate = float('inf')
                for u, v in path_links:
                    link = (u, v)
                    current_load = link_load.get(link, 0.0)
                    capacity = network.bandwidth_mapping.get(link, 0.0)
                    available = capacity - current_load
                    min_rate = min(min_rate, available)
                rate += min_rate
            per_worker_rates.append(rate)
        
        # Job effective rate is the minimum of per-worker rates
        job_rate = min(per_worker_rates) if per_worker_rates else 0.0
        job_rates.append(job_rate)
        
        # Completion time
        completion_time = jobs_size[j] / (job_rate + 1e-6)
        job_completion_times.append(completion_time)
        
        # Update bottleneck job if this one is slower
        if completion_time > max_completion_time:
            max_completion_time = completion_time
            bottleneck_job_idx = j
    
    # The auxiliary variable alpha = 1 / makespan
    # But we don't return it; we only use it for future refinement
    # The bottleneck job is now identified as job `bottleneck_job_idx`
    
    # --- TASK 6: Local Search Refinement via High-Impact Worker Reassignment to Improve the Bottleneck Job Rate ---
    # Start refinement only if a bottleneck job exists
    if bottleneck_job_idx == -1:
        return ina_placement_expanded, jobs_routing_expanded
    
    # Make a copy of current state for safe rollback
    # We'll use the current link_load and residual_switch_capacity as base
    # But we'll simulate reassignments on a temporary copy
    
    # Create temporary copies
    temp_residual_switch_capacity = residual_switch_capacity[:]
    temp_link_load = link_load.copy()
    
    # Identify bottleneck job
    j_star = bottleneck_job_idx
    job_size_star = jobs_size[j_star]
    workers_in_job = workers_num[j_star]
    
    # Build priority queue: workers with highest impact on bottleneck
    # Priority: (negative_rank, m_j, current_rate, path_congestion)
    # We'll use a simple heuristic: prioritize based on gradient volume and low current rate
    candidates = []
    for w in range(workers_in_job):
        # Get current assignment
        assigned_to_ina = False
        assigned_to_ps = False
        assigned_switch_idx = -1
        current_rate = 0.0
        
        # Check INA assignment
        for s_idx, s in enumerate(ina_candidates):
            if y_j_w_s_full[j_star][w][s_idx] == 1:
                assigned_to_ina = True
                assigned_switch_idx = s_idx
                # Compute current rate based on path
                src = workers_id[j_star][w]
                dst = s
                path = network.allPathDict.get(src, {}).get(dst, [])
                if path:
                    path_links = links_set(src, dst, network.allPathDict)
                    min_rate = float('inf')
                    for u, v in path_links:
                        link = (u, v)
                        current_load = temp_link_load.get(link, 0.0)
                        capacity = network.bandwidth_mapping.get(link, 0.0)
                        available = capacity - current_load
                        min_rate = min(min_rate, available)
                    current_rate = min_rate
                break
        if not assigned_to_ina:
            # Check PS assignment
            if y_j_w_d[j_star][w] == 1:
                assigned_to_ps = True
                src = workers_id[j_star][w]
                dst = ps_id[j_star]
                path = network.allPathDict.get(src, {}).get(dst, [])
                if path:
                    path_links = links_set(src, dst, network.allPathDict)
                    min_rate = float('inf')
                    for u, v in path_links:
                        link = (u, v)
                        current_load = temp_link_load.get(link, 0.0)
                        capacity = network.bandwidth_mapping.get(link, 0.0)
                        available = capacity - current_load
                        min_rate = min(min_rate, available)
                    current_rate = min_rate
        
        # Compute congestion metric: number of links with high utilization
        # Use a simple heuristic: sum of (current_load / capacity) for all links in path
        congestion = 0.0
        if assigned_to_ina:
            src = workers_id[j_star][w]
            dst = ina_candidates[assigned_switch_idx]
            path = network.allPathDict.get(src, {}).get(dst, [])
            if path:
                path_links = links_set(src, dst, network.allPathDict)
                for u, v in path_links:
                    link = (u, v)
                    capacity = network.bandwidth_mapping.get(link, 0.0)
                    if capacity > 0:
                        utilization = temp_link_load.get(link, 0.0) / capacity
                        congestion += utilization
        elif assigned_to_ps:
            src = workers_id[j_star][w]
            dst = ps_id[j_star]
            path = network.allPathDict.get(src, {}).get(dst, [])
            if path:
                path_links = links_set(src, dst, network.allPathDict)
                for u, v in path_links:
                    link = (u, v)
                    capacity = network.bandwidth_mapping.get(link, 0.0)
                    if capacity > 0:
                        utilization = temp_link_load.get(link, 0.0) / capacity
                        congestion += utilization
        
        # Priority: higher volume, lower rate, higher congestion
        priority = (-jobs_size[j_star], -current_rate, -congestion, w)
        candidates.append((priority, w))
    
    # Sort candidates by priority
    candidates.sort()
    
    # Refinement loop: up to 20 iterations
    max_iterations = 20
    iterations = 0
    improved = True
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Process candidates in order
        for _, w in candidates:
            # Skip if worker is not active
            if w >= workers_in_job:
                continue
                
            # Current assignment
            current_assigned_to_ina = False
            current_assigned_switch_idx = -1
            current_assigned_to_ps = False
            current_rate = 0.0
            
            # Determine current assignment
            for s_idx, s in enumerate(ina_candidates):
                if y_j_w_s_full[j_star][w][s_idx] == 1:
                    current_assigned_to_ina = True
                    current_assigned_switch_idx = s_idx
                    current_rate = 0.0  # We'll recompute
                    break
            if not current_assigned_to_ina:
                if y_j_w_d[j_star][w] == 1:
                    current_assigned_to_ps = True
                    current_rate = 0.0  # We'll recompute
                    
            # Compute current effective rate for the job if we keep current assignment
            # We only need to know the bottleneck job rate
            # But we'll simulate the change
            # First, remove current contribution from temp state
            temp_residual_switch_capacity = residual_switch_capacity[:]
            temp_link_load = link_load.copy()
            
            # Remove current assignment
            if current_assigned_to_ina:
                src = workers_id[j_star][w]
                dst = ina_candidates[current_assigned_switch_idx]
                path = network.allPathDict.get(src, {}).get(dst, [])
                if path:
                    path_links = links_set(src, dst, network.allPathDict)
                    for u, v in path_links:
                        link = (u, v)
                        temp_link_load[link] -= current_rate
                        if temp_link_load[link] < 0:
                            temp_link_load[link] = 0
                    temp_residual_switch_capacity[current_assigned_switch_idx] += current_rate
            elif current_assigned_to_ps:
                src = workers_id[j_star][w]
                dst = ps_id[j_star]
                path = network.allPathDict.get(src, {}).get(dst, [])
                if path:
                    path_links = links_set(src, dst, network.allPathDict)
                    for u, v in path_links:
                        link = (u, v)
                        temp_link_load[link] -= current_rate
                        if temp_link_load[link] < 0:
                            temp_link_load[link] = 0
            
            # Now try all alternative destinations
            best_rate = 0.0
            best_dest = None  # (type, idx, target)
            
            # Active INA switches
            for s_idx, s in enumerate(ina_candidates):
                if ina_placement_expanded[s_idx] == 1:
                    # Check if we can route to this switch
                    src = workers_id[j_star][w]
                    dst = s
                    path = network.allPathDict.get(src, {}).get(dst, [])
                    if not path:
                        continue
                        
                    # Compute available rate
                    min_rate = float('inf')
                    path_links = links_set(src, dst, network.allPathDict)
                    for u, v in path_links:
                        link = (u, v)
                        current_load = temp_link_load.get(link, 0.0)
                        capacity = network.bandwidth_mapping.get(link, 0.0)
                        available = capacity - current_load
                        min_rate = min(min_rate, available)
                    
                    # Switch capacity
                    switch_capacity = temp_residual_switch_capacity[s_idx]
                    min_rate = min(min_rate, switch_capacity)
                    
                    if min_rate > best_rate:
                        best_rate = min_rate
                        best_dest = ('ina', s_idx, s)
            
            # PS
            src = workers_id[j_star][w]
            dst = ps_id[j_star]
            path = network.allPathDict.get(src, {}).get(dst, [])
            if path:
                min_rate = float('inf')
                path_links = links_set(src, dst, network.allPathDict)
                for u, v in path_links:
                    link = (u, v)
                    current_load = temp_link_load.get(link, 0.0)
                    capacity = network.bandwidth_mapping.get(link, 0.0)
                    available = capacity - current_load
                    min_rate = min(min_rate, available)
                
                # PS bandwidth not limited by capacity in this model
                if min_rate > best_rate:
                    best_rate = min_rate
                    best_dest = ('ps', None, ps_id[j_star])
            
            # If we found a better alternative and best_rate > 0
            if best_dest is not None and best_rate > 0:
                # Compute new job rate after reassignment
                # We only change this worker
                # Recompute the minimum rate among all workers in job j_star
                new_per_worker_rates = []
                for w2 in range(workers_in_job):
                    if w2 == w:
                        new_per_worker_rates.append(best_rate)
                    else:
                        # Reuse existing rate
                        rate = 0.0
                        # Check INA
                        for s_idx, s in enumerate(ina_candidates):
                            if y_j_w_s_full[j_star][w2][s_idx] == 1:
                                src = workers_id[j_star][w2]
                                dst = s
                                path = network.allPathDict.get(src, {}).get(dst, [])
                                if path:
                                    path_links = links_set(src, dst, network.allPathDict)
                                    min_rate = float('inf')
                                    for u, v in path_links:
                                        link = (u, v)
                                        current_load = link_load.get(link, 0.0)
                                        capacity = network.bandwidth_mapping.get(link, 0.0)
                                        available = capacity - current_load
                                        min_rate = min(min_rate, available)
                                    rate = min_rate
                                break
                        # Check PS
                        if rate == 0.0 and y_j_w_d[j_star][w2] == 1:
                            src = workers_id[j_star][w2]
                            dst = ps_id[j_star]
                            path = network.allPathDict.get(src, {}).get(dst, [])
                            if path:
                                path_links = links_set(src, dst, network.allPathDict)
                                min_rate = float('inf')
                                for u, v in path_links:
                                    link = (u, v)
                                    current_load = link_load.get(link, 0.0)
                                    capacity = network.bandwidth_mapping.get(link, 0.0)
                                    available = capacity - current_load
                                    min_rate = min(min_rate, available)
                                rate = min_rate
                        new_per_worker_rates.append(rate)
                
                new_job_rate = min(new_per_worker_rates) if new_per_worker_rates else 0.0
                original_job_rate = min(job_rates[j_star], job_rates[j_star])
                
                # If the new job rate is higher, accept the change
                if new_job_rate > original_job_rate:
                    # Accept the reassignment
                    # Update the assignment
                    # First, remove old assignment
                    if current_assigned_to_ina:
                        y_j_w_s_full[j_star][w][current_assigned_switch_idx] = 0
                    elif current_assigned_to_ps:
                        y_j_w_d[j_star][w] = 0
                    
                    # Apply new assignment
                    if best_dest[0] == 'ina':
                        y_j_w_s_full[j_star][w][best_dest[1]] = 1
                        # Update residual capacity
                        residual_switch_capacity[best_dest[1]] -= best_rate
                    else:  # PS
                        y_j_w_d[j_star][w] = 1
                    
                    # Update link loads
                    src = workers_id[j_star][w]
                    dst = best_dest[2]
                    path = network.allPathDict.get(src, {}).get(dst, [])
                    if path:
                        path_links = links_set(src, dst, network.allPathDict)
                        for u, v in path_links:
                            link = (u, v)
                            link_load[link] = link_load.get(link, 0.0) + best_rate
                    
                    # Update job rate
                    job_rates[j_star] = new_job_rate
                    
                    # Mark that improvement was found
                    improved = True
                    
                    # Update the candidates list for next iteration
                    # Since we changed the state, restart the priority queue
                    break
        
        # Break if no improvement or max iterations reached
        if not improved:
            break
    
    # Return final result
    return ina_placement_expanded, jobs_routing_expanded