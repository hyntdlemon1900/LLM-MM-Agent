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
                if w in network.allPathDict and s in network.allPathDict[w]:
                    path = network.allPathDict[w][s]
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
    # TASK 2: Per-Worker Assignment with Cost-Based Min-Cost Flow Approximation and Capacity Enforcement
    # --------------------------
    
    # Precompute link capacity and path information
    # For each (src, dst) pair, get path and link set
    path_links_dict = path_to_links_dict(
        src_set=ina_candidates + [pid for pid in ps_id] + [wid for j in workers_id for wid in j],
        dst_set=ina_candidates + [pid for pid in ps_id],
        allPathDict=network.allPathDict
    )
    
    # Initialize tracking for per-switch and per-link loads
    switch_load = {s: 0.0 for s in ina_candidates}
    link_load = {edge: 0.0 for edge in network.bandwidth_mapping.keys()}
    
    # Build a list of jobs sorted by job size (descending) to prioritize large jobs
    job_priority = sorted(range(jobs_num), key=lambda j: jobs_size[j], reverse=True)
    
    # For each job in descending order of size
    for j in job_priority:
        job_workers = workers_id[j]
        job_workers_count = workers_num[j]
        job_ps = ps_id[j]
        
        # For each worker in job j
        for w in range(job_workers_count):
            worker = job_workers[w]
            
            # Compute cost for routing to each INA switch and PS
            costs = {}
            for s_idx, s in enumerate(ina_candidates):
                if ina_placement_expanded[s_idx] == 1:  # Only consider selected INA switches
                    # Get path from worker to INA switch
                    if worker in network.allPathDict and s in network.allPathDict[worker]:
                        path = network.allPathDict[worker][s]
                        # Find bottleneck link: min bandwidth along path
                        bottleneck_bw = float('inf')
                        for i in range(len(path) - 1):
                            edge = (path[i], path[i + 1])
                            if edge in network.bandwidth_mapping:
                                bottleneck_bw = min(bottleneck_bw, network.bandwidth_mapping[edge])
                            else:
                                # Reverse edge check
                                edge_rev = (path[i + 1], path[i])
                                if edge_rev in network.bandwidth_mapping:
                                    bottleneck_bw = min(bottleneck_bw, network.bandwidth_mapping[edge_rev])
                        if bottleneck_bw == float('inf'):
                            # No known bandwidth, skip or use default
                            cost = float('inf')
                        else:
                            # Cost is inverse of effective throughput (lower cost = better)
                            cost = 1.0 / bottleneck_bw
                        costs[(s, 'ina')] = cost
            # Also consider direct routing to PS
            if worker in network.allPathDict and job_ps in network.allPathDict[worker]:
                path = network.allPathDict[worker][job_ps]
                bottleneck_bw = float('inf')
                for i in range(len(path) - 1):
                    edge = (path[i], path[i + 1])
                    if edge in network.bandwidth_mapping:
                        bottleneck_bw = min(bottleneck_bw, network.bandwidth_mapping[edge])
                    else:
                        edge_rev = (path[i + 1], path[i])
                        if edge_rev in network.bandwidth_mapping:
                            bottleneck_bw = min(bottleneck_bw, network.bandwidth_mapping[edge_rev])
                if bottleneck_bw == float('inf'):
                    cost = float('inf')
                else:
                    cost = 1.0 / bottleneck_bw
                costs[('ps', 'ps')] = cost
            
            # Find best candidate with minimal cost
            best_candidate = None
            min_cost = float('inf')
            for cand, cost in costs.items():
                if cost < min_cost:
                    min_cost = cost
                    best_candidate = cand
            
            # Try to assign worker to best candidate
            assigned = False
            if best_candidate and best_candidate[1] == 'ina':
                s, _ = best_candidate
                s_idx = ina_candidates.index(s)
                
                # Check switch capacity: total rate must not exceed C_s
                # Assume worker sends at a rate proportional to job size (m_j), but we only care about
                # relative load; use m_j as proxy for rate
                proposed_load = switch_load[s] + jobs_size[j]
                if proposed_load <= Cs:
                    # Check link load constraints for all links on path from worker to s
                    path = network.allPathDict[worker][s]
                    links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                    valid_path = True
                    for edge in links:
                        edge_rev = edge[::-1]
                        link_capacity = network.bandwidth_mapping.get(edge, network.bandwidth_mapping.get(edge_rev, float('inf')))
                        if link_load[edge] + jobs_size[j] > link_capacity:
                            valid_path = False
                            break
                
                    if valid_path:
                        # Assign worker to INA switch
                        y_j_w_s_full[j][w][s_idx] = 1
                        switch_load[s] += jobs_size[j]
                        for edge in links:
                            link_load[edge] += jobs_size[j]
                        assigned = True
            
            if not assigned:
                # Try PS assignment
                if best_candidate and best_candidate[1] == 'ps':
                    # Check link load on path from worker to PS
                    path = network.allPathDict[worker][job_ps]
                    links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                    valid_path = True
                    for edge in links:
                        edge_rev = edge[::-1]
                        link_capacity = network.bandwidth_mapping.get(edge, network.bandwidth_mapping.get(edge_rev, float('inf')))
                        if link_load[edge] + jobs_size[j] > link_capacity:
                            valid_path = False
                            break
                
                    if valid_path:
                        y_j_w_d[j][w] = 1
                        for edge in links:
                            link_load[edge] += jobs_size[j]
                        assigned = True
            
            # If still not assigned, try alternative INA switches or PS
            if not assigned:
                # Try all other INA switches in order of increasing cost
                for s_idx, s in enumerate(ina_candidates):
                    if ina_placement_expanded[s_idx] == 1 and best_candidate and best_candidate[1] == 'ina' and s != best_candidate[0]:
                        path = network.allPathDict[worker][s]
                        links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                        valid_path = True
                        for edge in links:
                            edge_rev = edge[::-1]
                            link_capacity = network.bandwidth_mapping.get(edge, network.bandwidth_mapping.get(edge_rev, float('inf')))
                            if link_load[edge] + jobs_size[j] > link_capacity:
                                valid_path = False
                                break
                        if valid_path:
                            proposed_load = switch_load[s] + jobs_size[j]
                            if proposed_load <= Cs:
                                y_j_w_s_full[j][w][s_idx] = 1
                                switch_load[s] += jobs_size[j]
                                for edge in links:
                                    link_load[edge] += jobs_size[j]
                                assigned = True
                                break
                # If still not assigned, try PS
                if not assigned:
                    path = network.allPathDict[worker][job_ps]
                    links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                    valid_path = True
                    for edge in links:
                        edge_rev = edge[::-1]
                        link_capacity = network.bandwidth_mapping.get(edge, network.bandwidth_mapping.get(edge_rev, float('inf')))
                        if link_load[edge] + jobs_size[j] > link_capacity:
                            valid_path = False
                            break
                    if valid_path:
                        y_j_w_d[j][w] = 1
                        for edge in links:
                            link_load[edge] += jobs_size[j]
                        assigned = True
            
            # If still not assigned, reassign from other workers in this job or other jobs
            if not assigned:
                # Backtrack: try to reassign other workers in the same job to reduce load
                # For now, assign to PS as fallback
                y_j_w_d[j][w] = 1
                path = network.allPathDict[worker][job_ps]
                links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                for edge in links:
                    link_load[edge] += jobs_size[j]
    
    # --------------------------
    # TASK 3: Link-Load Aggregation and Global Capacity Validation via Path-Based Rate Summation
    # --------------------------
    
    # Initialize load tracking for each link
    link_load = {edge: 0.0 for edge in network.bandwidth_mapping.keys()}
    
    # Aggregate all flows: worker→INA, worker→PS, INA→PS
    for j in range(jobs_num):
        job_workers = workers_id[j]
        job_ps = ps_id[j]
        job_size = jobs_size[j]
        
        # For each worker in job j
        for w in range(workers_num[j]):
            worker = job_workers[w]
            worker_size = job_size  # Use job size as rate proxy
            
            # Check worker-to-INA assignment
            assigned_to_ina = False
            for s_idx, s in enumerate(ina_candidates):
                if y_j_w_s_full[j][w][s_idx] == 1:
                    # Get path from worker to INA switch
                    if worker in network.allPathDict and s in network.allPathDict[worker]:
                        path = network.allPathDict[worker][s]
                        links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                        # Add load to each link
                        for edge in links:
                            link_load[edge] += worker_size
                    assigned_to_ina = True
                    break
            
            # If not assigned to INA, check worker-to-PS
            if not assigned_to_ina and y_j_w_d[j][w] == 1:
                # Get path from worker to PS
                if worker in network.allPathDict and job_ps in network.allPathDict[worker]:
                    path = network.allPathDict[worker][job_ps]
                    links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                    # Add load to each link
                    for edge in links:
                        link_load[edge] += worker_size
            
            # Add INA-to-PS flow: sum of all worker→INA flows for this job
            # For each INA switch used by this job
            for s_idx, s in enumerate(ina_candidates):
                if ina_placement_expanded[s_idx] == 1:
                    # Check if any worker in this job is assigned to this INA
                    has_worker = any(y_j_w_s_full[j][w][s_idx] == 1 for w in range(workers_num[j]))
                    if has_worker:
                        # Get path from INA to PS
                        if s in network.allPathDict and job_ps in network.allPathDict[s]:
                            path = network.allPathDict[s][job_ps]
                            links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                            # Add load to each link
                            for edge in links:
                                link_load[edge] += job_size
    
    # Validate link capacity constraints
    capacity_violations = False
    max_load_ratio = 0.0
    most_congested_edge = None
    
    for edge, total_load in link_load.items():
        capacity = network.bandwidth_mapping.get(edge, float('inf'))
        if total_load > capacity:
            capacity_violations = True
            load_ratio = total_load / capacity
            if load_ratio > max_load_ratio:
                max_load_ratio = load_ratio
                most_congested_edge = edge
    
    # If any link violates capacity, apply load mitigation protocol
    if capacity_violations:
        # Reduce load on most congested link by 10% and reassign workers
        # Reassign workers from high-cost paths to alternative aggregation points
        # Start with largest jobs and most congested links
        for j in job_priority:
            job_workers = workers_id[j]
            job_ps = ps_id[j]
            job_size = jobs_size[j]
            
            # Reassign workers that use the most congested edge
            for w in range(workers_num[j]):
                worker = job_workers[w]
                
                # Check if worker is routed to INA and path includes most_congested_edge
                assigned_to_ina = False
                for s_idx, s in enumerate(ina_candidates):
                    if y_j_w_s_full[j][w][s_idx] == 1:
                        if worker in network.allPathDict and s in network.allPathDict[worker]:
                            path = network.allPathDict[worker][s]
                            links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                            if most_congested_edge in links or most_congested_edge[::-1] in links:
                                # Try reassigning to PS
                                if worker in network.allPathDict and job_ps in network.allPathDict[worker]:
                                    path_to_ps = network.allPathDict[worker][job_ps]
                                    links_to_ps = [(path_to_ps[i], path_to_ps[i + 1]) for i in range(len(path_to_ps) - 1)]
                                    valid_path = True
                                    for edge in links_to_ps:
                                        capacity = network.bandwidth_mapping.get(edge, float('inf'))
                                        if link_load[edge] + job_size > capacity:
                                            valid_path = False
                                            break
                                    if valid_path:
                                        # Reassign worker to PS
                                        y_j_w_s_full[j][w][s_idx] = 0
                                        y_j_w_d[j][w] = 1
                                        
                                        # Update link loads
                                        for edge in links:
                                            link_load[edge] -= job_size
                                        for edge in links_to_ps:
                                            link_load[edge] += job_size
                                        assigned_to_ina = True
                                        break
                # If still assigned to INA and path uses congested link, try other INA switches
                if not assigned_to_ina:
                    for s_idx, s in enumerate(ina_candidates):
                        if ina_placement_expanded[s_idx] == 1 and s != s:
                            if worker in network.allPathDict and s in network.allPathDict[worker]:
                                path = network.allPathDict[worker][s]
                                links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                                valid_path = True
                                for edge in links:
                                    capacity = network.bandwidth_mapping.get(edge, float('inf'))
                                    if link_load[edge] + job_size > capacity:
                                        valid_path = False
                                        break
                                if valid_path:
                                    # Reassign
                                    y_j_w_s_full[j][w][s_idx] = 1
                                    y_j_w_d[j][w] = 0
                                    
                                    # Update link loads
                                    for edge in links:
                                        link_load[edge] += job_size
                                    assigned_to_ina = True
                                    break
    
    # --------------------------
    # TASK 4: Job Rate Consistency and Makespan Estimation via Bottleneck-Driven Completion Time Calculation
    # --------------------------
    
    # Compute the effective rate for each worker in each job
    # γ_jw = sum(γ_jws) + γ_jwd_j
    # Then γ_j = min_w(γ_jw), t_j = m_j / γ_j, t = max_j(t_j)
    
    # Initialize job completion times
    job_completion_times = [0.0] * jobs_num
    
    # For each job, compute the effective rate γ_j
    for j in range(jobs_num):
        job_workers = workers_id[j]
        job_size = jobs_size[j]
        job_workers_count = workers_num[j]
        
        # Track individual worker sending rates
        worker_rates = []
        
        # For each worker in job j
        for w in range(job_workers_count):
            worker_rate = 0.0
            
            # Sum rates from all assigned INA switches
            for s_idx, s in enumerate(ina_candidates):
                if y_j_w_s_full[j][w][s_idx] == 1:
                    # Use job size as proxy for transmission rate
                    worker_rate += job_size
            
            # Add direct PS rate if assigned
            if y_j_w_d[j][w] == 1:
                worker_rate += job_size
            
            worker_rates.append(worker_rate)
        
        # Job rate is the minimum worker rate (bottleneck)
        job_rate = min(worker_rates) if worker_rates else 0.0
        
        # Completion time for job j
        if job_rate > 0:
            completion_time = job_size / job_rate
        else:
            completion_time = float('inf')
        
        job_completion_times[j] = completion_time
    
    # Overall makespan is the maximum completion time
    makespan = max(job_completion_times) if job_completion_times else float('inf')
    
    # The auxiliary variable α = 1 / makespan
    # This value is used to evaluate solution quality during refinement
    # It will be used in Task 5 for dual decomposition and in Task 6 for final validation
    
    # --------------------------
    # TASK 5: Iterative Local Search with Perturbation-Based Refinement and Feasibility Preservation
    # --------------------------
    
    # Set refinement parameters
    max_iterations = 15
    patience = 5
    best_makespan = makespan
    best_placement = ina_placement_expanded[:]
    best_routing = [y_j_w_s_full[j][:] for j in range(jobs_num)]
    best_routing_d = [y_j_w_d[j][:] for j in range(jobs_num)]
    
    # Track convergence
    no_improvement_count = 0
    
    # Create a list of all top-2K candidates for swapping (ranked by score)
    top_k_candidates = sorted_candidates[:2 * K]
    
    # Perform iterative refinement
    for iteration in range(max_iterations):
        improved = False
        
        # Perturbation 1: Switch swapping
        # Try replacing a selected INA switch with a non-selected one from top-2K
        selected_indices = [i for i, val in enumerate(best_placement) if val == 1]
        non_selected_indices = [i for i, val in enumerate(best_placement) if val == 0]
        
        # Only consider non-selected candidates in top-2K
        valid_non_selected = [i for i in non_selected_indices if i < len(top_k_candidates) and i in [ina_candidates.index(s) for s in top_k_candidates]]
        
        if len(selected_indices) > 0 and valid_non_selected:
            # Randomly pick one to swap
            idx_to_remove = random.choice(selected_indices)
            idx_to_add = random.choice(valid_non_selected)
            
            # Create new placement
            new_placement = best_placement[:]
            new_placement[idx_to_remove] = 0
            new_placement[idx_to_add] = 1
            
            # Recompute routing with new placement
            # Reinitialize routing
            new_y_j_w_s_full = [[[0] * num_candidates for _ in range(max_workers)] for _ in range(jobs_num)]
            new_y_j_w_d = [[0] * max_workers for _ in range(jobs_num)]
            
            # Reassign workers using the same min-cost flow logic
            # Initialize switch and link loads
            new_switch_load = {s: 0.0 for s in ina_candidates}
            new_link_load = {edge: 0.0 for edge in network.bandwidth_mapping.keys()}
            
            # Reassign in job priority order
            for j in job_priority:
                job_workers = workers_id[j]
                job_ps = ps_id[j]
                job_size = jobs_size[j]
                
                for w in range(workers_num[j]):
                    worker = job_workers[w]
                    
                    # Compute costs
                    costs = {}
                    for s_idx, s in enumerate(ina_candidates):
                        if new_placement[s_idx] == 1:  # Only selected switches
                            if worker in network.allPathDict and s in network.allPathDict[worker]:
                                path = network.allPathDict[worker][s]
                                bottleneck_bw = float('inf')
                                for i in range(len(path) - 1):
                                    edge = (path[i], path[i + 1])
                                    if edge in network.bandwidth_mapping:
                                        bottleneck_bw = min(bottleneck_bw, network.bandwidth_mapping[edge])
                                    else:
                                        edge_rev = (path[i + 1], path[i])
                                        if edge_rev in network.bandwidth_mapping:
                                            bottleneck_bw = min(bottleneck_bw, network.bandwidth_mapping[edge_rev])
                                if bottleneck_bw == float('inf'):
                                    cost = float('inf')
                                else:
                                    cost = 1.0 / bottleneck_bw
                                costs[(s, 'ina')] = cost
                    
                    if worker in network.allPathDict and job_ps in network.allPathDict[worker]:
                        path = network.allPathDict[worker][job_ps]
                        bottleneck_bw = float('inf')
                        for i in range(len(path) - 1):
                            edge = (path[i], path[i + 1])
                            if edge in network.bandwidth_mapping:
                                bottleneck_bw = min(bottleneck_bw, network.bandwidth_mapping[edge])
                            else:
                                edge_rev = (path[i + 1], path[i])
                                if edge_rev in network.bandwidth_mapping:
                                    bottleneck_bw = min(bottleneck_bw, network.bandwidth_mapping[edge_rev])
                        if bottleneck_bw == float('inf'):
                            cost = float('inf')
                        else:
                            cost = 1.0 / bottleneck_bw
                        costs[('ps', 'ps')] = cost
                    
                    # Find best candidate
                    best_candidate = None
                    min_cost = float('inf')
                    for cand, cost in costs.items():
                        if cost < min_cost:
                            min_cost = cost
                            best_candidate = cand
                    
                    # Try assignment
                    assigned = False
                    if best_candidate and best_candidate[1] == 'ina':
                        s, _ = best_candidate
                        s_idx = ina_candidates.index(s)
                        
                        proposed_load = new_switch_load[s] + jobs_size[j]
                        if proposed_load <= Cs:
                            path = network.allPathDict[worker][s]
                            links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                            valid_path = True
                            for edge in links:
                                edge_rev = edge[::-1]
                                link_capacity = network.bandwidth_mapping.get(edge, network.bandwidth_mapping.get(edge_rev, float('inf')))
                                if new_link_load[edge] + jobs_size[j] > link_capacity:
                                    valid_path = False
                                    break
                            
                            if valid_path:
                                new_y_j_w_s_full[j][w][s_idx] = 1
                                new_switch_load[s] += jobs_size[j]
                                for edge in links:
                                    new_link_load[edge] += jobs_size[j]
                                assigned = True
                    
                    if not assigned:
                        if best_candidate and best_candidate[1] == 'ps':
                            path = network.allPathDict[worker][job_ps]
                            links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                            valid_path = True
                            for edge in links:
                                edge_rev = edge[::-1]
                                link_capacity = network.bandwidth_mapping.get(edge, network.bandwidth_mapping.get(edge_rev, float('inf')))
                                if new_link_load[edge] + jobs_size[j] > link_capacity:
                                    valid_path = False
                                    break
                            
                            if valid_path:
                                new_y_j_w_d[j][w] = 1
                                for edge in links:
                                    new_link_load[edge] += jobs_size[j]
                                assigned = True
                    
                    # Backtrack if needed
                    if not assigned:
                        for s_idx, s in enumerate(ina_candidates):
                            if new_placement[s_idx] == 1 and best_candidate and best_candidate[1] == 'ina' and s != best_candidate[0]:
                                path = network.allPathDict[worker][s]
                                links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                                valid_path = True
                                for edge in links:
                                    edge_rev = edge[::-1]
                                    link_capacity = network.bandwidth_mapping.get(edge, network.bandwidth_mapping.get(edge_rev, float('inf')))
                                    if new_link_load[edge] + jobs_size[j] > link_capacity:
                                        valid_path = False
                                        break
                                if valid_path:
                                    proposed_load = new_switch_load[s] + jobs_size[j]
                                    if proposed_load <= Cs:
                                        new_y_j_w_s_full[j][w][s_idx] = 1
                                        new_switch_load[s] += jobs_size[j]
                                        for edge in links:
                                            new_link_load[edge] += jobs_size[j]
                                        assigned = True
                                        break
                        if not assigned:
                            path = network.allPathDict[worker][job_ps]
                            links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                            valid_path = True
                            for edge in links:
                                edge_rev = edge[::-1]
                                link_capacity = network.bandwidth_mapping.get(edge, network.bandwidth_mapping.get(edge_rev, float('inf')))
                                if new_link_load[edge] + jobs_size[j] > link_capacity:
                                    valid_path = False
                                    break
                            if valid_path:
                                new_y_j_w_d[j][w] = 1
                                for edge in links:
                                    new_link_load[edge] += jobs_size[j]
                                assigned = True
                    
                    if not assigned:
                        new_y_j_w_d[j][w] = 1
                        path = network.allPathDict[worker][job_ps]
                        links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                        for edge in links:
                            new_link_load[edge] += jobs_size[j]
            
            # Validate link capacity
            link_load_check = {edge: 0.0 for edge in network.bandwidth_mapping.keys()}
            for j in range(jobs_num):
                job_workers = workers_id[j]
                job_ps = ps_id[j]
                job_size = jobs_size[j]
                
                for w in range(workers_num[j]):
                    worker = job_workers[w]
                    worker_size = job_size
                    
                    assigned_to_ina = False
                    for s_idx, s in enumerate(ina_candidates):
                        if new_y_j_w_s_full[j][w][s_idx] == 1:
                            if worker in network.allPathDict and s in network.allPathDict[worker]:
                                path = network.allPathDict[worker][s]
                                links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                                for edge in links:
                                    link_load_check[edge] += worker_size
                            assigned_to_ina = True
                            break
                    
                    if not assigned_to_ina and new_y_j_w_d[j][w] == 1:
                        if worker in network.allPathDict and job_ps in network.allPathDict[worker]:
                            path = network.allPathDict[worker][job_ps]
                            links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                            for edge in links:
                                link_load_check[edge] += worker_size
                    
                    for s_idx, s in enumerate(ina_candidates):
                        if new_placement[s_idx] == 1:
                            has_worker = any(new_y_j_w_s_full[j][w][s_idx] == 1 for w in range(workers_num[j]))
                            if has_worker:
                                if s in network.allPathDict and job_ps in network.allPathDict[s]:
                                    path = network.allPathDict[s][job_ps]
                                    links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                                    for edge in links:
                                        link_load_check[edge] += job_size
            
            # Check for violations
            capacity_violated = False
            for edge, total_load in link_load_check.items():
                capacity = network.bandwidth_mapping.get(edge, float('inf'))
                if total_load > capacity:
                    capacity_violated = True
                    break
            
            if not capacity_violated:
                # Compute makespan for new solution
                new_job_completion_times = [0.0] * jobs_num
                for j in range(jobs_num):
                    job_workers = workers_id[j]
                    job_size = jobs_size[j]
                    job_workers_count = workers_num[j]
                    
                    worker_rates = []
                    for w in range(job_workers_count):
                        worker_rate = 0.0
                        for s_idx, s in enumerate(ina_candidates):
                            if new_y_j_w_s_full[j][w][s_idx] == 1:
                                worker_rate += job_size
                        if new_y_j_w_d[j][w] == 1:
                            worker_rate += job_size
                        worker_rates.append(worker_rate)
                    
                    job_rate = min(worker_rates) if worker_rates else 0.0
                    if job_rate > 0:
                        completion_time = job_size / job_rate
                    else:
                        completion_time = float('inf')
                    new_job_completion_times[j] = completion_time
                
                new_makespan = max(new_job_completion_times) if new_job_completion_times else float('inf')
                
                if new_makespan < best_makespan:
                    best_makespan = new_makespan
                    best_placement = new_placement[:]
                    best_routing = [new_y_j_w_s_full[j][:] for j in range(jobs_num)]
                    best_routing_d = [new_y_j_w_d[j][:] for j in range(jobs_num)]
                    improved = True
        
        # Perturbation 2: Worker reassignment from bottleneck job
        # Find job with highest completion time
        bottleneck_job = None
        max_completion = -1
        for j in range(jobs_num):
            if job_completion_times[j] > max_completion:
                max_completion = job_completion_times[j]
                bottleneck_job = j
        
        if bottleneck_job is not None:
            job_workers = workers_id[bottleneck_job]
            job_size = jobs_size[bottleneck_job]
            job_ps = ps_id[bottleneck_job]
            
            # Try reassigning a worker from this job
            # Use the best routing from current solution
            for w in range(workers_num[bottleneck_job]):
                # Current assignment
                current_ina = -1
                current_d = False
                
                for s_idx, s in enumerate(ina_candidates):
                    if y_j_w_s_full[bottleneck_job][w][s_idx] == 1:
                        current_ina = s_idx
                        break
                if y_j_w_d[bottleneck_job][w] == 1:
                    current_d = True
                
                # Only proceed if worker is assigned to INA or PS
                if current_ina == -1 and not current_d:
                    continue
                
                # Try reassigning to alternative
                new_y_j_w_s_full = [row[:] for row in y_j_w_s_full]
                new_y_j_w_d = [row[:] for row in y_j_w_d]
                
                # Try PS assignment if currently assigned to INA
                if current_ina != -1:
                    # Check PS path
                    worker = job_workers[w]
                    if worker in network.allPathDict and job_ps in network.allPathDict[worker]:
                        path = network.allPathDict[worker][job_ps]
                        links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                        valid_path = True
                        for edge in links:
                            edge_rev = edge[::-1]
                            link_capacity = network.bandwidth_mapping.get(edge, network.bandwidth_mapping.get(edge_rev, float('inf')))
                            if link_load[edge] + job_size > link_capacity:
                                valid_path = False
                                break
                        if valid_path:
                            # Reassign to PS
                            new_y_j_w_s_full[bottleneck_job][w][current_ina] = 0
                            new_y_j_w_d[bottleneck_job][w] = 1
                            
                            # Check capacity
                            # Recompute load
                            new_link_load = {edge: 0.0 for edge in network.bandwidth_mapping.keys()}
                            
                            # Aggregate flows
                            for j in range(jobs_num):
                                job_size_j = jobs_size[j]
                                for w in range(workers_num[j]):
                                    worker = workers_id[j][w]
                                    worker_size = job_size_j
                                    
                                    assigned_to_ina = False
                                    for s_idx, s in enumerate(ina_candidates):
                                        if new_y_j_w_s_full[j][w][s_idx] == 1:
                                            if worker in network.allPathDict and s in network.allPathDict[worker]:
                                                path = network.allPathDict[worker][s]
                                                links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                                                for edge in links:
                                                    new_link_load[edge] += worker_size
                                            assigned_to_ina = True
                                            break
                                    
                                    if not assigned_to_ina and new_y_j_w_d[j][w] == 1:
                                        if worker in network.allPathDict and ps_id[j] in network.allPathDict[worker]:
                                            path = network.allPathDict[worker][ps_id[j]]
                                            links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                                            for edge in links:
                                                new_link_load[edge] += worker_size
                                    
                                    for s_idx, s in enumerate(ina_candidates):
                                        if ina_placement_expanded[s_idx] == 1:
                                            has_worker = any(new_y_j_w_s_full[j][w][s_idx] == 1 for w in range(workers_num[j]))
                                            if has_worker:
                                                if s in network.allPathDict and ps_id[j] in network.allPathDict[s]:
                                                    path = network.allPathDict[s][ps_id[j]]
                                                    links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                                                    for edge in links:
                                                        new_link_load[edge] += job_size_j
                            
                            # Check violations
                            capacity_violated = False
                            for edge, total_load in new_link_load.items():
                                capacity = network.bandwidth_mapping.get(edge, float('inf'))
                                if total_load > capacity:
                                    capacity_violated = True
                                    break
                            
                            if not capacity_violated:
                                # Compute makespan
                                new_job_completion_times = [0.0] * jobs_num
                                for j in range(jobs_num):
                                    job_workers = workers_id[j]
                                    job_size = jobs_size[j]
                                    job_workers_count = workers_num[j]
                                    
                                    worker_rates = []
                                    for w in range(job_workers_count):
                                        worker_rate = 0.0
                                        for s_idx, s in enumerate(ina_candidates):
                                            if new_y_j_w_s_full[j][w][s_idx] == 1:
                                                worker_rate += job_size
                                        if new_y_j_w_d[j][w] == 1:
                                            worker_rate += job_size
                                        worker_rates.append(worker_rate)
                                    
                                    job_rate = min(worker_rates) if worker_rates else 0.0
                                    if job_rate > 0:
                                        completion_time = job_size / job_rate
                                    else:
                                        completion_time = float('inf')
                                    new_job_completion_times[j] = completion_time
                                
                                new_makespan = max(new_job_completion_times) if new_job_completion_times else float('inf')
                                
                                if new_makespan < best_makespan:
                                    best_makespan = new_makespan
                                    best_placement = ina_placement_expanded[:]
                                    best_routing = [new_y_j_w_s_full[j][:] for j in range(jobs_num)]
                                    best_routing_d = [new_y_j_w_d[j][:] for j in range(jobs_num)]
                                    improved = True
                
                # Try reassigning to another INA switch
                if not current_d:
                    for s_idx, s in enumerate(ina_candidates):
                        if s_idx != current_ina and ina_placement_expanded[s_idx] == 1:
                            worker = job_workers[w]
                            if worker in network.allPathDict and s in network.allPathDict[worker]:
                                path = network.allPathDict[worker][s]
                                links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                                valid_path = True
                                for edge in links:
                                    edge_rev = edge[::-1]
                                    link_capacity = network.bandwidth_mapping.get(edge, network.bandwidth_mapping.get(edge_rev, float('inf')))
                                    if link_load[edge] + job_size > link_capacity:
                                        valid_path = False
                                        break
                                if valid_path:
                                    # Check switch capacity
                                    proposed_load = switch_load[ina_candidates[s_idx]] + job_size
                                    if proposed_load <= Cs:
                                        # Reassign
                                        new_y_j_w_s_full[bottleneck_job][w][current_ina] = 0
                                        new_y_j_w_s_full[bottleneck_job][w][s_idx] = 1
                                        new_y_j_w_d[bottleneck_job][w] = 0
                                        
                                        # Check link capacity
                                        new_link_load = {edge: 0.0 for edge in network.bandwidth_mapping.keys()}
                                        
                                        for j in range(jobs_num):
                                            job_size_j = jobs_size[j]
                                            for w in range(workers_num[j]):
                                                worker = workers_id[j][w]
                                                worker_size = job_size_j
                                                
                                                assigned_to_ina = False
                                                for s_idx, s in enumerate(ina_candidates):
                                                    if new_y_j_w_s_full[j][w][s_idx] == 1:
                                                        if worker in network.allPathDict and s in network.allPathDict[worker]:
                                                            path = network.allPathDict[worker][s]
                                                            links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                                                            for edge in links:
                                                                new_link_load[edge] += worker_size
                                                        assigned_to_ina = True
                                                        break
                                                
                                                if not assigned_to_ina and new_y_j_w_d[j][w] == 1:
                                                    if worker in network.allPathDict and ps_id[j] in network.allPathDict[worker]:
                                                        path = network.allPathDict[worker][ps_id[j]]
                                                        links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                                                        for edge in links:
                                                            new_link_load[edge] += worker_size
                                                
                                                for s_idx, s in enumerate(ina_candidates):
                                                    if ina_placement_expanded[s_idx] == 1:
                                                        has_worker = any(new_y_j_w_s_full[j][w][s_idx] == 1 for w in range(workers_num[j]))
                                                        if has_worker:
                                                            if s in network.allPathDict and ps_id[j] in network.allPathDict[s]:
                                                                path = network.allPathDict[s][ps_id[j]]
                                                                links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                                                                for edge in links:
                                                                    new_link_load[edge] += job_size_j
                                        
                                        # Check violations
                                        capacity_violated = False
                                        for edge, total_load in new_link_load.items():
                                            capacity = network.bandwidth_mapping.get(edge, float('inf'))
                                            if total_load > capacity:
                                                capacity_violated = True
                                                break
                                        
                                        if not capacity_violated:
                                            # Compute makespan
                                            new_job_completion_times = [0.0] * jobs_num
                                            for j in range(jobs_num):
                                                job_workers = workers_id[j]
                                                job_size = jobs_size[j]
                                                job_workers_count = workers_num[j]
                                                
                                                worker_rates = []
                                                for w in range(job_workers_count):
                                                    worker_rate = 0.0
                                                    for s_idx, s in enumerate(ina_candidates):
                                                        if new_y_j_w_s_full[j][w][s_idx] == 1:
                                                            worker_rate += job_size
                                                    if new_y_j_w_d[j][w] == 1:
                                                        worker_rate += job_size
                                                    worker_rates.append(worker_rate)
                                                
                                                job_rate = min(worker_rates) if worker_rates else 0.0
                                                if job_rate > 0:
                                                    completion_time = job_size / job_rate
                                                else:
                                                    completion_time = float('inf')
                                                new_job_completion_times[j] = completion_time
                                            
                                            new_makespan = max(new_job_completion_times) if new_job_completion_times else float('inf')
                                            
                                            if new_makespan < best_makespan:
                                                best_makespan = new_makespan
                                                best_placement = ina_placement_expanded[:]
                                                best_routing = [new_y_j_w_s_full[j][:] for j in range(jobs_num)]
                                                best_routing_d = [new_y_j_w_d[j][:] for j in range(jobs_num)]
                                                improved = True
        
        # Update counters
        if improved:
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Check early termination
        if no_improvement_count >= patience:
            break
    
    # Update final solution
    ina_placement_expanded = best_placement
    y_j_w_s_full = best_routing
    y_j_w_d = best_routing_d
    jobs_routing_expanded = [y_j_w_s_full, y_j_w_d]
    
    # --------------------------
    # TASK 6: Final feasibility validation and output formatting
    # --------------------------
    
    # Reconstruct the final placement and routing in required format
    # Ensure exactly K switches are selected
    assert sum(ina_placement_expanded) == K, f"Expected K={K} INA switches, got {sum(ina_placement_expanded)}"
    
    # Validate that each active worker has exactly one aggregation point assigned
    for j in range(jobs_num):
        for w in range(workers_num[j]):
            total_assignment = sum(y_j_w_s_full[j][w]) + y_j_w_d[j][w]
            assert total_assignment == 1, f"Worker {w} in job {j} must have exactly one assignment, got {total_assignment}"
    
    # Validate that no worker is assigned to a non-INA switch
    for j in range(jobs_num):
        for w in range(workers_num[j]):
            for s_idx, s in enumerate(ina_candidates):
                if y_j_w_s_full[j][w][s_idx] == 1:
                    assert ina_placement_expanded[s_idx] == 1, f"Worker {w} in job {j} assigned to INA switch {s} which is not selected (x_s = 0)"
    
    # Validate that inactive workers are zero-padded
    for j in range(jobs_num):
        for w in range(workers_num[j], max_workers):
            for s_idx in range(num_candidates):
                assert y_j_w_s_full[j][w][s_idx] == 0, f"Inactive worker {w} in job {j} should be zero-padded"
            assert y_j_w_d[j][w] == 0, f"Inactive worker {w} in job {j} should be zero-padded"
    
    # Final validation: compute link loads using allPathDict and verify capacity
    link_load = {edge: 0.0 for edge in network.bandwidth_mapping.keys()}
    
    # Aggregate flows: worker→INA, worker→PS, INA→PS
    for j in range(jobs_num):
        job_workers = workers_id[j]
        job_ps = ps_id[j]
        job_size = jobs_size[j]
        
        for w in range(workers_num[j]):
            worker = job_workers[w]
            worker_size = job_size
            
            # Check worker-to-INA assignment
            assigned_to_ina = False
            for s_idx, s in enumerate(ina_candidates):
                if y_j_w_s_full[j][w][s_idx] == 1:
                    if worker in network.allPathDict and s in network.allPathDict[worker]:
                        path = network.allPathDict[worker][s]
                        links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                        for edge in links:
                            link_load[edge] += worker_size
                    assigned_to_ina = True
                    break
            
            # If not assigned to INA, check worker-to-PS
            if not assigned_to_ina and y_j_w_d[j][w] == 1:
                if worker in network.allPathDict and job_ps in network.allPathDict[worker]:
                    path = network.allPathDict[worker][job_ps]
                    links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                    for edge in links:
                        link_load[edge] += worker_size
            
            # Add INA-to-PS flow: sum of all worker→INA flows for this job
            for s_idx, s in enumerate(ina_candidates):
                if ina_placement_expanded[s_idx] == 1:
                    has_worker = any(y_j_w_s_full[j][w][s_idx] == 1 for w in range(workers_num[j]))
                    if has_worker:
                        if s in network.allPathDict and job_ps in network.allPathDict[s]:
                            path = network.allPathDict[s][job_ps]
                            links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                            for edge in links:
                                link_load[edge] += job_size
    
    # Check all link capacities
    for edge, total_load in link_load.items():
        capacity = network.bandwidth_mapping.get(edge, float('inf'))
        assert total_load <= capacity, f"Link {edge} has load {total_load} > capacity {capacity}"
    
    # Final validation passed - return solution
    return ina_placement_expanded, jobs_routing_expanded