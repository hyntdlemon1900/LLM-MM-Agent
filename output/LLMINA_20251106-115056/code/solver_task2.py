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
    # TASK 3-6: Placeholder Implementations (Will be optimized in future tasks)
    # --------------------------
    # Task 3: Feedback loop to detect bottlenecks and re-optimize
    # Task 4: Iterative refinement via local search (swap, reassign)
    # Task 5: Dual-decomposition-inspired flow assignment (refine rates)
    # Task 6: Final feasibility validation and output formatting
    
    # TODO: Will be optimized in Task 3
    # TODO: Will be optimized in Task 4
    # TODO: Will be optimized in Task 5
    # TODO: Will be optimized in Task 6
    
    return ina_placement_expanded, jobs_routing_expanded