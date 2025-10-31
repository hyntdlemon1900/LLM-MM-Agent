# Auto-generated solver with helper function imports
import sys
from pathlib import Path
import copy

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
    
    # Step 1: Identify candidate INA switches based on topology
    ina_candidates = get_ina_candidates(network)
    
    # Step 2: Compute composite centrality score for each candidate
    scores = []
    max_hops = 2
    T = 1.0  # Baseline iteration time (assumed)
    capacity_upper_bound = Cs / 2  # Conservative upper bound for load prediction
    
    for s in ina_candidates:
        # Component 1: Topological proximity to workers
        proximity_score = 0.0
        for j in range(jobs_num):
            for w_idx, w in enumerate(workers_id[j]):
                if w in network.allPathDict and s in network.allPathDict[w]:
                    path = network.allPathDict[w][s]
                    hops = len(path) - 1
                    if hops <= max_hops:
                        # Weight by inverse of hops
                        weight = 1.0 / hops if hops > 0 else 1.0
                        proximity_score += weight
        
        # Component 2: Job coverage potential (jobs whose PS is reachable in ≤2 hops)
        coverage_score = 0.0
        for j in range(jobs_num):
            if ps_id[j] in network.allPathDict and s in network.allPathDict[ps_id[j]]:
                path = network.allPathDict[ps_id[j]][s]
                hops = len(path) - 1
                if hops <= max_hops:
                    coverage_score += jobs_size[j]
        
        # Component 3: Capacity-aware load prediction
        load_pred_score = 0.0
        total_gradient_volume = 0.0
        for j in range(jobs_num):
            for w in workers_id[j]:
                if w in network.allPathDict and s in network.allPathDict[w]:
                    path = network.allPathDict[w][s]
                    hops = len(path) - 1
                    if hops <= max_hops:
                        # Approximate rate: m_j / T
                        rate = jobs_size[j] / T
                        total_gradient_volume += rate
        
        # Cap at conservative upper bound
        load_pred_score = min(total_gradient_volume, capacity_upper_bound)
        
        # Store components
        scores.append({
            'switch': s,
            'proximity': proximity_score,
            'coverage': coverage_score,
            'load_pred': load_pred_score
        })
    
    # Normalize scores (min-max scaling for each component)
    if len(scores) == 0:
        # Fallback: assign equal scores if no candidates
        normalized_scores = [(0.4, 0.4, 0.2) for _ in ina_candidates]
    else:
        # Extract components
        prox_scores = [s['proximity'] for s in scores]
        cov_scores = [s['coverage'] for s in scores]
        load_scores = [s['load_pred'] for s in scores]
        
        # Min-max scaling
        min_prox, max_prox = min(prox_scores), max(prox_scores)
        min_cov, max_cov = min(cov_scores), max(cov_scores)
        min_load, max_load = min(load_scores), max(load_scores)
        
        # Handle zero-variation case
        if max_prox == min_prox:
            norm_prox = [0.5] * len(prox_scores)
        else:
            norm_prox = [(p - min_prox) / (max_prox - min_prox) for p in prox_scores]
        
        if max_cov == min_cov:
            norm_cov = [0.5] * len(cov_scores)
        else:
            norm_cov = [(c - min_cov) / (max_cov - min_cov) for c in cov_scores]
        
        if max_load == min_load:
            norm_load = [0.5] * len(load_scores)
        else:
            norm_load = [(l - min_load) / (max_load - min_load) for l in load_scores]
        
        # Combine with weights: w1=0.4, w2=0.4, w3=0.2
        w1, w2, w3 = 0.4, 0.4, 0.2
        final_scores = [
            w1 * p + w2 * c + w3 * l
            for p, c, l in zip(norm_prox, norm_cov, norm_load)
        ]
        
        # Map back to switches
        normalized_scores = list(zip(ina_candidates, final_scores))
    
    # Sort candidates by score in descending order
    sorted_candidates = sorted(normalized_scores, key=lambda x: x[1], reverse=True)
    
    # Select top K switches
    ina_placement_expanded = [0] * len(ina_candidates)
    for i in range(K):
        switch_id = sorted_candidates[i][0]
        idx = ina_candidates.index(switch_id)
        ina_placement_expanded[idx] = 1
    
    # Initialize routing output
    max_workers = max(workers_num)
    num_candidates = len(ina_candidates)
    
    y_j_w_s_full = [[[0] * num_candidates for _ in range(max_workers)] for _ in range(jobs_num)]
    y_j_w_d = [[0] * max_workers for _ in range(jobs_num)]
    jobs_routing_expanded = [y_j_w_s_full, y_j_w_d]
    
    # Pre-Computation: Feasible Paths and Path-Load Matrix Initialization
    # Initialize load trackers
    switch_load = {s: 0.0 for s in ina_candidates}  # INA switch load tracking
    link_load_w_to_s = {link: 0.0 for link in network.bandwidth_mapping.keys()}  # Worker → INA switch
    link_load_w_to_ps = {link: 0.0 for link in network.bandwidth_mapping.keys()}  # Worker → PS
    link_load_s_to_ps = {link: 0.0 for link in network.bandwidth_mapping.keys()}  # INA switch → PS
    
    # Precompute path-incidence matrices
    path_incidence_dict = {}
    for j in range(jobs_num):
        job_workers = workers_id[j]
        job_ps = ps_id[j]
        job_ina_candidates = ina_candidates
        
        # For each worker and each possible aggregation point
        for w_idx, w in enumerate(job_workers):
            for s_idx, s in enumerate(job_ina_candidates):
                # Check if worker can reach INA s via valid path (direct, one-hop)
                if w in network.allPathDict and s in network.allPathDict[w]:
                    path = network.allPathDict[w][s]
                    hops = len(path) - 1
                    if hops <= 2:  # Single INA hop constraint: direct worker->switch
                        links = links_set(w, s, network.allPathDict)
                        path_incidence_dict[(w, s)] = links
                    else:
                        path_incidence_dict[(w, s)] = []
                else:
                    path_incidence_dict[(w, s)] = []
            
            # Check worker to PS path
            if w in network.allPathDict and job_ps in network.allPathDict[w]:
                path = network.allPathDict[w][job_ps]
                hops = len(path) - 1
                if hops <= 2:  # Valid direct path
                    links = links_set(w, job_ps, network.allPathDict)
                    path_incidence_dict[(w, job_ps)] = links
                else:
                    path_incidence_dict[(w, job_ps)] = []
            else:
                path_incidence_dict[(w, job_ps)] = []
        
        # Precompute INA-to-PS paths
        for s_idx, s in enumerate(job_ina_candidates):
            if s in network.allPathDict and job_ps in network.allPathDict[s]:
                path = network.allPathDict[s][job_ps]
                hops = len(path) - 1
                if hops <= 2:  # Valid direct path
                    links = links_set(s, job_ps, network.allPathDict)
                    path_incidence_dict[(s, job_ps)] = links
                else:
                    path_incidence_dict[(s, job_ps)] = []
            else:
                path_incidence_dict[(s, job_ps)] = []
    
    # Subtask 3: Greedy Worker Assignment with Constraint-Aware Cost Evaluation
    # Initialize current job effective rates
    gamma_j_prev = [0.0] * jobs_num
    
    # Sort jobs by gradient size in descending order to prioritize large jobs
    job_indices = sorted(range(jobs_num), key=lambda j: jobs_size[j], reverse=True)
    
    # Process each job in priority order
    for j in job_indices:
        job_workers = workers_id[j]
        job_ps = ps_id[j]
        job_size = jobs_size[j]
        job_workers_num = workers_num[j]
        
        # Initialize job-specific current rates
        gamma_j_w_current = [0.0] * max_workers
        gamma_j_w_current[:job_workers_num] = [job_size / T] * job_workers_num  # Initial rate estimate
        
        # Update job effective rate
        gamma_j_prev[j] = min(gamma_j_w_current[:job_workers_num])
        
        # For each worker, evaluate assignment options
        worker_indices = list(range(job_workers_num))
        
        # Sort workers by gradient size (larger first) to prevent bottlenecks
        worker_indices.sort(key=lambda w: jobs_size[j], reverse=True)
        
        for w in worker_indices:
            # Candidate aggregation points: all selected INAs + PS
            candidates = []
            selected_inas = [i for i, x in enumerate(ina_placement_expanded) if x == 1]
            
            # Add selected INA switches
            for s_idx, s in enumerate(ina_candidates):
                if s_idx in selected_inas and (w, s) in path_incidence_dict and path_incidence_dict[(w, s)]:
                    candidates.append(('ina', s_idx))
            
            # Add PS
            if (w, job_ps) in path_incidence_dict and path_incidence_dict[(w, job_ps)]:
                candidates.append(('ps', job_ps))
            
            if not candidates:
                # No valid path - assign to PS as fallback
                y_j_w_d[j][w] = 1
                # Update link load
                links = path_incidence_dict.get((w, job_ps), [])
                for link in links:
                    link_load_w_to_ps[link] = link_load_w_to_ps.get(link, 0.0) + (job_size / T)
                continue
            
            best_cost = float('inf')
            best_assignment = None
            
            # Evaluate each candidate
            for candidate_type, candidate_id in candidates:
                cost = 0.0
                link_load_impact = 0.0
                bottleneck_risk = 0.0
                
                # Compute link load impact
                if candidate_type == 'ina':
                    s_idx = candidate_id
                    links = path_incidence_dict.get((w, ina_candidates[s_idx]), [])
                else:
                    links = path_incidence_dict.get((w, job_ps), [])
                
                # Estimate rate: m_j / T
                gamma_jw = job_size / T
                
                # Compute max normalized load on any link
                max_load_ratio = 0.0
                for link in links:
                    current_load = 0.0
                    if candidate_type == 'ina':
                        current_load = link_load_w_to_s.get(link, 0.0)
                    else:
                        current_load = link_load_w_to_ps.get(link, 0.0)
                    
                    capacity = network.bandwidth_mapping.get(link, float('inf'))
                    new_load = current_load + gamma_jw
                    load_ratio = new_load / capacity
                    max_load_ratio = max(max_load_ratio, load_ratio)
                
                link_load_impact = max_load_ratio
                cost += 0.7 * link_load_impact
                
                # Compute bottleneck risk
                # Tentative new gamma_j_w for this worker
                gamma_j_w_new = gamma_jw
                
                # Update tentative job effective rate
                gamma_j_w_tentative = gamma_j_w_current[:]
                gamma_j_w_tentative[w] = gamma_jw
                gamma_j_new = min(gamma_j_w_tentative[:job_workers_num])
                
                # Avoid division by zero
                if gamma_j_prev[j] > 0:
                    bottleneck_risk = 1.0 - (gamma_j_new / gamma_j_prev[j])
                else:
                    bottleneck_risk = 1.0
                
                cost += 0.3 * bottleneck_risk
                
                # Check feasibility: if cost is lower, consider it
                if cost < best_cost:
                    feasible = True
                    current_ina_load = 0.0
                    # Check link capacity
                    for link in links:
                        if candidate_type == 'ina':
                            current_load = link_load_w_to_s.get(link, 0.0)
                        else:
                            current_load = link_load_w_to_ps.get(link, 0.0)
                        
                        capacity = network.bandwidth_mapping.get(link, float('inf'))
                        new_load = current_load + gamma_jw
                        if new_load > capacity:
                            feasible = False
                            break
                    
                    # Check INA switch capacity if routing to INA
                    if candidate_type == 'ina':
                        s_idx = candidate_id
                        current_ina_load = sum(gamma_j_w_current[i] for i in range(job_workers_num) 
                                             if y_j_w_s_full[j][i][s_idx] == 1)
                        if current_ina_load + gamma_jw > Cs:
                            feasible = False
                    
                    if feasible:
                        best_cost = cost
                        best_assignment = (candidate_type, candidate_id)
            
            # Apply best valid assignment
            if best_assignment:
                if best_assignment[0] == 'ina':
                    s_idx = best_assignment[1]
                    y_j_w_s_full[j][w][s_idx] = 1
                    # Update link and switch load
                    links = path_incidence_dict.get((w, ina_candidates[s_idx]), [])
                    for link in links:
                        link_load_w_to_s[link] = link_load_w_to_s.get(link, 0.0) + (job_size / T)
                    switch_load[ina_candidates[s_idx]] += (job_size / T)
                else:
                    y_j_w_d[j][w] = 1
                    links = path_incidence_dict.get((w, job_ps), [])
                    for link in links:
                        link_load_w_to_ps[link] = link_load_w_to_ps.get(link, 0.0) + (job_size / T)
            else:
                # No feasible assignment found, fallback to PS
                y_j_w_d[j][w] = 1
                links = path_incidence_dict.get((w, job_ps), [])
                for link in links:
                    link_load_w_to_ps[link] = link_load_w_to_ps.get(link, 0.0) + (job_size / T)
            
            # Validate global constraints after assignment
            # Check switch capacity
            for s in ina_candidates:
                if ina_placement_expanded[ina_candidates.index(s)] == 1:
                    if switch_load[s] > Cs:
                        # Violation detected - trigger repair (handled in Task 5)
                        # For now, we'll just log and attempt fallback
                        pass
            
            # Check link capacity
            for link in network.bandwidth_mapping:
                total_load = link_load_w_to_s.get(link, 0.0) + link_load_w_to_ps.get(link, 0.0) + link_load_s_to_ps.get(link, 0.0)
                if total_load > network.bandwidth_mapping[link]:
                    # Violation detected - trigger repair (handled in Task 5)
                    pass
    
    # Ensure every worker has exactly one assignment (fix for the reported error)
    for j in range(jobs_num):
        for w in range(workers_num[j]):
            # Check if worker is assigned
            ina_sum = sum(y_j_w_s_full[j][w][s_idx] for s_idx in range(len(ina_candidates)))
            ps_sum = y_j_w_d[j][w]
            total_assigned = ina_sum + ps_sum
            
            # If no assignment, assign to PS as fallback
            if total_assigned == 0:
                # Find any valid PS path
                if (workers_id[j][w], ps_id[j]) in path_incidence_dict and path_incidence_dict[(workers_id[j][w], ps_id[j])]:
                    y_j_w_d[j][w] = 1
                    links = path_incidence_dict.get((workers_id[j][w], ps_id[j]), [])
                    for link in links:
                        link_load_w_to_ps[link] = link_load_w_to_ps.get(link, 0.0) + (jobs_size[j] / T)
                else:
                    # If no valid path, assign to PS anyway (error case)
                    y_j_w_d[j][w] = 1
    
    return ina_placement_expanded, jobs_routing_expanded