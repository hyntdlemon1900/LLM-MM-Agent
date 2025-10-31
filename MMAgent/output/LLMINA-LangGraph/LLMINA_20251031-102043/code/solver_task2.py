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
    
    # Initialize routing output (placeholder for now)
    max_workers = max(workers_num)
    num_candidates = len(ina_candidates)
    
    y_j_w_s_full = [[[0] * num_candidates for _ in range(max_workers)] for _ in range(jobs_num)]
    y_j_w_d = [[0] * max_workers for _ in range(jobs_num)]
    jobs_routing_expanded = [y_j_w_s_full, y_j_w_d]
    
    # Pre-Computation: Feasible Paths and Path-Load Matrix Initialization
    # Initialize link load trackers
    link_load_w_to_s = {link: 0.0 for link in network.bandwidth_mapping.keys()}
    link_load_w_to_ps = {link: 0.0 for link in network.bandwidth_mapping.keys()}
    link_load_s_to_ps = {link: 0.0 for link in network.bandwidth_mapping.keys()}
    
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
    
    # TODO: Will be optimized in Task 3-6
    # For now, assign randomly (but valid) — placeholder only
    for j in range(jobs_num):
        for w in range(workers_num[j]):
            # Randomly assign to either PS or one of the INAs (if any selected)
            if any(ina_placement_expanded):
                # Choose a random INA from selected ones
                selected_inas = [i for i, x in enumerate(ina_placement_expanded) if x == 1]
                if selected_inas:
                    chosen_s = selected_inas[0]  # Simple: pick first selected
                    y_j_w_s_full[j][w][chosen_s] = 1
                else:
                    y_j_w_d[j][w] = 1
            else:
                y_j_w_d[j][w] = 1
    
    return ina_placement_expanded, jobs_routing_expanded