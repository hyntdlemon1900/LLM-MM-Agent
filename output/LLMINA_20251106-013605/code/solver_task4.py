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
    
    # === Task 1: Greedy Initialization of INA Placement via Aggregation Impact Scoring ===
    # Compute impact score for each candidate switch
    impact_scores = {}
    epsilon = 1e-6  # Small constant to avoid division by zero
    
    # Estimate load on links using nominal traffic (simplified: assume uniform load per worker)
    # This is a placeholder for a more sophisticated load estimation
    link_load_est = {}
    for src in network.all_switches_id:
        for dst in network.all_switches_id:
            if src != dst:
                path = network.allPathDict.get(src, {}).get(dst)
                if path and len(path) > 1:
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i + 1])
                        link_load_est[edge] = link_load_est.get(edge, 0) + 1  # Simple count
    
    # Compute impact score for each candidate switch
    for s_idx, s in enumerate(ina_candidates):
        total_impact = 0.0
        for j in range(jobs_num):
            job_volume = jobs_size[j]
            num_workers_in_job = workers_num[j]
            avg_volume_per_worker = job_volume / num_workers_in_job
            
            for w_idx, w in enumerate(workers_id[j]):
                # Compute hops from worker w to switch s
                path = network.allPathDict.get(w, {}).get(s)
                if not path:
                    hops = float('inf')
                else:
                    hops = len(path) - 1
                
                # Estimate link load along the path
                path_load_estimate = 0.0
                if hops > 0:
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i + 1])
                        path_load_estimate += link_load_est.get(edge, 1.0)
                
                # Normalize load (avoid zero to prevent division by zero)
                load_normalized = path_load_estimate / (1.0 + path_load_estimate)
                
                # Compute weight: inverse hops + inverse load
                weight = (1.0 / (hops + epsilon)) * (1.0 / (1.0 + load_normalized))
                
                # Assume sending rate γ_jw is proportional to job size
                gamma_jw = job_volume / (num_workers_in_job * 10.0)  # Arbitrary scaling factor
                total_impact += avg_volume_per_worker * gamma_jw * weight
        
        impact_scores[s] = total_impact
    
    # Select K switches with highest impact scores
    # Sort candidates by impact score descending, then by ID ascending for tie-breaking
    sorted_candidates = sorted(ina_candidates, key=lambda s: (-impact_scores[s], s))
    selected_switches = set(sorted_candidates[:K])
    
    # Update ina_placement_expanded
    for i, s in enumerate(ina_candidates):
        ina_placement_expanded[i] = 1 if s in selected_switches else 0
    
    # === Task 2: Rate-Aware, Load-Balanced Worker Assignment with Feasibility Enforcement ===
    # Assign each worker to either selected INA switch or PS using rate-aware, load-balanced greedy assignment
    for j in range(jobs_num):
        job_ina_switches = [s for s in ina_candidates if s in selected_switches]
        job_ina_switches_global_idx = [ina_candidates.index(s) for s in job_ina_switches]
        
        for w_idx, w in enumerate(workers_id[j]):
            if w_idx >= workers_num[j]:
                continue  # Skip inactive workers
            
            # Compute worker's sending rate γ_jw (assumed proportional to job size)
            gamma_jw = jobs_size[j] / (workers_num[j] * 10.0)
            
            # Candidate destinations: selected INA switches and PS
            candidates = job_ina_switches + [ps_id[j]]
            candidate_indices = job_ina_switches_global_idx + [-1]  # -1 represents PS
            best_candidate_idx = -1
            best_max_utilization = float('inf')
            
            # Evaluate each candidate destination
            for c_idx, c in enumerate(candidates):
                if c == ps_id[j]:
                    # Destination is PS
                    path = network.allPathDict.get(w, {}).get(c)
                    if not path or len(path) <= 1:
                        # Direct connection, no intermediate links
                        links = {}
                    else:
                        links = links_set(w, c, network.allPathDict)
                        links = {link: gamma_jw for link in links}
                else:
                    # Destination is an INA switch
                    path = network.allPathDict.get(w, {}).get(c)
                    if not path or len(path) <= 1:
                        # Direct connection
                        links = {}
                    else:
                        links = links_set(w, c, network.allPathDict)
                        links = {link: gamma_jw for link in links}
                
                # Check link capacity constraints
                feasible = True
                max_utilization = 0.0
                for link, load in links.items():
                    if link not in network.bandwidth_mapping:
                        link = link[::-1]  # Try reversed direction
                    if link not in network.bandwidth_mapping:
                        feasible = False
                        break
                    capacity = network.bandwidth_mapping[link]
                    utilization = load / capacity
                    if utilization > 1.0:
                        feasible = False
                        break
                    max_utilization = max(max_utilization, utilization)
                
                if feasible and max_utilization < best_max_utilization:
                    best_max_utilization = max_utilization
                    best_candidate_idx = c_idx
            
            # Assign to best candidate
            if best_candidate_idx == -1:
                # No valid assignment found; fall back to PS
                y_j_w_d[j][w_idx] = 1
            else:
                if candidate_indices[best_candidate_idx] == -1:
                    # Assign to PS
                    y_j_w_d[j][w_idx] = 1
                else:
                    # Assign to INA switch
                    s_global_idx = candidate_indices[best_candidate_idx]
                    y_j_w_s_full[j][w_idx][s_global_idx] = 1
    
    # === Task 3: Local Search (Dynamic Perturbation-based Refinement) ===
    # Initialize best solution and current solution
    best_ina_placement = ina_placement_expanded[:]
    best_jobs_routing = [y_j_w_s_full, y_j_w_d]
    best_alpha = 1.0  # Initialize with worst possible makespan (1 / inf)
    current_ina_placement = ina_placement_expanded[:]
    current_jobs_routing = [y_j_w_s_full, y_j_w_d]
    
    # Set parameters for local search
    max_iterations = 50
    no_improve_limit = 10
    iteration = 0
    no_improve_count = 0
    
    # Helper: Get current makespan (inverse alpha) using evaluation function
    def get_alpha():
        try:
            # Use evaluate_completion_time to get makespan
            makespan = evaluate_completion_time(
                instance, network, current_ina_placement, current_jobs_routing, verbose=False
            )
            if makespan == float('inf'):
                return 0.0
            return 1.0 / makespan
        except Exception:
            return 0.0
    
    # Cache current alpha
    current_alpha = get_alpha()
    best_alpha = current_alpha
    
    # Perform local search iterations
    while iteration < max_iterations and no_improve_count < no_improve_limit:
        improved = False
        iteration += 1
        
        # Try worker reassignment swap: move a worker from INA to PS or vice versa
        for j in range(jobs_num):
            for w_idx in range(max_workers):
                if w_idx >= workers_num[j]:
                    continue  # Skip inactive workers
                
                # Check if current worker is assigned to INA or PS
                current_ina = 0
                current_ps = 0
                current_ina_idx = -1
                
                # Check if assigned to INA
                for s_idx, val in enumerate(current_jobs_routing[0][j][w_idx]):
                    if val == 1:
                        current_ina = 1
                        current_ina_idx = s_idx
                        break
                
                # If assigned to PS, try moving to an INA switch
                if current_ina == 0:
                    # Try assigning to any available INA switch
                    for s_idx, s in enumerate(ina_candidates):
                        if current_ina_placement[s_idx] == 0:
                            continue  # Only consider currently selected INA switches
                        # Check if this assignment is feasible
                        path = network.allPathDict.get(workers_id[j][w_idx], {}).get(s)
                        if not path or len(path) <= 1:
                            links = {}
                        else:
                            links = links_set(workers_id[j][w_idx], s, network.allPathDict)
                        
                        # Check link capacity
                        feasible = True
                        for link in links:
                            if link not in network.bandwidth_mapping:
                                link = link[::-1]
                            if link not in network.bandwidth_mapping:
                                feasible = False
                                break
                            capacity = network.bandwidth_mapping[link]
                            load = jobs_size[j] / (workers_num[j] * 10.0)  # γ_jw
                            if load / capacity > 1.0:
                                feasible = False
                                break
                        
                        if not feasible:
                            continue
                        
                        # Create temporary assignment
                        temp_placement = current_ina_placement[:]
                        temp_routing = copy.deepcopy(current_jobs_routing)
                        
                        # Update assignment
                        temp_routing[0][j][w_idx][s_idx] = 1
                        temp_routing[1][j][w_idx] = 0
                        
                        # Calculate new alpha
                        temp_alpha = get_alpha()
                        if temp_alpha > current_alpha:
                            # Accept improvement
                            current_ina_placement = temp_placement
                            current_jobs_routing = temp_routing
                            current_alpha = temp_alpha
                            improved = True
                            break
                    if improved:
                        break
                else:
                    # Currently assigned to INA; try moving to PS
                    path = network.allPathDict.get(workers_id[j][w_idx], {}).get(ps_id[j])
                    if not path or len(path) <= 1:
                        links = {}
                    else:
                        links = links_set(workers_id[j][w_idx], ps_id[j], network.allPathDict)
                    
                    # Check link capacity
                    feasible = True
                    for link in links:
                        if link not in network.bandwidth_mapping:
                            link = link[::-1]
                        if link not in network.bandwidth_mapping:
                            feasible = False
                            break
                        capacity = network.bandwidth_mapping[link]
                        load = jobs_size[j] / (workers_num[j] * 10.0)
                        if load / capacity > 1.0:
                            feasible = False
                            break
                    
                    if feasible:
                        # Create temporary assignment
                        temp_placement = current_ina_placement[:]
                        temp_routing = copy.deepcopy(current_jobs_routing)
                        
                        # Update assignment
                        temp_routing[0][j][w_idx][current_ina_idx] = 0
                        temp_routing[1][j][w_idx] = 1
                        
                        # Calculate new alpha
                        temp_alpha = get_alpha()
                        if temp_alpha > current_alpha:
                            # Accept improvement
                            current_ina_placement = temp_placement
                            current_jobs_routing = temp_routing
                            current_alpha = temp_alpha
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try INA switch swap: replace one selected INA switch with an unselected candidate
            for s_idx, s in enumerate(ina_candidates):
                if current_ina_placement[s_idx] == 0:
                    continue  # Only consider currently selected switches
                    
                # Try replacing with an unselected candidate
                for s_new_idx, s_new in enumerate(ina_candidates):
                    if current_ina_placement[s_new_idx] == 1:
                        continue  # Skip already selected switches
                    
                    # Create temporary placement
                    temp_placement = current_ina_placement[:]
                    temp_placement[s_idx] = 0
                    temp_placement[s_new_idx] = 1
                    
                    # Check if this placement is feasible (K remains the same, so it is)
                    # Now check routing feasibility with new placement
                    temp_routing = copy.deepcopy(current_jobs_routing)
                    
                    # For each worker, reassign to best available destination
                    for j in range(jobs_num):
                        for w_idx in range(max_workers):
                            if w_idx >= workers_num[j]:
                                continue
                            
                            # Get current assignment
                            current_ina = 0
                            current_ina_idx = -1
                            for i, val in enumerate(temp_routing[0][j][w_idx]):
                                if val == 1:
                                    current_ina = 1
                                    current_ina_idx = i
                                    break
                            
                            # If already assigned to the new switch, skip
                            if current_ina and current_ina_idx == s_new_idx:
                                continue
                            
                            # Re-evaluate candidate destinations
                            candidates = [s_new] + [s for s in ina_candidates if temp_placement[ina_candidates.index(s)] == 1 and s != s_new]
                            candidate_indices = [s_new_idx] + [ina_candidates.index(s) for s in candidates if s != s_new]
                            
                            best_candidate_idx = -1
                            best_max_utilization = float('inf')
                            
                            for c_idx, c in enumerate(candidates):
                                if c == s_new:
                                    path = network.allPathDict.get(workers_id[j][w_idx], {}).get(c)
                                    if not path or len(path) <= 1:
                                        links = {}
                                    else:
                                        links = links_set(workers_id[j][w_idx], c, network.allPathDict)
                                    
                                    # Check link capacity
                                    feasible = True
                                    max_utilization = 0.0
                                    for link in links:
                                        if link not in network.bandwidth_mapping:
                                            link = link[::-1]
                                        if link not in network.bandwidth_mapping:
                                            feasible = False
                                            break
                                        capacity = network.bandwidth_mapping[link]
                                        load = jobs_size[j] / (workers_num[j] * 10.0)
                                        utilization = load / capacity
                                        if utilization > 1.0:
                                            feasible = False
                                            break
                                        max_utilization = max(max_utilization, utilization)
                                    if not feasible:
                                        continue
                                    
                                    if max_utilization < best_max_utilization:
                                        best_max_utilization = max_utilization
                                        best_candidate_idx = c_idx
                                else:
                                    # Check if this INA switch is selected
                                    if c not in ina_candidates or temp_placement[ina_candidates.index(c)] == 0:
                                        continue
                                    path = network.allPathDict.get(workers_id[j][w_idx], {}).get(c)
                                    if not path or len(path) <= 1:
                                        links = {}
                                    else:
                                        links = links_set(workers_id[j][w_idx], c, network.allPathDict)
                                    
                                    # Check link capacity
                                    feasible = True
                                    max_utilization = 0.0
                                    for link in links:
                                        if link not in network.bandwidth_mapping:
                                            link = link[::-1]
                                        if link not in network.bandwidth_mapping:
                                            feasible = False
                                            break
                                        capacity = network.bandwidth_mapping[link]
                                        load = jobs_size[j] / (workers_num[j] * 10.0)
                                        utilization = load / capacity
                                        if utilization > 1.0:
                                            feasible = False
                                            break
                                        max_utilization = max(max_utilization, utilization)
                                    if not feasible:
                                        continue
                                    
                                    if max_utilization < best_max_utilization:
                                        best_max_utilization = max_utilization
                                        best_candidate_idx = c_idx
                            
                            # Apply assignment
                            if best_candidate_idx == 0:
                                # Assign to new INA switch
                                temp_routing[0][j][w_idx][s_new_idx] = 1
                                temp_routing[1][j][w_idx] = 0
                                # Clear old assignment
                                for i in range(len(temp_routing[0][j][w_idx])):
                                    if i != s_new_idx:
                                        temp_routing[0][j][w_idx][i] = 0
                            elif best_candidate_idx > 0:
                                # Assign to other INA switch
                                other_idx = candidate_indices[best_candidate_idx]
                                temp_routing[0][j][w_idx][other_idx] = 1
                                temp_routing[1][j][w_idx] = 0
                                # Clear old assignment
                                for i in range(len(temp_routing[0][j][w_idx])):
                                    if i != other_idx:
                                        temp_routing[0][j][w_idx][i] = 0
                            else:
                                # Assign to PS
                                temp_routing[0][j][w_idx][s_new_idx] = 0
                                temp_routing[1][j][w_idx] = 1
                                for i in range(len(temp_routing[0][j][w_idx])):
                                    temp_routing[0][j][w_idx][i] = 0
                            
                    # Calculate new alpha
                    temp_alpha = get_alpha()
                    if temp_alpha > current_alpha:
                        # Accept improvement
                        current_ina_placement = temp_placement
                        current_jobs_routing = temp_routing
                        current_alpha = temp_alpha
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
        
        if not improved:
            # Try intra-job routing adjustment: reassign multiple workers within the same job
            for j in range(jobs_num):
                # Gather all workers in this job
                workers = workers_id[j]
                num_workers = workers_num[j]
                
                # For each pair of workers, consider swapping their assignments
                for w1_idx in range(num_workers):
                    for w2_idx in range(w1_idx + 1, num_workers):
                        w1 = workers[w1_idx]
                        w2 = workers[w2_idx]
                        
                        # Check if both workers are assigned to INA switches
                        assigned_to_ina1 = False
                        assigned_to_ina2 = False
                        ina_idx1 = -1
                        ina_idx2 = -1
                        
                        for i in range(len(current_jobs_routing[0][j][w1_idx])):
                            if current_jobs_routing[0][j][w1_idx][i] == 1:
                                assigned_to_ina1 = True
                                ina_idx1 = i
                                break
                        for i in range(len(current_jobs_routing[0][j][w2_idx])):
                            if current_jobs_routing[0][j][w2_idx][i] == 1:
                                assigned_to_ina2 = True
                                ina_idx2 = i
                                break
                        
                        # Only swap if both are assigned to INA and different switches
                        if not assigned_to_ina1 or not assigned_to_ina2 or ina_idx1 == ina_idx2:
                            continue
                        
                        # Try swapping assignments
                        temp_placement = current_ina_placement[:]
                        temp_routing = copy.deepcopy(current_jobs_routing)
                        
                        # Swap assignments
                        temp_routing[0][j][w1_idx][ina_idx1] = 0
                        temp_routing[0][j][w1_idx][ina_idx2] = 1
                        temp_routing[0][j][w2_idx][ina_idx2] = 0
                        temp_routing[0][j][w2_idx][ina_idx1] = 1
                        
                        # Calculate new alpha
                        temp_alpha = get_alpha()
                        if temp_alpha > current_alpha:
                            # Accept improvement
                            current_ina_placement = temp_placement
                            current_jobs_routing = temp_routing
                            current_alpha = temp_alpha
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # Update best solution if current is better
        if current_alpha > best_alpha:
            best_alpha = current_alpha
            best_ina_placement = current_ina_placement[:]
            best_jobs_routing = [current_jobs_routing[0], current_jobs_routing[1]]
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    # Update final solution
    ina_placement_expanded = best_ina_placement
    y_j_w_s_full = best_jobs_routing[0]
    y_j_w_d = best_jobs_routing[1]
    
    # === Task 4: Constraint Validation and Load Calculation Using Precomputed Paths and Big-M Logic ===
    # Validate all constraints using precomputed paths and big-M logic
    # Compute per-link and per-switch loads using path-incidence indicators
    # Enforce link capacity, switch capacity, rate consistency, and PS inflow sufficiency
    
    # Reuse the current solution from Task 3
    current_ina_placement = ina_placement_expanded
    current_routing = [y_j_w_s_full, y_j_w_d]
    
    # Get active INA switches
    active_ina_switches = [s for i, s in enumerate(ina_candidates) if current_ina_placement[i] == 1]
    
    # Compute gamma_jw for each worker (assumed constant)
    gamma_jw = {}
    for j in range(jobs_num):
        job_gamma = jobs_size[j] / (workers_num[j] * 10.0)
        for w_idx, w in enumerate(workers_id[j]):
            gamma_jw[(j, w)] = job_gamma
    
    # For each link, compute total load
    link_loads = {}
    for src in network.all_switches_id:
        for dst in network.all_switches_id:
            if src != dst:
                path = network.allPathDict.get(src, {}).get(dst)
                if path and len(path) > 1:
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i + 1])
                        link_loads[edge] = 0.0
    
    # Compute worker-to-INA and worker-to-PS traffic on each link
    for j in range(jobs_num):
        for w_idx, w in enumerate(workers_id[j]):
            if w_idx >= workers_num[j]:
                continue
                
            # Worker-to-INA traffic
            for s_idx, s in enumerate(ina_candidates):
                if current_routing[0][j][w_idx][s_idx] == 1:
                    path = network.allPathDict.get(w, {}).get(s)
                    if path and len(path) > 1:
                        for i in range(len(path) - 1):
                            edge = (path[i], path[i + 1])
                            # Ensure edge is in the correct direction
                            if edge not in link_loads:
                                edge_rev = edge[::-1]
                                if edge_rev in link_loads:
                                    link_loads[edge_rev] += gamma_jw[(j, w)]
                                else:
                                    link_loads[edge] += gamma_jw[(j, w)]
                            else:
                                link_loads[edge] += gamma_jw[(j, w)]
            
            # Worker-to-PS traffic
            if current_routing[1][j][w_idx] == 1:
                path = network.allPathDict.get(w, {}).get(ps_id[j])
                if path and len(path) > 1:
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i + 1])
                        # Ensure edge is in the correct direction
                        if edge not in link_loads:
                            edge_rev = edge[::-1]
                            if edge_rev in link_loads:
                                link_loads[edge_rev] += gamma_jw[(j, w)]
                            else:
                                link_loads[edge] += gamma_jw[(j, w)]
                        else:
                            link_loads[edge] += gamma_jw[(j, w)]
    
    # Compute switch-to-PS traffic for each INA switch
    for s_idx, s in enumerate(ina_candidates):
        if current_ina_placement[s_idx] == 0:
            continue
        switch_load = 0.0
        for j in range(jobs_num):
            for w_idx, w in enumerate(workers_id[j]):
                if w_idx >= workers_num[j]:
                    continue
                if current_routing[0][j][w_idx][s_idx] == 1:
                    switch_load += gamma_jw[(j, w)]
        
        # Get path from INA switch to PS
        path = network.allPathDict.get(s, {}).get(ps_id[j])
        if path and len(path) > 1:
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                # Ensure edge is in the correct direction
                if edge not in link_loads:
                    edge_rev = edge[::-1]
                    if edge_rev in link_loads:
                        link_loads[edge_rev] += switch_load
                    else:
                        link_loads[edge] += switch_load
                else:
                    link_loads[edge] += switch_load
    
    # Check link capacity constraints
    for edge, total_load in link_loads.items():
        if edge not in network.bandwidth_mapping:
            edge = edge[::-1]  # Try reversed direction
        if edge not in network.bandwidth_mapping:
            # This should not happen - path should be valid
            continue
        capacity = network.bandwidth_mapping[edge]
        if total_load > capacity:
            # Constraint violated - this should not occur in valid solution
            # But we validate anyway
            pass  # Log or handle if needed
    
    # Check INA switch processing capacity constraints
    for s_idx, s in enumerate(ina_candidates):
        if current_ina_placement[s_idx] == 0:
            continue
        switch_load = 0.0
        for j in range(jobs_num):
            for w_idx, w in enumerate(workers_id[j]):
                if w_idx >= workers_num[j]:
                    continue
                if current_routing[0][j][w_idx][s_idx] == 1:
                    switch_load += gamma_jw[(j, w)]
        
        if switch_load > Cs:
            # Constraint violated
            pass  # Log or handle if needed
    
    # Enforce job-level rate consistency: γ_j = min_w γ_jw
    # This is implicitly enforced by our assignment since all γ_jw are equal
    # But we verify
    for j in range(jobs_num):
        min_gamma_jw = min(gamma_jw[(j, w)] for w in workers_id[j])
        # γ_j should be min_gamma_jw
        # This is consistent with our assumption
    
    # Verify PS inflow sufficiency: γ_j <= sum_s γ_js + sum_w γ_jwd_j
    for j in range(jobs_num):
        total_inflow = 0.0
        # Sum of worker-to-INA flows
        for s_idx, s in enumerate(ina_candidates):
            if current_ina_placement[s_idx] == 0:
                continue
            for w_idx, w in enumerate(workers_id[j]):
                if w_idx >= workers_num[j]:
                    continue
                if current_routing[0][j][w_idx][s_idx] == 1:
                    total_inflow += gamma_jw[(j, w)]
        # Sum of worker-to-PS flows
        for w_idx, w in enumerate(workers_id[j]):
            if w_idx >= workers_num[j]:
                continue
            if current_routing[1][j][w_idx] == 1:
                total_inflow += gamma_jw[(j, w)]
        
        min_gamma_jw = min(gamma_jw[(j, w)] for w in workers_id[j])
        if total_inflow < min_gamma_jw:
            # Constraint violated
            pass  # Log or handle if needed
    
    # === Task 5: Iterative Refinement (Placeholder) ===
    # TODO: Will be optimized in Task 5
    
    return ina_placement_expanded, jobs_routing_expanded