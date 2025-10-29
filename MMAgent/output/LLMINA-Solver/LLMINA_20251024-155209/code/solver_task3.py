# Auto-generated solver with helper function imports
import sys
from pathlib import Path

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
    
    # Get sorted candidate INA switches
    ina_candidates = get_ina_candidates(network)
    max_workers = max(workers_num)
    num_candidates = len(ina_candidates)
    
    # Initialize output (default: all zeros - REPLACE WITH YOUR SOLUTION)
    ina_placement_expanded = [0] * num_candidates
    y_j_w_s_full = [[[0] * num_candidates for _ in range(max_workers)] for _ in range(jobs_num)]
    y_j_w_d = [[0] * max_workers for _ in range(jobs_num)]
    jobs_routing_expanded = [y_j_w_s_full, y_j_w_d]
    
    # === TASK 1: INA Placement via Greedy Heuristic with Capacity-Driven Benefit Estimation ===
    
    # Initialize tracking variables
    selected_switches = set()
    current_link_loads = {edge: 0.0 for edge in network.bandwidth_mapping.keys()}
    
    # Precompute worker-to-switch mappings
    worker_to_switch = {}
    for j in range(jobs_num):
        for w_idx, w_id in enumerate(workers_id[j]):
            # Find which switch connects this worker (assume ToR switch)
            for switch_id in network.tors_id:
                if w_id in network.G.adj[switch_id]:
                    worker_to_switch[(j, w_id)] = switch_id
                    break
    
    # Greedy selection loop for K switches
    for _ in range(K):
        best_switch = -1
        best_score = -1
        
        # Evaluate each unselected candidate switch
        for idx, switch_id in enumerate(ina_candidates):
            if switch_id in selected_switches:
                continue
                
            # Compute proximity gain: number of workers directly connected
            connected_workers = 0
            connected_jobs = set()
            total_volume = 0.0
            for j in range(jobs_num):
                for w_id in workers_id[j]:
                    if (j, w_id) in worker_to_switch and worker_to_switch[(j, w_id)] == switch_id:
                        connected_workers += 1
                        connected_jobs.add(j)
                        total_volume += jobs_size[j]
            
            # Compute congestion alleviation: impact on high-traffic links
            congestion_score = 0.0
            for edge in network.bandwidth_mapping.keys():
                u, v = edge
                if u == switch_id or v == switch_id:
                    load = current_link_loads[edge]
                    capacity = network.bandwidth_mapping[edge]
                    if capacity > 0:
                        congestion_score += (1 - load / capacity)
            
            # Weighted benefit score
            alpha, beta, gamma = 1.0, 1.0, 1.0
            benefit_score = alpha * connected_workers + beta * total_volume + gamma * congestion_score
            
            # Capacity feasibility check: worst-case load on switch
            worst_case_load = sum(jobs_size[j] for j in connected_jobs)
            if worst_case_load > Cs:
                continue  # Skip if exceeds processing capacity
            
            # Update best candidate
            if benefit_score > best_score:
                best_score = benefit_score
                best_switch = idx
        
        # If no valid switch found, break (should not happen if K <= available switches)
        if best_switch == -1:
            break
            
        # Select the best switch
        switch_id = ina_candidates[best_switch]
        selected_switches.add(switch_id)
        ina_placement_expanded[best_switch] = 1
        
        # Update link load estimates: all paths from workers to PS via this switch
        for j in range(jobs_num):
            for w_id in workers_id[j]:
                if (j, w_id) in worker_to_switch and worker_to_switch[(j, w_id)] == switch_id:
                    # Get path from worker to PS
                    path = network.allPathDict[w_id][ps_id[j]]
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i + 1])
                        if edge in current_link_loads:
                            current_link_loads[edge] += jobs_size[j]
                        # Also handle reverse direction if needed
                        rev_edge = (path[i + 1], path[i])
                        if rev_edge in current_link_loads:
                            current_link_loads[rev_edge] += jobs_size[j]
        
    # === TASK 2: Constraint-Aware Worker-to-Aggregation Point Assignment Using Modified Min-Cost Flow with Dynamic Load Propagation ===
    
    # Reset routing assignment to zero
    for j in range(jobs_num):
        for w in range(max_workers):
            for s in range(num_candidates):
                y_j_w_s_full[j][w][s] = 0
            y_j_w_d[j][w] = 0
    
    # Initialize per-switch load tracking
    switch_load = {switch_id: 0.0 for switch_id in ina_candidates}
    
    # For each job, process workers in descending order of gradient volume
    job_worker_list = []
    for j in range(jobs_num):
        workers_with_volume = [(w, jobs_size[j]) for w in range(workers_num[j])]
        # Sort by volume descending
        workers_with_volume.sort(key=lambda x: x[1], reverse=True)
        job_worker_list.append(workers_with_volume)
    
    # Process each job and its workers
    for j in range(jobs_num):
        workers_to_assign = job_worker_list[j]
        job_volume = jobs_size[j]
        
        for w, volume in workers_to_assign:
            best_option = None  # (switch_idx, cost)
            best_cost = float('inf')
            
            # Option 1: Direct to PS
            valid_direct = True
            # Check if direct path exists and doesn't violate link or PS capacity
            direct_path = network.allPathDict[workers_id[j][w]][ps_id[j]]
            direct_links = [(direct_path[i], direct_path[i+1]) for i in range(len(direct_path)-1)]
            
            # Check all links in path
            for edge in direct_links:
                if edge not in current_link_loads:
                    valid_direct = False
                    break
                new_load = current_link_loads[edge] + volume
                capacity = network.bandwidth_mapping[edge]
                if new_load > capacity:
                    valid_direct = False
                    break
            
            # Check PS inflow capacity
            # Find a neighboring switch for PS (assumed to be connected to a switch)
            neighbors = list(network.G.adj[ps_id[j]].keys())
            if neighbors:
                # Use the first neighbor as SD (switch connected to PS)
                sd_id = neighbors[0]
                ps_link = (ps_id[j], sd_id)
                if ps_link in network.bandwidth_mapping:
                    ps_capacity = network.bandwidth_mapping[ps_link]
                    # Estimate current load on PS link from other jobs
                    ps_current_load = 0.0
                    for prev_j in range(jobs_num):
                        if prev_j == j:
                            continue
                        # Check if any worker from prev_j is connected to PS
                        for w_prev in range(workers_num[prev_j]):
                            if workers_id[prev_j][w_prev] in network.G.adj[ps_id[prev_j]]:
                                ps_current_load += jobs_size[prev_j]
                                break
                    if ps_current_load + volume > ps_capacity:
                        valid_direct = False
            
            if valid_direct:
                cost = -volume  # Higher volume = better rate gain
                if cost < best_cost:
                    best_cost = cost
                    best_option = ('direct', -1)
            
            # Option 2: Assign to each selected INA switch
            for s_idx, switch_id in enumerate(ina_candidates):
                if ina_placement_expanded[s_idx] == 0:
                    continue  # Skip unselected switches
                
                # Check if worker is connected to this switch (topology feasibility)
                worker_switch = worker_to_switch.get((j, workers_id[j][w]))
                if worker_switch is None:
                    continue
                
                # Check direct adjacency
                if switch_id not in network.G.adj[worker_switch]:
                    continue
                
                # Get path from worker to INA switch
                path_to_ina = network.allPathDict[workers_id[j][w]][switch_id]
                if len(path_to_ina) < 2:
                    continue  # No valid path
                ina_links = [(path_to_ina[i], path_to_ina[i+1]) for i in range(len(path_to_ina)-1)]
                
                # Check link capacity for this path
                valid_ina = True
                for edge in ina_links:
                    if edge not in current_link_loads:
                        valid_ina = False
                        break
                    new_load = current_link_loads[edge] + volume
                    capacity = network.bandwidth_mapping[edge]
                    if new_load > capacity:
                        valid_ina = False
                        break
                
                # Check INA switch processing capacity
                if valid_ina and switch_load[switch_id] + volume > Cs:
                    valid_ina = False
                
                if valid_ina:
                    # Compute cost: negative of effective rate gain
                    # Higher volume, lower link load → higher gain
                    link_load_sum = sum(current_link_loads[edge] for edge in ina_links if edge in current_link_loads)
                    cost = -(volume / (1 + link_load_sum))
                    if cost < best_cost:
                        best_cost = cost
                        best_option = ('ina', s_idx)
            
            # Assign best valid option
            if best_option is not None:
                if best_option[0] == 'direct':
                    y_j_w_d[j][w] = 1
                    # Update link loads for direct path
                    direct_path = network.allPathDict[workers_id[j][w]][ps_id[j]]
                    for i in range(len(direct_path) - 1):
                        edge = (direct_path[i], direct_path[i+1])
                        if edge in current_link_loads:
                            current_link_loads[edge] += volume
                        rev_edge = (direct_path[i+1], direct_path[i])
                        if rev_edge in current_link_loads:
                            current_link_loads[rev_edge] += volume
                else:
                    s_idx = best_option[1]
                    y_j_w_s_full[j][w][s_idx] = 1
                    # Update link loads for ina path
                    path_to_ina = network.allPathDict[workers_id[j][w]][ina_candidates[s_idx]]
                    for i in range(len(path_to_ina) - 1):
                        edge = (path_to_ina[i], path_to_ina[i+1])
                        if edge in current_link_loads:
                            current_link_loads[edge] += volume
                        rev_edge = (path_to_ina[i+1], path_to_ina[i])
                        if rev_edge in current_link_loads:
                            current_link_loads[rev_edge] += volume
                    # Update switch load
                    switch_load[ina_candidates[s_idx]] += volume
            else:
                # Fallback: assign to PS if still valid
                # Re-check direct path validity
                direct_path = network.allPathDict[workers_id[j][w]][ps_id[j]]
                direct_links = [(direct_path[i], direct_path[i+1]) for i in range(len(direct_path)-1)]
                valid_direct = True
                for edge in direct_links:
                    if edge not in current_link_loads:
                        valid_direct = False
                        break
                    new_load = current_link_loads[edge] + volume
                    capacity = network.bandwidth_mapping[edge]
                    if new_load > capacity:
                        valid_direct = False
                        break
                
                # Check PS inflow capacity
                neighbors = list(network.G.adj[ps_id[j]].keys())
                if neighbors:
                    sd_id = neighbors[0]
                    ps_link = (ps_id[j], sd_id)
                    if ps_link in network.bandwidth_mapping:
                        ps_capacity = network.bandwidth_mapping[ps_link]
                        ps_current_load = 0.0
                        for prev_j in range(jobs_num):
                            if prev_j == j:
                                continue
                            for w_prev in range(workers_num[prev_j]):
                                if workers_id[prev_j][w_prev] in network.G.adj[ps_id[prev_j]]:
                                    ps_current_load += jobs_size[prev_j]
                                    break
                        if ps_current_load + volume > ps_capacity:
                            valid_direct = False
                
                if valid_direct:
                    y_j_w_d[j][w] = 1
                    # Update link loads
                    for i in range(len(direct_path) - 1):
                        edge = (direct_path[i], direct_path[i+1])
                        if edge in current_link_loads:
                            current_link_loads[edge] += volume
                        rev_edge = (direct_path[i+1], direct_path[i])
                        if rev_edge in current_link_loads:
                            current_link_loads[rev_edge] += volume
                else:
                    # No valid assignment: keep as zero (should not happen with proper topology)
                    # But we must ensure exactly one assignment per worker
                    # This is a critical fix: if no valid assignment, we must assign to PS anyway
                    # But only if PS path exists and doesn't violate capacity
                    # Re-check PS path again with current state
                    direct_path = network.allPathDict[workers_id[j][w]][ps_id[j]]
                    direct_links = [(direct_path[i], direct_path[i+1]) for i in range(len(direct_path)-1)]
                    valid_direct = True
                    for edge in direct_links:
                        if edge not in current_link_loads:
                            valid_direct = False
                            break
                        new_load = current_link_loads[edge] + volume
                        capacity = network.bandwidth_mapping[edge]
                        if new_load > capacity:
                            valid_direct = False
                            break
                    
                    neighbors = list(network.G.adj[ps_id[j]].keys())
                    if neighbors:
                        sd_id = neighbors[0]
                        ps_link = (ps_id[j], sd_id)
                        if ps_link in network.bandwidth_mapping:
                            ps_capacity = network.bandwidth_mapping[ps_link]
                            ps_current_load = 0.0
                            for prev_j in range(jobs_num):
                                if prev_j == j:
                                    continue
                                for w_prev in range(workers_num[prev_j]):
                                    if workers_id[prev_j][w_prev] in network.G.adj[ps_id[prev_j]]:
                                        ps_current_load += jobs_size[prev_j]
                                        break
                            if ps_current_load + volume > ps_capacity:
                                valid_direct = False
                    
                    if valid_direct:
                        y_j_w_d[j][w] = 1
                        for i in range(len(direct_path) - 1):
                            edge = (direct_path[i], direct_path[i+1])
                            if edge in current_link_loads:
                                current_link_loads[edge] += volume
                            rev_edge = (direct_path[i+1], direct_path[i])
                            if rev_edge in current_link_loads:
                                current_link_loads[rev_edge] += volume
                    else:
                        # Final fallback: assign to PS even if it violates capacity
                        # This should not happen in valid inputs, but we must ensure constraint is met
                        # Set PS assignment regardless of capacity (though this may be infeasible)
                        y_j_w_d[j][w] = 1
                        # Update link loads for PS path
                        for i in range(len(direct_path) - 1):
                            edge = (direct_path[i], direct_path[i+1])
                            if edge in current_link_loads:
                                current_link_loads[edge] += volume
                            rev_edge = (direct_path[i+1], direct_path[i])
                            if rev_edge in current_link_loads:
                                current_link_loads[rev_edge] += volume
            # Ensure exactly one assignment per worker
            # This is guaranteed by the above logic
            total_assignments = sum(y_j_w_s_full[j][w][s] for s in range(num_candidates)) + y_j_w_d[j][w]
            if total_assignments != 1:
                # Force assignment to PS as last resort
                # Reset all assignments
                for s in range(num_candidates):
                    y_j_w_s_full[j][w][s] = 0
                y_j_w_d[j][w] = 1
                # Update link loads for PS path
                direct_path = network.allPathDict[workers_id[j][w]][ps_id[j]]
                for i in range(len(direct_path) - 1):
                    edge = (direct_path[i], direct_path[i+1])
                    if edge in current_link_loads:
                        current_link_loads[edge] += volume
                    rev_edge = (direct_path[i+1], direct_path[i])
                    if rev_edge in current_link_loads:
                        current_link_loads[rev_edge] += volume
    
    # === TASK 3: Flow Consistency Enforcement and Binary-Variable Structuring for Feasibility and Model Compatibility ===
    
    # Create a mapping from selected INA switch IDs to their indices in ina_candidates
    # This ensures proper alignment of y_j_w_s_full with the MILP variable indexing
    ina_id = [ina_candidates[i] for i in range(len(ina_candidates)) if ina_placement_expanded[i] == 1]
    switch_id_to_idx = {switch_id: idx for idx, switch_id in enumerate(ina_candidates)}
    
    # Reconstruct the final assignment matrices with correct binary structure
    # Ensure y_j_w_s_full[j][w][s] is 1 only if worker w is assigned to INA switch s
    # and that each worker is assigned to exactly one aggregation point
    for j in range(jobs_num):
        for w in range(max_workers):
            # Reset assignments for this worker
            for s in range(num_candidates):
                y_j_w_s_full[j][w][s] = 0
            y_j_w_d[j][w] = 0
            
            # If worker is inactive, keep all zero
            if w >= workers_num[j]:
                continue
                
            # Find the assigned aggregation point from previous routing
            assigned_to_ina = False
            assigned_to_ps = False
            for s_idx in range(num_candidates):
                if y_j_w_s_full[j][w][s_idx] == 1:
                    # Verify that the assigned switch is actually selected
                    switch_id = ina_candidates[s_idx]
                    if switch_id in ina_id:
                        # Set the binary flag only if valid
                        y_j_w_s_full[j][w][s_idx] = 1
                        assigned_to_ina = True
                        break
            if not assigned_to_ina:
                # Check direct PS assignment
                if y_j_w_d[j][w] == 1:
                    y_j_w_d[j][w] = 1
                    assigned_to_ps = True
            
            # Enforce exclusivity: exactly one assignment per worker
            total_assignments = sum(y_j_w_s_full[j][w][s] for s in range(num_candidates)) + y_j_w_d[j][w]
            if total_assignments != 1:
                # Force assignment to PS if no valid assignment was found
                for s in range(num_candidates):
                    y_j_w_s_full[j][w][s] = 0
                y_j_w_d[j][w] = 1
            
            # Enforce y_jws <= x_s: only selected switches can be assigned
            for s_idx in range(num_candidates):
                switch_id = ina_candidates[s_idx]
                if ina_placement_expanded[s_idx] == 0:  # Unselected switch
                    y_j_w_s_full[j][w][s_idx] = 0  # Cannot assign to unselected switch
    
    # === TASK 4: Constraint-Enforced Flow Conservation and Load Propagation (Placeholder) ===
    # TODO: Will be optimized in Task 4
    
    # === TASK 5: Makespan Evaluation and Solution Finalization (Placeholder) ===
    # TODO: Will be optimized in Task 5
    
    # Final output
    return ina_placement_expanded, jobs_routing_expanded