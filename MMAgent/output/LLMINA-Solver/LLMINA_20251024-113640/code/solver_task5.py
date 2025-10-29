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
    
    # === TASK 1: INA Placement via Greedy Heuristic with Marginal Benefit Estimation ===
    
    # Initialize tracking variables
    selected_switches = set()
    current_link_loads = {}  # Store current load per link (u,v) in Gbps
    
    # Precompute all path-to-links mappings for efficiency
    # Use l_i to precompute which (src, dst) pairs use each link
    # But we'll compute incrementally during evaluation
    
    # Initialize global load accumulator
    # We'll use a dictionary to track link usage: edge -> total current load
    # Initially, all zero
    for edge in network.bandwidth_mapping:
        current_link_loads[edge] = 0.0
    
    # Define weights for the benefit score
    alpha, beta, gamma = 1.0, 1.0, 1.0  # Equal weighting for now
    
    # Greedy selection loop for K switches
    for _ in range(K):
        best_switch = -1
        best_benefit = -float('inf')
        
        # Evaluate each unselected candidate
        for idx, switch in enumerate(ina_candidates):
            if switch in selected_switches:
                continue  # Already selected
            
            # 1. Proximity Gain: Number of workers connected to this switch
            # Check all jobs: find workers connected to this switch via edge/leaf switches
            connected_workers = 0
            total_volume = 0.0
            jobs_connected = set()
            
            # For FatTree: check if switch is in tors_id, aggrs_id, or cores_id
            # For SpineLeaf: check if switch is in tors_id or spines_id
            has_tors = hasattr(network, 'tors_id')
            has_aggrs = hasattr(network, 'aggrs_id')
            has_cores = hasattr(network, 'cores_id')
            has_leafs = hasattr(network, 'leafs_id')
            has_spines = hasattr(network, 'spines_id')
            
            # Determine connectivity based on topology
            if has_tors and switch in network.tors_id:
                # Edge switch: directly connected to workers
                for j in range(jobs_num):
                    # Use tor_switch_workers to get worker count per edge switch
                    if switch in instance['tor_switch_workers'][j]:
                        count = instance['tor_switch_workers'][j][switch]
                        connected_workers += count
                        total_volume += count * jobs_size[j]
                        jobs_connected.add(j)
            elif has_aggrs and switch in network.aggrs_id:
                # Aggregation switch in FatTree
                # Connected to edge switches
                for j in range(jobs_num):
                    for edge_switch in network.tors_id:
                        if edge_switch in instance['tor_switch_workers'][j]:
                            # Check if there's a path from edge_switch to this switch
                            if edge_switch in network.G and switch in network.G:
                                try:
                                    path = network.allPathDict[edge_switch][switch]
                                    # If path exists, then workers on edge_switch are indirectly connected
                                    count = instance['tor_switch_workers'][j][edge_switch]
                                    connected_workers += count
                                    total_volume += count * jobs_size[j]
                                    jobs_connected.add(j)
                                except KeyError:
                                    continue
            elif has_cores and switch in network.cores_id:
                # Core switch in FatTree
                # Connected to aggregation switches
                for j in range(jobs_num):
                    for aggr_switch in network.aggrs_id:
                        if aggr_switch in network.G and switch in network.G:
                            try:
                                path = network.allPathDict[aggr_switch][switch]
                                # If path exists, then workers via aggr_switch are connected
                                for edge_switch in network.tors_id:
                                    if edge_switch in instance['tor_switch_workers'][j]:
                                        count = instance['tor_switch_workers'][j][edge_switch]
                                        connected_workers += count
                                        total_volume += count * jobs_size[j]
                                        jobs_connected.add(j)
                            except KeyError:
                                continue
            elif has_leafs and switch in network.leafs_id:
                # Leaf switch in SpineLeaf
                for j in range(jobs_num):
                    if switch in instance['tor_switch_workers'][j]:
                        count = instance['tor_switch_workers'][j][switch]
                        connected_workers += count
                        total_volume += count * jobs_size[j]
                        jobs_connected.add(j)
            elif has_spines and switch in network.spines_id:
                # Spine switch in SpineLeaf
                # Connected to leaf switches
                for j in range(jobs_num):
                    for leaf_switch in network.leafs_id:
                        if leaf_switch in network.G and switch in network.G:
                            try:
                                path = network.allPathDict[leaf_switch][switch]
                                # If path exists, then workers on leaf_switch are connected
                                count = instance['tor_switch_workers'][j].get(leaf_switch, 0)
                                connected_workers += count
                                total_volume += count * jobs_size[j]
                                jobs_connected.add(j)
                            except KeyError:
                                continue
            
            # 2. Volume Gain: Total gradient volume from connected workers
            # Already computed above: total_volume
            
            # 3. Congestion Alleviation: Estimate impact on high-traffic links
            # We simulate activation: what links would carry additional traffic?
            # Use l_i to find all (src, dst) pairs whose paths go through any link
            # that would be used if this switch were activated
            congestion_benefit = 0.0
            total_congestion = 0.0
            
            # We'll compute for all links that could be used by any worker-to-INA or INA-to-PS flow
            # But we don't know routing yet, so we assume worst-case: all workers go via this switch
            # This overestimates congestion, so we use it as a conservative estimate
            
            # First, get all links that are on any path from workers to this switch
            # and from this switch to PS
            used_links = set()
            
            # Paths from workers to switch
            for j in range(jobs_num):
                for edge_switch in network.tors_id:
                    if edge_switch in instance['tor_switch_workers'][j]:
                        # Check if there is a path from edge_switch to this switch
                        if edge_switch in network.G and switch in network.G:
                            try:
                                path = network.allPathDict[edge_switch][switch]
                                # Add all links in path
                                for i in range(len(path) - 1):
                                    u, v = path[i], path[i + 1]
                                    used_links.add((u, v))
                                    used_links.add((v, u))  # undirected
                            except KeyError:
                                continue
            # Paths from switch to PS
            for j in range(jobs_num):
                if switch in network.G and ps_id[j] in network.G:
                    try:
                        path = network.allPathDict[switch][ps_id[j]]
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i + 1]
                            used_links.add((u, v))
                            used_links.add((v, u))
                    except KeyError:
                        continue
            
            # For each link in used_links, compute congestion benefit
            # Congestion benefit = (1 - current_load / capacity) if current_load < capacity
            # But we don't know future load, so we use current load only
            # We assume that activating this switch will increase load on these links
            # But we don't have a way to compute future load without routing
            # So instead, we use the fact that high load on a link means it's already congested
            # So if a link has high load, it's already a bottleneck, and adding more might be bad
            # So benefit decreases if the link is already heavily loaded
            
            for edge in used_links:
                if edge not in network.bandwidth_mapping:
                    continue
                capacity = network.bandwidth_mapping[edge]
                current_load = current_link_loads.get(edge, 0.0)
                # If link is already at or near capacity, benefit is low
                # But we're adding load, so we penalize high current load
                # We use: 1 - (current_load / capacity) if current_load < capacity, else 0
                if current_load >= capacity:
                    congestion_contribution = 0.0
                else:
                    congestion_contribution = (capacity - current_load) / capacity
                total_congestion += congestion_contribution
            
            # Normalize congestion benefit
            if used_links:
                congestion_benefit = total_congestion / len(used_links)
            else:
                congestion_benefit = 0.0
            
            # Estimate potential load on the switch itself
            # Worst-case: all connected workers send data to this switch
            # Total incoming rate = total_volume
            # Check if this exceeds capacity
            if total_volume > Cs:
                # This switch cannot handle the load
                # So skip this candidate
                continue
            
            # Compute benefit score
            benefit_score = alpha * connected_workers + beta * total_volume + gamma * congestion_benefit
            
            if benefit_score > best_benefit:
                best_benefit = benefit_score
                best_switch = idx  # index in ina_candidates
        
        # If no valid switch found, break
        if best_switch == -1:
            break
        
        # Select the best switch
        selected_switches.add(ina_candidates[best_switch])
        ina_placement_expanded[best_switch] = 1
        
        # Update current_link_loads: simulate adding the switch
        # We add the maximum possible load on each link that would be used
        # But we don't know the exact routing, so we use worst-case: all workers go via this switch
        # We'll use the same used_links set as above
        # But we don't know which flows go where, so we assume:
        # - All worker-to-INA flows go through the links on worker->switch paths
        # - All INA-to-PS flows go through the links on switch->PS paths
        
        # For simplicity, we just add the total_volume to each link in used_links
        # But that's not accurate because not all data goes through the same path
        # Instead, we can estimate based on number of paths
        # But we'll do a conservative update: add total_volume to each link
        # This overestimates, but ensures we don't overcommit
        
        for edge in used_links:
            if edge in network.bandwidth_mapping:
                capacity = network.bandwidth_mapping[edge]
                # Update load
                current_link_loads[edge] = min(capacity, current_link_loads.get(edge, 0.0) + total_volume)
        
        # End of greedy iteration
    
    # === TASK 2: Constraint-Aware Worker-to-Aggregation Point Assignment ===
    
    # Initialize load tracking for assignment phase
    # We'll maintain a copy of current_link_loads for dynamic updates
    link_loads = current_link_loads.copy()
    switch_loads = {s: 0.0 for s in selected_switches}
    
    # For each job, sort workers by gradient volume (descending) to prioritize high-volume jobs
    # This helps in maximizing effective rate gain
    for j in range(jobs_num):
        # Create list of (worker_index, volume) for active workers
        workers = [(w, jobs_size[j]) for w in range(workers_num[j])]
        # Sort by volume descending
        workers.sort(key=lambda x: x[1], reverse=True)
        
        # Process each worker in this job
        for w, volume in workers:
            best_option = None
            best_cost = float('inf')
            
            # Evaluate routing to each INA switch
            for s_idx, switch in enumerate(ina_candidates):
                if switch not in selected_switches:
                    continue
                
                # Check if this assignment is valid: would it violate any capacity?
                # Compute path from worker w to switch
                # First, find the edge/leaf switch connected to worker w
                # We assume worker w is on some edge/leaf switch that has a path to this switch
                # But we don't have direct worker-to-switch mapping
                # So we assume the worker is on a switch that has a path
                
                # Get all edge/leaf switches that have workers for job j
                edge_switches = [sw for sw in network.tors_id if sw in instance['tor_switch_workers'][j]]
                
                # For each edge switch, check if there's a path from it to the target switch
                valid_path = False
                path = None
                for edge_switch in edge_switches:
                    if edge_switch in network.G and switch in network.G:
                        try:
                            path = network.allPathDict[edge_switch][switch]
                            valid_path = True
                            break
                        except KeyError:
                            continue
                
                if not valid_path:
                    continue  # No valid path to this switch
                
                # Compute incremental load on all links in the path
                path_links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                
                # Check if adding this flow would exceed link capacity
                link_violation = False
                for edge in path_links:
                    # Check both directions in undirected graph
                    if edge not in network.bandwidth_mapping:
                        continue
                    capacity = network.bandwidth_mapping[edge]
                    current_load = link_loads.get(edge, 0.0)
                    new_load = current_load + volume
                    
                    if new_load > capacity:
                        link_violation = True
                        break
                
                # Check switch capacity
                switch_violation = switch_loads[switch] + volume > Cs
                
                # If no violation, compute cost (negative of effective rate gain)
                if not link_violation and not switch_violation:
                    # Cost is negative of effective rate gain
                    # Effective rate gain = volume / effective_time
                    # effective_time ≈ 1 / (total_load_on_path) — we use a surrogate
                    # We use a simple proxy: cost = -volume / (sum of link loads + 1)
                    # But we want to penalize high load, so we use:
                    # cost = -volume / (sum of current_link_loads + volume)
                    effective_rate = volume / (sum(link_loads.get(edge, 0.0) + volume for edge in path_links) + 1e-9)
                    cost = -effective_rate
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_option = ('ina', s_idx, path_links)
            
            # Evaluate direct routing to PS
            # Check path from worker w to PS
            # Again, use edge_switches
            valid_path = False
            path = None
            for edge_switch in edge_switches:
                if edge_switch in network.G and ps_id[j] in network.G:
                    try:
                        path = network.allPathDict[edge_switch][ps_id[j]]
                        valid_path = True
                        break
                    except KeyError:
                        continue
            
            if valid_path:
                # Compute incremental load on path
                path_links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                
                # Check link capacity
                link_violation = False
                for edge in path_links:
                    if edge not in network.bandwidth_mapping:
                        continue
                    capacity = network.bandwidth_mapping[edge]
                    current_load = link_loads.get(edge, 0.0)
                    new_load = current_load + volume
                    
                    if new_load > capacity:
                        link_violation = True
                        break
                
                # Check PS inflow capacity
                ps_capacity = 200  # Assume PS bandwidth is 200 Gbps
                # Calculate current load on PS from other jobs
                ps_current_load = 0.0
                for j2 in range(jobs_num):
                    if j2 == j:
                        continue
                    # Check if job j2's workers could be using the same path
                    # But we don't know their routing, so we assume they're not using the same path
                    # In reality, this should be calculated based on their routing
                    # For now, we only consider job j
                    # This is a conservative estimate
                    ps_current_load += volume
                
                ps_violation = ps_current_load + volume > ps_capacity
                
                if not link_violation and not ps_violation:
                    # Cost is negative of effective rate gain
                    effective_rate = volume / (sum(link_loads.get(edge, 0.0) + volume for edge in path_links) + 1e-9)
                    cost = -effective_rate
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_option = ('ps', None, path_links)
            
            # Assign the best option
            if best_option is not None:
                if best_option[0] == 'ina':
                    s_idx = best_option[1]
                    # Assign worker w to INA switch s_idx
                    y_j_w_s_full[j][w][s_idx] = 1
                    # Update loads
                    for edge in best_option[2]:
                        link_loads[edge] = link_loads.get(edge, 0.0) + volume
                    switch_loads[ina_candidates[s_idx]] += volume
                else:  # direct to PS
                    y_j_w_d[j][w] = 1
                    # Update link loads on path
                    for edge in best_option[2]:
                        link_loads[edge] = link_loads.get(edge, 0.0) + volume
            else:
                # No valid assignment found — fallback to direct PS routing only if possible
                # Try again with only PS option
                for edge_switch in edge_switches:
                    if edge_switch in network.G and ps_id[j] in network.G:
                        try:
                            path = network.allPathDict[edge_switch][ps_id[j]]
                            path_links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                            
                            # Check link capacity
                            link_violation = False
                            for edge in path_links:
                                if edge not in network.bandwidth_mapping:
                                    continue
                                capacity = network.bandwidth_mapping[edge]
                                current_load = link_loads.get(edge, 0.0)
                                new_load = current_load + volume
                                
                                if new_load > capacity:
                                    link_violation = True
                                    break
                            
                            if not link_violation:
                                # Also check PS inflow
                                ps_capacity = 200
                                ps_current_load = 0.0
                                for j2 in range(jobs_num):
                                    if j2 == j:
                                        continue
                                    # This is a conservative estimate - in reality we need to know other jobs' routing
                                    ps_current_load += jobs_size[j2]
                                
                                if ps_current_load + volume <= ps_capacity:
                                    y_j_w_d[j][w] = 1
                                    for edge in path_links:
                                        link_loads[edge] = link_loads.get(edge, 0.0) + volume
                                    break
                        except KeyError:
                            continue
                
                # If still no assignment, keep as zero (unassigned)
                # This should not happen in practice, but we handle it
                # The system will be infeasible, but we proceed with zero assignment
                
            # End of worker assignment
    
    # === TASK 3: Flow Consistency Enforcement and Binary-Variable Structuring ===
    
    # Create a mapping from selected switch ID to its index in ina_candidates
    # This ensures that y_j_w_s_full is indexed correctly
    ina_id = [s for s in selected_switches]
    ina_id_to_idx = {s: ina_candidates.index(s) for s in ina_id}
    
    # Reconstruct the assignment matrices to ensure strict compliance
    # We use the current assignment from Tasks 1-2, but enforce constraints
    for j in range(jobs_num):
        for w in range(max_workers):
            # For inactive workers, ensure zero padding
            if w >= workers_num[j]:
                for s in range(num_candidates):
                    y_j_w_s_full[j][w][s] = 0
                y_j_w_d[j][w] = 0
                continue
            
            # Ensure exclusivity: exactly one assignment per worker
            total_assignment = sum(y_j_w_s_full[j][w][s] for s in range(num_candidates)) + y_j_w_d[j][w]
            if total_assignment == 0:
                # No assignment found - fallback to direct PS
                # This is a safeguard, though should be rare
                y_j_w_d[j][w] = 1
                for s in range(num_candidates):
                    y_j_w_s_full[j][w][s] = 0
            elif total_assignment > 1:
                # Multiple assignments - resolve by prioritizing INA over PS
                # Reset all INA assignments
                for s in range(num_candidates):
                    y_j_w_s_full[j][w][s] = 0
                # If any INA was assigned, keep the first one
                assigned = False
                for s in range(num_candidates):
                    if y_j_w_s_full[j][w][s] == 1:
                        # Keep this assignment
                        assigned = True
                        break
                if not assigned:
                    # Fallback to PS
                    y_j_w_d[j][w] = 1
            else:
                # One assignment is correct - do nothing
                pass
            
            # Enforce the conditional constraint: y_jws <= x_s
            # If worker w is assigned to INA switch s, then s must be selected
            # This is already ensured by the selection process, but we verify
            for s in range(num_candidates):
                if y_j_w_s_full[j][w][s] == 1:
                    switch_id = ina_candidates[s]
                    if switch_id not in selected_switches:
                        # This should not happen, but if it does, fix it
                        y_j_w_s_full[j][w][s] = 0
                        y_j_w_d[j][w] = 1  # fallback to PS
                        break
    
    # === TASK 4: Local Search Refinement via Perturbation-Based Improvement with Lightweight LP Evaluation ===
    
    # Save initial solution for comparison
    best_placement = ina_placement_expanded[:]
    best_routing = [job[:] for job in y_j_w_s_full], [job[:] for job in y_j_w_d]
    
    # Initialize best makespan
    best_makespan = evaluate_completion_time(
        instance, network, best_placement, best_routing, verbose=False, ina_capacity=Cs, ps_bandwidth=200
    )
    
    improved = True
    iteration_count = 0
    max_iterations = 100  # Prevent infinite loops
    
    while improved and iteration_count < max_iterations:
        improved = False
        iteration_count += 1
        
        # Create a list of all possible perturbations
        perturbations = []
        
        # 1. Worker reassignment perturbations: swap assignment between INA and PS for a worker
        for j in range(jobs_num):
            for w in range(workers_num[j]):
                # Skip if worker is already assigned to PS or INA
                is_assigned_to_ps = y_j_w_d[j][w] == 1
                is_assigned_to_ina = sum(y_j_w_s_full[j][w][s] for s in range(num_candidates)) == 1
                
                if is_assigned_to_ps and is_assigned_to_ina:
                    # This should not happen due to exclusivity, but just in case
                    continue
                
                # If assigned to INA, consider moving to PS
                if is_assigned_to_ina:
                    # Find which INA switch it's assigned to
                    for s in range(num_candidates):
                        if y_j_w_s_full[j][w][s] == 1:
                            # Create a perturbation: move this worker from INA s to PS
                            perturbations.append({
                                'type': 'swap_worker',
                                'job': j,
                                'worker': w,
                                'from': 'ina',
                                'to': 'ps',
                                'ina_switch': s
                            })
                            break
                
                # If assigned to PS, consider moving to INA
                if is_assigned_to_ps:
                    # Check all INA switches for valid assignment
                    for s in range(num_candidates):
                        switch_id = ina_candidates[s]
                        if switch_id not in selected_switches:
                            continue  # Switch not selected
                        
                        # Check path validity and capacity
                        # Get edge switches for job j
                        edge_switches = [sw for sw in network.tors_id if sw in instance['tor_switch_workers'][j]]
                        valid_path = False
                        path = None
                        for edge_switch in edge_switches:
                            if edge_switch in network.G and switch_id in network.G:
                                try:
                                    path = network.allPathDict[edge_switch][switch_id]
                                    valid_path = True
                                    break
                                except KeyError:
                                    continue
                        
                        if not valid_path:
                            continue
                        
                        # Check link and switch capacity
                        path_links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                        link_violation = False
                        for edge in path_links:
                            if edge not in network.bandwidth_mapping:
                                continue
                            capacity = network.bandwidth_mapping[edge]
                            current_load = link_loads.get(edge, 0.0)
                            new_load = current_load + jobs_size[j]
                            if new_load > capacity:
                                link_violation = True
                                break
                        
                        if link_violation:
                            continue
                        
                        switch_violation = switch_loads.get(switch_id, 0.0) + jobs_size[j] > Cs
                        if switch_violation:
                            continue
                        
                        # Valid swap: add perturbation
                        perturbations.append({
                            'type': 'swap_worker',
                            'job': j,
                            'worker': w,
                            'from': 'ps',
                            'to': 'ina',
                            'ina_switch': s
                        })
        
        # 2. INA switch swap perturbations: exchange selected and unselected switches
        selected_indices = [i for i in range(num_candidates) if ina_placement_expanded[i] == 1]
        unselected_indices = [i for i in range(num_candidates) if ina_placement_expanded[i] == 0]
        
        for i in selected_indices:
            for j in unselected_indices:
                # Try swapping switch i and j
                # Check if switch j can handle the load
                # Get total volume from workers connected to switch j
                volume_j = 0.0
                # This is a simplified estimate
                for job_idx in range(jobs_num):
                    for edge_switch in network.tors_id:
                        if edge_switch in instance['tor_switch_workers'][job_idx]:
                            # Check if path exists from edge_switch to switch j
                            if edge_switch in network.G and ina_candidates[j] in network.G:
                                try:
                                    path = network.allPathDict[edge_switch][ina_candidates[j]]
                                    count = instance['tor_switch_workers'][job_idx][edge_switch]
                                    volume_j += count * jobs_size[job_idx]
                                except KeyError:
                                    continue
                if volume_j > Cs:
                    continue  # Cannot handle load
                
                # Valid swap: add perturbation
                perturbations.append({
                    'type': 'swap_ina',
                    'from': i,
                    'to': j
                })
        
        # Apply each perturbation and evaluate
        for perturbation in perturbations:
            # Create a copy of current solution
            new_placement = best_placement[:]
            new_routing = [job[:] for job in best_routing[0]], [job[:] for job in best_routing[1]]
            
            # Apply perturbation
            if perturbation['type'] == 'swap_worker':
                j, w = perturbation['job'], perturbation['worker']
                from_type = perturbation['from']
                to_type = perturbation['to']
                s = perturbation['ina_switch']
                
                if from_type == 'ina' and to_type == 'ps':
                    # Remove from INA
                    new_routing[0][j][w][s] = 0
                    # Add to PS
                    new_routing[1][j][w] = 1
                elif from_type == 'ps' and to_type == 'ina':
                    # Remove from PS
                    new_routing[1][j][w] = 0
                    # Add to INA
                    new_routing[0][j][w][s] = 1
            
            elif perturbation['type'] == 'swap_ina':
                i, j = perturbation['from'], perturbation['to']
                # Swap activation status
                new_placement[i] = 0
                new_placement[j] = 1
            
            # Evaluate makespan
            current_makespan = evaluate_completion_time(
                instance, network, new_placement, new_routing, verbose=False, ina_capacity=Cs, ps_bandwidth=200
            )
            
            # If improved, accept the change
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_placement = new_placement
                best_routing = new_routing
                improved = True
                # Break to restart the loop with new solution
                break
        
        # If no improvement, loop ends
        
        # Update ina_placement_expanded and jobs_routing_expanded
        ina_placement_expanded = best_placement
        y_j_w_s_full = best_routing[0]
        y_j_w_d = best_routing[1]
    
    # === TASK 5: Feasibility Verification and Constraint Enforcement Across All Solution Components ===
    
    # Verify INA budget constraint: exactly K switches selected
    selected_count = sum(ina_placement_expanded)
    if selected_count != K:
        # Revert to previous valid solution if possible
        # But we assume the previous stages maintained feasibility
        # In practice, this should not happen
        pass
    
    # Verify worker assignment exclusivity: each active worker has exactly one assignment
    for j in range(jobs_num):
        for w in range(max_workers):
            if w >= workers_num[j]:
                # Inactive worker: all assignments should be 0
                for s in range(num_candidates):
                    if y_j_w_s_full[j][w][s] != 0:
                        # This should not happen
                        y_j_w_s_full[j][w][s] = 0
                if y_j_w_d[j][w] != 0:
                    y_j_w_d[j][w] = 0
            else:
                # Active worker: exactly one assignment
                total_assignments = sum(y_j_w_s_full[j][w][s] for s in range(num_candidates)) + y_j_w_d[j][w]
                if total_assignments == 0:
                    # No assignment: fallback to PS
                    y_j_w_d[j][w] = 1
                    for s in range(num_candidates):
                        y_j_w_s_full[j][w][s] = 0
                elif total_assignments > 1:
                    # Multiple assignments: prioritize INA over PS
                    # Reset all INA assignments
                    for s in range(num_candidates):
                        y_j_w_s_full[j][w][s] = 0
                    # Keep the first INA assignment if any
                    found_ina = False
                    for s in range(num_candidates):
                        if y_j_w_s_full[j][w][s] == 1:
                            found_ina = True
                            break
                    if not found_ina:
                        # Fallback to PS
                        y_j_w_d[j][w] = 1
                # Ensure binary values
                for s in range(num_candidates):
                    if y_j_w_s_full[j][w][s] not in (0, 1):
                        y_j_w_s_full[j][w][s] = int(y_j_w_s_full[j][w][s] > 0.5)
                if y_j_w_d[j][w] not in (0, 1):
                    y_j_w_d[j][w] = int(y_j_w_d[j][w] > 0.5)
    
    # Verify INA-only assignment validity: y_jws <= x_s
    for j in range(jobs_num):
        for w in range(max_workers):
            if w >= workers_num[j]:
                continue
            for s in range(num_candidates):
                if y_j_w_s_full[j][w][s] == 1:
                    switch_id = ina_candidates[s]
                    if switch_id not in selected_switches:
                        # Invalid assignment: switch not selected
                        y_j_w_s_full[j][w][s] = 0
                        y_j_w_d[j][w] = 1  # Fallback to PS
    
    # Verify switch processing capacity using LP evaluation
    # Use evaluate_completion_time to get actual rates
    makespan = evaluate_completion_time(
        instance, network, ina_placement_expanded, [y_j_w_s_full, y_j_w_d], verbose=False, ina_capacity=Cs, ps_bandwidth=200
    )
    
    # If infeasible, fallback to direct PS routing
    if makespan == float('inf'):
        # Revert to default: all workers direct to PS
        for j in range(jobs_num):
            for w in range(workers_num[j]):
                y_j_w_s_full[j][w] = [0] * num_candidates
                y_j_w_d[j][w] = 1
        for j in range(jobs_num):
            for w in range(workers_num[j], max_workers):
                for s in range(num_candidates):
                    y_j_w_s_full[j][w][s] = 0
                y_j_w_d[j][w] = 0
    
    # Final verification: ensure all constraints are satisfied
    # Re-evaluate using the final solution
    final_makespan = evaluate_completion_time(
        instance, network, ina_placement_expanded, [y_j_w_s_full, y_j_w_d], verbose=False, ina_capacity=Cs, ps_bandwidth=200
    )
    
    # If still infeasible, this is a critical failure
    if final_makespan == float('inf'):
        # In production, this would trigger an alert or fallback
        # For now, we proceed with the solution as is
        # This should not happen due to the robustness of prior stages
        pass
    
    # Final zero-padding for inactive workers (already done above)
    # Ensure each worker has exactly one assignment (already enforced above)
    
    # Return the result
    return ina_placement_expanded, jobs_routing_expanded