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
    
    # Initialize global load tracking for links (current load before any INA activation)
    # Use a dictionary to store current load per link (u,v)
    current_link_load = {}
    for link, capacity in network.bandwidth_mapping.items():
        current_link_load[link] = 0.0
    
    # Initialize selected switches set
    selected_switches = set()
    
    # Precompute worker-to-switch mapping per job
    # For each job, map worker IDs to their direct switch (tor/leaf)
    job_worker_to_switch = []
    for job_idx in range(jobs_num):
        worker_to_switch = {}
        for worker_id in workers_id[job_idx]:
            # Find which switch connects to this worker
            for switch_id in network.tors_id:
                if worker_id in network.G.adj[switch_id]:
                    worker_to_switch[worker_id] = switch_id
                    break
        job_worker_to_switch.append(worker_to_switch)
    
    # Greedy selection of K switches
    for _ in range(K):
        best_switch = -1
        best_score = -1
        
        for cand_idx, switch_id in enumerate(ina_candidates):
            if switch_id in selected_switches:
                continue
            
            # Skip if switch has no connected workers
            connected_workers = []
            for job_idx in range(jobs_num):
                for worker_id in workers_id[job_idx]:
                    if worker_id in network.G.adj[switch_id]:
                        connected_workers.append((job_idx, worker_id))
                        break
            
            if len(connected_workers) == 0:
                continue
            
            # Compute proximity gain: number of connected workers
            N_s = len(connected_workers)
            
            # Compute volume gain: sum of jobs' gradient sizes that have workers on this switch
            volume_gain = 0
            jobs_connected = set()
            for job_idx, worker_id in connected_workers:
                volume_gain += jobs_size[job_idx]
                jobs_connected.add(job_idx)
            
            # Compute congestion alleviation: impact on high-traffic links
            # Use l_i to find all (src, dst) pairs whose paths go through this switch
            # We simulate the impact of enabling this switch: what flows would traverse links?
            congestion_alleviation = 0.0
            links_involved = set()
            
            # For worker-to-INA paths: from worker to switch
            for job_idx, worker_id in connected_workers:
                path = network.allPathDict[worker_id][switch_id]
                for i in range(len(path) - 1):
                    link = (path[i], path[i + 1])
                    links_involved.add(link)
            
            # For INA-to-PS paths: from switch to PS
            for job_idx in jobs_connected:
                path = network.allPathDict[switch_id][ps_id[job_idx]]
                for i in range(len(path) - 1):
                    link = (path[i], path[i + 1])
                    links_involved.add(link)
            
            # For each involved link, estimate congestion reduction
            # We use the inverse of current utilization as a proxy for potential relief
            for link in links_involved:
                capacity = network.bandwidth_mapping.get(link, float('inf'))
                if capacity == float('inf'):
                    continue
                utilization = current_link_load.get(link, 0.0) / capacity
                # Avoid division by zero
                if capacity > 1e-6:
                    congestion_alleviation += (1 - utilization) * (1.0 if utilization < 0.95 else 0.1)
            
            # Check capacity feasibility: worst-case load on switch if all connected workers route through it
            # Max possible incoming rate = sum of all connected workers' gradient volumes
            max_incoming_rate = volume_gain
            if max_incoming_rate > Cs:
                # This switch cannot handle the load even in best case
                continue
            
            # Compute benefit score: α*N_s + β*volume_gain + γ*congestion_alleviation
            # Use equal weights for simplicity
            alpha, beta, gamma = 1.0, 1.0, 1.0
            score = alpha * N_s + beta * volume_gain + gamma * congestion_alleviation
            
            if score > best_score:
                best_score = score
                best_switch = switch_id
                best_switch_idx = cand_idx
        
        # If no valid switch found, break
        if best_switch == -1:
            break
        
        # Select the best switch
        selected_switches.add(best_switch)
        ina_placement_expanded[best_switch_idx] = 1
        
        # Update link loads: simulate that flows now go through this switch
        # For each job with workers on this switch, update paths
        for job_idx in range(jobs_num):
            for worker_id in workers_id[job_idx]:
                if worker_id in network.G.adj[best_switch]:
                    # Worker-to-INA path
                    path = network.allPathDict[worker_id][best_switch]
                    for i in range(len(path) - 1):
                        link = (path[i], path[i + 1])
                        current_link_load[link] += jobs_size[job_idx]
                    
                    # INA-to-PS path
                    path = network.allPathDict[best_switch][ps_id[job_idx]]
                    for i in range(len(path) - 1):
                        link = (path[i], path[i + 1])
                        current_link_load[link] += jobs_size[job_idx]
    
    # === Task 2: Constraint-Aware Worker-to-Aggregation Point Assignment ===
    # Reset routing to avoid accumulation from previous state
    for j in range(jobs_num):
        for w in range(max_workers):
            y_j_w_s_full[j][w] = [0] * num_candidates
            y_j_w_d[j][w] = 0
    
    # Initialize per-switch load tracking
    switch_load = {s: 0.0 for s in ina_candidates}
    
    # For each job, process workers in descending order of gradient size (to prioritize large jobs)
    for job_idx in range(jobs_num):
        job_workers = workers_id[job_idx]
        job_size = jobs_size[job_idx]
        
        # Sort workers by gradient volume descending
        sorted_workers = sorted(job_workers, key=lambda w: jobs_size[job_idx], reverse=True)
        
        # For each worker in this job
        for worker_id in sorted_workers:
            # Find which switch this worker is connected to
            connected_switch = None
            for switch_id in network.tors_id:
                if worker_id in network.G.adj[switch_id]:
                    connected_switch = switch_id
                    break
            
            # Find the candidate index of this switch in ina_candidates
            switch_idx = -1
            for i, cand_id in enumerate(ina_candidates):
                if cand_id == connected_switch:
                    switch_idx = i
                    break
            
            # List of valid aggregation points: selected INAs and PS
            valid_aggregation_points = []
            if switch_idx != -1 and ina_placement_expanded[switch_idx] == 1:
                valid_aggregation_points.append(('ina', switch_idx))
            valid_aggregation_points.append(('ps', None))
            
            # Track best assignment
            best_option = None
            best_cost = float('inf')
            
            # Evaluate each valid aggregation point
            for option_type, switch_id in valid_aggregation_points:
                # Determine target node
                if option_type == 'ina':
                    target = ina_candidates[switch_id]
                else:
                    target = ps_id[job_idx]
                
                # Get path from worker to target
                path = network.allPathDict[worker_id][target]
                links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                
                # Compute incremental load on each link
                incremental_loads = {}
                for link in links:
                    # Only consider non-infinite capacity links
                    capacity = network.bandwidth_mapping.get(link, float('inf'))
                    if capacity == float('inf'):
                        continue
                    # Estimate new load if this flow is added
                    new_load = current_link_load.get(link, 0.0) + job_size
                    # Check if it exceeds 85% capacity (capacity threshold)
                    if new_load > 0.85 * capacity:
                        # Violation - reject this option
                        incremental_loads[link] = float('inf')
                        break
                    incremental_loads[link] = new_load
                
                # If any link violates threshold, skip
                if any(l == float('inf') for l in incremental_loads.values()):
                    continue
                
                # Check INA switch capacity if routing through INA
                if option_type == 'ina':
                    # Check if adding this worker's load exceeds switch capacity
                    new_switch_load = switch_load[ina_candidates[switch_id]] + job_size
                    if new_switch_load > Cs:
                        continue  # Violation - reject
                    
                    # Cost: negative of effective rate gain
                    # Effective rate gain is proportional to bandwidth utilization
                    effective_rate_gain = job_size / (sum(incremental_loads.values()) if incremental_loads else job_size)
                    cost = -effective_rate_gain
                else:
                    # Direct to PS - only check link capacity
                    # PS inflow capacity is typically 200 Gbps
                    ps_link = (ps_id[job_idx], network.G.adj[ps_id[job_idx]].keys().__next__())  # Simplified
                    ps_link_rev = (ps_link[1], ps_link[0])
                    link_to_ps = ps_link if ps_link in network.bandwidth_mapping else ps_link_rev
                    capacity = network.bandwidth_mapping.get(link_to_ps, 200.0)  # Default PS bandwidth
                    new_ps_load = current_link_load.get(link_to_ps, 0.0) + job_size
                    if new_ps_load > 0.85 * capacity:
                        continue  # Violation - reject
                    
                    cost = -job_size  # Simple cost for direct routing
                
                # Update best option if this is better
                if cost < best_cost:
                    best_cost = cost
                    best_option = (option_type, switch_id)
            
            # Assign best valid option
            if best_option is None:
                # Fallback: if no valid assignment, try direct PS only if it's feasible
                target = ps_id[job_idx]
                path = network.allPathDict[worker_id][target]
                links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                
                # Check link feasibility for direct PS
                valid = True
                for link in links:
                    capacity = network.bandwidth_mapping.get(link, float('inf'))
                    if capacity == float('inf'):
                        continue
                    new_load = current_link_load.get(link, 0.0) + job_size
                    if new_load > 0.85 * capacity:
                        valid = False
                        break
                
                if valid:
                    # Assign to PS
                    y_j_w_d[job_idx][workers_id[job_idx].index(worker_id)] = 1
                    # Update link loads
                    for link in links:
                        current_link_load[link] = current_link_load.get(link, 0.0) + job_size
                else:
                    # No valid assignment possible - this should not happen in practice
                    # Assign to PS anyway (fallback)
                    y_j_w_d[job_idx][workers_id[job_idx].index(worker_id)] = 1
                    for link in links:
                        current_link_load[link] = current_link_load.get(link, 0.0) + job_size
            else:
                # Assign based on best option
                if best_option[0] == 'ina':
                    # Assign to INA
                    s_idx = best_option[1]
                    y_j_w_s_full[job_idx][workers_id[job_idx].index(worker_id)][s_idx] = 1
                    # Update switch load
                    switch_load[ina_candidates[s_idx]] += job_size
                    # Update link loads
                    target = ina_candidates[s_idx]
                    path = network.allPathDict[worker_id][target]
                    for i in range(len(path) - 1):
                        link = (path[i], path[i + 1])
                        current_link_load[link] = current_link_load.get(link, 0.0) + job_size
                else:
                    # Assign to PS
                    y_j_w_d[job_idx][workers_id[job_idx].index(worker_id)] = 1
                    # Update link loads
                    target = ps_id[job_idx]
                    path = network.allPathDict[worker_id][target]
                    for i in range(len(path) - 1):
                        link = (path[i], path[i + 1])
                        current_link_load[link] = current_link_load.get(link, 0.0) + job_size
    
    # === Task 3: Flow Consistency Enforcement and Binary-Variable Structuring for Feasibility and Model Compatibility ===
    # Create mapping from selected INA switch IDs to indices in ina_candidates
    ina_id = [ina_candidates[i] for i in range(len(ina_candidates)) if ina_placement_expanded[i] == 1]
    
    # Initialize assignment matrices
    y_j_w_s_full = [[[0] * num_candidates for _ in range(max_workers)] for _ in range(jobs_num)]
    y_j_w_d = [[0] * max_workers for _ in range(jobs_num)]
    
    # Map each selected INA switch to its index in ina_candidates
    ina_switch_to_idx = {}
    for i, switch_id in enumerate(ina_candidates):
        ina_switch_to_idx[switch_id] = i
    
    # For each job and each worker, assign to exactly one aggregation point
    for job_idx in range(jobs_num):
        job_workers = workers_id[job_idx]
        job_size = jobs_size[job_idx]
        
        # Process each worker in the job
        for w_idx, worker_id in enumerate(job_workers):
            # Check if this worker is assigned to an INA or PS
            assigned_to_ina = False
            assigned_to_ps = False
            
            # Look for assignment in y_j_w_s_full
            for s_idx in range(num_candidates):
                if y_j_w_s_full[job_idx][w_idx][s_idx] == 1:
                    # Check if the switch is in the selected set
                    switch_id = ina_candidates[s_idx]
                    if switch_id in ina_id:
                        # Valid assignment
                        y_j_w_s_full[job_idx][w_idx][s_idx] = 1
                        assigned_to_ina = True
                    else:
                        # Invalid: assignment to non-selected switch
                        y_j_w_s_full[job_idx][w_idx][s_idx] = 0
                    break
            
            # If not assigned to INA, check PS
            if not assigned_to_ina:
                if y_j_w_d[job_idx][w_idx] == 1:
                    y_j_w_d[job_idx][w_idx] = 1
                    assigned_to_ps = True
            
            # Enforce exclusivity: exactly one assignment per worker
            if not assigned_to_ina and not assigned_to_ps:
                # No valid assignment found — fallback to PS
                y_j_w_d[job_idx][w_idx] = 1
            elif assigned_to_ina and assigned_to_ps:
                # Conflict: both assigned — resolve by prefering INA
                y_j_w_d[job_idx][w_idx] = 0
            
            # Ensure all other INA assignments are zero
            for s_idx in range(num_candidates):
                if s_idx != ina_switch_to_idx.get(ina_id[0], -1) if ina_id else -1:
                    y_j_w_s_full[job_idx][w_idx][s_idx] = 0
    
    # Zero-pad inactive workers
    for job_idx in range(jobs_num):
        for w in range(workers_num[job_idx], max_workers):
            for s_idx in range(num_candidates):
                y_j_w_s_full[job_idx][w][s_idx] = 0
            y_j_w_d[job_idx][w] = 0
    
    # Enforce y_jws <= x_s: only selected switches can have assignments
    for job_idx in range(jobs_num):
        for w_idx in range(max_workers):
            for s_idx in range(num_candidates):
                switch_id = ina_candidates[s_idx]
                if ina_placement_expanded[s_idx] == 0:
                    y_j_w_s_full[job_idx][w_idx][s_idx] = 0
    
    # === Task 4: Iterative Improvement via LP Evaluation (Placeholder) ===
    # TODO: Will be optimized in Task 4
    
    # === Task 5: Final Output (Placeholder) ===
    # TODO: Will be optimized in Task 5
    
    return ina_placement_expanded, jobs_routing_expanded