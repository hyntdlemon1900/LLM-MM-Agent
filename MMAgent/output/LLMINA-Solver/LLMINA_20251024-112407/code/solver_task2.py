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
    
    # ========================
    # TASK 1: INA Placement (Greedy Selection with Capacity-Driven Marginal Benefit)
    # ========================
    
    # Precompute worker connectivity per candidate switch
    # For each candidate switch, store which jobs' workers are connected
    candidate_worker_jobs = {s: set() for s in ina_candidates}
    candidate_worker_count = {s: 0 for s in ina_candidates}
    
    # Determine switch types and connectivities based on topology
    if hasattr(network, 'tors_id'):
        tor_switches = network.tors_id
    else:
        tor_switches = set()
    
    # Map workers to their direct switches
    worker_to_switch = {}
    for j in range(jobs_num):
        for w in workers_id[j]:
            # Use the fact that in most topologies, workers are connected to edge/leaf switches
            # We'll assume worker w connects to switch w if w is in tor_switches
            # Otherwise, we'll use a heuristic: find the nearest candidate switch
            if w in tor_switches:
                worker_to_switch[w] = w
            else:
                # Try to find a candidate switch that is on the path from worker to PS
                # For simplicity, assume worker w connects to switch w if w is in ina_candidates
                worker_to_switch[w] = w if w in ina_candidates else None
    
    # Build candidate_worker_jobs and candidate_worker_count
    for j in range(jobs_num):
        for w in workers_id[j]:
            sw = worker_to_switch.get(w)
            if sw is not None and sw in ina_candidates:
                candidate_worker_jobs[sw].add(j)
                candidate_worker_count[sw] += 1
    
    # Initialize current load on each link (from all flows, excluding candidate switches)
    # We'll use a dictionary: link -> total current load
    current_link_load = {edge: 0.0 for edge in network.bandwidth_mapping}
    
    # We'll also track the worst-case load on each candidate switch (for capacity check)
    # For now, we assume that if a switch is activated, all its connected workers will route to it
    # This gives an upper bound on incoming rate
    worst_case_switch_load = {s: 0.0 for s in ina_candidates}
    
    # We need to compute the volume gain: sum of m_j for jobs connected to switch
    volume_gain = {s: 0.0 for s in ina_candidates}
    for s in ina_candidates:
        for j in candidate_worker_jobs[s]:
            volume_gain[s] += jobs_size[j]
    
    # Weights for benefit function
    alpha, beta, gamma = 1.0, 1.0, 1.0
    
    # Greedy selection of K switches
    selected_switches = set()
    remaining_candidates = set(ina_candidates)
    
    for step in range(K):
        best_score = -float('inf')
        best_switch = None
        
        for s in remaining_candidates:
            # Skip if switch is already selected
            if s in selected_switches:
                continue
                
            # Check feasibility: if activating s would exceed capacity, skip
            # Worst-case incoming rate = sum of all gradient volumes from connected workers
            worst_case_rate = volume_gain[s]
            if worst_case_rate > Cs:
                continue  # Skip infeasible switch
            
            # Compute benefit score
            # Proximity gain: number of connected workers
            N_s = candidate_worker_count[s]
            
            # Volume gain: total gradient volume from connected workers
            V_s = volume_gain[s]
            
            # Congestion alleviation: estimate impact on high-traffic links
            # For each link that would be traversed by worker->s or s->PS paths
            workers_to_s = []
            for j in candidate_worker_jobs[s]:
                for w in workers_id[j]:
                    if w in network.allPathDict and s in network.allPathDict[w]:
                        path = network.allPathDict[w][s]
                        links = [(path[i], path[i+1]) for i in range(len(path)-1)]
                        workers_to_s.extend(links)
            
            s_to_ps = []
            if s in network.allPathDict and ps_id[0] in network.allPathDict[s]:
                path = network.allPathDict[s][ps_id[0]]
                s_to_ps = [(path[i], path[i+1]) for i in range(len(path)-1)]
            
            all_links = set(workers_to_s + s_to_ps)
            
            congestion_score = 0.0
            for link in all_links:
                if link not in current_link_load:
                    rev_link = link[::-1]
                    if rev_link not in current_link_load:
                        continue
                    link = rev_link
                
                bandwidth = network.bandwidth_mapping.get(link, 0.0)
                if bandwidth == 0.0:
                    continue
                    
                load = current_link_load.get(link, 0.0)
                ratio = load / bandwidth if bandwidth > 0 else 0.0
                congestion_score += (1.0 - ratio)
            
            # Total benefit
            benefit = alpha * N_s + beta * V_s + gamma * congestion_score
            
            if benefit > best_score:
                best_score = benefit
                best_switch = s
        
        # If no valid switch found, break
        if best_switch is None:
            break
            
        # Select the best switch
        selected_switches.add(best_switch)
        ina_placement_expanded[ina_candidates.index(best_switch)] = 1
        
        # Update current_link_load: add the worst-case load from this switch
        workers_to_s = []
        for j in candidate_worker_jobs[best_switch]:
            for w in workers_id[j]:
                if w in network.allPathDict and best_switch in network.allPathDict[w]:
                    path = network.allPathDict[w][best_switch]
                    links = [(path[i], path[i+1]) for i in range(len(path)-1)]
                    workers_to_s.extend(links)
        
        s_to_ps = []
        if best_switch in network.allPathDict and ps_id[0] in network.allPathDict[best_switch]:
            path = network.allPathDict[best_switch][ps_id[0]]
            s_to_ps = [(path[i], path[i+1]) for i in range(len(path)-1)]
        
        for link in set(workers_to_s + s_to_ps):
            rev_link = link[::-1]
            actual_link = link
            if link not in current_link_load and rev_link in current_link_load:
                actual_link = rev_link
            
            bandwidth = network.bandwidth_mapping.get(actual_link, 0.0)
            if bandwidth > 0:
                # Add worst-case load (volume gain) to the link
                load_increase = volume_gain[best_switch] / len(set(workers_to_s + s_to_ps)) if len(set(workers_to_s + s_to_ps)) > 0 else volume_gain[best_switch]
                current_link_load[actual_link] += load_increase
        
        # Remove from remaining candidates
        remaining_candidates.remove(best_switch)
    
    # ========================
    # TASK 2: Constraint-Aware Worker-to-Aggregation Point Assignment
    # ========================
    
    # Extract selected INA switches (in sorted order)
    ina_id = sorted([ina_candidates[i] for i, val in enumerate(ina_placement_expanded) if val == 1])
    
    # Map selected INA switches to indices for routing
    ina_to_idx = {s: i for i, s in enumerate(ina_id)}
    
    # Initialize per-link and per-switch load tracking
    link_load = {link: 0.0 for link in network.bandwidth_mapping}
    switch_load = {s: 0.0 for s in ina_id}
    
    # For each job, assign workers greedily based on maximum effective rate gain
    for j in range(jobs_num):
        # Sort workers by gradient volume (descending) to prioritize large jobs
        worker_indices = sorted(range(workers_num[j]), key=lambda w: jobs_size[j], reverse=True)
        
        for w in worker_indices:
            # Current worker ID
            worker = workers_id[j][w]
            
            # List of valid assignment options: (type, id, cost)
            valid_options = []
            
            # Option 1: Route to PS directly
            direct_path = network.allPathDict.get(worker, {}).get(ps_id[j])
            if direct_path is not None:
                direct_links = [(direct_path[i], direct_path[i+1]) for i in range(len(direct_path)-1)]
                # Check if direct routing violates link or PS capacity
                valid_direct = True
                for link in direct_links:
                    rev_link = link[::-1]
                    actual_link = link
                    if link not in link_load and rev_link in link_load:
                        actual_link = rev_link
                    
                    bandwidth = network.bandwidth_mapping.get(actual_link, 0.0)
                    if bandwidth == 0.0:
                        continue
                    if link_load.get(actual_link, 0.0) + jobs_size[j] > bandwidth:
                        valid_direct = False
                        break
                
                # Also check PS inflow capacity
                if valid_direct:
                    ps_link = (ps_id[j], sd_id[j]) if hasattr(network, 'sd_id') else (ps_id[j], list(network.G.adj[ps_id[j]].keys())[0])
                    rev_ps_link = ps_link[::-1]
                    ps_link_actual = ps_link
                    if ps_link not in link_load and rev_ps_link in link_load:
                        ps_link_actual = rev_ps_link
                    ps_capacity = network.bandwidth_mapping.get(ps_link_actual, 200)  # default 200 Gbps
                    if link_load.get(ps_link_actual, 0.0) + jobs_size[j] > ps_capacity:
                        valid_direct = False
                
                if valid_direct:
                    # Cost is negative of effective rate gain
                    # Estimate effective rate = jobs_size[j] / (time to transmit)
                    # Time is proportional to (current_load + jobs_size[j]) / bandwidth
                    # So rate gain ~ bandwidth / (current_load + jobs_size[j])
                    # We use -rate_gain as cost
                    effective_rate = jobs_size[j] / (1 + sum(1 for link in direct_links if link in network.bandwidth_mapping))
                    cost = -effective_rate
                    valid_options.append(('direct', -1, cost))
            
            # Option 2: Route to each selected INA switch
            for s in ina_id:
                ina_path = network.allPathDict.get(worker, {}).get(s)
                if ina_path is None:
                    continue  # No path exists
                ina_links = [(ina_path[i], ina_path[i+1]) for i in range(len(ina_path)-1)]
                
                # Check link capacity for worker->INA path
                valid_ina = True
                for link in ina_links:
                    rev_link = link[::-1]
                    actual_link = link
                    if link not in link_load and rev_link in link_load:
                        actual_link = rev_link
                    
                    bandwidth = network.bandwidth_mapping.get(actual_link, 0.0)
                    if bandwidth == 0.0:
                        continue
                    if link_load.get(actual_link, 0.0) + jobs_size[j] > bandwidth:
                        valid_ina = False
                        break
                
                # Check INA switch capacity
                if valid_ina and switch_load.get(s, 0.0) + jobs_size[j] > Cs:
                    valid_ina = False
                
                if valid_ina:
                    # Cost: negative of effective rate gain
                    effective_rate = jobs_size[j] / (1 + len(ina_links))
                    cost = -effective_rate
                    valid_options.append(('ina', s, cost))
            
            # Select best valid option
            if valid_options:
                # Choose option with minimum cost (i.e., maximum effective rate gain)
                best_option = min(valid_options, key=lambda x: x[2])
                option_type, switch_id, cost = best_option
                
                if option_type == 'direct':
                    y_j_w_d[j][w] = 1
                    # Update link loads
                    direct_path = network.allPathDict[worker][ps_id[j]]
                    direct_links = [(direct_path[i], direct_path[i+1]) for i in range(len(direct_path)-1)]
                    for link in direct_links:
                        rev_link = link[::-1]
                        actual_link = link
                        if link not in link_load and rev_link in link_load:
                            actual_link = rev_link
                        if actual_link in link_load:
                            link_load[actual_link] += jobs_size[j]
                
                elif option_type == 'ina':
                    # Assign to INA switch
                    idx = ina_to_idx[switch_id]
                    y_j_w_s_full[j][w][idx] = 1
                    # Update link loads
                    ina_path = network.allPathDict[worker][switch_id]
                    ina_links = [(ina_path[i], ina_path[i+1]) for i in range(len(ina_path)-1)]
                    for link in ina_links:
                        rev_link = link[::-1]
                        actual_link = link
                        if link not in link_load and rev_link in link_load:
                            actual_link = rev_link
                        if actual_link in link_load:
                            link_load[actual_link] += jobs_size[j]
                    
                    # Update switch load
                    switch_load[switch_id] += jobs_size[j]
            
            # If no valid option, fall back to direct PS routing only if feasible
            else:
                direct_path = network.allPathDict.get(worker, {}).get(ps_id[j])
                if direct_path is not None:
                    direct_links = [(direct_path[i], direct_path[i+1]) for i in range(len(direct_path)-1)]
                    valid_direct = True
                    for link in direct_links:
                        rev_link = link[::-1]
                        actual_link = link
                        if link not in link_load and rev_link in link_load:
                            actual_link = rev_link
                        bandwidth = network.bandwidth_mapping.get(actual_link, 0.0)
                        if bandwidth == 0.0:
                            continue
                        if link_load.get(actual_link, 0.0) + jobs_size[j] > bandwidth:
                            valid_direct = False
                            break
                    
                    # Check PS inflow
                    if valid_direct:
                        ps_link = (ps_id[j], sd_id[j]) if hasattr(network, 'sd_id') else (ps_id[j], list(network.G.adj[ps_id[j]].keys())[0])
                        rev_ps_link = ps_link[::-1]
                        ps_link_actual = ps_link
                        if ps_link not in link_load and rev_ps_link in link_load:
                            ps_link_actual = rev_ps_link
                        ps_capacity = network.bandwidth_mapping.get(ps_link_actual, 200)
                        if link_load.get(ps_link_actual, 0.0) + jobs_size[j] > ps_capacity:
                            valid_direct = False
                    
                    if valid_direct:
                        y_j_w_d[j][w] = 1
                        for link in direct_links:
                            rev_link = link[::-1]
                            actual_link = link
                            if link not in link_load and rev_link in link_load:
                                actual_link = rev_link
                            if actual_link in link_load:
                                link_load[actual_link] += jobs_size[j]
    
    # ========================
    # TASK 3-5: Placeholder for Future Optimization
    # ========================
    # TODO: Will be optimized in Task 3: Local Search Refinement with LP Evaluation
    # TODO: Will be optimized in Task 4: Flow Conservation Enforcement
    # TODO: Will be optimized in Task 5: Dynamic Load Rebalancing
    
    # Return final result
    return ina_placement_expanded, jobs_routing_expanded