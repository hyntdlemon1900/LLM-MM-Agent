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
                    for w in workers_id[j]:
                        # Check if worker w is connected to this switch
                        # In typical topologies, worker -> edge switch
                        # So check if worker w's switch is this one
                        # This assumes workers are mapped to switches via a mapping
                        # But we don't have that here, so we use tor_switch_workers
                        # tor_switch_workers[j][switch] gives count of workers of job j on this tor
                        if switch in instance['tor_switch_workers'][j] and instance['tor_switch_workers'][j][switch] > 0:
                            connected_workers += instance['tor_switch_workers'][j][switch]
                            total_volume += instance['tor_switch_workers'][j][switch] * jobs_size[j]
                            jobs_connected.add(j)
            elif has_aggrs and switch in network.aggrs_id:
                # Aggregation switch in FatTree
                # Connected to edge switches
                # We assume each edge switch connects to some workers
                # So we can estimate connectivity via edge switches
                for j in range(jobs_num):
                    for edge_switch in network.tors_id:
                        if edge_switch in instance['tor_switch_workers'][j]:
                            # If this edge switch is connected to the aggregation switch
                            # and we know the path from edge_switch to switch
                            if edge_switch in network.G and switch in network.G:
                                # Check if there's a path (should be)
                                # But we can use allPathDict to verify
                                try:
                                    path = network.allPathDict[edge_switch][switch]
                                    # If path exists, then edge_switch is connected
                                    # So workers on edge_switch are indirectly connected
                                    connected_workers += instance['tor_switch_workers'][j].get(edge_switch, 0)
                                    total_volume += instance['tor_switch_workers'][j].get(edge_switch, 0) * jobs_size[j]
                                    jobs_connected.add(j)
                                except KeyError:
                                    # No path, skip
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
                                        # Workers on edge_switch are connected via aggr_switch
                                        connected_workers += instance['tor_switch_workers'][j].get(edge_switch, 0)
                                        total_volume += instance['tor_switch_workers'][j].get(edge_switch, 0) * jobs_size[j]
                                        jobs_connected.add(j)
                            except KeyError:
                                continue
            elif has_leafs and switch in network.leafs_id:
                # Leaf switch in SpineLeaf
                for j in range(jobs_num):
                    for w in workers_id[j]:
                        # Leaf switch is directly connected to workers
                        # So check if worker w is on this leaf switch
                        # Again, use tor_switch_workers as proxy
                        if switch in instance['tor_switch_workers'][j] and instance['tor_switch_workers'][j][switch] > 0:
                            connected_workers += instance['tor_switch_workers'][j][switch]
                            total_volume += instance['tor_switch_workers'][j][switch] * jobs_size[j]
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
                                connected_workers += instance['tor_switch_workers'][j].get(leaf_switch, 0)
                                total_volume += instance['tor_switch_workers'][j].get(leaf_switch, 0) * jobs_size[j]
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
            total_link_load_increase = 0.0
            total_congestion = 0.0
            
            # We'll compute for all links that could be used by any worker-to-INA or INA-to-PS flow
            # But we don't know routing yet, so we assume worst-case: all workers go via this switch
            # This overestimates congestion, so we use it as a conservative estimate
            
            # First, get all links that are on any path from workers to this switch
            # and from this switch to PS
            used_links = set()
            
            # Paths from workers to switch
            for j in range(jobs_num):
                for w in workers_id[j]:
                    # Find worker switch (assume tor_switch_workers maps worker to switch)
                    # But we don't have worker->switch mapping directly
                    # Use tor_switch_workers: for job j, which switches have workers?
                    # We can infer: worker w is on some switch s such that s in tor_switch_workers[j]
                    # But we don't know which one
                    # So we take a conservative approach: assume all workers on all switches
                    # But that's not accurate
                    # Alternative: assume each worker is connected to the same switch as their job's edge
                    # But we don't have that info
                    # So we use the tor_switch_workers distribution
                    
                    # Instead, we assume that for each job, workers are distributed across edge switches
                    # So we consider all edge switches that have workers for job j
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
            # But we want to alleviate congestion, so we want to avoid switches that increase load on already congested links
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
    
    # === TASK 2-5: Placeholder Implementations ===
    # These will be refined in future tasks
    # For now, use a simple baseline: assign all workers to PS
    # But we must satisfy the constraint that each worker is assigned to exactly one point
    # and that INA switches are placed
    
    # For now, assign all workers to PS
    for j in range(jobs_num):
        for w in range(workers_num[j]):
            y_j_w_d[j][w] = 1
    
    # For unused workers (padding), ensure zero
    for j in range(jobs_num):
        for w in range(workers_num[j], max_workers):
            y_j_w_d[j][w] = 0
            for s in range(num_candidates):
                y_j_w_s_full[j][w][s] = 0
    
    # Return the result
    return ina_placement_expanded, jobs_routing_expanded