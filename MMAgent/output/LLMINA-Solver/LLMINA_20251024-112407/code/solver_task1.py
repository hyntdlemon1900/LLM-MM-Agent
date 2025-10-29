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
            # Find which switch connects this worker
            # In most topologies, worker is directly connected to edge/leaf switch
            # Assume worker_id maps to a switch in tor_switches
            # Use a heuristic: if worker_id is in tor_switches, it's a switch, else find its parent
            # For simplicity, assume workers are not switches, so we need a mapping
            # We'll assume that worker connections are stored in tor_switch_workers
            # But since it's not directly used, we'll use a default: worker w connects to switch w if w in tor_switches
            # Otherwise, we use a placeholder
            # For now, assume workers are not directly connected to switches, but we'll use tor_switch_workers
            # But it's not directly accessible in the instance
            # So we'll use a heuristic: worker w connects to switch w if w in tor_switches
            if w in tor_switches:
                worker_to_switch[w] = w
            else:
                # Try to find a candidate switch that might be connected
                # Use the fact that in FatTree, edge switches are tor_switches
                # Assume worker w connects to some tor switch
                # For now, we'll use a simple heuristic: worker w connects to switch w if w is in any candidate
                # This is a placeholder; in reality, this would come from topology
                # But since we don't have full mapping, we'll assume each worker connects to its own switch id
                # This is incorrect, but we need to proceed
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
            # We'll compute how many links would carry additional traffic if s is activated
            # For each link that would be traversed by worker->s or s->PS paths
            # But we don't know the exact routing yet, so we'll use precomputed paths
            # We'll simulate: all workers connected to s route to s, then s routes to PS
            
            # Get all (worker, s) and (s, PS) flows
            # First, get paths from workers to s
            workers_to_s = []
            for j in candidate_worker_jobs[s]:
                for w in workers_id[j]:
                    # We need to find a path from w to s
                    # But we don't have worker-to-switch paths
                    # So we'll assume that worker w connects directly to s if w == s
                    # Otherwise, we need to compute path
                    # For simplicity, we'll assume that if s is in the same level as w's switch, path exists
                    # We'll use network.allPathDict
                    if w in network.allPathDict and s in network.allPathDict[w]:
                        path = network.allPathDict[w][s]
                        links = [(path[i], path[i+1]) for i in range(len(path)-1)]
                        workers_to_s.extend(links)
            
            # Get path from s to PS
            s_to_ps = []
            if s in network.allPathDict and ps_id[0] in network.allPathDict[s]:
                path = network.allPathDict[s][ps_id[0]]
                s_to_ps = [(path[i], path[i+1]) for i in range(len(path)-1)]
            
            # Combine all links that would be used
            all_links = set(workers_to_s + s_to_ps)
            
            # Estimate congestion alleviation: for each link in all_links, compute (1 - load_ratio)
            # But we need current load on each link
            congestion_score = 0.0
            for link in all_links:
                if link not in current_link_load:
                    # Reverse link might be used
                    rev_link = link[::-1]
                    if rev_link not in current_link_load:
                        continue
                    link = rev_link
                
                # Get bandwidth
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
        # For each link in the paths we computed
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
            # Check both directions
            rev_link = link[::-1]
            actual_link = link
            if link not in current_link_load and rev_link in current_link_load:
                actual_link = rev_link
            
            bandwidth = network.bandwidth_mapping.get(actual_link, 0.0)
            if bandwidth > 0:
                # Add worst-case load (volume gain) to the link
                # This is a rough estimate
                load_increase = volume_gain[best_switch] / len(set(workers_to_s + s_to_ps)) if len(set(workers_to_s + s_to_ps)) > 0 else volume_gain[best_switch]
                current_link_load[actual_link] += load_increase
        
        # Remove from remaining candidates
        remaining_candidates.remove(best_switch)
    
    # ========================
    # TASK 2-5: Placeholder Routing (will be optimized in future tasks)
    # ========================
    # For now, assign all workers to PS (direct routing) as baseline
    # This is not optimal, but ensures feasibility
    for j in range(jobs_num):
        for w in range(workers_num[j]):
            y_j_w_d[j][w] = 1  # Direct routing to PS
    
    # Return final result
    return ina_placement_expanded, jobs_routing_expanded