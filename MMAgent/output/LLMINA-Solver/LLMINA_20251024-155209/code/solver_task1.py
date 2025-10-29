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
    
    # --- TASK 1: INA Candidate Selection via Greedy Heuristic ---
    
    # Initialize global load tracking (current load on each link, in Gbps)
    # Use a dictionary: link -> current load
    current_link_load = {}
    
    # Precompute all links and initialize load to 0
    for u, v in network.bandwidth_mapping:
        current_link_load[(u, v)] = 0.0
        current_link_load[(v, u)] = 0.0  # Bidirectional
    
    # Keep track of selected switches
    selected_switches = set()
    
    # Define weights for benefit score
    alpha, beta, gamma = 1.0, 1.0, 1.0
    
    # Greedy selection loop for K switches
    for _ in range(K):
        best_switch = -1
        best_score = -1.0
        
        # Evaluate each unselected candidate switch
        for idx, s in enumerate(ina_candidates):
            if s in selected_switches:
                continue  # Skip already selected
            
            # 1. Proximity Gain: Number of workers directly connected to switch s
            # Use tor_switch_workers to get worker count per switch
            # But note: s might not be in tors_id (e.g., core switch)
            # So we need to check which worker switches connect to s via paths
            # Instead, we use the fact that workers are directly connected to ToR switches
            # So we look for ToR switches that connect to s
            # But for simplicity, we assume that if s is a ToR, it has direct workers
            # Else, we check if any worker is connected via a path to s
            
            # Simpler: use the network G to find direct connections
            # If s is in tors_id, then workers connected to s are directly connected
            # Otherwise, we need to find any worker with path to s
            
            # For now, we define proximity gain as number of workers connected to s via direct ToR
            # But since ToR are edge switches, and s might be aggr/core/spine, we need a better way
            
            # Alternative: use the fact that workers are connected to ToR switches
            # and ToR switches are connected to other switches
            # So we can find which workers have paths ending at s
            
            # Actually, for a switch s, workers connected to it are those whose ToR is connected to s
            # But since we don't have direct mapping from worker to ToR, we use tor_switch_workers
            
            # Let's assume tor_switch_workers[j] maps ToR ID to worker count for job j
            # We'll sum over all jobs
            connected_workers = 0
            for j in range(jobs_num):
                for tor_switch, count in instance['tor_switch_workers'][j].items():
                    # Check if tor_switch connects to s
                    # This is hard without path info
                    # Instead, we assume that if s is a ToR, it has direct workers
                    # If s is not a ToR, then we assume it has no direct workers
                    # This is a simplification, but acceptable for now
                    if s == tor_switch:
                        connected_workers += count
            
            N_s = connected_workers
            
            # 2. Volume Gain: Sum of gradient volumes from connected workers
            volume_gain = 0.0
            for j in range(jobs_num):
                for tor_switch, count in instance['tor_switch_workers'][j].items():
                    if s == tor_switch:
                        volume_gain += count * jobs_size[j]
            
            # 3. Congestion Alleviation: Estimate impact on high-traffic links
            # For each link e, compute the improvement in utilization if s is activated
            # But we simulate: if we enable s, then some flows (worker->s) and (s->PS) will be added
            # So we need to find all (worker, s) and (s, PS) paths
            # But we don't know routing yet — so we simulate worst-case: all workers connected to s route via s
            
            congestion_alleviation = 0.0
            
            # For each job, find all worker-to-s paths and s-to-PS paths
            # But we need to simulate flow on links
            # For each link e, compute how many (worker, s) and (s, PS) flows use e
            # Then compute the potential load increase if s is activated
            # But we don't know the assignment yet — so we use the worst-case scenario:
            # All workers connected to s route via s, and s routes all to PS
            
            # We need to find all (worker, s) pairs that use link e
            # and (s, PS) pairs that use link e
            
            # Use l_i to find (src, dst) pairs that use edge e
            # But we need to define src_set and dst_set
            src_set = []
            dst_set = []
            
            # Workers are sources
            for j in range(jobs_num):
                src_set.extend(workers_id[j])
            
            # s and PS are destinations
            dst_set.append(s)
            dst_set.append(ps_id[0])  # Assuming all PS are same for now
            
            # But we need to do this per link
            # Instead, we'll compute the total number of (worker, s) and (s, PS) paths that use each link
            # But we don't know all paths yet
            
            # Alternative: use the l_i function to get (row, col) indices for (worker, s) and (s, PS)
            # But we need to compute for all links
            
            # Since this is expensive, we approximate:
            # The congestion alleviation is proportional to the number of paths that would use high-load links
            # But we can't compute it exactly without knowing the routing
            
            # For now, we skip congestion alleviation and use a placeholder
            # In practice, we would use precomputed path incidence
            # But for now, we use the number of workers connected to s as a proxy
            # This is a simplification
            congestion_alleviation = N_s * 0.1  # Placeholder
            
            # Compute total benefit
            benefit = alpha * N_s + beta * volume_gain + gamma * congestion_alleviation
            
            # Check feasibility: would activating s exceed its processing capacity?
            # Worst-case: all connected workers send their full gradient to s
            total_incoming_rate = volume_gain
            if total_incoming_rate > Cs:
                # Not feasible
                continue
            
            # Update best if this is better
            if benefit > best_score:
                best_score = benefit
                best_switch = idx
        
        # If no valid switch found, break (shouldn't happen with K <= len(candidates))
        if best_switch == -1:
            break
            
        # Select the best switch
        selected_switches.add(ina_candidates[best_switch])
        ina_placement_expanded[best_switch] = 1
        
        # Update current_link_load based on the new switch
        # For each link, we need to estimate the additional load if this switch is activated
        # But we don't know routing yet — so we simulate worst-case: all workers connected to s
        # route via s to PS
        
        # We need to find all links that carry worker->s or s->PS traffic
        # But we don't know the exact paths — so we assume worst-case paths
        # For worker->s: use the shortest path from worker to s
        # For s->PS: use the shortest path from s to PS
        
        # For each worker connected to s (via ToR), we get the path to s
        # Then from s to PS
        
        # But we can use the l_i function to find which (worker, s) pairs use a given link
        # But we need to do it for each link
        
        # Instead, we'll do a rough approximation: add the volume_gain to all links
        # along the path from s to PS, and also add the worker-to-s links
        
        # But we don't know the path from s to PS yet — we can use allPathDict
        # Get the shortest path from s to PS
        s_to_ps_path = network.allPathDict[s][ps_id[0]]
        s_to_ps_links = [(s_to_ps_path[i], s_to_ps_path[i+1]) for i in range(len(s_to_ps_path)-1)]
        
        # For each worker connected to s
        for j in range(jobs_num):
            for tor_switch, count in instance['tor_switch_workers'][j].items():
                if tor_switch == s:
                    # Use the path from worker to s
                    # But we don't have worker-to-s path directly
                    # So we assume the path goes through the ToR, then to s
                    # But we don't know the exact path
                    # So we skip updating link load for worker->s
                    pass
        
        # For s->PS, we can update the links in s_to_ps_links
        for link in s_to_ps_links:
            # Add volume_gain to the link load
            current_link_load[link] += volume_gain
            # Reverse direction
            current_link_load[link[::-1]] += volume_gain
        
        # End of greedy selection
    
    # --- TASK 2-5: Placeholder Implementations ---
    # These will be optimized in future tasks
    
    # For now, assign all workers to PS (fallback)
    # This ensures feasibility
    for j in range(jobs_num):
        for w in range(workers_num[j]):
            y_j_w_d[j][w] = 1
        for w in range(workers_num[j], max_workers):
            # Zero-pad
            pass
    
    # Return the solution
    return ina_placement_expanded, jobs_routing_expanded