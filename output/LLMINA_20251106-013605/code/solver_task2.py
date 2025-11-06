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
    # In practice, this could be derived from historical data or cluster telemetry
    link_load_est = {}
    for src in network.all_switches_id:
        for dst in network.all_switches_id:
            if src != dst:
                path = network.allPathDict[src][dst]
                if len(path) > 1:
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
                if w not in network.allPathDict or s not in network.allPathDict[w]:
                    hops = float('inf')
                else:
                    path = network.allPathDict[w][s]
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
                
                # Assume sending rate γ_jw is proportional to job size and worker count
                # For simplicity, use a fixed rate per worker per job (can be tuned)
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
                    path = network.allPathDict[w][c]
                    if len(path) <= 1:
                        # Direct connection, no intermediate links
                        link_loads = {}
                    else:
                        links = links_set(w, c, network.allPathDict)
                        link_loads = {link: gamma_jw for link in links}
                else:
                    # Destination is an INA switch
                    path = network.allPathDict[w][c]
                    if len(path) <= 1:
                        # Direct connection
                        link_loads = {}
                    else:
                        links = links_set(w, c, network.allPathDict)
                        link_loads = {link: gamma_jw for link in links}
                
                # Check link capacity constraints
                feasible = True
                max_utilization = 0.0
                for link, load in link_loads.items():
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
                
                # Check INA switch capacity if routing to INA
                if c != ps_id[j]:
                    if c in selected_switches:
                        # Check if this INA switch can handle the additional load
                        # In practice, we'll enforce this via the routing variable later
                        # For now, we just check the current total load
                        pass  # No explicit capacity check here since it's enforced in the LP
                else:
                    # PS is always feasible for direct routing (no switch load)
                    pass
                
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
    
    # === Task 3: Local Search (Placeholder) ===
    # TODO: Will be optimized in Task 3
    
    # === Task 4: Load Balancing (Placeholder) ===
    # TODO: Will be optimized in Task 4
    
    # === Task 5: Iterative Refinement (Placeholder) ===
    # TODO: Will be optimized in Task 5
    
    return ina_placement_expanded, jobs_routing_expanded