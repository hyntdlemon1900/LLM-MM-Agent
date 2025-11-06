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
    
    # --- TASK 1: Candidate INA Switch Identification and Ranking ---
    # Step 1: For each candidate switch, compute traffic centrality and bottleneck potential
    scores = {}
    
    # Weights for composite score
    weight_central = 0.6
    weight_bottleneck = 0.4
    
    for s in ina_candidates:
        traffic_central_score = 0.0
        bottleneck_potential = 0.0
        
        # Iterate over all jobs
        for j in range(jobs_num):
            # Get workers connected to switch s for this job
            # Use tor_switch_workers to get worker count per switch
            worker_count_dict = instance['tor_switch_workers'][j]
            worker_count = worker_count_dict.get(s, 0)
            if worker_count == 0:
                continue  # No workers connected to this switch for job j
            
            # Get PS ID for job j
            ps = ps_id[j]
            
            # Get shortest path from switch s to PS
            path = network.allPathDict.get(s, {}).get(ps, [])
            if not path:
                # No path found, skip
                continue
            path_hops = len(path) - 1
            
            # Traffic centrality: sum over workers connected to s of (m_j / path_hops)
            # Weight by job size and inverse path length (penalize long paths)
            centrality_contrib = jobs_size[j] / (path_hops + 1e-6)  # avoid division by zero
            traffic_central_score += centrality_contrib * worker_count
            
            # Bottleneck potential: sum of link bandwidths along path
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                link = (u, v)
                bandwidth = network.bandwidth_mapping.get(link, 0.0)
                bottleneck_potential += bandwidth
        
        # Normalize scores to avoid bias from scale
        # Compute composite score
        composite_score = weight_central * traffic_central_score + weight_bottleneck * bottleneck_potential
        scores[s] = composite_score
    
    # Step 2: Sort candidates by composite score in descending order
    ranked_candidates = sorted(ina_candidates, key=lambda s: scores[s], reverse=True)
    
    # Step 3: Select top-K switches
    selected_switches = ranked_candidates[:K]
    
    # Step 4: Create expanded placement vector
    for i, s in enumerate(ina_candidates):
        if s in selected_switches:
            ina_placement_expanded[i] = 1
    
    # --- TASK 2-6: Placeholder implementations ---
    # These will be optimized in future tasks, but we must satisfy constraints
    
    # For now, assign all workers to PSs (default fallback)
    for j in range(jobs_num):
        for w in range(workers_num[j]):
            y_j_w_d[j][w] = 1
    
    # Ensure inactive workers are zero-padded (already done by initialization)
    
    # Return final result
    return ina_placement_expanded, jobs_routing_expanded