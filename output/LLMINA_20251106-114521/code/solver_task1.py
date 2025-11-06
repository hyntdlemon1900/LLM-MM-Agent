# Auto-generated solver with helper function imports
import sys
from pathlib import Path
import copy

# Add problem directory to path (using absolute path)
problem_dir = r'/home/hyn/LLM-MM-Agent-LLMINA-clean/MMBench/problem/LLMINA/runtime'
if problem_dir not in sys.path:
    sys.path.insert(0, problem_dir)

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
    # Task 1: Candidate INA Switch Identification and Initial Centrality Scoring
    # Step 1: Identify candidate switches based on topology
    candidates = get_ina_candidates(network)
    
    # Determine the relevant candidate set (aggregation in FatTree, spine in SpineLeaf)
    if topo_name == 'FatTree':
        # Use aggregation switches (aggrs_id) as primary candidates
        candidate_set = set(network.aggrs_id) if hasattr(network, 'aggrs_id') else set()
    elif topo_name == 'SpineLeaf':
        # Use spine switches (spines_id) as primary candidates
        candidate_set = set(network.spines_id) if hasattr(network, 'spines_id') else set()
    else:
        # Fallback: use all candidates
        candidate_set = set(candidates)
    
    # Filter candidates to only include those in the relevant layer
    valid_candidates = [s for s in candidates if s in candidate_set]
    
    # Step 2: Compute worker-attached gradient volume for each candidate switch
    # Map each worker to its ToR switch
    tor_switch_workers = instance['tor_switch_workers']  # job_id -> worker_id -> tor_switch_id
    
    # Initialize volume and exposure counters
    volume = {s: 0.0 for s in valid_candidates}
    exposure = {s: 0 for s in valid_candidates}
    
    # Define lambda for score weighting (default 0.6: more weight to volume)
    lambda_weight = 0.6
    
    # For each job, compute contribution of workers connected to switches adjacent to candidate switches
    for job_idx in range(jobs_num):
        job_size = instance['jobs_size'][job_idx]
        worker_ids = instance['workers_id'][job_idx]
        ps_id = instance['ps_id'][job_idx]
        
        # Get all worker-to-PS paths
        for worker_id in worker_ids:
            # Find which ToR switch the worker is connected to
            tor_switch = None
            for tor, workers in tor_switch_workers[job_idx].items():
                # workers is a list of worker IDs
                if worker_id in workers:
                    tor_switch = tor
                    break
            
            if tor_switch is None:
                continue  # Skip if worker not found in any ToR
            
            # Find all switches directly connected to this ToR (i.e., next-hop in path)
            next_hops = []
            for neighbor in network.G.neighbors(tor_switch):
                if topo_name == 'FatTree' and neighbor in network.aggrs_id:
                    next_hops.append(neighbor)
                elif topo_name == 'SpineLeaf' and neighbor in network.spines_id:
                    next_hops.append(neighbor)
            
            # For each potential next-hop (aggr/spine), add worker's job size to volume
            for s in next_hops:
                if s in valid_candidates:
                    volume[s] += job_size
            
            # Now compute path exposure: how many unique worker-to-PS paths pass through s
            # Use precomputed allPathDict
            if worker_id not in network.allPathDict or ps_id not in network.allPathDict[worker_id]:
                continue  # Skip if no path exists
            path = network.allPathDict[worker_id][ps_id]
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                # Check if current switch is in path and is a valid candidate
                if u in valid_candidates:
                    exposure[u] += 1
                if v in valid_candidates:
                    exposure[v] += 1
        
        # Also consider direct paths from worker to PS if worker is directly connected to PS
        # But in most topologies, workers are not directly connected to PS, so skip unless needed
    
    # Step 3: Compute composite centrality score
    score = {}
    max_volume = max(volume.values()) if volume.values() else 1
    max_exposure = max(exposure.values()) if exposure.values() else 1

    for s in valid_candidates:
        vol_norm = volume[s] / max_volume
        exp_norm = exposure[s] / max_exposure
        score[s] = lambda_weight * vol_norm + (1 - lambda_weight) * exp_norm
    
    # Sort candidates by descending score
    ranked_candidates = sorted(valid_candidates, key=lambda s: score[s], reverse=True)
    
    # Step 4: Select top K candidates using greedy selection
    selected_candidates = ranked_candidates[:K]
    
    # Convert selected candidates to expanded placement vector (aligned with sorted full candidate list)
    ina_placement_expanded = [1 if s in selected_candidates else 0 for s in candidates]
    
    # Task 2-6: Placeholder implementations (will be optimized in later tasks)
    # For now, initialize dummy routing (will be refined in later tasks)
    max_workers = max(instance['workers_num'])
    jobs_routing_expanded = [
        # y_j_w_s_full: [jobs_num][max_workers][len(candidates)]
        [
            [
                [1 if s == selected_candidates[0] else 0 for s in candidates]  # Dummy: assign to first INA
                for w in range(max_workers)
            ]
            for j in range(jobs_num)
        ],
        # y_j_w_d: [jobs_num][max_workers] - direct to PS
        [
            [1 for w in range(max_workers)]  # All workers go directly to PS for now
            for j in range(jobs_num)
        ]
    ]
    
    # Return the final output
    return ina_placement_expanded, jobs_routing_expanded