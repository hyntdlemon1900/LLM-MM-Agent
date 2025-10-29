def llm_solver(instance, network, K, jobs_num, Cs, topo_name):
    # ===== Tasks 1-0: Copied Implementations =====
    # (No previous tasks to copy)

    # ===== Task 1: Candidate INA Switch Selection via Topology-Aware Greedy Heuristic =====
    # This subtask selects exactly K INA switches from a candidate set to minimize makespan,
    # using a capacity-aware, topology-informed greedy heuristic with marginal benefit estimation.

    # Step 0: Initialize data structures
    selected_switches = set()
    updated_loads = instance['current_link_loads'].copy()  # Copy baseline load (l_e_current)

    # Define candidate switch sets based on topology
    if topo_name == "FatTree":
        ina_candidates = sorted(set(instance['edges_id'] + instance['aggrs_id'] + instance['cores_id']))
    elif topo_name == "SpineLeaf":
        ina_candidates = sorted(set(instance['leafs_id'] + instance['spines_id']))
    else:
        raise ValueError(f"Unsupported topology: {topo_name}")

    # Precompute worker-to-switch mapping and job volumes
    workers_per_switch = {}
    total_volume_s = {}  # total_gradient_volume per switch
    N_s = {}  # number of workers per switch

    # Initialize mapping from switch to list of connected workers
    for switch_id in ina_candidates:
        workers_per_switch[switch_id] = []

    # Populate worker-per-switch mapping
    for worker_id in instance['workers']:
        # Determine which switch this worker is connected to (via edge or leaf)
        # Assume instance['worker_switch_map'] maps worker_id -> switch_id
        connected_switch = instance['worker_switch_map'].get(worker_id)
        if connected_switch in workers_per_switch:
            workers_per_switch[connected_switch].append(worker_id)

    # Precompute N_s and total_volume_s for each candidate switch
    for switch_id in ina_candidates:
        workers = workers_per_switch[switch_id]
        N_s[switch_id] = len(workers)
        total_volume = 0.0
        for w in workers:
            job_id = instance['worker_job_map'][w]
            total_volume += instance['gradient_volumes'].get(job_id, 0.0)
        total_volume_s[switch_id] = total_volume

    # Define weights for benefit score
    alpha, beta, gamma = 1.0, 1.0, 1.0  # Normalized weights

    # Main greedy loop: K iterations
    for iteration in range(K):
        best_switch = None
        best_benefit = float('-inf')

        # Evaluate all unselected candidate switches
        for switch_id in ina_candidates:
            if switch_id in selected_switches:
                continue  # Skip already selected

            # ✅ Feasibility Check: Can the switch handle worst-case incoming rate?
            if total_volume_s[switch_id] > Cs.get(switch_id, float('inf')):
                continue  # Exceeds processing capacity → invalid

            # ✅ Compute E_s: set of links that would carry additional traffic if switch is activated
            E_s = set()

            # Add paths from workers connected to switch_id to the switch itself
            for w in workers_per_switch[switch_id]:
                path_to_switch = network.allPathDict.get((w, switch_id))
                if path_to_switch is None:
                    continue  # Skip if path not found
                for link in path_to_switch:
                    E_s.add(link)

            # Add path from switch_id to PS (if switch is not PS)
            path_to_PS = network.allPathDict.get((switch_id, instance['PS']))
            if path_to_PS is not None:
                for link in path_to_PS:
                    E_s.add(link)

            # ✅ Compute congestion alleviation term
            congestion_alleviation = 0.0
            for link in E_s:
                if link not in instance['link_bandwidths']:
                    continue  # Skip invalid or missing link
                C_e_bw = instance['link_bandwidths'][link]
                if C_e_bw <= 0:
                    continue
                current_load = updated_loads.get(link, 0.0)
                utilization = current_load / C_e_bw
                if utilization >= 1.0:
                    continue  # Already saturated → minimal gain
                congestion_alleviation += (1.0 - utilization)

            # ✅ Compute total benefit score
            benefit = (
                alpha * N_s[switch_id] +
                beta * total_volume_s[switch_id] +
                gamma * congestion_alleviation
            )

            # Update best candidate if this one is better
            if benefit > best_benefit:
                best_benefit = benefit
                best_switch = switch_id

        # ❗ If no valid switch found, fail gracefully
        if best_switch is None:
            raise RuntimeError(
                f"No feasible INA switch available for selection in iteration {iteration + 1}. "
                f"Consider increasing K or relaxing capacity constraints."
            )

        # ✅ Select best switch
        selected_switches.add(best_switch)

        # ✅ Update global link loads: simulate adding traffic from workers to best_switch
        # Add worker → best_switch traffic
        for w in workers_per_switch[best_switch]:
            path_to_switch = network.allPathDict.get((w, best_switch))
            if path_to_switch is None:
                continue
            job_id = instance['worker_job_map'][w]
            vol = instance['gradient_volumes'].get(job_id, 0.0)
            for link in path_to_switch:
                updated_loads[link] += vol

        # Add best_switch → PS traffic
        path_to_PS = network.allPathDict.get((best_switch, instance['PS']))
        if path_to_PS is not None:
            for link in path_to_PS:
                updated_loads[link] += total_volume_s[best_switch]

    # Final output for Task 1: Selected INA switches and updated link loads
    ina_placement_expanded = {
        'selected_switches': list(selected_switches),
        'updated_loads': updated_loads
    }

    # ===== Tasks 2-5: Defaults (PLACEHOLDERS) =====
    # TODO: Will be optimized in Task 2
    # Default: Assign all workers directly to PS (no INA usage)
    jobs_routing_expanded = {}
    for j in range(jobs_num):
        job_id = f"job_{j}"
        jobs_routing_expanded[job_id] = {
            'workers': [f"worker_{i}" for i in range(100)],  # Sample workers
            'aggregation_point': instance['PS'],
            'path': [f"worker_{i}" for i in range(100)] + [instance['PS']],
            'rate': 10.0  # MB/s
        }

    # TODO: Will be optimized in Task 3
    # Default: All jobs use direct PS routing
    ina_placement_expanded['routing'] = {
        'worker_assignment': {},
        'ina_usage': {},
        'flow_rates': {}
    }

    # TODO: Will be optimized in Task 4
    # Default: Assume all jobs complete at same time
    makespan = 1.0  # Placeholder

    # TODO: Will be optimized in Task 5
    # Default: No refinement
    refinement_log = []

    # ===== Assemble and Return =====
    return ina_placement_expanded, jobs_routing_expanded, makespan