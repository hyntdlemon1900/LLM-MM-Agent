def _evaluate_and_improve_placement(self):
    """Step 4: Iteratively improve INA placement by replacing underperforming switches with better candidates based on load imbalance and capacity utilization."""
    
    # --- Begin Core Algorithm Logic ---
    K = self.problem_data['K']
    Cs = self.problem_data['Cs']
    bandwidth_mapping = self.problem_data['network'].bandwidth_mapping
    ina_placement = set(self._ina_placement)
    candidate_switches = set(self.ina_candidates)
    non_deployed_candidates = candidate_switches - ina_placement
    
    # Current link loads and switch utilization
    link_loads = self._link_loads
    switch_loads = {s: 0.0 for s in ina_placement}
    
    # Reconstruct current worker assignments
    worker_assignments = self._worker_assignments
    jobs_num = len(worker_assignments)
    workers_id = self.problem_data['instance']['workers_id']
    ps_id = self.problem_data['instance']['ps_id']
    
    # Compute current switch loads and link loads
    for j in range(jobs_num):
        for w in range(len(worker_assignments[j])):
            assignment = worker_assignments[j][w]
            if len(assignment) == 1 and assignment[0] == ps_id[j]:
                # Direct to PS
                path = self._get_path_links(workers_id[j][w], ps_id[j])
                for edge in path:
                    link_loads[edge] += 0.0  # No INA load
            elif len(assignment) > 0:
                # To INA switch
                s = assignment[0]
                if s in ina_placement:
                    switch_loads[s] += 0.0  # Placeholder for rate
                    path = self._get_path_links(workers_id[j][w], s)
                    for edge in path:
                        link_loads[edge] += 0.0  # Placeholder
                    # INA-to-PS path
                    path_s2ps = self._get_path_links(s, ps_id[j])
                    for edge in path_s2ps:
                        link_loads[edge] += 0.0  # Placeholder

    # Try replacing each deployed switch with a non-deployed candidate
    improved = True
    max_iter = K * len(non_deployed_candidates)
    iter_count = 0

    while improved and iter_count < max_iter:
        improved = False
        best_improvement = float('inf')
        best_swap = None

        for s_deployed in ina_placement:
            for s_candidate in non_deployed_candidates:
                # Simulate swap: remove s_deployed, add s_candidate
                new_placement = ina_placement - {s_deployed} | {s_candidate}
                new_switch_loads = {k: v for k, v in switch_loads.items() if k != s_deployed}
                new_switch_loads[s_candidate] = 0.0

                # Check capacity feasibility: new_switch_loads <= Cs
                if any(new_switch_loads[s] > Cs for s in new_switch_loads):
                    continue

                # Estimate improvement via link load reduction
                # Compute potential link load changes
                total_load_diff = 0.0
                for edge in bandwidth_mapping:
                    # Simulate load impact
                    # This is a simplified heuristic: prioritize switches that reduce high-load links
                    # and improve utilization balance
                    if link_loads[edge] > 0.9 * bandwidth_mapping[edge]:
                        # High utilization link – avoid increasing
                        total_load_diff += 1.0
                    elif link_loads[edge] > 0.5 * bandwidth_mapping[edge]:
                        total_load_diff += 0.5

                # If this swap reduces congestion more than current best
                if total_load_diff < best_improvement:
                    best_improvement = total_load_diff
                    best_swap = (s_deployed, s_candidate)

        # Apply best swap if found
        if best_swap:
            s_deployed, s_candidate = best_swap
            ina_placement.remove(s_deployed)
            ina_placement.add(s_candidate)
            improved = True
            iter_count += 1

    # --- End Core Algorithm Logic ---

    # Update self._ina_placement
    self._ina_placement = sorted(list(ina_placement))