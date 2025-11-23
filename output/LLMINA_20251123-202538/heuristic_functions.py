# Heuristic Functions - Auto-generated

from typing import Dict, List, Any, Tuple

# ============ Function 1: _initialize_placement_candidates ============

def _initialize_placement_candidates(self):
    """Rank and filter INA candidate switches based on centrality and job access patterns to prioritize high-impact placement locations."""

    # --- Begin Core Algorithm Logic ---
    job_workers = self.problem_data["instance"]["workers_id"]
    ps_id = self.problem_data["instance"]["ps_id"]
    ina_candidates = self.ina_candidates
    allPathDict = self.allPathDict

    # Compute centrality score: number of jobs with workers reachable via short paths, weighted by worker density
    centrality_scores = {}
    for s in ina_candidates:
        score = 0.0
        for j, worker_ids in enumerate(job_workers):
            # Count how many workers in job j can reach switch s with a single hop (direct connection)
            for w in worker_ids:
                if s in allPathDict[w][s] or s in allPathDict[s][w]:
                    # Direct path: worker to switch or switch to worker
                    score += 1.0
            # Add bonus if PS is directly connected to switch s
            if ps_id[j] in allPathDict[s]:
                score += 0.5
        centrality_scores[s] = score

    # Sort candidates by centrality score in descending order
    ranked_candidates = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)
    # --- End Core Algorithm Logic ---

    # Write results to self
    self._placement_candidates = ranked_candidates

# ============ Function 2: _construct_initial_placement ============

def _construct_initial_placement(self):
    """Construct a feasible initial INA deployment by greedily selecting top-ranked switches under budget K."""

    # --- Begin Core Algorithm Logic ---
    # Select the first K switches from the pre-ranked candidates
    # This ensures a feasible placement under the budget constraint
    k = self.problem_data["K"]
    self._ina_placement = [
        switch_id for switch_id, _ in self._placement_candidates[:k]
    ]

    # --- End Core Algorithm Logic ---

    # Write results to self, e.g.: self.some_new_state = result
    # (Must match 'Member Variables Written')
    # self._ina_placement is updated with selected switch IDs

# ============ Function 3: _assign_workers_to_aggregation_points ============

def _assign_workers_to_aggregation_points(self):
    """Step 3: Assign each worker to an aggregation point (INA or PS) to minimize job completion time, respecting INA deployment and capacity constraints."""

    jobs_num = self.problem_data['jobs_num']
    workers_num = self.problem_data['instance']['workers_num']
    jobs_size = self.problem_data['instance']['jobs_size']
    ps_id = self.problem_data['instance']['ps_id']
    ina_placement = set(self._ina_placement)
    ina_candidates = self.ina_candidates
    allPathDict = self.allPathDict

    # Initialize output state
    self._worker_assignments = []
    self._job_rates = []
    self._link_loads = {}

    # Process each job
    for j in range(jobs_num):
        job_workers = self.problem_data['instance']['workers_id'][j]
        job_ps = ps_id[j]
        job_size = jobs_size[j]

        # Track link loads for this job
        job_link_loads = {}

        # Compute effective rate: gamma_j is limited by the slowest worker
        # Initially, assume all workers send at full rate
        worker_rates = [1.0] * len(job_workers)  # Placeholder; actual rate will be assigned

        # For each worker, decide optimal assignment: INA or PS
        worker_assignments = []
        total_gamma_jws = 0.0  # Total rate assigned to INA for this job
        total_gamma_jwd = 0.0  # Total rate assigned to PS for this job

        for w_idx, w in enumerate(job_workers):
            best_switch = None
            best_rate = 0.0
            best_path_load = 0.0

            # Try assigning to PS (direct path)
            path_to_ps = allPathDict[w][job_ps]
            links_to_ps = [(path_to_ps[i], path_to_ps[i + 1]) for i in range(len(path_to_ps) - 1)]
            path_load = sum(1.0 for link in links_to_ps)  # Simple path length as proxy

            # Check if PS path is feasible (already assumed)
            # Assume PS rate limit is not binding here; handled in model
            # Direct rate is full if no bottleneck
            direct_rate = 1.0
            total_gamma_jwd += direct_rate

            # Consider all INA switches that are deployed
            for s_idx, s in enumerate(ina_candidates):
                if s not in ina_placement:
                    continue  # Skip non-deployed INA switches

                path_to_s = allPathDict[w][s]
                links_to_s = [(path_to_s[i], path_to_s[i + 1]) for i in range(len(path_to_s) - 1)]
                path_load_to_s = sum(1.0 for link in links_to_s)

                # Use path length as proxy for link congestion (simplified)
                # In real algorithm, compute actual load using precomputed routing
                load = path_load_to_s
                # Better rate if less load and feasible
                if load < path_load:
                    # This path is better; update best
                    path_load = load
                    best_switch = s
                    best_rate = 1.0  # Assume rate can be fully utilized
                    best_path_load = load

            # Assign worker to best option
            if best_switch is not None:
                # Assign to INA
                worker_assignments.append([best_switch])
                total_gamma_jws += 1.0
            else:
                # Assign to PS
                worker_assignments.append([job_ps])
                total_gamma_jwd += 1.0

        # Compute effective job rate: min over all workers
        # But rate is bounded by slowest worker
        # Here, we assume all workers are equally fast unless capacity constraints
        # So job rate is min of (total_gamma_jws, total_gamma_jwd) but constrained by worker
        # In reality, gamma_j = min over w of gamma_jw
        # But since we are using 1.0 per worker, gamma_j = 1.0 unless constrained
        effective_rate = 1.0  # Simplified for now

        # Check INA capacity: if total_gamma_jws > Cs, must reduce rate
        Cs = self.problem_data['Cs']
        if total_gamma_jws > Cs:
            # Scale down rate proportionally
            scale_factor = Cs / total_gamma_jws
            effective_rate = min(effective_rate, scale_factor)
            # But we only reduce rate if needed
            # In assignment, we keep assignment but reduce rate

        # Update job rate
        self._job_rates.append(effective_rate)

        # Update worker assignments
        self._worker_assignments.append(worker_assignments)

        # Update link loads for all flows
        # For each worker
        for w_idx, w in enumerate(job_workers):
            if worker_assignments[w_idx][0] == job_ps:
                # Direct to PS
                path = allPathDict[w][job_ps]
                links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                for link in links:
                    self._link_loads[link] = self._link_loads.get(link, 0.0) + 1.0
            else:
                # To INA switch
                s = worker_assignments[w_idx][0]
                path = allPathDict[w][s]
                links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                for link in links:
                    self._link_loads[link] = self._link_loads.get(link, 0.0) + 1.0

        # For INA-to-PS flow
        for s in ina_placement:
            if s in ina_candidates:
                s_idx = ina_candidates.index(s)
                path = allPathDict[s][job_ps]
                links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                for link in links:
                    self._link_loads[link] = self._link_loads.get(link, 0.0) + 1.0

# ============ Function 4: _evaluate_and_improve_placement ============

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

# ============ Function 5: solve_with_heuristic ============

def solve_with_heuristic(self):
    """Orchestrates the full heuristic workflow: initializes candidate ranking, constructs initial placement, assigns workers, iteratively improves placement, and formats the final solution dictionary."""

    # --- Begin Core Algorithm Logic ---
    self._initialize_placement_candidates()
    self._construct_initial_placement()
    self._assign_workers_to_aggregation_points()
    self._evaluate_and_improve_placement()
    # --- End Core Algorithm Logic ---

    # Write results to self
    alpha_val = 1.0 / max(self._job_rates) if self._job_rates else float('inf')
    makespan = 1.0 / alpha_val if alpha_val > 1e-9 else float('inf')

    self.solution = {
        "ina_placement_switches": self._ina_placement,
        "worker_to_ina": [
            [
                [1 if s in self._worker_assignments[j][w] and s != self.problem_data["instance"]["ps_id"][j] else 0
                 for s in self.ina_candidates]
                for w in range(max(self.problem_data["instance"]["workers_num"]))]
            for j in range(self.problem_data["jobs_num"])
        ],
        "worker_to_ps": [
            [1 if self._worker_assignments[j][w] == [self.problem_data["instance"]["ps_id"][j]] else 0
             for w in range(max(self.problem_data["instance"]["workers_num"]))]
            for j in range(self.problem_data["jobs_num"])
        ],
        "job_rates": self._job_rates,
        "makespan": makespan,
    }

