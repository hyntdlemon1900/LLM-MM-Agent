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