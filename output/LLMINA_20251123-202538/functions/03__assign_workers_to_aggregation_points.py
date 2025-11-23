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