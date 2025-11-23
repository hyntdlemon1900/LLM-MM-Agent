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