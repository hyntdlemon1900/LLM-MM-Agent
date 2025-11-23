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