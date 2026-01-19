from __future__ import annotations
from typing import List, Dict, Tuple, Any, Optional
import copy
import networkx as nx
from MMBench.problem.LLMINA.runtime.TemplateSolver import TemplateSolver
import random

class ModelSolver(TemplateSolver):
    """
    MILP Solver for the LLMINA (In-Network Aggregation) Problem.
    
    This class provides the **Architectural Context** for the Data Center Network.
    It unpacks the physical topology (Fat-Tree) and logical job requirements 
    into a structured format usable by heuristic algorithms.
    """

    def __init__(
        self,
        instance: Dict[str, Any],
        network: Any,
        ina_budget: int,
        jobs_num: int,
        Cs: float = 750.0,
        base_bw: float = 100.0,
        topo_name: str = "FatTree",
    ):
        """
        Initializes the solver context.
        
        Args:
            instance (dict): Job specifications (Workers, PS, Volume).
            network (object): Physical Network Topology Object.
            ina_budget(int): Max number of INA-enabled switches allowed.
            jobs_num (int): Number of concurrent jobs.
            Cs (float): Switch Processing Throughput (Gbps).
            base_bw (float): Link Bandwidth (Gbps).
            topo_name (str): Topology type (e.g., "FatTree").
        """
        # Store raw inputs
        problem_data = {
            "instance": instance,
            "network": network,
            "ina_budget": ina_budget,
            "jobs_num": jobs_num,
            "Cs": Cs,
            "base_bw": base_bw,
            "topo_name": topo_name,
        }
        super().__init__(problem_data)
        
        self.solution: Optional[Dict[str, Any]] = None
        
        # Trigger explicit data unpacking
        self._preprocess_data()

    def _preprocess_data(self) -> None:
        """
        DATA PREPROCESSING & ARCHITECTURE DEFINITION
        
        This method documents the **Physical Network Specifications**.
        It defines the topology structure and hardware throughput limits that constitute 
        the constraint environment for the algorithm.
        """
        
        # =========================================================================
        # 1. PHYSICAL TOPOLOGY STRUCTURE (Fat-Tree)
        # =========================================================================
        # The cluster uses a standard 3-tier Fat-Tree topology.
        #
        # [A] CONNECTIVITY HIERARCHY:
        #    1. Edge Layer (ToR Switches): 
        #       - Connectivity: Directly connected to Servers.
        #       - Specification: Each ToR connects to exactly **20 Servers**.
        #    2. Aggregation Layer (Aggr Switches): 
        #       - Connectivity: Interconnects ToR switches.
        #    3. Core Layer (Core Switches): 
        #       - Connectivity: Interconnects different Pods.
        #
        # [B] LOGICAL GROUPING (PODs):
        #    - A "Pod" consists of a set of ToR and Aggr switches.
        #    - Intra-Pod Path: Server -> ToR -> Aggr -> ToR -> Server.
        #    - Inter-Pod Path: Server -> ToR -> Aggr -> Core -> Aggr -> ToR -> Server.
        #
        # =========================================================================
        # 2. HARDWARE THROUGHPUT SPECIFICATIONS (The Constraints)
        # =========================================================================
        # Congestion occurs whenever Traffic Rate > Link Bandwidth at ANY location.
        #
        # [A] SERVER ACCESS LINKS (The "Last Hop"):
        #    - Definition: The physical link connecting a Server (Worker/PS) to a ToR.
        #    - Bandwidth: Fixed at `base_bw` (Gbps).
        #    - Constraint: The sum of bidirectional traffic on this link cannot exceed `base_bw`.
        #
        # [B] SWITCH UPLINKS (The Fabric Links):
        #    - Definition: Links connecting ToR -> Aggr and Aggr -> Core.
        #    - Bandwidth ratio: The topology design has a **2:1 Oversubscription Ratio** #      at the Edge layer (Total Downlink Bandwidth = 2 * Total Uplink Bandwidth).
        #    - Constraint: Traffic leaving a ToR is limited by this physical ratio.
        #
        # [C] SWITCH PROCESSING UNIT:
        #    - Definition: The computation capability of a switch if INA is enabled.
        #    - Throughput: Fixed at `Cs` (Gbps).
        #    - Constraint: Sum(Incoming Flow Rates) <= Cs.
        #
        # =========================================================================
        # 3. PHYSICS OF FLOW (Conservation Laws)
        # =========================================================================
        #    1. Rate Definition: Rate (Gbps) = Volume (Gb) / Time (s).
        #    2. Flow Conservation: A flow with Rate R consumes R bandwidth on 
        #       **every single link** it traverses.
        # =========================================================================

        # Load raw source objects
        raw_instance = self.problem_data["instance"]
        raw_network = self.problem_data["network"]

        # -------------------------------------------------------------------------
        # 1. Global Scalar Constraints
        # -------------------------------------------------------------------------
        self.ina_budget: int = self.problem_data["ina_budget"]                # INA Deployment Budget
        self.jobs_num: int = self.problem_data["jobs_num"]  # Total concurrent jobs
        self.Cs: float = float(self.problem_data["Cs"])     # Switch Processing Cap (Gbps)
        self.base_bw: float = float(self.problem_data["base_bw"])    # Server-ToR Link Bandwidth (Gbps)
        self.topo_name: str = self.problem_data["topo_name"]

        # -------------------------------------------------------------------------
        # 2. Job Specifications (Unpacking 'instance' Dictionary)
        # -------------------------------------------------------------------------
        # [Data Structure] instance["workers_id"]: List[List[int]]
        # [Meaning] A nested list where index j corresponds to Job j.
        #           The inner list contains the Node IDs of all workers for that job.
        # [Example] [[101, 102], [201, 202, 203]] -> Job 0 has workers 101, 102.
        self.workers_id: List[List[int]] = raw_instance["workers_id"]

        # [Data Structure] instance["ps_id"]: List[int]
        # [Meaning] A list where index j is the Node ID of the Parameter Server for Job j.
        # [Physical Loc] These nodes are servers attached to ToR switches.
        self.ps_id: List[int] = raw_instance["ps_id"]

        # [Data Structure] instance["jobs_size"]: List[float]
        # [Meaning] The gradient data volume (in Gigabits) Job j generates per iteration.
        self.jobs_size: List[float] = raw_instance["jobs_size"]

        # [Data Structure] instance["workers_num"]: List[int]
        # [Meaning] The count of workers belonging to Job j.
        self.workers_num: List[int] = raw_instance["workers_num"]

        # -------------------------------------------------------------------------
        # 3. Network Topology Details (Unpacking 'network' Object)
        # -------------------------------------------------------------------------
        # [Graph Object] network.G (networkx.Graph)
        # Represents the physical wiring. Nodes are Switches or Servers.
        self.G: nx.Graph = copy.deepcopy(raw_network.G)

        # [Path Data] network.allPathDict: Dict[int, Dict[int, List[int]]]
        # [Meaning] Pre-computed shortest paths. 
        #           Query: self.allPathDict[src_id][dst_id] -> List of Node IDs.
        #           Used to determine which links a flow traverses.
        self.allPathDict: Dict[int, Dict[int, List[int]]] = copy.deepcopy(raw_network.allPathDict)

        # [Bandwidth Map] network.bandwidth_mapping: Dict[Tuple[int, int], float]
        # [Meaning] Bandwidth of directed physical links.
        #           Key: (u, v), Value: Bandwidth in Gbps.
        self.bandwidth_mapping: Dict[Tuple[int, int], float] = (
            copy.deepcopy(raw_network.bandwidth_mapping) 
            if hasattr(raw_network, "bandwidth_mapping") else {}
        )

        # [Switch Classification]
        # The 'network' object exposes lists of switch IDs based on their tier.
        # These are crucial for determining valid INA placement candidates.
        self.tors_id: List[int] = list(getattr(raw_network, "tors_id", []))     # Edge Layer
        self.aggrs_id: List[int] = list(getattr(raw_network, "aggrs_id", []))   # Aggregation Layer
        self.cores_id: List[int] = list(getattr(raw_network, "cores_id", []))   # Core Layer
        self.all_switches_id: List[int] = list(getattr(raw_network, "all_switches_id", []))

        # -------------------------------------------------------------------------
        # 4. Derived Variables (Calculated for Optimization Context)
        # -------------------------------------------------------------------------
        
        # [Variable] self.ina_candidates: List[int]
        # [Logic] Based on the topology, select which switches are programmable.
        # For Fat-Tree, usually ToR, Aggr, and Core are all candidates.
        self.ina_candidates: List[int] = self._compute_ina_candidates()
        self.ina_candidates_num: int = len(self.ina_candidates)

    # =============================================================================
    # Helper Methods
    # =============================================================================

    def _compute_ina_candidates(self) -> List[int]:
        """
        Returns a sorted list of unique switch IDs eligible for INA placement.
        """
        candidates = []
        if self.topo_name == "FatTree":
            # In FatTree, logical aggregation can happen at any switch layer
            candidates = self.tors_id + self.aggrs_id + self.cores_id
        elif self.topo_name == "SpineLeaf":
            candidates = self.tors_id + self.spines_id
        else:
            candidates = self.all_switches_id
        return sorted(list(set(candidates)))

    def _get_path_links(self, src: int, dst: int) -> List[Tuple[int, int]]:
        """
        Returns the list of directed edges [(u,v), (v,w)...] representing the 
        shortest path from src to dst using self.allPathDict.
        """
        if src not in self.allPathDict or dst not in self.allPathDict[src]:
            return []
        path = self.allPathDict[src][dst]
        return [(path[i], path[i+1]) for i in range(len(path) - 1)]

    def _get_node_pod(self, node_id: int) -> Optional[int]:
        """
        Determines the logical Pod ID for a given Node (Server or Switch).
        
        In a Fat-Tree topology, ToR and Aggregation switches (and their connected servers)
        belong to specific Pods. 
        
        **CRITICAL**: Core Switches constitute the backbone and DO NOT belong to any Pod.
        
        Returns:
            int: The Pod ID (0-indexed) if applicable.
            None: If the node is a Core Switch or cannot be mapped.
        """
        if self.topo_name != "FatTree":
            return 0
            
        try:
            num_tors = len(self.tors_id)
            k_half = int((num_tors / 2) ** 0.5)
            if k_half == 0: return 0
        except Exception:
            return 0

        # Check ToR
        if node_id in self.tors_id:
            return self.tors_id.index(node_id) // k_half
            
        # Check Aggr
        if node_id in self.aggrs_id:
            return self.aggrs_id.index(node_id) // k_half
            
        # Check Server (Worker/PS) - use uplink ToR
        neighbors = list(self.G.neighbors(node_id))
        for n in neighbors:
            if n in self.tors_id:
                return self.tors_id.index(n) // k_half
                
        # Core Switches or isolated nodes have no Pod
        return None

    def get_makespan(self, ina_placement_switches, worker_agg_id) -> Tuple[float, Dict[int, float]]:
        """
        Calculates the theoretically optimal makespan given a fixed INA deployment and routing configuration.

        ---------------------------------------------------------------------------
        ### FUNCTIONAL ROLE: THE ORACLE (SIMULATOR)
        ---------------------------------------------------------------------------
        This method acts as a "Black Box Simulator" or "Physics Engine" for your heuristic.
        
        - **Input**: A static configuration (Placement of INA switches + Routing of workers).
        - **Operation**: It constructs and solves a continuous Linear Programming (LP) model.
          It attempts to maximize the global traffic speed (Rate) while strictly respecting:
            1. Link Bandwidth limits (Sum of Rates <= C_e).
            2. Switch Processing limits (Sum of Rates <= C_s).
            3. Flow Conservation (Rate is constant end-to-end).
        - **Output**: The minimum possible Time (Makespan) required to finish the job under 
          these constraints.

        ---------------------------------------------------------------------------
        ### USAGE GUIDE FOR ALGORITHM DESIGNERS
        ---------------------------------------------------------------------------
        You do NOT need to implement rate calculations or throughput checks manually inside this function.
        You can use this function to **validate and score** the solutions generated 
        by your heuristic logic.
        WARNING: This function involves solving a Linear Programming (LP) model via an external solver.
        COST: High (~50ms - 200ms per call).
        Args:
            ina_placement_switches (List[int]): The specific set of switches enabled with In-Network Aggregation.
            worker_agg_id (Dict[int, Dict[int, int]]): The routing map defining where each worker sends its gradients.
                **Structure**: { job_idx: { worker_idx: Aggregation_Node_ID } }
        
        Returns:
            Tuple[float, Dict[int, float]]: 
                - float: The minimum achievable makespan (Seconds).
                - Dict[int, float]: A dictionary mapping job index to its effective rate (Gbps).

        """
        return super().get_makespan(ina_placement_switches, worker_agg_id)

    # ============ Heuristic Functions ============

    # Function 1: _init_state
    # Role: Initialize all persistent state variables used in the heuristic workflow.
    def _init_state(self):
        """
        Initializes the persistent state variables used in the heuristic search workflow.
        This method sets up the foundation for the optimization process by defining
        the initial best solution, iteration control, and convergence tracking.
        """
        # Step 1: Initialize best makespan to infinity
        self.best_makespan = float('inf')

        # Step 2: Initialize best placement as an empty list
        self.best_placement = []

        # Step 3: Initialize best routing as a list of empty lists, one per job
        self.best_routing = [[] for _ in range(self.jobs_num)]

        # Step 4: Initialize iteration counter
        self.iteration_count = 0

        # Step 5: Set maximum number of iterations for the search
        self.max_iterations = 100

        # Step 6: Initialize acceptance flag to track progress in current cycle
        self.accepted_change = False

    # Function 2: _get_switch_utilization
    # Role: Estimate current load on a switch (sum of incoming flow rates) assuming all workers are routed directly to PS.
    def _get_switch_utilization(self, switch_id: int) -> float:
        """
        Estimates the total traffic load (in Gbps equivalent) on a given switch
        by simulating all workers sending gradients directly to their respective PS
        under the assumption of shortest-path routing.

        Args:
            switch_id (int): Node ID of the switch to evaluate.

        Returns:
            float: Estimated total traffic load (in Gbps equivalent) on the switch.
        """
        total_load = 0.0

        # Iterate over each job
        for job_idx in range(self.jobs_num):
            workers = self.workers_id[job_idx]
            ps = self.ps_id[job_idx]

            # Iterate over each worker in the job
            for worker in workers:
                # Retrieve the shortest path from worker to PS
                path_edges = self._get_path_links(worker, ps)

                # Process each link in the path
                for u, v in path_edges:
                    # Get bandwidth for the link (u, v), default to base_bw if missing
                    bandwidth = self.bandwidth_mapping.get((u, v), self.base_bw)

                    # Avoid division by zero
                    if bandwidth <= 0:
                        continue

                    # Add the approximate flow rate (volume / bandwidth) to total load
                    rate = self.jobs_size[job_idx] / bandwidth
                    total_load += rate

        return total_load

    # Function 3: _select_top_k_switches
    # Role: Select the best `k` switches (up to B_INA) based on `switch_scores`.
    def _select_top_k_switches(self):
        """
        Selects the top k switches for INA placement based on their computed scores.

        Returns:
            List[int]: List of switch IDs selected for INA, sorted by score in descending order.
                       Returns empty list if ina_budget is 0.
        """
        # Step 1: If ina_budget is 0, return empty list
        if self.ina_budget == 0:
            return []

        # Step 2: Retrieve switch scores; ensure safe access using .get with default
        switch_scores = self.switch_scores or {}

        # Step 3: Sort switches by score in descending order
        sorted_switches = sorted(
            switch_scores.keys(),
            key=lambda switch_id: switch_scores.get(switch_id, 0),
            reverse=True
        )

        # Step 4: Select top min(len(switch_scores), ina_budget) switches
        k = min(len(sorted_switches), self.ina_budget)
        selected_switches = sorted_switches[:k]

        return selected_switches

    # Function 4: _assign_workers_to_aggregators
    # Role: Assign each worker to either an INA switch (if enabled) or the PS, maximizing job-level throughput under current INA placement.
    def _assign_workers_to_aggregators(self):
        """
        Assigns each worker to an optimal aggregation node (either a PS or an INA-enabled switch)
        based on minimizing path congestion and maximizing effective throughput.

        This method implements a greedy assignment strategy per job, where each worker is routed
        to the aggregator (PS or INA switch) that provides the highest possible rate under
        the current INA placement configuration.
        """
        # Step 1: Initialize the assignment list for all jobs
        self.worker_agg_id = [[] for _ in range(self.jobs_num)]

        # Step 2: Iterate over each job
        for job_idx in range(self.jobs_num):
            workers = self.workers_id[job_idx]
            ps = self.ps_id[job_idx]

            # Step 3: Define candidate aggregation nodes
            candidates = [ps]  # PS is always a candidate

            # Step 4: Add all INA-enabled switches that are valid candidates
            for switch_id in self.ina_placement_switches:
                if switch_id in self.ina_candidates:
                    candidates.append(switch_id)

            # Step 5: Assign each worker to the best candidate
            for worker in workers:
                best_rate = 0.0
                best_agg = ps  # Default fallback: send to PS

                # Step 6: Evaluate each candidate
                for candidate in candidates:
                    path_edges = self._get_path_links(worker, candidate)

                    # Skip if no path exists
                    if not path_edges:
                        continue

                    # Compute total load on the path (sum of bandwidth constraints)
                    total_load = 0.0
                    valid_path = True

                    for u, v in path_edges:
                        bandwidth = self.bandwidth_mapping.get((u, v), self.base_bw)
                        if bandwidth <= 0:
                            valid_path = False
                            break
                        total_load += bandwidth

                    if not valid_path:
                        continue

                    # Compute achievable rate: volume / total bandwidth of the path
                    rate = self.jobs_size[job_idx] / total_load

                    # Update best assignment if this rate is higher
                    if rate > best_rate:
                        best_rate = rate
                        best_agg = candidate

                # Append the best-aggregator choice for this worker
                self.worker_agg_id[job_idx].append(best_agg)

        # Return the final assignment
        return self.worker_agg_id

    # Function 5: _evaluate_solution
    # Role: Use the oracle `get_makespan` to evaluate the current solution and update the best solution if improved.
    def _evaluate_solution(self):
        """
        Evaluates the current solution (INA placement and worker routing) using the oracle simulator.
        Updates the best solution if the current one is better.

        Returns:
            float: The makespan (minimum time to complete all jobs) of the current configuration.
        """
        # Step 1: Call the oracle to evaluate the current configuration
        try:
            makespan, job_rates = self.get_makespan(
                self.ina_placement_switches,
                self.worker_agg_id
            )
        except Exception:
            # If evaluation fails (e.g., LP solver error), return a large makespan
            # to discourage this solution
            return float('inf')

        # Step 2: Check if current makespan is better than the best found so far
        if makespan < self.best_makespan:
            # Update best makespan
            self.best_makespan = makespan

            # Update best placement (deep copy to avoid reference issues)
            self.best_placement = self.ina_placement_switches.copy()

            # Update best routing (deep copy of each job's assignment list)
            self.best_routing = [row[:] for row in self.worker_agg_id]

            # Signal that a better solution was accepted
            self.accepted_change = True

        # Return the makespan of the current solution
        return makespan

    # Function 6: _perturb_placement
    # Role: Modify the current INA placement by swapping in/out switches to escape local minima.
    def _perturb_placement(self):
        """
        Perturbs the current INA placement by either swapping out an existing switch
        with a new candidate or adding a new candidate switch, maintaining the budget constraint.

        This method implements a local search perturbation to escape local minima.
        """
        # Step 1: Check if current placement is at budget limit
        if len(self.ina_placement_switches) == self.ina_budget:
            # Step 2: Select a random switch to remove from current placement
            to_remove = random.choice(self.ina_placement_switches)

            # Step 3: Select a random switch from candidates not currently placed
            available_candidates = [
                switch_id for switch_id in self.ina_candidates
                if switch_id not in self.ina_placement_switches
            ]

            # Defensive: If no available candidates, return current placement (no change)
            if not available_candidates:
                return self.ina_placement_switches

            to_add = random.choice(available_candidates)

            # Step 4: Replace the removed switch with the new one
            self.ina_placement_switches.remove(to_remove)
            self.ina_placement_switches.append(to_add)

        else:
            # Step 5: Current placement has space; add a new switch
            # Select a random switch from candidates not in current placement
            available_candidates = [
                switch_id for switch_id in self.ina_candidates
                if switch_id not in self.ina_placement_switches
            ]

            # Defensive: If no candidates available, return current placement
            if not available_candidates:
                return self.ina_placement_switches

            to_add = random.choice(available_candidates)

            # Step 6: Append the new switch to the placement
            self.ina_placement_switches.append(to_add)

        # Return the updated placement
        return self.ina_placement_switches

    # Function 7: _compute_switch_load_benefit_score
    # Role: Compute a composite score for each candidate switch, indicating its potential to reduce global makespan if selected for INA.
    def _compute_switch_load_benefit_score(self):
        """
        Computes a benefit score for each candidate switch based on its potential to reduce global makespan
        if selected for In-Network Aggregation (INA).

        The score is a composite metric that considers:
        1. Bandwidth savings from shorter paths to the aggregation switch.
        2. Available processing capacity at the switch.

        The result is stored in `self.switch_scores` and returned.

        Returns:
            Dict[int, float]: Mapping of switch ID to computed benefit score.
        """
        # Step 1: Initialize the scores dictionary
        self.switch_scores = {}

        # Step 2: Iterate over each candidate switch
        for switch_id in self.ina_candidates:
            score = 0.0

            # Step 3: For each job, evaluate the benefit of placing INA at `switch_id`
            for job_idx in range(self.jobs_num):
                workers = self.workers_id[job_idx]
                ps = self.ps_id[job_idx]
                job_size = self.jobs_size[job_idx]

                # Skip if no workers or invalid job size
                if not workers or job_size <= 0:
                    continue

                # Step 7: Compute average path length from workers to switch_id
                total_path_length_to_s = 0.0
                valid_paths_to_s = 0

                for worker in workers:
                    path_links = self._get_path_links(worker, switch_id)
                    if path_links:
                        total_path_length_to_s += len(path_links)
                        valid_paths_to_s += 1

                # Compute average path length to switch_id
                avg_path_length_to_s = (total_path_length_to_s / valid_paths_to_s) if valid_paths_to_s > 0 else 0.0

                # Step 9: Compute average path length from workers to PS
                total_path_length_to_ps = 0.0
                valid_paths_to_ps = 0

                for worker in workers:
                    path_links = self._get_path_links(worker, ps)
                    if path_links:
                        total_path_length_to_ps += len(path_links)
                        valid_paths_to_ps += 1

                # Compute average path length to PS
                avg_path_length_to_ps = (total_path_length_to_ps / valid_paths_to_ps) if valid_paths_to_ps > 0 else 0.0

                # Step 10: Compute saved bandwidth (approximation)
                saved_bandwidth = (avg_path_length_to_ps - avg_path_length_to_s) * job_size

                # Step 11: Add benefit if saved bandwidth is positive
                if saved_bandwidth > 0:
                    # Normalize by (path_length_to_ps + 1) to prevent division by zero and bias toward shorter paths
                    score += saved_bandwidth / (avg_path_length_to_ps + 1)

                # Step 12: Add benefit based on available switch processing capacity
                # Get current utilization (estimated load)
                current_utilization = self._get_switch_utilization(switch_id)
                # Available capacity
                available_capacity = self.Cs - current_utilization
                # Normalize by Cs
                capacity_bonus = available_capacity / self.Cs if self.Cs > 0 else 0.0
                score += capacity_bonus

            # Step 13: Store the final score for this switch
            self.switch_scores[switch_id] = score

        # Step 14: Return the computed scores
        return self.switch_scores

    # Function 8: _run_local_search
    # Role: Perform iterative refinement of the solution using local perturbations and reassignment.
    def _run_local_search(self):
        """
        Performs iterative local search to refine the INA placement and worker routing
        using a heuristic optimization strategy based on the provided blueprint.
        """
        # Step 1: Initialize the persistent state
        self._init_state()

        # Step 2: Iterate up to max_iterations
        while self.iteration_count < self.max_iterations:
            # Step 3: Compute benefit scores for all candidate switches
            try:
                self._compute_switch_load_benefit_score()
            except Exception:
                # In case of error, proceed with current state (safe fallback)
                pass

            # Step 4: Select top-k switches based on benefit scores
            try:
                self.ina_placement_switches = self._select_top_k_switches()
            except Exception:
                # If selection fails, maintain current placement
                pass

            # Step 5: Assign workers to optimal aggregators (PS or INA switches)
            try:
                self.worker_agg_id = self._assign_workers_to_aggregators()
            except Exception:
                # If assignment fails, keep previous routing
                pass

            # Step 6: Evaluate current solution using the oracle simulator
            current_makespan = 0.0
            try:
                current_makespan = self._evaluate_solution()
            except Exception:
                # If evaluation fails, set a large makespan to discourage this solution
                current_makespan = float('inf')

            # Step 7: Check if no improvement was accepted in this round
            if not self.accepted_change:
                # Step 8: Perturb the current INA placement to escape local minima
                try:
                    self._perturb_placement()
                except Exception:
                    # If perturbation fails, keep current placement
                    pass

                # Step 9: Reset acceptance flag for next iteration
                self.accepted_change = False

            # Step 10: Break condition is implicitly handled by loop condition
            # Step 11: Increment iteration counter
            self.iteration_count += 1

        # Step 12: Return the best found placement and routing
        return self.best_placement, self.best_routing

    # Function 9: solve_with_heuristic
    # Role: Orchestrator: Main driver of the algorithm.
    def solve_with_heuristic(self):
        """
        Orchestrates the heuristic optimization pipeline to find the best INA placement and worker routing.

        This method follows the Architect's blueprint precisely:
        1. Initializes the search state.
        2. Runs a local search to refine placement and routing.
        3. Finalizes the solution by storing the best results in self.solution.
        4. Returns the final solution.

        Returns:
            Dict: Final solution containing INA placement and worker aggregation assignments.
        """
        # Step 1: Initialize the persistent state for the optimization loop
        try:
            self._init_state()
        except Exception:
            # Fallback: Proceed with default state if initialization fails
            self.best_makespan = float('inf')
            self.best_placement = []
            self.best_routing = [[] for _ in range(self.jobs_num)]
            self.iteration_count = 0
            self.max_iterations = 100
            self.accepted_change = False

        # Step 2: Run the local search to optimize INA placement and worker routing
        try:
            best_placement, best_routing = self._run_local_search()
        except Exception:
            # Fallback: Use the current best state if local search fails
            best_placement = self.best_placement
            best_routing = self.best_routing

        # Step 3: Assign the best-found INA placement to the solution
        self.solution = self.solution or {}
        self.solution['ina_placement_switches'] = best_placement.copy() if best_placement is not None else []

        # Step 4: Assign the best-found worker routing to the solution
        self.solution['worker_agg_id'] = [row[:] for row in best_routing] if best_routing is not None else [[] for _ in range(self.jobs_num)]

        # Step 5: Return the finalized solution
        return self.solution
