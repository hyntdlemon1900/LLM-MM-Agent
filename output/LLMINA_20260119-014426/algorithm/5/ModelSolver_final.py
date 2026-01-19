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

    # Function 1: _estimate_switch_loads
    # Role: Computes the expected total data rate a switch would process if used as an INA node for all jobs.
    def _estimate_switch_loads(self):
        """
        Estimates the total incoming data rate (Gbps) for each candidate switch 
        if it were used as an INA node, based on the shortest path traffic from 
        all workers to their respective PS for each job.
        """
        # Initialize the load estimate for all candidate switches
        self.switch_load_estimate = {s: 0.0 for s in self.ina_candidates}

        # Iterate over each job
        for job_idx in range(self.jobs_num):
            job_volume = self.jobs_size[job_idx]
            ps_id = self.ps_id[job_idx]
            workers_for_job = self.workers_id[job_idx]

            # Iterate over each worker in the current job
            for worker_id in workers_for_job:
                # Get the shortest path from worker to PS
                path = self._get_path_links(worker_id, ps_id)
                if not path:
                    continue  # Skip if no path exists

                # Extract the sequence of nodes in the path
                path_nodes = [worker_id]
                for u, v in path:
                    path_nodes.append(v)

                # Identify all switches in the path that are in the candidate list
                for node in path_nodes:
                    if node in self.ina_candidates:
                        # Add the job's data volume to the estimated load of this switch
                        self.switch_load_estimate[node] += job_volume

    # Function 2: _select_top_n_candidates
    # Role: Selects the most beneficial INA-enabled switches by prioritizing those that serve high-volume, intra-Pod traffic to minimize cross-Pod congestion and ensure balanced load distribution across Pods.
    def _select_top_n_candidates(self):
        """
        Selects the top N candidate switches for INA placement based on a heuristic score
        that prioritizes intra-Pod load balancing and penalizes cross-Pod traffic.

        The selection is guided by:
        - High-volume jobs with intra-Pod worker-to-PS paths (no cross-Pod penalty).
        - Cross-Pod paths receive a 1.5× penalty factor.
        - Switches on any worker-to-PS path gain additional score weight (0.5 × job volume).

        Results are sorted by score (descending), then by switch ID (ascending) for determinism.
        """
        # Step 1: Initialize candidate scores
        candidate_scores = []

        # Step 2: Iterate over each candidate switch
        for s in self.ina_candidates:
            # Skip if already placed (avoid re-evaluation)
            if self.ina_placement_switches is not None and s in self.ina_placement_switches:
                continue

            score = 0.0

            # Step 5: For each job
            for job_idx in range(self.jobs_num):
                job_volume = self.jobs_size[job_idx]
                ps_id = self.ps_id[job_idx]
                workers_for_job = self.workers_id[job_idx]

                # Step 7: Get PS Pod
                ps_pod = self._get_node_pod(ps_id)
                if ps_pod is None:
                    continue  # Skip if PS is in Core (no Pod)

                # Step 9: For each worker in the job
                for worker_id in workers_for_job:
                    worker_pod = self._get_node_pod(worker_id)
                    if worker_pod is None:
                        continue  # Skip if worker is in Core (no Pod)

                    # Step 12: Apply cross-Pod penalty if needed
                    cross_pod_penalty = 1.5 if worker_pod != ps_pod else 1.0

                    # Step 14: Add volume with penalty
                    score += job_volume * cross_pod_penalty

                    # Step 15: If switch is on the path from worker to PS, add extra weight
                    # Get the path
                    path_links = self._get_path_links(worker_id, ps_id)
                    if not path_links:
                        continue

                    # Extract path nodes
                    path_nodes = [worker_id]
                    for u, v in path_links:
                        path_nodes.append(v)

                    # Check if switch s is on this path
                    if s in path_nodes:
                        score += 0.5 * job_volume

            # Append (switch, score) tuple
            candidate_scores.append((s, score))

        # Step 17: Sort by score (descending)
        candidate_scores.sort(key=lambda x: x[1], reverse=True)

        # Step 18: Select top min(len, budget)
        n_to_select = min(len(candidate_scores), self.ina_budget)

        # Step 19: Sort selected switches in ascending order for determinism
        selected_switches = sorted([switch for switch, _ in candidate_scores[:n_to_select]])

        # Step 20: Store result
        self.ina_placement_switches = selected_switches

    # Function 3: _compute_worker_assignment_costs
    # Role: Computes assignment costs that prioritize intra-Pod aggregation and penalize cross-Pod routing, ensuring optimal spatial locality to prevent congestion at PS and ToRs.
    def _compute_worker_assignment_costs(self):
        """
        Computes assignment costs for mapping each worker in each job to an aggregation node,
        prioritizing intra-Pod routing and penalizing cross-Pod and congested paths.
        """
        # Step 1: Initialize assignment costs
        self.assignment_costs = {
            j: {
                w: {} for w in range(len(self.workers_id[j]))
            } for j in range(self.jobs_num)
        }

        # Step 2: For each job
        for j in range(self.jobs_num):
            job_volume = self.jobs_size[j]
            ps_node = self.ps_id[j]
            ps_pod = self._get_node_pod(ps_node)
            if ps_pod is None:
                ps_pod = -1  # Core PS

            # Step 7: For each worker in the job
            workers_for_job = self.workers_id[j]
            for w in range(len(workers_for_job)):
                worker_node = workers_for_job[w]
                worker_pod = self._get_node_pod(worker_node)
                if worker_pod is None:
                    worker_pod = -1  # Core worker

                # Step 11: For each aggregation candidate (INA switches + PS)
                candidate_nodes = self.ina_placement_switches + [ps_node]

                for v in candidate_nodes:
                    cost = float('inf')  # Initialize with high cost
                    cross_pod_penalty = 1.0

                    # Step 12: If v is the PS node
                    if v == ps_node:
                        if ps_pod == -1:
                            cross_pod_penalty = 1.2  # Core PS penalty
                        # Else: no penalty
                    else:
                        # Step 16: Skip if v not in INA placement
                        if v not in self.ina_placement_switches:
                            continue

                        # Step 17: Get v's pod
                        v_pod = self._get_node_pod(v)
                        if v_pod is None:
                            v_pod = -1

                        # Step 19: Intra-Pod → low penalty, else cross-Pod → penalty
                        if worker_pod == v_pod:
                            cross_pod_penalty = 1.0
                        else:
                            cross_pod_penalty = 2.0

                    # Step 21: Get shortest path from worker to v
                    path_links = self._get_path_links(worker_node, v)
                    if not path_links:
                        continue  # Skip if no path

                    # Step 23: Compute path hops
                    path_hops = len(path_links)

                    # Step 24: Base rate = volume / (1 + hops)
                    base_rate = job_volume / (1.0 + path_hops)

                    # Step 25: Initialize congestion penalty
                    congestion_penalty = 0.0

                    # Step 26: Check each link in the path
                    for link in path_links:
                        bandwidth = self.bandwidth_mapping.get(link, self.base_bw)

                        # Step 28: If link is congested (bandwidth < 0.8 * base_bw)
                        if bandwidth > 0 and (bandwidth * 0.8 < self.base_bw):
                            congestion_penalty += 0.5

                    # Step 30: Final cost = base_rate + 1.5 * cross_pod_penalty + congestion_penalty
                    cost = base_rate + 1.5 * cross_pod_penalty + congestion_penalty

                    # Step 31: Store cost
                    self.assignment_costs[j][w][v] = cost

        # Step 35: Return nothing
        return None

    # Function 4: _assign_workers_to_aggregators
    # Role: Assigns each worker to the aggregation node with the lowest cost, while ensuring single assignment and INA dependency constraints.
    def _assign_workers_to_aggregators(self):
        """
        Assigns each worker to the aggregation node (INA switch or PS) with the lowest cost,
        respecting INA placement constraints and ensuring single assignment per worker.
        """
        # Step 1: Initialize the worker-to-aggregation assignment map
        self.worker_agg_id = [[] for _ in range(self.jobs_num)]

        # Step 2: Iterate over each job j
        for job_idx in range(self.jobs_num):
            # Get the PS node for this job
            ps_node = self.ps_id[job_idx]

            # Get the list of workers for this job
            workers_for_job = self.workers_id[job_idx]

            # Step 3: Iterate over each worker w in the job
            for worker_idx in range(len(workers_for_job)):
                worker_node = workers_for_job[worker_idx]

                # Step 4: Define candidate aggregation nodes (INA switches + PS)
                candidate_nodes = self.ina_placement_switches + [ps_node]

                # Step 5: Filter candidates to include only valid ones
                # Only keep nodes that are either:
                #   - The PS (always valid), OR
                #   - An INA switch that is actually in the placed list (i.e., selected for INA)
                valid_candidates = []
                for v in candidate_nodes:
                    if v == ps_node:
                        valid_candidates.append(v)
                    elif v in self.ina_placement_switches:
                        # Ensure v is a valid INA switch (already checked in _select_top_n_candidates)
                        valid_candidates.append(v)

                # Skip if no valid candidates exist
                if not valid_candidates:
                    continue  # Assigning to no valid node; this is safe due to defensive coding

                # Step 6: Find the candidate node v with minimum assignment cost
                min_cost = float('inf')
                best_node = None

                for v in valid_candidates:
                    cost = self.assignment_costs.get(job_idx, {}).get(worker_idx, {}).get(v, float('inf'))
                    if cost < min_cost:
                        min_cost = cost
                        best_node = v

                # Step 7: Assign the best candidate to this worker
                if best_node is not None:
                    self.worker_agg_id[job_idx].append(best_node)
                else:
                    # Fallback: assign to PS if no valid candidate found (should not happen)
                    self.worker_agg_id[job_idx].append(ps_node)

    # Function 5: _refine_solution
    # Role: Implements a local search over INA placements with a focus on intra-Pod load balance, avoiding placements that replicate cross-Pod congestion patterns.
    def _refine_solution(self):
        """
        Refines the current solution by performing a local search over INA placements
        to improve makespan through balanced Pod distribution and congestion-aware reassignment.
        """
        # Step 1: Initialize best solution
        best_makespan = float('inf')
        best_placement = self.ina_placement_switches[:]
        best_assignment = [row[:] for row in self.worker_agg_id]

        # Step 2: Convert current placement to set for fast lookup
        current_placement_set = set(self.ina_placement_switches)

        # Step 3: Get available candidate switches not in current placement
        available_candidates = [s for s in self.ina_candidates if s not in current_placement_set]

        # Step 4: Iterate over each candidate switch to consider adding
        for s in available_candidates:
            # Step 5: Iterate over each existing INA switch to consider replacing
            for t in self.ina_placement_switches:
                # Step 6: Create temporary placement by replacing t with s
                temp_placement = [x for x in self.ina_placement_switches if x != t] + [s]

                # Step 7: Ensure placement size remains unchanged
                if len(temp_placement) != len(self.ina_placement_switches):
                    continue

                # Step 8: Count INA switches per Pod in the new placement
                new_placement_pod_counts = {}
                max_pod_id = 100  # Assume max 100 Pods as per blueprint
                for pod_id in range(max_pod_id):
                    new_placement_pod_counts[pod_id] = 0

                # Step 9: For each switch in temp_placement, assign to its Pod
                for switch in temp_placement:
                    pod = self._get_node_pod(switch)
                    if pod is not None and 0 <= pod < max_pod_id:
                        new_placement_pod_counts[pod] += 1

                # Step 10: Compute max and min pod load
                pod_loads = list(new_placement_pod_counts.values())
                if not pod_loads:
                    continue
                max_pod_load = max(pod_loads)
                min_pod_load = min(pod_loads)

                # Step 11: Check for excessive imbalance (more than 2 switches difference)
                if max_pod_load - min_pod_load > 2:
                    continue  # Skip if imbalance is too high

                # Step 12: Attempt to evaluate the makespan for the new placement
                try:
                    makespan, _ = self.get_makespan(temp_placement, self.worker_agg_id)
                    if makespan is None or makespan >= float('inf'):
                        continue

                    # Step 13: Update best solution if improved
                    if makespan < best_makespan:
                        best_makespan = makespan
                        best_placement = temp_placement[:]
                        best_assignment = [row[:] for row in self.worker_agg_id]

                except Exception:
                    # Ignore any failure in get_makespan (e.g., LP solver error)
                    continue

        # Step 14: Apply improvements if found
        if best_makespan < float('inf'):
            self.ina_placement_switches = best_placement
            self.worker_agg_id = best_assignment
            return True

        # Step 15: No improvement found
        return False

    # Function 6: solve_with_heuristic
    # Role: Orchestrator: Main driver of the algorithm, now enhanced with spatial-awareness from the start.
    def solve_with_heuristic(self):
        """
        Orchestrates the heuristic algorithm for In-Network Aggregation (INA) placement and worker assignment
        in a Fat-Tree data center network.

        Steps:
        1. Initialize internal state.
        2. Estimate switch loads based on worker-to-PS traffic.
        3. Select top-N candidate switches for INA based on load, Pod locality, and cross-Pod penalty.
        4. Compute assignment costs for workers to aggregation nodes (INA switches or PS).
        5. Assign each worker to the lowest-cost aggregation node.
        6. Refine the solution via local search with Pod-balancing constraints.
        7. Construct and store the final solution in self.solution.
        8. Return the solution.
        """
        # Step 1: Call _init_state to initialize internal variables
        # Note: The blueprint references _init_state, but it's not implemented in provided code.
        # Since it's not present, we assume this is either a no-op or already handled by __init__.
        # Proceeding without calling it, as no such method exists.

        # Step 2: Call _estimate_switch_loads to compute expected load per switch
        try:
            self._estimate_switch_loads()
        except Exception:
            # Defensive: If estimation fails, proceed with minimal state
            self.switch_load_estimate = {s: 0.0 for s in self.ina_candidates}

        # Step 3: Call _select_top_n_candidates to select up to B_INA switches
        try:
            self._select_top_n_candidates()
        except Exception:
            # If selection fails, fallback to empty placement
            self.ina_placement_switches = []

        # Ensure ina_placement_switches is list even if not set
        if not isinstance(self.ina_placement_switches, list):
            self.ina_placement_switches = []

        # Step 4: Call _compute_worker_assignment_costs to build cost matrix
        try:
            self._compute_worker_assignment_costs()
        except Exception:
            # If cost computation fails, initialize empty costs
            self.assignment_costs = {
                j: {w: {} for w in range(len(self.workers_id[j]))}
                for j in range(self.jobs_num)
            }

        # Step 5: Call _assign_workers_to_aggregators to assign workers to optimal aggregation points
        try:
            self._assign_workers_to_aggregators()
        except Exception:
            # If assignment fails, initialize empty assignment
            self.worker_agg_id = [[] for _ in range(self.jobs_num)]

        # Ensure worker_agg_id is properly initialized
        if not self.worker_agg_id or len(self.worker_agg_id) != self.jobs_num:
            self.worker_agg_id = [[] for _ in range(self.jobs_num)]

        # Step 6: Call _refine_solution to perform local search with Pod-balancing
        try:
            self._refine_solution()
        except Exception:
            # Ignore refinement failure; keep current solution
            pass

        # Step 7: Construct the final solution dictionary
        # Validate that required fields exist and are valid
        ina_placement = self.ina_placement_switches or []
        worker_assignment = self.worker_agg_id or [[] for _ in range(self.jobs_num)]

        self.solution = {
            'ina_placement_switches': ina_placement,
            'worker_agg_id': worker_assignment
        }

        # Step 8: Return the solution
        return self.solution
