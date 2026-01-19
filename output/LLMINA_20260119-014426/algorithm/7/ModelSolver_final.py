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
    # Role: Initializes persistent state required for heuristic execution.
    def _init_state(self):
        """
        Initializes the persistent state required for heuristic execution.

        This method sets up the core data structures used throughout the heuristic algorithm.
        It must be called before any optimization or assignment logic is applied.
        """
        # Step 1. Initialize `self.assigned_workers` = [] as a jagged list of length `self.jobs_num`, where each inner list is empty.
        self.assigned_workers = [[] for _ in range(self.jobs_num)]

        # Step 2. Initialize `self.ina_placement` = [] as a list of selected switch IDs.
        self.ina_placement = []

        # Step 3. Initialize `self.workers_to_switch_load` = {} as a dictionary mapping (job_id, switch_id) to total incoming rate (Gbps).
        self.workers_to_switch_load = {}

        # Step 4. Initialize `self.switch_load` = {} as a dictionary mapping switch_id to total aggregate incoming rate from all jobs.
        self.switch_load = {}

        # Step 5. Initialize `self.job_effective_rate` = {} as a dictionary mapping job_id to its effective rate (Gbps).
        self.job_effective_rate = {}

    # Function 2: _compute_worker_criticality_score
    # Role: Scores each worker by its potential to bottleneck job completion, based on volume and path length.
    def _compute_worker_criticality_score(self):
        """
        Computes and stores a criticality score for each worker in each job based on 
        the worker's data volume and the path length to its Parameter Server.

        The criticality score is defined as: volume / path_length (in number of links).
        This score helps identify workers that are likely to bottleneck job completion 
        due to high volume and long paths.
        """
        # Step 1: Initialize the criticality dictionary
        self.worker_criticality = {}

        # Step 2: Iterate over each job
        for j in range(self.jobs_num):
            # Step 2.1: Get the job's total volume
            m_j = self.jobs_size[j]

            # Step 2.2: Get the PS ID for this job
            d_j = self.ps_id[j]

            # Step 2.3: Iterate over each worker in the job
            workers_for_job = self.workers_id[j]
            if not workers_for_job:
                continue  # Skip if no workers assigned to this job

            for w in workers_for_job:
                # Step 2.3.1: Get the shortest path from worker w to PS d_j
                path_links = self._get_path_links(w, d_j)

                # Step 2.3.2: Compute path length (number of links)
                path_length = len(path_links)

                # Step 2.3.3: Avoid division by zero; assign a very high score if path is empty
                if path_length <= 0:
                    criticality = float('inf')  # Prevents zero or undefined; indicates unreachable
                else:
                    criticality = m_j / path_length

                # Step 2.3.4: Store the criticality score
                self.worker_criticality[(j, w)] = criticality

        # Step 3: Return nothing (as required)
        return None

    # Function 3: _assign_workers_to_agg_points
    # Role: Assigns each worker in every job to the optimal aggregation point (INA switch or PS) based on shortest path, ensuring correct initialization and assignment with strict cardinality guarantees.
    def _assign_workers_to_agg_points(self):
        """
        Assigns each worker in every job to the optimal aggregation point (INA switch or PS)
        based on shortest path hop count, ensuring correct initialization and cardinality.
        """
        # Step 1: Initialize self.assigned_workers if not already done
        if not hasattr(self, 'assigned_workers') or not isinstance(self.assigned_workers, list) or len(self.assigned_workers) != self.jobs_num:
            self.assigned_workers = [[] for _ in range(self.jobs_num)]

        # Step 2: For each job j
        for j in range(self.jobs_num):
            workers_for_job = self.workers_id[j]

            # Step 2.2: If no workers, ensure empty assignment and continue
            if not workers_for_job:
                self.assigned_workers[j] = []
                continue

            # Step 2.3: Ensure self.assigned_workers[j] is initialized and empty
            if not isinstance(self.assigned_workers[j], list):
                self.assigned_workers[j] = []
            if len(self.assigned_workers[j]) > 0:
                self.assigned_workers[j].clear()

            # Step 2.4: For each worker w in workers_for_job
            for w in workers_for_job:
                # Step 2.4.1: Default assignment to PS
                best_node = self.ps_id[j]
                best_cost = float('inf')

                # Step 2.4.3: Candidate aggregation nodes: INA switches + PS
                candidate_nodes = self.ina_placement + [self.ps_id[j]]

                for v in candidate_nodes:
                    # Step 2.4.3.1: Skip if v is invalid
                    if v is None or v not in self.G:
                        continue

                    # Step 2.4.3.2: Get path from w to v
                    path_links = self._get_path_links(w, v)
                    if path_links is None:
                        continue  # Skip unreachable or invalid paths

                    # Step 2.4.3.3: Cost = hop count
                    cost = len(path_links)

                    # Step 2.4.3.4: Update best if cost is lower
                    if cost < best_cost:
                        best_cost = cost
                        best_node = v

                # Step 2.4.4: Assign best aggregation point
                self.assigned_workers[j].append(best_node)

            # Step 2.5: Validate assignment count
            if len(self.assigned_workers[j]) != len(workers_for_job):
                raise AssertionError(
                    f'Assignment count mismatch for job {j}: expected {len(workers_for_job)}, got {len(self.assigned_workers[j])}'
                )

    # Function 4: _evaluate_load_and_update_rates
    # Role: Computes the effective rate and load distribution for all switches and jobs using the simulator.
    def _evaluate_load_and_update_rates(self):
        """
        Evaluates the current load distribution and effective rates for all jobs and switches
        using the simulator, then updates the internal state accordingly.

        This method computes the makespan and effective rates based on the current
        assignment of workers to aggregation points and the placement of INA switches.
        """
        # Step 1. Call _init_state to reset the state before evaluation
        self._init_state()

        # Step 2. Check if required data is available
        if not self.assigned_workers or not self.ina_placement:
            return None

        # Step 3. Call the simulator to compute makespan and job rates
        try:
            makespan, job_rates = self.get_makespan(self.ina_placement, self.assigned_workers)
        except Exception:
            # If simulator fails, return None to indicate failure
            return None

        # Step 4. Store the job effective rates
        self.job_effective_rate = job_rates

        # Step 5. Update switch load and per-worker-to-switch load
        for j in range(self.jobs_num):
            # Skip if job has no workers or rate is not available
            if j not in job_rates or not self.workers_id[j]:
                continue

            rate_j = job_rates[j]

            # Iterate over each worker in the job
            for w_idx, w in enumerate(self.workers_id[j]):
                # Get the assigned aggregation point for this worker
                v = self.assigned_workers[j][w_idx] if len(self.assigned_workers[j]) > w_idx else None

                # Only update load if the worker is not assigned to its PS
                if v is not None and v != self.ps_id[j]:
                    # Update per-switch load from job j
                    key = (j, v)
                    self.workers_to_switch_load[key] = self.workers_to_switch_load.get(key, 0.0) + rate_j

                    # Update total load at switch v
                    self.switch_load[v] = self.switch_load.get(v, 0.0) + rate_j

        # Step 6. Return the makespan
        return makespan

    # Function 5: _compute_switch_criticality_score
    # Role: Computes a priority score for each candidate switch based on traffic concentration and job proximity.
    def _compute_switch_criticality_score(self):
        """
        Computes a criticality score for each candidate switch based on the aggregate
        worker criticality weighted by the distance from the switch to its job's PS.

        The score reflects the strategic importance of a switch for aggregation:
        - Higher scores indicate switches that are both close to PSs and handle high-criticality workers.
        """
        # Step 1. Initialize self.switch_criticality as an empty dictionary
        self.switch_criticality = {}

        # Step 2. Ensure worker_criticality is initialized before use
        if not hasattr(self, 'worker_criticality') or self.worker_criticality is None:
            self.worker_criticality = {}

        # Step 2.1. Iterate over each candidate switch in self.ina_candidates
        for s in self.ina_candidates:
            total_score = 0.0

            # Step 2.1. Iterate over each job j
            for j in range(self.jobs_num):
                # Step 2.1.1. Get the PS ID for job j
                d_j = self.ps_id[j]

                # Step 2.1.2. Get the shortest path from switch s to PS d_j
                path_to_ps = self._get_path_links(s, d_j)

                # Step 2.1.3. Skip if path is empty or unreachable
                if not path_to_ps:
                    continue

                # Step 2.1.4. Compute the hop distance to PS
                distance_to_ps = len(path_to_ps)

                # Step 2.1.5. Iterate over each worker w in job j
                workers_for_job = self.workers_id[j]
                for w in workers_for_job:
                    # Step 2.1.5.1. Retrieve the worker's criticality score
                    c = self.worker_criticality.get((j, w), 0.0)

                    # Step 2.1.5.2. Add normalized criticality to total score
                    # Avoid division by zero; ensure distance_to_ps > 0
                    if distance_to_ps > 0:
                        total_score += c / distance_to_ps

            # Step 2.2. Store the computed score for switch s
            self.switch_criticality[s] = total_score

        # Step 3. Return nothing (as per contract)
        return None

    # Function 6: _local_improvement_swap
    # Role: Refines the solution via worker swaps to reduce switch load and improve makespan.
    def _local_improvement_swap(self):
        """
        Performs a local improvement search by attempting to swap each worker's assignment
        to a different aggregation point (INA switch or PS) to reduce makespan.
        """
        # Step 1: Call _evaluate_load_and_update_rates and store the initial makespan
        best_makespan = self._evaluate_load_and_update_rates()
        if best_makespan is None:
            return  # Early return if evaluation fails

        # Step 2: Iterate over each job j
        for j in range(self.jobs_num):
            # Skip if job has no workers
            if not self.workers_num[j]:
                continue

            # Step 2.1: Iterate over each worker w in job j
            for w_idx in range(self.workers_num[j]):
                # Step 2.1.1: Get current aggregation point
                current_agg = self.assigned_workers[j][w_idx] if len(self.assigned_workers[j]) > w_idx else None

                # Step 2.1.2: If current_agg is the PS, skip (no benefit from swapping to PS)
                if current_agg is not None and current_agg == self.ps_id[j]:
                    continue

                # Step 2.1.3: Consider alternative aggregation points
                candidates = self.ina_placement + [self.ps_id[j]]

                # Step 2.1.3.1: Temporary copy of assignments to simulate swaps
                original_assignment = current_agg

                for alternative_agg in candidates:
                    if alternative_agg == current_agg:
                        continue  # Skip same assignment

                    # Step 2.1.3.2: Simulate swap by temporarily updating assignment
                    # Only update the current worker's assignment in a temporary state
                    self.assigned_workers[j][w_idx] = alternative_agg

                    # Step 2.1.3.3: Re-evaluate the configuration with the new assignment
                    try:
                        new_makespan = self._evaluate_load_and_update_rates()
                    except Exception:
                        # Revert assignment and continue if evaluation fails
                        self.assigned_workers[j][w_idx] = original_assignment
                        continue

                    # Step 2.1.3.4: If better makespan found, update global best and apply change permanently
                    if new_makespan is not None and new_makespan < best_makespan:
                        best_makespan = new_makespan

                        # Permanently apply the swap
                        # (already applied in self.assigned_workers, so no further action needed)
                        # But we still need to revert if we want to continue testing other swaps
                        # So we only update best_makespan and proceed — the assignment remains changed
                        # until reverted later
                    else:
                        # Revert the assignment to simulate next candidate
                        self.assigned_workers[j][w_idx] = original_assignment

                # End of candidate loop: assignment has been reverted to original

        # Step 3: Return nothing (as per contract)
        return None

    # Function 7: _greedy_ina_placement
    # Role: Selects the top B_INA switches with highest criticality scores for INA deployment.
    def _greedy_ina_placement(self):
        """
        Selects the top INA-enabled switches for deployment based on their criticality scores
        using a greedy heuristic.

        This method follows the Architect's blueprint:
        1. Computes switch criticality scores.
        2. Sorts candidate switches by criticality in descending order.
        3. Selects up to ina_budget switches with the highest scores.
        4. Updates self.ina_placement with the selected switches.
        """
        # Step 1: Compute criticality scores for all candidate switches
        self._compute_switch_criticality_score()

        # Step 2: Sort candidate switches by criticality in descending order
        # Safely handle case where no switches are in candidates or scores are missing
        sorted_candidates = sorted(
            self.ina_candidates,
            key=lambda s: self.switch_criticality.get(s, 0.0),
            reverse=True
        )

        # Step 3: Initialize list to hold selected switches
        selected_switches = []

        # Step 4: Greedily select switches up to the budget limit
        for switch in sorted_candidates:
            if len(selected_switches) >= self.ina_budget:
                break
            selected_switches.append(switch)

        # Step 5: Assign the selected switches to the final placement
        self.ina_placement = selected_switches

        # Step 6: Return nothing (per contract)
        return None

    # Function 8: solve_with_heuristic
    # Role: Orchestrator: Main driver of the algorithm.
    def solve_with_heuristic(self):
        """
        Orchestrates the heuristic algorithm to solve the LLMINA problem.

        This method follows the Architect's blueprint precisely:
        1. Initializes the persistent state.
        2. Computes worker criticality scores.
        3. Selects INA switches using a greedy heuristic.
        4. Assigns workers to aggregation points.
        5. Applies local improvement via swap moves.
        6. Constructs and returns the final solution.

        Returns:
            Dict[str, Any]: Final solution with 'ina_placement' and 'worker_agg_id'.
        """
        # Step 1. Call _init_state to initialize persistent state once.
        self._init_state()

        # Step 2. Call _compute_worker_criticality_score to compute worker criticality scores.
        self._compute_worker_criticality_score()

        # Step 3. Call _greedy_ina_placement to select INA switches based on criticality.
        self._greedy_ina_placement()

        # Step 4. Call _assign_workers_to_agg_points to assign workers to aggregation points.
        self._assign_workers_to_agg_points()

        # Step 5. Call _local_improvement_swap to refine assignments locally.
        self._local_improvement_swap()

        # Step 6. Construct self.solution with ina_placement and assigned workers.
        # Ensure worker_agg_id is a List[List[int]] where inner list matches order of self.workers_id[j].
        self.solution = {
            "ina_placement": self.ina_placement[:],  # Safe copy
            "worker_agg_id": [
                self.assigned_workers[j][:] if j < len(self.assigned_workers) else []
                for j in range(self.jobs_num)
            ]
        }

        # Step 7. Return self.solution as the final output.
        return self.solution
