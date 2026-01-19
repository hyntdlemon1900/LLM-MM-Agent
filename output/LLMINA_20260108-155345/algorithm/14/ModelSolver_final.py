from __future__ import annotations
from typing import List, Dict, Tuple, Any, Optional
import copy
import networkx as nx
from MMBench.problem.LLMINA.runtime.TemplateSolver import TemplateSolver

class ModelSolver(TemplateSolver):
    """
    MILP Solver for the LLMINA (In-Network Aggregation) Problem.
    
    This class prepares the data context for the optimization model. 
    It explicitly unpacks the complex 'instance' and 'network' objects into 
    flat member variables to ensure the downstream LLM understands the physical 
    topology and logical job requirements.
    """

    def __init__(
        self,
        instance: Dict[str, Any],
        network: Any,
        ina_budget: int,
        jobs_num: int,
        Cs: float = 750.0,
        Ps: float = 200.0,
        topo_name: str = "FatTree",
    ):
        """
        Initializes the solver.
        
        Args:
            instance (dict): A dictionary containing Distributed ML Job specifications.
            network (object): A complex object representing the Datacenter Network Topology.
            ina_budget(int): Resource Budget - Max number of switches allowed to enable INA.
            jobs_num (int): Number of concurrent training jobs.
            Cs (float): Switching Capacity (throughput limit) for INA processing.
            Ps (float): Physical bandwidth limit of the Parameter Server's NIC (last hop).
            topo_name (str): Name of the topology (e.g., "FatTree").
        """
        # Store raw inputs
        problem_data = {
            "instance": instance,
            "network": network,
            "ina_budget": ina_budget,
            "jobs_num": jobs_num,
            "Cs": Cs,
            "Ps": Ps,
            "topo_name": topo_name,
        }
        super().__init__(problem_data)
        
        self.solution: Optional[Dict[str, Any]] = None
        
        # Trigger explicit data unpacking
        self._preprocess_data()

    def _preprocess_data(self) -> None:
        """
        DATA PREPROCESSING & SCHEMA DEFINITION
        
        This method unpacks the 'instance' dictionary and 'network' object into explicit 
        class member variables. It also documents the physical structure of the network 
        to ensure the optimization model respects the underlying topology.
        """
        
        # =========================================================================
        # [Context] Network Topology & Physical Constraints (Fat-Tree)
        # =========================================================================
        # The cluster uses a standard 3-tier Fat-Tree topology organized into "Pods".
        # Understanding the bandwidth hierarchy and physical limits is CRITICAL.
        #
        # 1. Physical Hierarchy (Three Layers):
        #    - Edge Layer (ToR Switches): The bottom layer. This is the ONLY layer
        #      where endpoints connect. **BOTH Workers and Parameter Servers (PS)
        #      are physically attached to these switches.**
        #    - Aggregation Layer (Aggr Switches): Middle layer. Connects multiple
        #      ToR switches to form a "Pod".
        #    - Core Layer (Core Switches): Top layer. Interconnects different Pods.
        #
        # 2. Critical Bandwidth Bottleneck (2:1 Oversubscription):
        #    The network is designed with a specific "Oversubscription Ratio" to
        #    reflect realistic Data Center constraints:
        #    - Tapering Rule: At both Edge and Aggregation layers, the total bandwidth
        #      of Downlink ports (facing servers) is approximately TWICE the total
        #      bandwidth of Uplink ports (facing the core).
        #    - Consequence: This creates a **2:1 Bottleneck** for any traffic moving
        #      upwards. Traffic leaving a Pod (Cross-Pod) fights for half the
        #      bandwidth available to traffic staying inside a Pod (Intra-Pod).
        #
        # 3. Server Access Links (The "Last Mile"):
        #    - Worker Connection: Workers connect to ToR switches with standard
        #      baseline bandwidth.
        #    - PS Connection: Parameter Servers also connect to ToR switches, but
        #      are provisioned with **DOUBLE (2x) the bandwidth** of a standard
        #      worker link.
        #    - Bottleneck Warning: Despite the 2x capacity, the PS link is a strict
        #      "Many-to-One" bottleneck (Incast) because all workers of a job send
        #      data to this single link simultaneously.
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
        self.Ps: float = float(self.problem_data["Ps"])     # PS Link Bandwidth (Gbps)
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
        # [Meaning] Capacity of directed physical links.
        #           Key: (u, v), Value: Capacity in Gbps.
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
        #         For Fat-Tree, usually ToR, Aggr, and Core are all candidates.
        self.ina_candidates: List[int] = self._compute_ina_candidates()
        self.ina_candidates_num: int = len(self.ina_candidates)

        # [Variable] self.sd_id: List[int]
        # [Meaning] The "Gateway Switch" for each Job's PS.
        #           self.sd_id[j] is the Switch Node ID directly connected to ps_id[j].
        # [Physical Constraint] Since this is a Fat-Tree, self.sd_id[j] IS ALWAYS A ToR SWITCH.
        #           This link (sd_id[j] <-> ps_id[j]) is the specific "PS Bottleneck".
        self.sd_id: List[int] = []
        for j in range(self.jobs_num):
            ps_node = self.ps_id[j]
            neighbors = list(self.G.neighbors(ps_node))
            if not neighbors:
                raise ValueError(f"PS node {ps_node} is isolated!")
            self.sd_id.append(neighbors[0]) # The unique ToR switch for this PS

        # -------------------------------------------------------------------------
        # 5. Final Capacity Adjustments
        # -------------------------------------------------------------------------
        # Enforce the 'Ps' parameter on the last-hop link.
        # The link between the PS and its ToR switch often has limited bandwidth (NIC limit).
        for j in range(self.jobs_num):
            s_node = self.sd_id[j]
            p_node = self.ps_id[j]
            self.bandwidth_mapping[(s_node, p_node)] = self.Ps
            self.bandwidth_mapping[(p_node, s_node)] = self.Ps

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

    def _get_flows_on_link(
        self,
        edge: Tuple[int, int],
        src_set: List[int],
        dst_set: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Identifies which flows (src_idx -> dst_idx) traverse the given edge.
        Used to construct Link Bandwidth Constraints.
        
        Returns:
            List of (src_index, dst_index) tuples.
        """
        flows = []
        edge_rev = (edge[1], edge[0])
        
        for s_idx, s_node in enumerate(src_set):
            for d_idx, d_node in enumerate(dst_set):
                path_edges = self._get_path_links(s_node, d_node)
                # Check if the edge (or its reverse) is part of the flow's path
                if edge in path_edges or edge_rev in path_edges:
                    flows.append((s_idx, d_idx))
        return flows

    """
    def solve_with_heuristic(self) -> Optional[Dict[str, Any]]:
        '''
        to be implemented below: solve the MILP model with heuristic methods
        '''         
        self.solution = {
            # List of integer Node IDs of selected switches. 
            # E.g., [101, 102, 205]
            "ina_placement_switches": [], 
            
            # Nested Dictionary: Mapping [job_index][worker_index] -> Aggregation Node ID
            # Key 1: Job index 'j' (int, 0 to jobs_num-1)
            # Key 2: Worker index 'w' (int, 0 to workers_num[j]-1)
            # Value: Node ID (int) of the chosen switch OR the PS node.
            "worker_agg_id": {} 
        }
        return self.solution
    """

    def get_makespan(self, ina_placement_switches, worker_agg_id) -> Optional[float]:
        """
        Calculates the theoretically optimal makespan given a fixed INA deployment and routing configuration.

        Methodology:
            With the discrete decisions (INA placement `x` and worker assignments `y`) fixed, the original Mixed-Integer 
            problem reduces to a continuous Linear Programming (LP) problem (specifically, a convex resource allocation problem).
            
            This function calculates the optimal gradient transmission rates by solving this continuous sub-problem, 
            often utilizing a "Water-Filling" algorithm or standard LP solver. It identifies the limiting bottleneck 
            (whether it be Link Bandwidth or Switch Processing Capacity) to determine the maximum feasible global 
            pacing rate `alpha`, where the Makespan t = 1/alpha.

        Args:
            ina_placement_switches (List[int]): The specific set of switches enabled with In-Network Aggregation.
            worker_agg_id (Dict[int, Dict[int, int]]): The routing map defining where each worker sends its gradients 
                                                       (Mapping: [job_idx][worker_idx] -> Aggregation Node ID).

        Returns:
            Optional[float]: The minimum achievable makespan (time to complete one iteration) under these settings. 
                             Returns None if the configuration is infeasible.
        """
        return super().get_makespan(ina_placement_switches, worker_agg_id)

    # ============ Heuristic Functions ============

    # Function 1: _decide_ina_placement_candidates
    # Role: Selects the most strategically impactful switches for INA deployment by evaluating their potential to alleviate aggregation bottlenecks across multiple jobs, based on topological centrality and job connectivity.
    def _decide_ina_placement_candidates(self):
        """Selects the most strategically impactful switches for INA deployment based on topological centrality, job connectivity, and load-balancing potential."""
        # Step 1: Ensure candidate switches are computed
        self._compute_ina_candidates()

        # Step 2: Initialize scoring dictionary
        switch_scores = {}

        # Step 3: For each candidate switch, compute a composite score
        for s in self.ina_candidates:
            score = 0.0

            # Track total traffic volume that could be aggregated at this switch
            total_agg_volume = 0.0

            # Track number of jobs whose worker-to-PS path includes this switch
            jobs_with_path = 0

            # Track whether switch is in the same Pod as any PS
            same_pod_jobs = 0

            # Iterate over all jobs
            for j in range(self.jobs_num):
                job_volume = self.jobs_size[j]
                ps_node = self.ps_id[j]
                ps_tor = self.sd_id[j]

                # Determine the Pod of the PS (ToR switch)
                ps_pod = ps_tor

                # Check if the switch is in the same Pod as the PS
                is_same_pod = (s in self.tors_id and s == ps_tor) or (s in self.aggrs_id and s // 100 == ps_tor // 100)  # Assuming Pod ID is derived from ToR ID

                # Get the shortest path from any worker in job j to the PS
                worker_path_exists = False
                for w in self.workers_id[j]:
                    path = self.allPathDict.get(w, {}).get(ps_node, [])
                    if path:
                        # Check if the switch s is on this path
                        if s in path:
                            worker_path_exists = True
                            # Increase job count for this path
                            jobs_with_path += 1
                            if is_same_pod:
                                same_pod_jobs += 1
                            break

                # If the switch is on at least one worker-to-PS path for this job, it can aggregate
                if worker_path_exists:
                    # Add job volume to total agg volume
                    total_agg_volume += job_volume

                    # Bonus for being in same Pod as PS (encourages local aggregation)
                    if is_same_pod:
                        score += job_volume * 1.5

                    # Penalty for cross-Pod traffic: penalize if the path goes through core layer
                    # Check if any path from worker to PS passes through a core switch
                    path_to_ps = self.allPathDict.get(self.workers_id[j][0], {}).get(ps_node, [])
                    has_core_hop = False
                    for i in range(len(path_to_ps) - 1):
                        u, v = path_to_ps[i], path_to_ps[i+1]
                        if v in self.cores_id:
                            has_core_hop = True
                            break

                    # Reward for reducing cross-Pod traffic (i.e., aggregating before core)
                    if has_core_hop and is_same_pod:
                        score += job_volume * 0.8  # Strong incentive for intra-Pod aggregation

            # Apply load-balancing penalty if switch is in a pod with high fragmentation
            # This is a proxy: if the switch is in a pod with many jobs and fragmented workers, it's more critical
            # But we use a simple heuristic: if it serves multiple jobs and has high total volume, it's high-value
            if jobs_with_path > 0:
                # Normalize score by total volume
                score += total_agg_volume * 0.5  # Weight by aggregate offload potential

                # Penalize if the switch is not on any high-volume path
                if total_agg_volume < 0.1:  # Very low potential
                    score = 0
            else:
                score = 0  # Not on any valid path

            switch_scores[s] = score

        # Step 4: Sort switches by score in descending order
        sorted_switches = sorted(switch_scores.keys(), key=lambda s: switch_scores[s], reverse=True)

        # Step 5: Apply budget constraint and enforce load-balancing
        selected_switches = []
        used_pods = set()

        # First, select top-ranked switches up to budget
        for s in sorted_switches:
            if len(selected_switches) >= self.ina_budget:
                break
            # Determine the Pod of the switch
            pod_id = None
            if s in self.tors_id:
                pod_id = s
            elif s in self.aggrs_id:
                pod_id = s // 100  # Assuming Aggr ID format like 172 -> Pod 17
            elif s in self.cores_id:
                pod_id = s // 100  # Core switches also associated with Pods

            # If this Pod is already covered by an INA switch, skip unless we need diversity
            if pod_id in used_pods:
                # Allow up to 2 switches per Pod to avoid hotspot
                pod_count = sum(1 for sw in selected_switches if sw in self.tors_id and sw == pod_id) + \
                            sum(1 for sw in selected_switches if sw in self.aggrs_id and sw // 100 == pod_id) + \
                            sum(1 for sw in selected_switches if sw in self.cores_id and sw // 100 == pod_id)
                if pod_count >= 2:
                    continue  # Avoid over-concentration

            selected_switches.append(s)
            if pod_id is not None:
                used_pods.add(pod_id)

        # Step 6: Ensure at least one INA switch per Pod with fragmented jobs
        # Identify Pods with jobs that have workers in multiple Pods
        fragmented_pods = set()
        for j in range(self.jobs_num):
            worker_pods = set()
            for w in self.workers_id[j]:
                # Find the ToR of the worker
                worker_tor = None
                for node in self.tors_id:
                    if w in self.G.neighbors(node):
                        worker_tor = node
                        break
                if worker_tor:
                    worker_pods.add(worker_tor)
            if len(worker_pods) > 1:  # Fragmented job
                for pod in worker_pods:
                    fragmented_pods.add(pod)

        # For each fragmented Pod, ensure at least one INA switch is selected
        for pod in fragmented_pods:
            # Check if any INA switch in this Pod is already selected
            has_switch_in_pod = any(s in self.tors_id and s == pod for s in selected_switches) or \
                                any(s in self.aggrs_id and s // 100 == pod for s in selected_switches)
            if not has_switch_in_pod:
                # Find the highest-scoring available switch in this Pod
                candidates = [s for s in sorted_switches if (s in self.tors_id and s == pod) or \
                                            (s in self.aggrs_id and s // 100 == pod)]
                if candidates:
                    # Add the best one not yet selected
                    best_candidate = None
                    for s in candidates:
                        if s not in selected_switches and len(selected_switches) < self.ina_budget:
                            if best_candidate is None or switch_scores[s] > switch_scores[best_candidate]:
                                best_candidate = s
                    if best_candidate:
                        selected_switches.append(best_candidate)
                        used_pods.add(pod)

        # Final sort by score
        selected_switches = sorted(selected_switches, key=lambda s: switch_scores[s], reverse=True)

        # Step 7: Store the result
        self._ina_placement_priority = selected_switches

    # Function 2: _assign_workers_to_aggregators
    # Role: Assigns each worker to a single aggregation point—either an INA-enabled switch or the PS—while respecting assignment validity and minimizing end-to-end congestion by prioritizing low-latency and high-capacity paths.
    def _assign_workers_to_aggregators(self):
        """Assigns each worker to a valid aggregation point—either an INA-enabled switch or the PS—while respecting assignment validity and minimizing end-to-end congestion."""
        # Initialize assignment map
        self._worker_assignment_map = {}

        # Get selected INA switches and valid candidates
        ina_placement_switches = self._ina_placement_priority[:self.ina_budget]
        valid_aggregation_candidates = set(ina_placement_switches) | set(self.ps_id)

        # For each job, assign workers to aggregation points
        for j in range(self.jobs_num):
            self._worker_assignment_map[j] = {}
            job_workers = self.workers_id[j]
            ps_node = self.ps_id[j]
            ps_tor = self.sd_id[j]

            # For each worker, find the best aggregation candidate on valid paths
            for w_idx, w_node in enumerate(job_workers):
                best_candidate = None
                best_score = -1.0

                # Check all valid candidates
                for candidate in valid_aggregation_candidates:
                    # Skip if candidate is not reachable via any valid path
                    path = self.allPathDict.get(w_node, {}).get(candidate, [])
                    if not path:
                        continue

                    # Compute score: prioritize closer switches (shorter path) and higher capacity
                    path_length = len(path) - 1  # Number of hops
                    capacity = self.Cs if candidate in ina_placement_switches else self.Ps  # Switch or PS capacity

                    # Score: inverse of path length + capacity bonus
                    score = (1.0 / (path_length + 1)) + (capacity / 100.0)

                    if score > best_score:
                        best_score = score
                        best_candidate = candidate

                # If no valid candidate found, fallback to PS
                if best_candidate is None:
                    best_candidate = ps_node

                self._worker_assignment_map[j][w_idx] = best_candidate

    # Function 3: solve_with_heuristic
    # Role: Orchestrates the complete solution pipeline by executing the placement and assignment steps in sequence, then constructs and returns the final solution object matching the required schema.
    def solve_with_heuristic(self) -> Optional[Dict[str, Any]]:
        """
        Orchestrates the complete heuristic pipeline by first determining optimal INA placement and then assigning workers to aggregation points.

        Steps:
        1. Determine candidate switches for INA placement based on topological and traffic-aware scoring.
        2. Select the top B_INA switches from the priority list for actual INA deployment.
        3. Assign each worker to a valid aggregation point (INA switch or PS), considering only switches that are both in the final placement and on valid paths.
        4. Finalize and store the solution in self.solution.
        """
        # Step 1: Determine INA placement candidates using topological and load-aware prioritization
        self._decide_ina_placement_candidates()

        # Step 2: Select the top B_INA switches based on priority score
        ina_placement_switches = self._ina_placement_priority[:self.ina_budget]

        # Step 3: Assign workers to aggregation points using capacity-aware, load-balanced routing
        self._assign_workers_to_aggregators()

        # Step 4: Validate that only selected INA switches are used in assignments
        # Filter valid aggregation candidates to only those in the final placement
        valid_aggregation_candidates = set(ina_placement_switches) | set(self.ps_id)

        # Construct worker_agg_id mapping, ensuring all assignments are to valid nodes
        worker_agg_id = {}
        for j in range(self.jobs_num):
            worker_agg_id[j] = {}
            for w in range(self.workers_num[j]):
                assigned_node = self._worker_assignment_map[j][w]
                if assigned_node not in valid_aggregation_candidates:
                    # Fallback to PS if no valid INA switch is available
                    assigned_node = self.ps_id[j]
                worker_agg_id[j][w] = assigned_node

        # Finalize the solution
        self.solution = {
            "ina_placement_switches": ina_placement_switches,
            "worker_agg_id": worker_agg_id
        }

        return self.solution
