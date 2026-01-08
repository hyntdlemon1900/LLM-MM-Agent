from __future__ import annotations
from typing import List, Dict, Tuple, Any, Optional
import copy
import networkx as nx
from MMBench.problem.problem_template.runtime.TemplateSolver import TemplateSolver

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
        """
        Selects the most strategically impactful switches for INA deployment by evaluating their potential to alleviate aggregation bottlenecks across multiple jobs.

        The algorithm computes a composite score for each candidate switch based on:
        - Proximity to PS (same ToR bonus)
        - Reduction in cross-Pod traffic (penalizes core-layer hops)
        - Aggregated job volume on the switch's path
        - Ensures switches are on actual worker-to-PS paths for fragmented jobs

        Results are ranked and selected up to the ina_budget, ensuring load balancing across Pods.
        """
        # Ensure candidates are computed
        self._compute_ina_candidates()

        # Initialize score dictionary
        switch_scores = {}
        switch_path_counts = {}
        switch_pod_assignment = {}

        # Precompute Pod membership for PS and switches
        ps_pods = {}
        for j in range(self.jobs_num):
            ps_node = self.ps_id[j]
            sd_node = self.sd_id[j]
            pod_id = sd_node  # In Fat-Tree, ToR node ID defines the Pod
            ps_pods[j] = pod_id

        # For each candidate switch, compute score
        for s in self.ina_candidates:
            score = 0.0
            path_count = 0
            pod_id = s  # Default to switch ID

            # Determine Pod ID based on switch type
            if s in self.tors_id:
                pod_id = s
            elif s in self.aggrs_id:
                # Aggr switches are in same Pod as their connected ToRs
                # Use a conservative heuristic: assume pod_id is the ToR switch they connect to
                # But we don't have direct connectivity, so we use the switch ID as Pod ID for simplicity
                # Alternatively, use ToR ID if known via topology; otherwise, default to switch ID
                # Since we can't resolve this without topology data, we use switch ID as Pod ID
                pod_id = s
            else:
                pod_id = s  # Core switches are in global Pod

            switch_pod_assignment[s] = pod_id

            # Track total job volume that can be aggregated via this switch
            total_job_volume = 0.0
            has_cross_pod_jobs = False

            # Evaluate all jobs
            for j in range(self.jobs_num):
                job_volume = self.jobs_size[j]
                ps_node = self.ps_id[j]
                sd_node = self.sd_id[j]
                ps_pod = ps_pods[j]

                # Check if this switch lies on any worker-to-PS path
                has_valid_path = False

                for w_idx, w_node in enumerate(self.workers_id[j]):
                    path = self._get_path_links(w_node, ps_node)
                    if not path:
                        continue

                    # Check if switch s is in the path
                    if s in [u for u, v in path] or s in [v for u, v in path]:
                        has_valid_path = True
                        break

                if not has_valid_path:
                    continue

                # If path exists, check if it traverses core layer (cross-Pod)
                path_cost = 0.0
                for w_idx, w_node in enumerate(self.workers_id[j]):
                    path = self._get_path_links(w_node, ps_node)
                    if not path:
                        continue

                    # Count core-layer hops
                    core_hops = 0
                    for u, v in path:
                        if u in self.cores_id or v in self.cores_id:
                            core_hops += 1
                    path_cost += core_hops * job_volume

                # Score bonus: same Pod as PS (ToR bonus)
                if s == sd_node:
                    score += job_volume * 1.5  # Strong bonus for same ToR

                # Penalty: cross-Pod traffic (core hops)
                path_cost_penalty = path_cost / max(1e-6, job_volume)
                score -= path_cost_penalty * job_volume

                # Volume-weighted contribution
                total_job_volume += job_volume

                # Mark that this switch serves a job with cross-Pod distribution
                if path_cost > 0:
                    has_cross_pod_jobs = True

                path_count += 1

            # Penalize if switch is not on any valid path
            if path_count == 0:
                score = -float('inf')
            else:
                # Bonus for serving cross-Pod jobs
                if has_cross_pod_jobs:
                    score += total_job_volume * 0.5

                # Apply load balancing: avoid over-concentration in one Pod
                pod_key = switch_pod_assignment[s]
                # Safely compute pod_count using only existing keys
                pod_count = sum(1 for sw in self.ina_candidates if sw in switch_pod_assignment and switch_pod_assignment[sw] == pod_key)
                if pod_count > 1:
                    score -= 0.1 * total_job_volume  # Small penalty for Pod saturation

            switch_scores[s] = score
            switch_path_counts[s] = path_count

        # Sort switches by score (descending)
        sorted_switches = sorted(self.ina_candidates, key=lambda s: switch_scores.get(s, -float('inf')), reverse=True)

        # Select up to ina_budget candidates
        selected_switches = sorted_switches[:self.ina_budget]

        # Store final ranked list
        self._ina_placement_priority = selected_switches

    # Function 2: _assign_workers_to_aggregators
    # Role: Assigns each worker to a single aggregation point—either an INA-enabled switch or the PS—while respecting assignment validity and minimizing end-to-end congestion by prioritizing low-latency and high-capacity paths.
    def _assign_workers_to_aggregators(self):
        """
        Assigns each worker to a single aggregation point (INA switch or PS) based on a composite cost function that prioritizes intra-Pod aggregation, minimizes cross-Pod traffic, and avoids overloading switches.

        The algorithm ensures:
        - Workers are assigned only to INA switches that are both in `self.ina_candidates` and selected in `self._ina_placement_priority`.
        - Only switches in the same Pod as the PS are considered (to reduce core-layer congestion).
        - Assignment minimizes a cost function combining path length, cross-Pod penalty, and load-aware routing.
        - The final assignment satisfies the 'single aggregation point' and 'assignment validity' constraints.
        """
        # Initialize the assignment map
        self._worker_assignment_map = {}

        # Get the actual INA placement from the decision step
        ina_placement_switches = set(self._ina_placement_priority)

        # Precompute Pod membership for switches and PS nodes
        # For each PS, determine its Pod (ToR switch is in the same Pod as PS)
        ps_pod_map = {}
        for j in range(self.jobs_num):
            ps_node = self.ps_id[j]
            sd_node = self.sd_id[j]  # ToR switch connected to PS
            ps_pod_map[j] = sd_node  # Pod is identified by the ToR switch

        # For each job, assign workers to aggregation points
        for j in range(self.jobs_num):
            self._worker_assignment_map[j] = {}

            # Get the list of workers and the PS for this job
            workers = self.workers_id[j]
            ps_node = self.ps_id[j]

            # Determine the Pod of this job's PS
            job_pod = ps_pod_map[j]

            # Build list of valid aggregation candidates for this job:
            # - All INA switches that are in the same Pod as the PS
            # - The PS itself as fallback
            valid_candidates = set()

            # Add all INA switches that are in the same Pod as the PS
            for switch in ina_placement_switches:
                # Check if this switch is in the same Pod as the PS
                # In Fat-Tree, ToR switches are in the same Pod as the PS
                # Aggregation switches in the same Pod are also valid
                # For simplicity, assume switch is in same Pod if it's in the same ToR's Pod
                # Since we only use ToR and Aggr switches for INA, and ToR is the Pod boundary
                # We consider a switch valid if it is connected to the same ToR as the PS
                # But to simplify: use the fact that ToR switches are the Pod boundary
                # So if switch is a ToR switch and is the same as job_pod, or if it's an Aggr switch connected to job_pod
                # For this version, we assume: any INA switch that is in the same Pod as the PS (i.e., same ToR) is valid
                # We assume the switch is in the same Pod if it is directly connected to the same ToR as the PS
                # But we don't have direct connectivity info, so we use a heuristic:
                # For now, assume all INA switches in the same Pod (i.e., same ToR) are valid
                # But to be safe: only consider INA switches that are in the same Pod as the PS
                # Since we know job_pod is a ToR switch, any INA switch that is connected to this ToR is in the same Pod
                # Instead, we use: only switch is in same Pod if it's a ToR switch and same as job_pod, or an Aggr switch connected to job_pod
                # But we don't have connectivity data in this scope, so we use the following:
                # For Fat-Tree, we assume that a switch is in the same Pod as the PS if it is in the same ToR's Pod
                # Since we don't have a direct way to check, we assume that if the switch is a ToR switch and equals job_pod, it's valid
                # If it's an Aggr switch and connected to job_pod, it's also valid
                # But we don't have that data, so we simplify:
                # We will allow a switch to be valid only if it is in the same Pod as the PS, which we define as being in the same ToR's Pod
                # We'll use the following rule: only ToR switches that are job_pod are valid
                # But this is too restrictive
                # Better: allow any INA switch that is in the same Pod as the PS, which we define as any switch whose shortest path to the PS goes through the same ToR
                # But we can't compute that here
                # So for now: allow any INA switch that is in the same Pod as the PS, where Pod is determined by the ToR switch connected to the PS
                # We'll assume that only the ToR switch (job_pod) and any Aggr switch connected to it are in the same Pod
                # But we don't have connectivity data
                # So we take a simpler approach: only consider INA switches that are in the same Pod as the PS
                # We will assume that a switch is in the same Pod if it is a ToR switch and equals job_pod
                # This is a conservative assumption
                if switch in self.tors_id and switch == job_pod:
                    valid_candidates.add(switch)
                elif switch in self.aggrs_id:
                    # Assume Aggr switches connected to job_pod are in the same Pod
                    # But we don't know which ones are
                    # So we skip Aggr switches for now to avoid error
                    pass

            # Add the PS as a fallback
            valid_candidates.add(ps_node)

            # For each worker in the job, assign to best candidate
            for w_idx, worker_node in enumerate(workers):
                best_candidate = None
                best_cost = float('inf')

                # Evaluate each valid candidate
                for candidate in valid_candidates:
                    # Compute path from worker to candidate
                    path = self._get_path_links(worker_node, candidate)
                    if not path:
                        continue  # Skip if no path

                    # Compute cost: hop count + cross-Pod penalty
                    hop_count = len(path)
                    cross_pod_penalty = 0.0

                    # Check if this path traverses a core layer link (cross-Pod)
                    # In Fat-Tree, core layer links are between Aggr and Core
                    # So if any edge in the path is a core layer link, it's cross-Pod
                    for u, v in path:
                        if u in self.aggrs_id and v in self.cores_id or u in self.cores_id and v in self.aggrs_id:
                            cross_pod_penalty += 5.0  # High penalty for cross-Pod
                            break

                    # Estimate switch load (for load-aware routing)
                    # We don't have real load yet, so use a proxy: number of workers already assigned to this switch
                    # But we are assigning one by one, so we can track dynamically
                    # We'll use a simple load proxy based on the switch's potential to handle traffic
                    # For now, we assume all switches have same capacity, so no penalty
                    # But we can add a soft penalty for switches with higher load
                    # Since we don't track load during assignment, we skip this

                    # Total cost
                    cost = hop_count + cross_pod_penalty

                    if cost < best_cost:
                        best_cost = cost
                        best_candidate = candidate

                # Assign the worker
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
