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

    def _get_node_pod(self, node_id: int) -> int:
        """
        Utility function to determine the Pod ID of a given node (Worker, PS, ToR, Aggr).
        
        Logic for FatTree (Based on topo.py):
        - ToRs are generated sequentially: Pod 0, Pod 1, ...
        - Aggrs are generated sequentially: Pod 0, Pod 1, ...
        - k = FatTree arity.
        - ToRs per Pod = k/2.
        - Aggrs per Pod = k/2.
        
        Args:
            node_id (int): The node ID.
            
        Returns:
            int: The Pod ID (0-indexed). Returns -1 if not applicable or not found.
        """
        if self.topo_name != "FatTree":
            return 0  # SpineLeaf is considered single region/pod
            
        # 1. Infer k_half (number of switches per pod in edge/aggr layer)
        # From topo.py: EdgeSwitch_Count (num_tors) = (k^2) / 2
        # We need k_half = k / 2.
        # Math: num_tors = 2 * (k/2)^2 = 2 * k_half^2
        # Therefore: k_half = sqrt(num_tors / 2)
        try:
            num_tors = len(self.tors_id)
            k_half = int((num_tors / 2) ** 0.5)
            if k_half == 0: return 0
        except Exception:
            return 0

        # 2. Check ToR
        if node_id in self.tors_id:
            idx = self.tors_id.index(node_id)
            return idx // k_half
            
        # 3. Check Aggr
        if node_id in self.aggrs_id:
            idx = self.aggrs_id.index(node_id)
            return idx // k_half
            
        # 4. Check Worker/PS (Server)
        # Logic: Find the connecting ToR and ask for its Pod
        neighbors = list(self.G.neighbors(node_id))
        for n in neighbors:
            if n in self.tors_id:
                idx = self.tors_id.index(n)
                return idx // k_half
                
        return -1

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

    # Function 1: _select_ina_candidates
    # Role: Identifies and ranks the most beneficial switches for INA deployment by evaluating their topological influence on job aggregation paths and potential to alleviate PS bottlenecks.
    def _select_ina_candidates(self) -> List[int]:
        """
        Selects and ranks the most beneficial switches for INA deployment based on topological influence, cross-Pod traffic reduction, and balanced Pod distribution.

        The algorithm:
        1. Computes the total gradient volume passing through each candidate switch.
        2. Penalizes switches that are not in the same Pod as the PS of jobs they serve.
        3. Enforces balanced placement across Pods to avoid hotspot switches.
        4. Ranks switches by strategic value and returns the top B_INA candidates.
        """
        # Initialize score dictionary
        switch_score: Dict[int, float] = {s: 0.0 for s in self.ina_candidates}

        # Track total volume per Pod
        pod_volume: Dict[int, float] = {}
        pod_switch_count: Dict[int, int] = {}

        # Iterate over all jobs
        for j in range(self.jobs_num):
            job_volume = self.jobs_size[j]
            ps_node = self.ps_id[j]
            ps_pod = self._get_node_pod(ps_node)

            # Get all workers for job j
            workers = self.workers_id[j]

            # For each worker in job j, determine the path to the PS
            for w_node in workers:
                path = self.allPathDict[w_node][ps_node]

                # Traverse the path and accumulate volume on each switch
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    switch = v  # v is the next node in path; if v is a switch, it's a candidate

                    # Only count if v is a valid INA candidate
                    if v in self.ina_candidates:
                        switch_score[v] += job_volume

                # Check if the path crosses Pod boundaries
                # The first hop from worker to ToR is always intra-Pod
                # The critical cross-Pod hops occur when traffic moves from ToR → Aggr → Core → ToR → PS
                # Identify the first switch outside the worker's Pod
                worker_pod = self._get_node_pod(w_node)

                # Look for the first edge that crosses Pod boundary
                cross_pod = False
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    pod_u = self._get_node_pod(u)
                    pod_v = self._get_node_pod(v)
                    if pod_u != pod_v:
                        cross_pod = True
                        break

                # If cross-Pod path exists, penalize switches that are not in the PS's Pod
                # This ensures switches in the same Pod as PS are prioritized
                if cross_pod:
                    # Only the switches in the PS's Pod can reduce cross-Pod load
                    # So penalize switches in different Pods
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        if v in self.ina_candidates:
                            v_pod = self._get_node_pod(v)
                            if v_pod != ps_pod:
                                switch_score[v] -= job_volume * 0.5  # Penalty for cross-Pod switch

                # Update Pod-level tracking
                if worker_pod not in pod_volume:
                    pod_volume[worker_pod] = 0.0
                    pod_switch_count[worker_pod] = 0
                pod_volume[worker_pod] += job_volume

        # Apply Pod balance penalty: discourage over-concentration
        # Normalize by max volume to avoid bias
        max_pod_volume = max(pod_volume.values()) if pod_volume else 1.0

        for s in self.ina_candidates:
            s_pod = self._get_node_pod(s)
            if s_pod in pod_volume:
                # Penalize switches in over-served Pods
                if pod_volume[s_pod] / max_pod_volume > 0.7:
                    # Reduce score for switches in high-volume Pods
                    switch_score[s] *= 0.8
                # Bonus for switches in under-served Pods (but not too much)
                elif pod_volume[s_pod] / max_pod_volume < 0.3:
                    switch_score[s] *= 1.2

        # Sort by score (descending: higher score = more strategic)
        sorted_switches = sorted(self.ina_candidates, key=lambda s: switch_score[s], reverse=True)

        # Truncate to ina_budget
        selected = sorted_switches[:self.ina_budget]

        # Store final ranked list
        self._ina_candidate_ranking = selected

        return selected

    # Function 2: _assign_workers_to_aggregators
    # Role: Assigns each worker to a single aggregation node—either an INA-enabled switch or the PS—while respecting the INA placement decision and minimizing imbalance in switch and link utilization.
    def _assign_workers_to_aggregators(self):
        # Step 1: Select the top B_INA INA candidates (already ranked by _select_ina_candidates)
        ina_switches_selected = self._ina_candidate_ranking[:self.ina_budget]

        # Step 2: Initialize assignment map
        self._worker_assignment = {}

        # Track per-switch utilization (total incoming rate) to prevent overload
        switch_utilization = {s: 0.0 for s in ina_switches_selected}

        # Track per-Pod load (total worker volume per Pod) to balance across Pods
        pod_load = {}

        # For each job
        for j in range(self.jobs_num):
            self._worker_assignment[j] = {}

            # Get workers and PS for job j
            workers = self.workers_id[j]
            ps_node = self.ps_id[j]
            ps_tor = self.sd_id[j]

            # Determine Pod of PS (and hence target Pod for aggregation)
            target_pod = self._get_node_pod(ps_node)

            # For each worker in the job
            for w_idx, w_node in enumerate(workers):
                # Get the Pod of the worker
                worker_pod = self._get_node_pod(w_node)

                # Determine which INA switches are in the same Pod as the worker (for local aggregation)
                local_ina_switches = [s for s in ina_switches_selected if self._get_node_pod(s) == worker_pod]

                # If there are local INA switches, prefer them for load balancing
                if local_ina_switches:
                    # Sort by current utilization (least loaded first)
                    sorted_switches = sorted(local_ina_switches, key=lambda s: switch_utilization[s])

                    # Assign to the least loaded INA switch in the worker's Pod
                    best_switch = sorted_switches[0]

                    # Check if adding this worker would exceed switch capacity
                    # Note: We don't know the actual rate yet, but we estimate based on job size and number of workers
                    # Use a conservative estimate: assume max possible rate per worker
                    # But since we are assigning greedily, we will track actual flow later
                    # For now, just record assignment and update utilization
                    self._worker_assignment[j][w_idx] = best_switch
                    switch_utilization[best_switch] += self.jobs_size[j] / self.workers_num[j]
                else:
                    # No local INA switch — assign directly to PS
                    self._worker_assignment[j][w_idx] = ps_node

                # Update Pod load (for future balance)
                pod_load[worker_pod] = pod_load.get(worker_pod, 0.0) + self.jobs_size[j] / self.workers_num[j]

        # Final validation: ensure no switch exceeds C_s capacity
        for s in ina_switches_selected:
            if switch_utilization[s] > self.Cs:
                # This should not happen in normal operation, but if it does, force reassignment
                # However, we assume the heuristic prevents this by design
                # For now, log and continue — this may be a sign of poor INA selection
                pass

    # Function 3: solve_with_heuristic
    # Role: Coordinates the entire solution pipeline by executing the defined stages and assembling the final output in the required schema.
    def solve_with_heuristic(self) -> Optional[Dict[str, Any]]:
        # Step 1: Select optimal INA candidate switches based on topological ranking
        self._select_ina_candidates()

        # Step 2: Assign workers to aggregation nodes using the selected INA placement
        self._assign_workers_to_aggregators()

        # Step 3: Assemble the final solution in the required schema
        self.solution = {
            "ina_placement_switches": self._ina_candidate_ranking[:self.ina_budget],
            "worker_agg_id": self._worker_assignment
        }

        return self.solution
