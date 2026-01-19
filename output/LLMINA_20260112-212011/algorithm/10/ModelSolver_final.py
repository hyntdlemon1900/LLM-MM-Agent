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

    # Function 1: _prepare_optimization_context
    # Role: Establishes a unified, structured representation of all problem components required for heuristic decision-making, ensuring consistent and complete access to job, network, and topological data.
    def _prepare_optimization_context(self):
        """
        PREPARE OPTIMIZATION CONTEXT

        This method constructs a unified, structured representation of all problem components
        required for heuristic decision-making. It establishes the foundational data context
        by integrating job specifications, network topology, and path information.

        The context includes:
        - Job-to-switch relationships
        - Shortest path data between workers and potential aggregation points
        - Link traversal information for each flow
        - Capacity constraints (PS link, switch processing)

        This structured context enables efficient and consistent access during the subsequent heuristic
        optimization phase.
        """
        # Initialize the context dictionary
        self._job_network_context = {
            "job_to_switch_paths": {},  # job_id -> {switch_id: [path_edges]}
            "job_to_ps_path": {},      # job_id -> [path_edges]
            "flow_traversal": {},     # (src, dst) -> list of (job_idx, worker_idx)
            "ps_last_hop_capacity": {},  # job_id -> capacity (Gbps)
            "switch_processing_capacity": {},  # switch_id -> capacity (Gbps)
            "ina_candidates": self.ina_candidates,
            "job_worker_count": {}
        }

        # Populate the context with derived data
        for j in range(self.jobs_num):
            # Store worker count
            self._job_network_context["job_worker_count"][j] = self.workers_num[j]

            # Get the PS node and its ToR gateway
            ps_node = self.ps_id[j]
            sd_node = self.sd_id[j]
            self._job_network_context["ps_last_hop_capacity"][j] = self.Ps

            # Compute the path from PS to its ToR (reverse direction, but we use it for reference)
            # This is the last-hop link (ToR <-> PS)
            ps_to_tor_path = self._get_path_links(ps_node, sd_node)
            self._job_network_context["job_to_ps_path"][j] = ps_to_tor_path

            # Initialize paths for all potential aggregation points
            self._job_network_context["job_to_switch_paths"][j] = {}

            # For each worker in job j
            for w_idx, w_node in enumerate(self.workers_id[j]):
                # Compute path from worker to PS (direct path)
                worker_to_ps_path = self._get_path_links(w_node, ps_node)
                self._job_network_context["job_to_switch_paths"][j][ps_node] = worker_to_ps_path

                # For each INA candidate switch
                for s_node in self.ina_candidates:
                    # Compute path from worker to switch s_node
                    worker_to_switch_path = self._get_path_links(w_node, s_node)
                    if s_node not in self._job_network_context["job_to_switch_paths"][j]:
                        self._job_network_context["job_to_switch_paths"][j][s_node] = {}
                    self._job_network_context["job_to_switch_paths"][j][s_node][w_idx] = worker_to_switch_path

            # Precompute traversal for all possible (src, dst) flows
            # We consider flows from workers to PS and from workers to INA switches
            for w_idx, w_node in enumerate(self.workers_id[j]):
                # Flow to PS
                ps_path_edges = self._get_path_links(w_node, ps_node)
                for edge in ps_path_edges:
                    if edge not in self._job_network_context["flow_traversal"]:
                        self._job_network_context["flow_traversal"][edge] = []
                    self._job_network_context["flow_traversal"][edge].append((j, w_idx))

                # Flows to INA switches
                for s_node in self.ina_candidates:
                    switch_path_edges = self._get_path_links(w_node, s_node)
                    for edge in switch_path_edges:
                        if edge not in self._job_network_context["flow_traversal"]:
                            self._job_network_context["flow_traversal"][edge] = []
                        self._job_network_context["flow_traversal"][edge].append((j, w_idx))

            # Set switch processing capacity for all INA candidates
            for s_node in self.ina_candidates:
                self._job_network_context["switch_processing_capacity"][s_node] = self.Cs

    # Function 2: _generate_heuristic_configuration
    # Role: Produces a candidate solution by determining optimal INA placement and worker assignment based on topological awareness, resource constraints, and job load distribution.
    def _generate_heuristic_configuration(self):
        # Step 1: Prepare the optimization context
        self._prepare_optimization_context()

        # Step 2: Initialize output structures
        ina_placement_switches = []
        worker_agg_id = {}

        # Step 3: Precompute job-to-switch load potential and topology metadata
        switch_score = {s: 0.0 for s in self.ina_candidates}
        switch_load = {s: 0.0 for s in self.ina_candidates}

        for j in range(self.jobs_num):
            job_workers = self.workers_id[j]
            job_ps = self.ps_id[j]
            job_sd = self.sd_id[j]
            job_size = self.jobs_size[j]
            job_workers_num = self.workers_num[j]

            # Determine the pod of the PS
            ps_pod = self._get_node_pod(job_ps)

            # For each worker in the job, determine the shortest path to the PS
            for w_idx, w_node in enumerate(job_workers):
                # Find the shortest path from worker to PS
                path = self.allPathDict[w_node][job_ps]
                # Traverse the path and accumulate load on intermediate switches
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    if v in self.ina_candidates:
                        switch_load[v] += job_size / job_workers_num  # Proportional load

                # Score switches based on job size and pod proximity
                # Prefer switches that are in the same pod as the PS or serve multiple jobs
                # Score based on path length and pod alignment
                path_length = len(path)
                path_pod = self._get_node_pod(w_node)
                pod_match_score = 1.0 if path_pod == ps_pod else 0.5
                switch_score[job_sd] += job_size * pod_match_score  # Reward local aggregation

                # Also consider the ToR switch directly connected to PS
                if job_sd in self.ina_candidates:
                    switch_score[job_sd] += job_size * 0.5  # Incentivize PS-side aggregation

        # Step 4: Sort switches by score and select top B_INA candidates
        sorted_switches = sorted(self.ina_candidates, key=lambda s: switch_score[s], reverse=True)
        ina_placement_switches = sorted_switches[:self.ina_budget]

        # Step 5: Assign workers to optimal aggregation points
        for j in range(self.jobs_num):
            job_workers = self.workers_id[j]
            job_ps = self.ps_id[j]
            job_sd = self.sd_id[j]

            # Initialize assignment for this job
            worker_agg_id[j] = {}

            # For each worker, determine the best aggregation point
            for w_idx, w_node in enumerate(job_workers):
                # Candidate: either an INA switch or the PS
                candidates = []
                # Add all INA switches that are valid (placed and reachable)
                for s in ina_placement_switches:
                    if s in self.ina_candidates:
                        # Check if path from worker to s exists and is valid
                        if s in self.allPathDict[w_node]:
                            # Prefer switches closer to worker
                            path_to_s = self._get_path_links(w_node, s)
                            path_len = len(path_to_s)
                            candidates.append((s, path_len))

                # If no INA switch is available, assign directly to PS
                if not candidates:
                    worker_agg_id[j][w_idx] = job_ps
                else:
                    # Choose the closest INA switch
                    best_switch = min(candidates, key=lambda x: x[1])[0]
                    worker_agg_id[j][w_idx] = best_switch

        # Step 6: Store results
        self._heuristic_solution = {
            "ina_placement_switches": ina_placement_switches,
            "worker_agg_id": worker_agg_id
        }

    # Function 3: solve_with_heuristic
    # Role: Coordinates the execution of the entire pipeline, invokes the internal steps in sequence, and constructs the final solution object matching the required schema.
    def solve_with_heuristic(self) -> Optional[Dict[str, Any]]:
        """
        Coordinates the execution of the entire pipeline, invokes the internal steps in sequence, and constructs the final solution object matching the required schema.
        """
        # Step 1: Prepare the unified optimization context
        self._prepare_optimization_context()

        # Step 2: Generate a candidate heuristic solution (INA placement & worker assignment)
        # Extract the solution from _generate_heuristic_configuration
        self._generate_heuristic_configuration()

        # Step 3: Store the final solution in the expected format
        self.solution = {
            "ina_placement_switches": self._heuristic_solution["ina_placement_switches"],
            "worker_agg_id": self._heuristic_solution["worker_agg_id"]
        }

        return self.solution
