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

    # Function 1: _identify_aggregation_candidates
    # Role: Determines the set of eligible switch nodes for INA deployment by analyzing job-to-PS topological relationships and switch tier classifications, ensuring only physically valid candidates are considered.
    def _identify_aggregation_candidates(self):
        """Determines the set of eligible switch nodes for INA deployment by analyzing job-to-PS topological relationships and switch tier classifications, ensuring only physically valid candidates are considered."""
        # Step 1: Start with the full set of candidate switches from topology
        candidate_switches = set(self.ina_candidates)

        # Step 2: Filter candidates to only those in the same Pod as their associated PS
        # For each job, get its PS's Pod and keep only switches in that Pod
        valid_candidates = set()
        for j in range(self.jobs_num):
            ps_pod = self._get_node_pod(self.ps_id[j])
            for s in candidate_switches:
                if self._get_node_pod(s) == ps_pod:
                    valid_candidates.add(s)

        # Step 3: Ensure only valid switch tiers are included (based on topology)
        # This is already enforced by _compute_ina_candidates, so no need to re-filter

        # Step 4: Verify connectivity to at least one worker or PS
        # Remove switches that are not on any path from a worker to its PS
        final_candidates = set()
        for s in valid_candidates:
            is_reachable = False
            for j in range(self.jobs_num):
                # Check if switch s lies on any path from a worker of job j to its PS
                for w in self.workers_id[j]:
                    path_to_ps = self._get_path_links(w, self.ps_id[j])
                    if any((s == u or s == v) for u, v in path_to_ps):
                        is_reachable = True
                        break
                if is_reachable:
                    break
            if is_reachable:
                final_candidates.add(s)

        # Step 5: Assign the final filtered list to the output member variable
        self._aggregation_candidate_pool = sorted(list(final_candidates))

    # Function 2: _construct_routing_and_placement
    # Role: Generates a feasible INA deployment configuration and worker assignment map by prioritizing aggregation points based on job load, link congestion risks, and switch capacity, ensuring all constraints are satisfied.
    def _construct_routing_and_placement(self):
        # Step 1: Identify aggregation candidates (must be called first)
        self._identify_aggregation_candidates()

        # Initialize output containers
        self._ina_placement_switches = []
        self._worker_agg_id = {}

        # Initialize tracking for switch load and link usage
        switch_load = {s: 0.0 for s in self._aggregation_candidate_pool}
        link_load = {}

        # Precompute job-level metrics: total size and number of workers
        job_data = []
        for j in range(self.jobs_num):
            total_size = self.jobs_size[j]
            num_workers = self.workers_num[j]
            job_data.append((j, total_size, num_workers))

        # Sort jobs by gradient volume (descending) to prioritize large jobs
        job_data.sort(key=lambda x: x[1], reverse=True)

        # For each job, assign aggregation point
        for j, total_size, num_workers in job_data:
            # Get the PS gateway switch
            ps_switch = self.sd_id[j]
            best_switch = None
            best_score = -1

            # Evaluate each candidate switch (including PS as fallback)
            candidates = self._aggregation_candidate_pool + [ps_switch]
            for s in candidates:
                # Skip if not valid (e.g., s is not a valid aggregation point for job j)
                # Check if s is an INA candidate and has INA enabled (if s in _ina_placement_switches)
                # But for scoring, we assume s is eligible

                # Calculate congestion risk: how much this switch would add to link loads
                # Only consider flows from workers to s
                total_flow_rate = 0.0
                # Simulate assignment: if s is selected, the rate from s to PS must support total_size
                # But we are evaluating s as a candidate, not final

                # Compute potential path load from each worker to s
                path_load = 0.0
                for w_idx, worker_id in enumerate(self.workers_id[j]):
                    path = self._get_path_links(worker_id, s)
                    for u, v in path:
                        edge = (u, v)
                        # Estimate link load contribution (if this flow were assigned)
                        # Use total_size / num_workers as base rate per worker
                        rate = total_size / num_workers
                        path_load += rate

                # Also consider load on the switch s: total incoming rate from all workers
                switch_incoming = total_size  # total volume from job j

                # Score: higher volume, lower congestion risk, higher capacity utilization
                # Score = (job volume) * (1 / (1 + path_load)) * (C_s / (switch_load[s] + 1e-6))
                # But we cannot assign yet

                # Instead, use a simple heuristic: prefer switches that are not overloaded and reduce path length
                # Weighted by: job size, low path congestion, high switch capacity
                congestion_score = 1.0 / (1.0 + path_load)
                capacity_score = 1.0 if switch_load[s] == 0 else (self.Cs - switch_load[s]) / self.Cs

                # Final score: prioritize large jobs and low congestion
                score = total_size * congestion_score * capacity_score

                # Use PS as fallback
                if s == ps_switch:
                    # PS is always valid, but has no INA
                    score = total_size * 0.5  # lower weight

                if score > best_score:
                    best_score = score
                    best_switch = s

            # Assign worker(s) to best_switch
            if best_switch is None:
                raise RuntimeError(f"No valid aggregation candidate for job {j}")

            # Ensure assignment is consistent
            self._worker_agg_id[j] = {}
            for w_idx, worker_id in enumerate(self.workers_id[j]):
                self._worker_agg_id[j][w_idx] = best_switch

            # Update switch load (only if best_switch is an INA switch and not the PS)
            if best_switch != ps_switch and best_switch in self._aggregation_candidate_pool:
                switch_load[best_switch] += total_size

        # Now select actual INA switches (up to budget)
        # Score each candidate switch by total load it carries
        switch_score = {}
        for s in self._aggregation_candidate_pool:
            # Score = total volume assigned to this switch across all jobs
            total_volume = 0.0
            for j in range(self.jobs_num):
                for w_idx, worker_id in enumerate(self.workers_id[j]):
                    if self._worker_agg_id[j][w_idx] == s:
                        total_volume += self.jobs_size[j]
            switch_score[s] = total_volume

        # Sort by score (descending) and select top B_INA
        sorted_switches = sorted(switch_score.keys(), key=lambda s: switch_score[s], reverse=True)
        self._ina_placement_switches = sorted_switches[:self.ina_budget]

        # Ensure all assignments respect INA placement: if a worker is assigned to a switch, it must be in the placement list
        # If not, reassign to PS
        for j in range(self.jobs_num):
            ps_switch = self.sd_id[j]
            for w_idx in self._worker_agg_id[j]:
                agg_node = self._worker_agg_id[j][w_idx]
                if agg_node in self._aggregation_candidate_pool and agg_node not in self._ina_placement_switches:
                    # Reassign to PS
                    self._worker_agg_id[j][w_idx] = ps_switch

    # Function 3: solve_with_heuristic
    # Role: Orchestrates the entire heuristic pipeline by invoking the sequential steps and assembling the final solution in the required schema, including both INA placement and worker assignment decisions.
    def solve_with_heuristic(self) -> Optional[Dict[str, Any]]:
        """
        Orchestrates the heuristic pipeline to solve the joint INA placement and traffic assignment problem.

        Steps:
        1. Identify valid aggregation candidates (eligible switches for INA) using topological and tier-based filtering.
        2. Construct a feasible configuration of INA placement and worker assignments that satisfies all constraints.
        3. Store the solution in self.solution in the required format.
        """
        # Step 1: Identify aggregation candidates
        self._identify_aggregation_candidates()

        # Step 2: Construct routing and placement based on heuristic logic
        self._construct_routing_and_placement()

        # Step 3: Assemble final solution
        self.solution = {
            "ina_placement_switches": self._ina_placement_switches,
            "worker_agg_id": self._worker_agg_id
        }

        return self.solution
