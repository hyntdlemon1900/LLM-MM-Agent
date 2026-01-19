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

    # Function 1: _initialize_aggregation_candidates
    # Role: Establishes the set of viable aggregation points for each job by identifying switch nodes that are both topologically favorable and eligible for INA, based on job-specific access patterns and network hierarchy.
    def _initialize_aggregation_candidates(self):
        """Initializes the set of viable aggregation points for each job by identifying topologically favorable INA-enabled switches."""
        self._aggregation_candidates_by_job = {}

        for j in range(self.jobs_num):
            candidates = set()
            ps_node = self.ps_id[j]
            sd_node = self.sd_id[j]

            # Get all workers for this job
            workers = self.workers_id[j]

            # For each worker, find the path to the PS
            for worker in workers:
                if worker not in self.allPathDict or ps_node not in self.allPathDict[worker]:
                    continue
                path = self.allPathDict[worker][ps_node]
                # Traverse the path and collect switches (excluding the PS)
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    # Check if v is an INA candidate and not the PS
                    if v in self.ina_candidates and v != ps_node:
                        candidates.add(v)
                    # Also check u if u is a valid candidate
                    if u in self.ina_candidates and u != ps_node:
                        candidates.add(u)

            # Convert to sorted list
            self._aggregation_candidates_by_job[j] = sorted(list(candidates))

    # Function 2: _construct_placement_and_routing
    # Role: Generates a feasible assignment of workers to aggregation nodes and selects a subset of switches for INA placement, ensuring all constraints—including budget, single assignment, and processing limits—are respected.
    def _construct_placement_and_routing(self):
        # Step 1: Initialize aggregation candidates for each job
        self._initialize_aggregation_candidates()

        # Step 2: Initialize output structures
        self._selected_ina_switches = []
        self._worker_aggregation_assignment = {j: {} for j in range(self.jobs_num)}

        # Step 3: For each job, sort aggregation candidates by path length (shortest first) and prioritize PS as fallback
        job_candidate_rankings = {}
        for j in range(self.jobs_num):
            candidates = self._aggregation_candidates_by_job[j]
            # Sort by shortest path length to PS (lower is better)
            # Use path length from worker to candidate, then candidate to PS
            def cost(node):
                # For each worker, find avg path length to this node and then to PS
                total_cost = 0.0
                for w in range(self.workers_num[j]):
                    worker_node = self.workers_id[j][w]
                    if node == self.ps_id[j]:
                        # Direct to PS
                        path = self._get_path_links(worker_node, self.ps_id[j])
                        total_cost += len(path)
                    else:
                        # To node, then to PS
                        path1 = self._get_path_links(worker_node, node)
                        path2 = self._get_path_links(node, self.ps_id[j])
                        total_cost += len(path1) + len(path2)
                return total_cost / self.workers_num[j]
            ranked = sorted(candidates, key=cost)
            job_candidate_rankings[j] = ranked

        # Step 4: Track switch load and assign workers greedily
        switch_load = {s: 0.0 for s in self.ina_candidates}
        # Use a set to track which switches are selected for INA
        selected_switches = set()

        # For each job, assign workers to the best aggregation candidate
        for j in range(self.jobs_num):
            for w in range(self.workers_num[j]):
                worker_node = self.workers_id[j][w]
                assigned = False
                # Try candidates in order of ranking
                for candidate in job_candidate_rankings[j]:
                    if candidate == self.ps_id[j]:
                        # Direct assignment to PS is always allowed
                        self._worker_aggregation_assignment[j][w] = self.ps_id[j]
                        assigned = True
                        break
                    elif candidate in self.ina_candidates and candidate not in selected_switches:
                        # If INA is not yet selected, assign to this switch
                        self._worker_aggregation_assignment[j][w] = candidate
                        # Mark switch as used (if not already)
                        selected_switches.add(candidate)
                        assigned = True
                        break
                if not assigned:
                    # Fallback: assign to PS
                    self._worker_aggregation_assignment[j][w] = self.ps_id[j]

        # Step 5: Enforce INA budget by selecting top K switches
        # Count how many workers are assigned to each switch
        switch_assignment_count = {s: 0 for s in self.ina_candidates}
        for j in range(self.jobs_num):
            for w in range(self.workers_num[j]):
                agg_node = self._worker_aggregation_assignment[j][w]
                if agg_node in self.ina_candidates:
                    switch_assignment_count[agg_node] += 1

        # Sort switches by number of assigned workers (highest first) and select top B_INA
        sorted_switches = sorted(switch_assignment_count.keys(), key=lambda s: switch_assignment_count[s], reverse=True)
        self._selected_ina_switches = sorted_switches[:self.ina_budget]

        # Step 6: Reassign any worker whose aggregation point is not in the selected set to PS
        for j in range(self.jobs_num):
            for w in range(self.workers_num[j]):
                agg_node = self._worker_aggregation_assignment[j][w]
                if agg_node in self.ina_candidates and agg_node not in self._selected_ina_switches:
                    self._worker_aggregation_assignment[j][w] = self.ps_id[j]


    # Function 3: solve_with_heuristic
    # Role: Coordinates the execution of the pipeline, invokes the intermediate steps in sequence, and assembles the final solution object in the required output schema.
    def solve_with_heuristic(self) -> Optional[Dict[str, Any]]:
        """
        Coordinates the execution of the pipeline by invoking the necessary heuristic steps to compute a feasible solution.

        First, initializes the set of valid aggregation candidates per job.
        Then, constructs the optimal INA placement and worker routing configuration.
        Finally, assembles and stores the solution in the required format.
        """
        # Step 1: Initialize aggregation candidates for each job
        self._initialize_aggregation_candidates()

        # Step 2: Construct the placement and routing solution
        self._construct_placement_and_routing()

        # Step 3: Assemble the final solution object
        self.solution = {
            # List of integer Node IDs of selected switches.
            "ina_placement_switches": self._selected_ina_switches[:],

            # Nested dictionary: Mapping [job_index][worker_index] -> Aggregation Node ID
            "worker_agg_id": {
                j: {w: self._worker_aggregation_assignment[j][w] for w in range(len(self._worker_aggregation_assignment[j]))}
                for j in range(self.jobs_num)
            }
        }
        return self.solution
