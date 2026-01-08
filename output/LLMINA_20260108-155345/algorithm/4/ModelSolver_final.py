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
        Selects the most impactful switches for In-Network Aggregation (INA) placement by evaluating their systemic contribution to reducing aggregation bottlenecks.

        The algorithm prioritizes switches that:
        - Serve high-volume jobs with fragmented worker distributions across Pods.
        - Enable local aggregation within a Pod, avoiding cross-Pod traffic.
        - Reduce congestion on the PS link by offloading large volumes of gradient data.

        Scoring is based on:
        - Job size (higher weight for larger jobs).
        - Path cost: penalizes cross-Pod hops (via core switches) with higher cost.
        - Pod affinity: favors intra-Pod aggregation over cross-Pod.
        - Aggregate traffic reduction potential: switches that can absorb multiple high-volume flows.

        Results are stored in self._ina_placement_priority.
        """
        # Ensure candidate list is computed
        self._compute_ina_candidates()

        # Initialize score dictionary for each candidate switch
        switch_scores = {s: 0.0 for s in self.ina_candidates}

        # Map each PS to its ToR (gateway switch) for Pod identification
        ps_to_tor = {j: self.sd_id[j] for j in range(self.jobs_num)}

        # Helper: Get the Pod ID for a switch (using ToR as Pod representative)
        def get_pod_id(switch_id):
            # For simplicity, assume ToRs are unique Pod identifiers
            if switch_id in self.tors_id:
                return switch_id
            # Aggregation and Core switches are associated with a Pod via their ToRs
            # Find a ToR that is connected to this switch (any one is sufficient for Pod assignment)
            for neighbor in self.G.neighbors(switch_id):
                if neighbor in self.tors_id:
                    return neighbor
            return None

        # Precompute Pod assignments
        switch_pod = {s: get_pod_id(s) for s in self.ina_candidates}

        # For each job, compute path cost from each worker to each candidate aggregation point
        # and accumulate scores based on job size and path cost
        for j in range(self.jobs_num):
            job_size = self.jobs_size[j]
            job_workers = self.workers_id[j]
            job_ps_tor = ps_to_tor[j]
            job_ps_pod = get_pod_id(job_ps_tor)

            # For each candidate switch s
            for s in self.ina_candidates:
                s_pod = switch_pod[s]
                is_cross_pod = (s_pod != job_ps_pod)

                # Initialize path cost for this switch
                total_path_cost = 0.0

                # Accumulate cost for all workers in job j
                for w in job_workers:
                    # Get shortest path from worker w to candidate switch s
                    path = self.allPathDict[w][s] if w in self.allPathDict and s in self.allPathDict[w] else []
                    if not path:
                        continue
                    # Convert path to list of edges
                    path_edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]
                    # Count core-layer hops (core switches in path)
                    core_hops = sum(1 for u, v in path_edges if u in self.cores_id or v in self.cores_id)
                    # Path cost: base cost + penalty for cross-Pod and core hops
                    # Cross-Pod traffic is penalized more heavily
                    cost = core_hops * 5.0  # High penalty for core-layer hops
                    if is_cross_pod:
                        cost += 10.0  # Additional penalty for cross-Pod
                    total_path_cost += cost

                # Weight the total path cost by job size, then subtract from score
                # Higher job size = higher priority to aggregate locally
                # Lower path cost = higher score
                if total_path_cost > 0:
                    score = job_size / total_path_cost
                else:
                    score = job_size  # Avoid division by zero

                # Apply a bonus for intra-Pod aggregation
                if not is_cross_pod:
                    score *= 1.5  # 50% bonus for local aggregation

                # Accumulate score for this switch
                switch_scores[s] += score

        # Sort switches by score in descending order
        ranked_switches = sorted(self.ina_candidates, key=lambda s: switch_scores[s], reverse=True)

        # Apply budget constraint: limit to self.ina_budget
        self._ina_placement_priority = ranked_switches[:self.ina_budget]


    # Function 2: _assign_workers_to_aggregators
    # Role: Assigns each worker to a single aggregation point—either an INA-enabled switch or the PS—while respecting assignment validity and minimizing end-to-end congestion by prioritizing low-latency and high-capacity paths.
    def _assign_workers_to_aggregators(self):
        # Initialize the assignment map
        self._worker_assignment_map = {}

        # Precompute the list of INA-enabled switches from the priority list
        ina_enabled_switches = set(self._ina_placement_priority)

        # For each job, process worker assignments
        for j in range(self.jobs_num):
            job_workers = self.workers_id[j]
            self._worker_assignment_map[j] = {}

            # Get the ToR switch connected to this job's PS
            ps_tor_switch = self.sd_id[j]

            # For each worker in the job
            for w_idx, worker_node in enumerate(job_workers):
                best_cost = float('inf')
                best_agg_node = None

                # Evaluate assignment to each potential aggregation point: INA switches and PS
                # First, check all INA-enabled switches (only those in the priority list)
                for s in ina_enabled_switches:
                    # Check if this switch is a valid candidate for job j (must be reachable)
                    path = self._get_path_links(worker_node, s)
                    if not path:
                        continue  # Skip unreachable switches

                    # Compute path cost: includes hop count and cross-Pod penalty
                    hop_count = len(path)
                    cross_pod_penalty = 0.0

                    # Check if the path traverses any core layer links (cross-Pod)
                    for u, v in path:
                        if u in self.cores_id or v in self.cores_id:
                            cross_pod_penalty += 5.0  # Penalty for cross-Pod traffic

                    # Compute total cost
                    cost = hop_count + cross_pod_penalty

                    # Optionally penalize if the switch is already overloaded
                    # However, since we don't track real-time load, use a proxy: switch centrality or static load estimate
                    # Here, we use a simple heuristic: penalize switches that are core switches more
                    if s in self.cores_id:
                        cost += 2.0

                    # Update best assignment if cost is lower
                    if cost < best_cost:
                        best_cost = cost
                        best_agg_node = s

                # If no valid INA switch found, assign to PS
                if best_agg_node is None:
                    best_agg_node = self.ps_id[j]

                # Assign worker to best aggregation point
                self._worker_assignment_map[j][w_idx] = best_agg_node


    # Function 3: solve_with_heuristic
    # Role: Orchestrates the complete solution pipeline by executing the placement and assignment steps in sequence, then constructs and returns the final solution object matching the required schema.
    def solve_with_heuristic(self) -> Optional[Dict[str, Any]]:
        """
        Orchestrates the complete solution pipeline by executing the placement and assignment steps in sequence, then constructs and returns the final solution object matching the required schema.
        """
        # Step 1: Decide optimal INA placement candidates based on strategic priority
        self._decide_ina_placement_candidates()

        # Step 2: Assign workers to aggregation points using the selected INA switches
        self._assign_workers_to_aggregators()

        # Step 3: Construct the final solution object
        self.solution = {
            "ina_placement_switches": self._ina_placement_priority[:self.ina_budget],
            "worker_agg_id": self._worker_assignment_map
        }

        return self.solution
