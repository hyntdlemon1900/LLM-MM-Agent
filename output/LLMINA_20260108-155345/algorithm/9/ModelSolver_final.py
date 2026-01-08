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
        # Ensure candidate switches are computed
        self._compute_ina_candidates()

        # Initialize score dictionary for each candidate switch
        switch_scores = {s: 0.0 for s in self.ina_candidates}

        # Track which jobs are already covered by a candidate in their Pod
        covered_jobs = set()

        # For each job, determine its Pod (via PS ToR)
        job_pods = {}
        for j in range(self.jobs_num):
            ps_tor = self.sd_id[j]
            job_pods[j] = ps_tor

        # For each candidate switch, compute its contribution
        for s in self.ina_candidates:
            total_volume = 0.0
            high_volume_jobs = 0
            in_same_pod = False
            for j in range(self.jobs_num):
                ps_tor = job_pods[j]
                if s == ps_tor:
                    in_same_pod = True
                # Check if switch s lies on the shortest path from any worker of job j to PS
                for w_idx, worker in enumerate(self.workers_id[j]):
                    path = self._get_path_links(worker, self.ps_id[j])
                    if any(edge[0] == s or edge[1] == s for edge in path):
                        total_volume += self.jobs_size[j]
                        if self.jobs_size[j] > 10.0:  # High-volume job
                            high_volume_jobs += 1

            # Bonus for same-ToR placement (PS Pod)
            if in_same_pod:
                switch_scores[s] += 100.0

            # Weight by total offload potential (volume of data that can be aggregated)
            switch_scores[s] += total_volume * 10.0

            # Penalize cross-Pod switches that don't serve high-volume jobs
            if not in_same_pod and high_volume_jobs == 0:
                switch_scores[s] -= 50.0

            # Avoid placing INA on switches that are already heavily used (simulated load from path usage)
            # Estimate load: number of jobs whose paths go through s
            path_jobs = 0
            for j in range(self.jobs_num):
                for worker in self.workers_id[j]:
                    path = self._get_path_links(worker, self.ps_id[j])
                    if any(edge[0] == s or edge[1] == s for edge in path):
                        path_jobs += 1
            # Apply penalty if too many jobs depend on this switch
            if path_jobs > 2 * self.jobs_num / len(self.ina_candidates):
                switch_scores[s] -= 10.0

        # Sort candidates by score in descending order
        ranked_candidates = sorted(self.ina_candidates, key=lambda s: switch_scores[s], reverse=True)

        # Apply budget constraint and ensure coverage of high-volume, fragmented jobs
        selected_switches = set()
        job_coverage = set()

        # First, ensure at least one INA switch is placed in each Pod with a high-volume, fragmented job
        for j in range(self.jobs_num):
            if self.jobs_size[j] > 10.0:  # High-volume job
                # Count how many workers are in different Pods
                worker_pods = set()
                for worker in self.workers_id[j]:
                    # Find the ToR connected to this worker
                    neighbors = list(self.G.neighbors(worker))
                    for n in neighbors:
                        if n in self.tors_id:
                            worker_pods.add(n)
                            break
                if len(worker_pods) > 1:  # Fragmented across Pods
                    # Assign a switch in the same Pod as the PS
                    ps_tor = job_pods[j]
                    # Prefer a switch in the same Pod as PS
                    for candidate in ranked_candidates:
                        if candidate in self.tors_id and candidate == ps_tor:
                            selected_switches.add(candidate)
                            job_coverage.add(j)
                            break

        # Now fill remaining budget with highest-scoring switches
        for s in ranked_candidates:
            if len(selected_switches) >= self.ina_budget:
                break
            if s not in selected_switches:
                selected_switches.add(s)

        # Convert to list in order of score
        self._ina_placement_priority = sorted(list(selected_switches), key=lambda s: switch_scores[s], reverse=True)

    # Function 2: _assign_workers_to_aggregators
    # Role: Assigns each worker to a single aggregation point—either an INA-enabled switch or the PS—while respecting assignment validity and minimizing end-to-end congestion by prioritizing low-latency and high-capacity paths.
    def _assign_workers_to_aggregators(self):
        # Initialize the worker assignment map
        self._worker_assignment_map = {}

        # Ensure INA placement has been decided
        if not hasattr(self, '_ina_placement_priority') or not self._ina_placement_priority:
            raise RuntimeError("INA placement must be decided before worker assignment.")

        # Extract the actual INA placement switches from the priority list
        ina_placement_switches = set(self._ina_placement_priority)

        # For each job, determine the PS's ToR switch
        for j in range(self.jobs_num):
            self._worker_assignment_map[j] = {}
            ps_node = self.ps_id[j]
            ps_tor_switch = self.sd_id[j]  # ToR switch directly connected to PS

            # Collect all valid aggregation points for this job
            valid_aggregators = {ps_node}  # Always include the PS
            # Add INA-enabled switches that are in the placement and are reachable
            for s in ina_placement_switches:
                if s in self.ina_candidates:
                    valid_aggregators.add(s)

            # For each worker in the job
            for w_idx, w_node in enumerate(self.workers_id[j]):
                best_cost = float('inf')
                best_aggregator = None

                # Evaluate each valid aggregator
                for v in valid_aggregators:
                    # Skip if v is not reachable (should not happen due to topology, but verify)
                    if v == ps_node:
                        # Direct path to PS
                        path = self._get_path_links(w_node, ps_node)
                    else:
                        # Path to INA switch
                        path = self._get_path_links(w_node, v)
                        if not path:
                            continue  # No valid path

                    # Compute path cost: hop count + cross-Pod penalty
                    hop_count = len(path)
                    cross_pod_penalty = 0.0
                    for u, v_edge in path:
                        # Check if this edge crosses Pod boundaries (e.g., ToR -> Aggr)
                        # For simplicity, assume any edge from ToR to Aggr is cross-Pod
                        if u in self.tors_id and v_edge in self.aggrs_id:
                            cross_pod_penalty += 5.0  # Penalty for cross-Pod

                    # Estimate switch processing load if v is an INA switch
                    estimated_load = 0.0
                    if v in ina_placement_switches and v in self.ina_candidates:
                        # Simulate load: assume current total traffic to v
                        # For now, use a simple heuristic: load = number of workers assigned to v
                        # This is a proxy for actual processing load
                        # We'll use a dictionary to track current assignments
                        # We need to avoid overloading
                        pass  # We'll handle this dynamically

                    # Composite cost: hop + cross-Pod + load penalty
                    # Use a load-aware penalty to discourage hotspots
                    load_penalty = 0.0
                    # We'll track current load per switch
                    # Initialize if not present
                    if not hasattr(self, '_switch_load'): 
                        self._switch_load = {}
                    current_load = self._switch_load.get(v, 0.0)
                    # Penalize if switch is near capacity (e.g., > 70% of Cs)
                    if current_load > 0.7 * self.Cs:
                        load_penalty = 10.0  # Strong penalty for near-capacity

                    cost = hop_count + cross_pod_penalty + load_penalty

                    # Prefer INA switch on same ToR as PS
                    if v == ps_tor_switch and v in ina_placement_switches:
                        cost -= 10.0  # Strong bonus for same-ToR INA

                    # Prefer INA switch if it's on the path
                    if v in ina_placement_switches and v != ps_node:
                        cost -= 2.0  # Bonus for using INA

                    # Update best cost
                    if cost < best_cost:
                        best_cost = cost
                        best_aggregator = v

                # Assign worker to best aggregator
                if best_aggregator is None:
                    raise RuntimeError(f"No valid aggregator found for worker {w_node} in job {j}")

                self._worker_assignment_map[j][w_idx] = best_aggregator

                # Update load estimate for the chosen aggregator
                if best_aggregator in self.ina_candidates and best_aggregator in ina_placement_switches:
                    if not hasattr(self, '_switch_load'):
                        self._switch_load = {}
                    self._switch_load[best_aggregator] = self._switch_load.get(best_aggregator, 0.0) + 1.0


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
