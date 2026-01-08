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
        """Selects the most strategically impactful switches for INA deployment based on topological centrality, job connectivity, and load distribution potential. The algorithm prioritizes switches that can reduce PS congestion and core-layer traffic, especially for high-volume jobs with cross-Pod worker distributions. It enforces diversity across Pods and avoids placing INA on switches that would become hotspots."""
        # Step 1: Ensure candidate switches are computed
        self._compute_ina_candidates()

        # Step 2: Initialize scoring dictionary
        switch_scores = {}

        # Helper to get pod ID for a switch
        def get_pod_id(switch):
            # In Fat-Tree, ToRs are grouped by pod; each ToR belongs to one pod
            # Assume each ToR ID is unique per pod, e.g., ToRs in pod 0: [1, 2, 3], pod 1: [4, 5, 6], etc.
            # Here we assume the pod ID is determined by the switch ID modulo the number of ToRs per pod
            # But since we don't have exact pod structure, we use a simple heuristic:
            # For a ToR switch, pod ID = switch_id // (number of ToRs per pod)
            # For Aggr and Core, we assign them to a default pod (e.g., 0) or use a placeholder
            if switch in self.tors_id:
                # Estimate number of ToRs per pod
                num_tors_per_pod = len(self.tors_id) // 3  # Assume 3 pods in total
                if num_tors_per_pod == 0:
                    num_tors_per_pod = 1
                return switch // num_tors_per_pod
            elif switch in self.aggrs_id:
                # Assign Aggr switches to pod 0 by default (they connect to multiple pods)
                return 0
            elif switch in self.cores_id:
                # Assign Core switches to a special pod (e.g., -1) to avoid conflicts
                return -1
            else:
                return 0

        # Step 3: For each candidate switch, compute a composite score
        for s in self.ina_candidates:
            total_job_volume = 0.0
            total_cross_pod_cost = 0.0
            num_jobs_served = 0
            core_hop_count = 0
            path_loads = {}

            # Track the number of jobs and total data volume that would be aggregated at this switch
            for j in range(self.jobs_num):
                job_workers = self.workers_id[j]
                job_size = self.jobs_size[j]
                job_cross_pod = False
                job_aggregated = False

                # Check if any worker in this job has a path through switch s
                for w in job_workers:
                    # Get path from worker w to the PS d_j
                    dst = self.ps_id[j]
                    path = self.allPathDict.get(w, {}).get(dst, [])
                    if not path or s not in path:
                        continue

                    job_aggregated = True
                    # Determine if the path goes through core layer (cross-Pod)
                    is_cross_pod = False
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        if u in self.cores_id or v in self.cores_id:
                            is_cross_pod = True
                            break

                    if is_cross_pod:
                        job_cross_pod = True
                        # Accumulate cost based on number of core hops
                        core_hop_count += 1

                    # Accumulate job volume
                    total_job_volume += job_size
                    num_jobs_served += 1

                # If this job is served via the switch, apply cross-Pod penalty
                if job_cross_pod:
                    total_cross_pod_cost += job_size * 2.0  # Penalty for cross-Pod traffic

            # Apply bonuses: same-ToR switch serving PS gets a bonus
            is_same_tor = False
            for j in range(self.jobs_num):
                if self.sd_id[j] == s:  # s is the ToR directly connected to PS
                    is_same_tor = True
                    break

            # Score components
            # Base score: job volume served
            score = total_job_volume

            # Bonus for same-ToR placement
            if is_same_tor:
                score += total_job_volume * 0.5

            # Penalty for cross-Pod traffic
            score -= total_cross_pod_cost

            # Penalty for high core hop count
            score -= core_hop_count * 10.0

            # Apply diversity penalty if switch is in a pod already heavily used
            # Simple heuristic: count how many INA candidates are in the same pod
            pod_id = get_pod_id(s)
            pod_switches = 0
            for switch in self.ina_candidates:
                if get_pod_id(switch) == pod_id:
                    pod_switches += 1

            # Penalize if too many switches are already in the same pod
            if pod_switches > 2:
                score *= 0.8

            switch_scores[s] = score

        # Step 4: Sort candidates by score in descending order
        sorted_candidates = sorted(switch_scores.items(), key=lambda x: x[1], reverse=True)

        # Step 5: Select up to ina_budget switches, ensuring diversity and avoiding hotspots
        selected_switches = []
        used_pods = set()

        for switch, score in sorted_candidates:
            pod_id = get_pod_id(switch)
            # Avoid selecting multiple switches from the same pod
            if len(selected_switches) >= self.ina_budget:
                break
            if pod_id in used_pods:
                continue

            # Final check: ensure switch can actually handle load
            # Estimate load: total data from jobs served via this switch
            est_load = 0.0
            for j in range(self.jobs_num):
                for w in self.workers_id[j]:
                    path = self.allPathDict.get(w, {}).get(self.ps_id[j], [])
                    if switch in path:
                        est_load += self.jobs_size[j]

            # If estimated load exceeds processing capacity, skip
            if est_load > self.Cs:
                continue

            selected_switches.append(s)
            used_pods.add(pod_id)

        # Step 6: Assign final ordered list to self._ina_placement_priority
        self._ina_placement_priority = selected_switches

    # Function 2: _assign_workers_to_aggregators
    # Role: Assigns each worker to a single aggregation point—either an INA-enabled switch or the PS—while respecting assignment validity and minimizing end-to-end congestion by prioritizing low-latency and high-capacity paths.
    def _assign_workers_to_aggregators(self):
        """
        Assigns each worker to a single aggregation point (either an INA-enabled switch or the PS) based on a composite cost function that balances path length, cross-Pod traffic, and PS ToR proximity.

        The algorithm ensures:
        - Each worker is assigned to exactly one valid aggregation point.
        - Only switches in the final INA placement are considered.
        - Workers prefer local INA switches (same ToR as PS) to reduce PS congestion.
        - Cross-Pod traffic is penalized but only when necessary.
        - Assignment respects physical constraints and avoids invalid nodes.
        """
        # Initialize the worker assignment map
        self._worker_assignment_map = {}

        # Get the actual INA placement switches from the decision method
        ina_placement_set = set(self._ina_placement_priority)

        # Ensure only valid candidates are considered
        valid_aggregation_candidates = set(ina_placement_set) | set(self.ps_id)

        # For each job, assign workers
        for j in range(self.jobs_num):
            self._worker_assignment_map[j] = {}
            job_workers = self.workers_id[j]
            ps_node = self.ps_id[j]
            sd_node = self.sd_id[j]  # ToR switch connected to PS

            # For each worker in the job, compute cost to all valid aggregation points
            worker_costs = {}
            for w_idx, w_node in enumerate(job_workers):
                costs = {}
                # Candidate 1: Direct to PS
                ps_cost = self._compute_path_cost(w_node, ps_node, sd_node)
                costs[ps_node] = ps_cost

                # Candidate 2: To each INA-enabled switch
                for s in ina_placement_set:
                    if s not in self.ina_candidates:
                        continue
                    path_cost = self._compute_path_cost(w_node, s, sd_node)
                    costs[s] = path_cost

                # Find the minimum cost candidate
                min_cost = min(costs.values())
                min_candidate = None
                for node, cost in costs.items():
                    if cost == min_cost:
                        min_candidate = node
                        break

                # Ensure the selected candidate is valid
                if min_candidate not in valid_aggregation_candidates:
                    raise ValueError(f"Worker {w_node} assigned to invalid aggregation node {min_candidate} for job {j}")

                self._worker_assignment_map[j][w_idx] = min_candidate

        # All assignments complete
        return

    def _compute_path_cost(self, src: int, dst: int, ps_tor: int) -> float:
        """
        Computes a composite cost for routing from src to dst.

        Cost components:
        - Base hop count (shortest path length)
        - Cross-Pod penalty: 5.0 if path traverses core layer
        - PS proximity bonus: Strong penalty if dst is not on the same ToR as PS
        """
        path = self.allPathDict[src][dst]
        hops = len(path) - 1

        # Check if path goes through core layer
        cross_pod = False
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if u in self.cores_id or v in self.cores_id:
                cross_pod = True
                break

        # Penalty for cross-Pod
        cross_pod_penalty = 5.0 if cross_pod else 0.0

        # Bonus for same ToR as PS (penalty if not)
        same_tor_penalty = 0.0
        if dst not in self.tors_id:
            # If destination is not a ToR, check if it's on same ToR as PS
            # Get ToR of destination
            dst_tor = None
            if dst in self.tors_id:
                dst_tor = dst
            else:
                for neighbor in self.G.neighbors(dst):
                    if neighbor in self.tors_id:
                        dst_tor = neighbor
                        break
            if dst_tor is None or dst_tor != ps_tor:
                same_tor_penalty = 10.0  # Strong penalty if not on same ToR

        return hops + cross_pod_penalty + same_tor_penalty

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
