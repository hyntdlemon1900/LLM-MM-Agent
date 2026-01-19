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