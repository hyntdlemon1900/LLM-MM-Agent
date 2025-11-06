def llm_solver(instance, network, K, jobs_num, Cs, topo_name):
  """
  LLMINA solver interface.
  
  Args:
      instance: dict
          - workers_id: List[List[int]] - Worker node IDs, shape [jobs_num][workers_num[j]]
            Example: [[0,1,2], [3,4,5,6]] means job 0 has workers 0,1,2
          - ps_id: List[int] - Parameter Server IDs, shape [jobs_num]
          - workers_num: List[int] - Worker counts per job, shape [jobs_num]
          - jobs_size: List[float] - Gradient volumes (m_j), shape [jobs_num]
          - tor_switch_workers: List[Dict[int,int]] - Worker distribution per tor switch
            Shape [jobs_num][edge_switch_id -> worker_count]
      
      network: object
          - G: networkx.Graph - Network topology
          - allPathDict: Dict[src][dst] -> List[int] - Precomputed paths [src, ..., dst]
            Usage: path = network.allPathDict[src][dst], hops = len(path) - 1
          - bandwidth_mapping: Dict[(u,v)] -> float - Link capacity C_e^bw (Gbps)
          - Switch ID sets:
            * tors_id: Edge switches (connect to workers, edge switches in fattree, leaf switches in spineleaf)
            * aggrs_id: Aggregation switches (FatTree only)
            * cores_id: Core switches (FatTree only)
            * spines_id: Spine switches (SpineLeaf only)
            * all_switches_id: All switches
          - Topology: FatTree (workers->edges->aggrs->cores), SpineLeaf (workers->leafs->spines)
      
      K: int - INA budget (corresponds to K in problem formulation)
      jobs_num: int - Number of concurrent jobs
      Cs: float - INA processing capacity C_s (e.g., 750 Gbps)
      topo_name: str - 'FatTree' or 'SpineLeaf'
  
  Returns:
      (ina_placement_expanded, jobs_routing_expanded):
      
      ina_placement_expanded: List[int]
          Binary vector, length len(ina_candidates), candidates sorted by ID
          Value 1 = selected, 0 = not selected (corresponds to x_s)
          Constraint: sum(ina_placement_expanded) == K
      
      jobs_routing_expanded: [y_j_w_s_full, y_j_w_d]
          y_j_w_s_full: List[List[List[int]]]
              Shape [jobs_num][max_workers][len(ina_candidates)]
              y_j_w_s_full[j][w][s]=1: worker w of job j routes to INA s (y_{jws})
          
          y_j_w_d: List[List[int]]
              Shape [jobs_num][max_workers]
              y_j_w_d[j][w]=1: worker w of job j routes directly to PS (y_{jw d_j})
          
          max_workers = max(workers_num)
          Constraints:
          - Active (w < workers_num[j]): sum(y_j_w_s_full[j][w]) + y_j_w_d[j][w] == 1
          - Inactive (w >= workers_num[j]): all entries = 0 (zero-padding)
  
  """
  
  # Extract instance data
  workers_id = instance['workers_id']
  ps_id = instance['ps_id']
  workers_num = instance['workers_num']
  jobs_size = instance['jobs_size']
  
  # Get sorted candidate INA switches
  ina_candidates = get_ina_candidates(network)
  max_workers = max(workers_num)
  num_candidates = len(ina_candidates)
  
  # Initialize output (default: all zeros - REPLACE WITH YOUR SOLUTION)
  ina_placement_expanded = [0] * num_candidates
  y_j_w_s_full = [[[0] * num_candidates for _ in range(max_workers)] for _ in range(jobs_num)]
  y_j_w_d = [[0] * max_workers for _ in range(jobs_num)]
  jobs_routing_expanded = [y_j_w_s_full, y_j_w_d]
  
  # TODO: Implement your solver logic
  # - Select K switches from ina_candidates
  # - Assign each worker to INA or PS
  # - Satisfy all constraints (K switches, one path per worker, capacities)

  
  return ina_placement_expanded, jobs_routing_expanded
