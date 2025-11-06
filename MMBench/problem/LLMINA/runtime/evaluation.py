from __future__ import annotations
from typing import Any, List, Dict
import numpy as np
import networkx as nx
from topo import FatTree, SpineLeaf
from dataset import generate_dataset_job_random_fattree, generate_dataset_job_sl
from ModelSolve import RelaxSolve
import copy
import signal
from contextlib import contextmanager

# Constants
DEFAULT_INA_CAPACITY = 750  # Gbps
DEFAULT_PS_BANDWIDTH = 200  # Gbps
SOLVER_TIMEOUT = 300  # 超时时间：5分钟 = 300秒


class TimeoutError(Exception):
    """超时异常"""
    pass


@contextmanager
def timeout(seconds):
    """
    超时上下文管理器
    
    Args:
        seconds: 超时时间（秒）
        
    Raises:
        TimeoutError: 如果执行时间超过指定秒数
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(f"执行超时（超过 {seconds} 秒）")
    
    # 设置信号处理器
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # 恢复原来的信号处理器
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

class FixedINAEvaluation:
    def __init__(self,
                 topo_name='FatTree',
                 ina_num_list = [5],
                 jobs_num_list = [10],
                 instances_num = 10
                 ):

        self.topo_name = topo_name
        self.ina_num_list = ina_num_list
        self.jobs_num_list = jobs_num_list # 任务数
        self.instances_num = instances_num

        #fat-tree拓扑参数
        self.k = 4  # k元拓扑
        self.hosts_num_in_fat = 20  # 每个tor（edge交换机）挂的host数

        #spine-leaf拓扑参数
        self.tops_num = 5   # spine交换机数量
        self.tors_num = 10  # leaf交换机数量
        self.hosts_num_in_tor = 20  # 每个tor（leaf交换机）挂的host数


        self.basic_band = 100  # 底层带宽(host和tor交换机)
        self.Cs = 750     # ina交换机处理速度
        self.k1 = 0.5     # 链路容量 k1×40(挂载server数)×basic_band (G)

    def evaluate_program(self) -> Any | None:
        Jct_dict = {}
        for ina_num in self.ina_num_list:
            for jobs_num in self.jobs_num_list:
                self.ina_num = ina_num
                self.jobs_num = jobs_num
                Jct_dict[(self.ina_num,self.jobs_num)] = self.evaluate()
        return Jct_dict

    def evaluate(self) -> float:
        """Evaluate heuristic function on a set of online binpacking instances."""
        if self.topo_name == 'FatTree':
            self.network = FatTree(self.k, self.basic_band, self.hosts_num_in_fat, 'fixed', self.k1)
        if self.topo_name == 'SpineLeaf':
            self.network = SpineLeaf(self.tops_num, self.tors_num, self.basic_band, self.hosts_num_in_tor, 'fixed', self.k1)
        
        all_workers_id = self.network.all_workers_id # server id列表
        
        Jct_list = []
        
        dataset = generate_dataset_job_random_fattree(self.jobs_num, all_workers_id, self.instances_num, True)

        instance_count = 0
        for name in dataset:            
            instance_count += 1

            print(f"\nProcessing instance {instance_count}")
            instance = dataset[name]

            ina_expanded, jobs_expanded= self.solver(instance)

            Jct = evaluate_completion_time(
                instance=instance,
                network=self.network,
                ina_placement=ina_expanded,
                jobs_routing=jobs_expanded,
                verbose=False,
                ina_capacity=self.Cs,
                ps_bandwidth=DEFAULT_PS_BANDWIDTH
            )
            print(f"Makespan (Jct): {Jct}")
            Jct_list.append(Jct)

        return Jct_list

#=======================================================================================================
    def solver(self, instance):
        ina_placement_expanded = self.get_inaid_placement(instance)
        jobs_routing_expanded = self.get_jobs_routing(instance, ina_placement_expanded)
        return ina_placement_expanded, jobs_routing_expanded

    def _get_ina_candidates(self):
        """
        Return sorted list of INA candidate switch IDs based on topology.
        
        For FatTree: ToR + Aggregation + Core switches
        For SpineLeaf: ToR (leaf) + Spine switches
        """
        if self.topo_name == 'FatTree':
            ina_candidates = self.network.tors_id + self.network.aggrs_id + self.network.cores_id
        elif self.topo_name == 'SpineLeaf':
            ina_candidates = self.network.tors_id + self.network.spines_id
        else:
            ina_candidates = self.network.all_switches_id
        return sorted(ina_candidates)

    # Removed normalization helper; the solver now accepts expanded inputs directly

    def get_inaid_placement(self, instance):   
        # Build sorted INA candidate list and compute priorities
        ina_candidates = self._get_ina_candidates()

        priority = {}
        for switch_id in ina_candidates:
            if switch_id in self.network.tors_id:
                label = 1
            elif switch_id in self.network.aggrs_id:
                label = 2
            else:
                label = 3
            priority[switch_id] = self.priority(switch_id, instance, label)

        # Select top-K INA by priority
        sorted_pairs = sorted(priority.items(), key=lambda x: x[1], reverse=True)
        chosen_ids = [sid for sid, _ in sorted_pairs[: self.ina_num]]
        chosen_ids.sort()

        # Map chosen IDs back to indices in candidate order
        id_to_index = {sid: idx for idx, sid in enumerate(ina_candidates)}
        chosen_idx_sorted_by_id = [id_to_index[sid] for sid in chosen_ids]

        # Build expanded x_s over all candidates (aligned to sorted ina_candidates)
        x_s = [0] * len(ina_candidates)
        for idx in chosen_idx_sorted_by_id:
            x_s[idx] = 1

        print("INA功能部署位置:", chosen_ids)
        # Return expanded vector aligned to candidate order
        return x_s

    def priority(self, switch_id, instance, switch_label: int) -> float:
        workers_id = instance['workers_id']
        pses_id = instance['ps_id']
        jobs_size = instance['jobs_size']

        total_transmission_time = 0.0
        total_aggregate_size = 0.0
        disruption_potential = 0.0

        for job_index in range(self.jobs_num):
            job_size = jobs_size[job_index]
            total_aggregate_size += job_size

            ps_id = pses_id[job_index]
            distance_to_ps = nx.shortest_path_length(self.network.G, switch_id, ps_id)

            if distance_to_ps > 0:
                total_transmission_time += (job_size * distance_to_ps)  # time taken to reach PS

            for worker_id in workers_id[job_index]:
                distance_to_worker = nx.shortest_path_length(self.network.G, switch_id, worker_id)

                if distance_to_worker > 0:
                    total_transmission_time += (job_size * distance_to_worker)  # time taken to reach worker
                    disruption_potential += (job_size / distance_to_worker)  # potential traffic reduction effect

        # Adjust total transmission time based on switch role
        if switch_label == 1:  # Edge
            total_transmission_time *= 0.8
        elif switch_label == 2:  # Aggregation
            total_transmission_time *= 0.5
        elif switch_label == 3:  # Core
            total_transmission_time *= 1.2

        # Calculate final score based on aggregation potential and total transmission time
        score = disruption_potential / (total_transmission_time + 1)  # adding 1 to avoid division by zero
        return score

    def get_jobs_routing(self, instance, ina_placement):
        """
        Expect expanded x_s input (length == number of INA candidates, ordered by switch ID).
        Returns expanded routing aligned to candidates.
        """
        # Determine candidate order
        ina_candidates = self._get_ina_candidates()

        # Validate and interpret expanded x_s
        if not isinstance(ina_placement, (list, tuple, np.ndarray)) or len(ina_placement) != len(ina_candidates):
            raise ValueError("get_jobs_routing expects expanded x_s aligned to INA candidates")

        vals = ina_placement.tolist() if isinstance(ina_placement, np.ndarray) else list(ina_placement)
        idx_sorted = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
        chosen_idx = idx_sorted[: self.ina_num]
        selected_indices = sorted(chosen_idx, key=lambda i: ina_candidates[i])
        selected_ids = [ina_candidates[i] for i in selected_indices]

        # Compute compact routing using the compact INA ID list
        method = RelaxSolve(self.ina_num, self.jobs_num, self.Cs, self.network, instance)
        y_compact, y_d, Jct_initial, bestjct = method.test_routing(selected_ids)

        # Expand y_compact into full candidate dimension aligned to ina_candidates
        jobs_num = self.jobs_num
        if isinstance(y_d, np.ndarray):
            max_workers = y_d.shape[1] if y_d.ndim >= 2 else (y_d.shape[0] if y_d.ndim == 1 else 0)
        elif isinstance(y_d, list):
            max_workers = len(y_d[0]) if y_d else 0
        else:
            max_workers = 0
        total_cands = len(ina_candidates)

        # Initialize zeros
        y_full = [
            [ [0] * total_cands for _ in range(max_workers) ]
            for _ in range(jobs_num)
        ]

        # Place compact columns into selected candidate positions
        for j in range(jobs_num):
            for w in range(max_workers):
                for k, cand_idx in enumerate(selected_indices):
                    y_full[j][w][cand_idx] = y_compact[j][w][k]

        expanded_jobs = [y_full, y_d]
        return expanded_jobs


def validate_solution_constraints(
    ina_placement_expanded,
    jobs_routing_expanded,
    instance,
    network,
    K,
    jobs_num,
    Cs,
    topo_name='FatTree'
):
    """
    验证求解方案的核心约束
    
    检查三个核心约束：
    1. INA数量不超过K
    2. 每个worker有且仅有一个聚合点
    3. Worker只能选择已部署INA的交换机 (y_j_w_s <= x_s)
    
    Args:
        ina_placement_expanded: INA部署方案 (expanded x_s vector)
        jobs_routing_expanded: 作业路由方案 [y_ina, y_ps]
        instance: 问题实例
        network: 网络拓扑
        K: INA预算
        jobs_num: 作业数量
        Cs: INA处理容量（未使用）
        topo_name: 拓扑名称
        
    Raises:
        ValueError: 如果违反任何核心约束，抛出详细错误信息
    """
    
    # 获取INA候选列表
    if topo_name == 'FatTree':
        ina_candidates = sorted(network.tors_id + network.aggrs_id + network.cores_id)
    elif topo_name == 'SpineLeaf':
        ina_candidates = sorted(network.tors_id + network.spines_id)
    else:
        ina_candidates = sorted(network.all_switches_id)
    
    # 解析输入
    workers_id = instance['workers_id']
    y_ina, y_ps = jobs_routing_expanded
    
    # ============================================================================
    # 约束1: INA数量不超过K
    # ============================================================================
    num_ina_deployed = sum(ina_placement_expanded)
    if num_ina_deployed > K + 1e-6:  # 允许小的数值误差
        raise ValueError(
            f"❌ Constraint Violation: INA数量超过预算\n"
            f"   约束: INA数量 ≤ {K}\n"
            f"   实际: {num_ina_deployed} 个INA\n"
            f"   超出: {num_ina_deployed - K} 个\n"
            f"   已部署的交换机: {[ina_candidates[i] for i, x in enumerate(ina_placement_expanded) if x > 0.5]}"
        )
    
    # ============================================================================
    # 约束2: 每个worker有且仅有一个聚合点
    # ============================================================================
    for j in range(jobs_num):
        for w_idx, worker_id in enumerate(workers_id[j]):
            # 统计worker选择的聚合点数量
            ina_selections = sum(y_ina[j][w_idx])  # 选择INA的数量
            ps_selection = y_ps[j][w_idx] if isinstance(y_ps[j][w_idx], (int, float)) else sum(y_ps[j][w_idx])
            
            total_selections = ina_selections + ps_selection
            
            if abs(total_selections - 1.0) > 1e-6:
                raise ValueError(
                    f"❌ Constraint Violation: Worker聚合点选择错误\n"
                    f"   约束: 每个worker必须选择恰好1个聚合点\n"
                    f"   Job: {j}, Worker: {worker_id} (索引 {w_idx})\n"
                    f"   实际: {total_selections:.4f} 个聚合点\n"
                    f"   分解: {ina_selections:.4f} (INA) + {ps_selection:.4f} (PS)"
                )
    
    # ============================================================================
    # 约束3: Worker只能选择已部署INA的交换机 (y_j_w_s <= x_s)
    # ============================================================================
    for j in range(jobs_num):
        for w_idx, worker_id in enumerate(workers_id[j]):
            for s_idx in range(len(ina_placement_expanded)):
                # 如果交换机s没有部署INA (x_s = 0)，则worker不能选择它 (y_j_w_s = 0)
                if ina_placement_expanded[s_idx] < 0.5 and y_ina[j][w_idx][s_idx] > 1e-6:
                    raise ValueError(
                        f"❌ Constraint Violation: Worker选择了未部署INA的交换机\n"
                        f"   约束: y_j_w_s[j][w][s] ≤ x_s[s] (只能选择已部署INA的交换机)\n"
                        f"   Job: {j}, Worker: {worker_id} (索引 {w_idx})\n"
                        f"   交换机: {ina_candidates[s_idx]} (索引 {s_idx})\n"
                        f"   x_s[{s_idx}] = {ina_placement_expanded[s_idx]:.4f} (未部署)\n"
                        f"   y_j_w_s[{j}][{w_idx}][{s_idx}] = {y_ina[j][w_idx][s_idx]:.4f} (不应选择)"
                    )
    
    print("✓ 核心约束验证通过")


def test_llm_solver(solver_func, topo_name='FatTree', ina_num_list=[3], jobs_num_list=[10], instances_num=10):
    """
    测试LLM生成的solver函数，包含完整的约束验证
    
    Args:
        solver_func: LLM生成的solver函数，签名为 solver_func(instance, network, K, jobs_num, Cs, topo_name)
        topo_name: 拓扑名称
        ina_num_list: INA数量列表
        jobs_num_list: 作业数量列表
        instances_num: 实例数量
    
    Returns:
        评估结果字典
        
    Raises:
        ValueError: 如果生成的解违反任何LLMINA约束
    """
    class LLMINAEvaluation(FixedINAEvaluation):
        def __init__(self, solver_func, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.llm_solver_func = solver_func
        
        def solver(self, instance):
            """使用LLM生成的solver并验证约束（含超时控制）"""
            try:
                # 使用超时控制调用LLM solver
                with timeout(SOLVER_TIMEOUT):
                    ina_placement_expanded, jobs_routing_expanded = self.llm_solver_func(
                        instance=instance,
                        network=self.network,
                        K=self.ina_num,
                        jobs_num=self.jobs_num,
                        Cs=self.Cs,
                        topo_name=self.topo_name
                    )
            except TimeoutError as e:
                print(f"\n{'='*80}")
                print("❌ SOLVER TIMEOUT")
                print(f"{'='*80}")
                print(f"求解器执行超时（限制：{SOLVER_TIMEOUT}秒 = {SOLVER_TIMEOUT//60}分钟）")
                print(f"{'='*80}\n")
                raise ValueError(f"Solver timeout after {SOLVER_TIMEOUT} seconds") from e
            
            # 验证约束
            try:
                validate_solution_constraints(
                    ina_placement_expanded=ina_placement_expanded,
                    jobs_routing_expanded=jobs_routing_expanded,
                    instance=instance,
                    network=self.network,
                    K=self.ina_num,
                    jobs_num=self.jobs_num,
                    Cs=self.Cs,
                    topo_name=self.topo_name
                )
            except ValueError as e:
                print(f"\n{'='*80}")
                print("约束验证失败")
                print(f"{'='*80}")
                print(str(e))
                print(f"{'='*80}\n")
                raise
            
            return ina_placement_expanded, jobs_routing_expanded
    
    evaluator = LLMINAEvaluation(
        solver_func=solver_func,
        topo_name=topo_name,
        ina_num_list=ina_num_list,
        jobs_num_list=jobs_num_list,
        instances_num=instances_num
    )
    
    return evaluator.evaluate_program()

def get_ina_candidates(network) -> List[int]:
    """
    Get candidate INA switch IDs based on topology type.
    
    According to LLMINA formulation, INA can be deployed on switches (set S).
    For FatTree: ToR (edge) + Aggregation + Core switches
    For SpineLeaf: ToR (leaf) + Spine switches
    
    Args:
        network: Network object with switch ID attributes:
            - FatTree: tors_id, aggrs_id, cores_id, all_switches_id
            - SpineLeaf: tors_id, spines_id, all_switches_id
    
    Returns:
        Sorted list of candidate switch IDs (ascending order)
    """
    # Check topology type by examining available attributes
    has_fattree_attrs = (hasattr(network, 'tors_id') and 
                         hasattr(network, 'aggrs_id') and 
                         hasattr(network, 'cores_id'))
    has_spineleaf_attrs = (hasattr(network, 'tors_id') and 
                           hasattr(network, 'spines_id'))
    
    if has_fattree_attrs:
        # FatTree: ToR + Aggregation + Core
        candidates = list(network.tors_id) + list(network.aggrs_id) + list(network.cores_id)
    elif has_spineleaf_attrs:
        # SpineLeaf: ToR (leaf) + Spine
        candidates = list(network.tors_id) + list(network.spines_id)
    elif hasattr(network, 'all_switches_id'):
        # Fallback: use all switches
        candidates = list(network.all_switches_id)
    
    return sorted(candidates)


def links_set(src: int, dst: int, allPathDict: Dict) -> List[tuple]:
    """
    Get all links (edges) along the path from src to dst.
    
    Args:
        src: Source node ID
        dst: Destination node ID
        allPathDict: Dictionary mapping [src][dst] to path [src, ..., dst]
    
    Returns:
        List of link tuples [(node_i, node_i+1), ...]
    """
    path = allPathDict[src][dst]
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def path_to_links_dict(src_set: List[int], dst_set: List[int], allPathDict: Dict) -> Dict[tuple, List[tuple]]:
    """
    Create a dictionary mapping (src, dst) pairs to their link sets.
    
    Args:
        src_set: List of source node IDs
        dst_set: List of destination node IDs
        allPathDict: Dictionary mapping [src][dst] to path
    
    Returns:
        Dict mapping (src, dst) -> list of links
    """
    return {
        (src, dst): links_set(src, dst, allPathDict)
        for src in src_set
        for dst in dst_set
    }


def l_i(edge: tuple, src_set: List[int], dst_set: List[int], allPathDict: Dict) -> List[tuple]:
    """
    Find all (src, dst) pairs whose paths contain edge e.
    
    Args:
        edge: Edge tuple (u, v)
        src_set: List of source node IDs
        dst_set: List of destination node IDs
        allPathDict: Dictionary mapping [src][dst] to path
    
    Returns:
        List of (row_index, col_index) tuples indicating which (src, dst) pairs use edge e
    """
    path_links_dict = path_to_links_dict(src_set, dst_set, allPathDict)
    edge_reversed = edge[::-1]
    
    index_set = []
    for (src, dst), links in path_links_dict.items():
        if edge in links or edge_reversed in links:
            row_idx = src_set.index(src)
            col_idx = dst_set.index(dst)
            index_set.append((row_idx, col_idx))
    
    return index_set


def link_capacity_limitation(job_idx: int, edge: tuple, workers_id: List[List[int]], 
                            ina_id: List[int], ps_id: List[int], allPathDict: Dict) -> tuple:
    """
    Identify which flows use edge e for job j.
    
    Args:
        job_idx: Job index
        edge: Edge tuple (u, v)
        workers_id: Worker IDs per job [jobs_num][workers_num[j]]
        ina_id: List of selected INA switch IDs
        ps_id: Parameter server IDs per job [jobs_num]
        allPathDict: Dictionary mapping [src][dst] to path
    
    Returns:
        Tuple of (i_j_w_s, i_j_w_d, i_s_d):
        - i_j_w_s: Worker-to-INA flows using edge e
        - i_j_w_d: Worker-to-PS flows using edge e
        - i_s_d: INA-to-PS flows using edge e
    """
    workers_to_ina = l_i(edge, workers_id[job_idx], ina_id, allPathDict)
    workers_to_ps = l_i(edge, workers_id[job_idx], [ps_id[job_idx]], allPathDict)
    ina_to_ps = l_i(edge, ina_id, [ps_id[job_idx]], allPathDict)
    
    return workers_to_ina, workers_to_ps, ina_to_ps


def evaluate_completion_time(
    instance: dict, 
    network, 
    ina_placement: List[int], 
    jobs_routing: List, 
    verbose: bool = False,
    ina_capacity = 750,
    ps_bandwidth = 200,
) -> float:
    """
    Evaluate makespan for given INA placement and routing decisions.
    
    Solves a reduced LP using Gurobi to compute optimal transmission rates.
    
    Args:
        instance: Dict with workers_id, ps_id, workers_num, jobs_size
        network: Object with G, allPathDict, bandwidth_mapping
        ina_placement: Binary vector [len(ina_candidates)], aligned to sorted candidates
        jobs_routing: [y_j_w_s_full, y_j_w_d] where
            - y_j_w_s_full: [jobs_num][max_workers][len(ina_candidates)]
            - y_j_w_d: [jobs_num][max_workers]
        verbose: Print detailed info
    
    Returns:
        Makespan (float), or inf if infeasible
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as e:
        if verbose:
            print(f"Gurobi import error: {e}")
        return float('inf')
    
    try:
        # Setup
        ina_num = sum(ina_placement)
        jobs_num = len(instance['jobs_size'])        
        workers_id = instance['workers_id']
        ps_id = instance['ps_id']
        workers_num = instance['workers_num']
        jobs_size = instance['jobs_size']
        workers_num_align = max(workers_num)
        
        G = copy.deepcopy(network.G)
        mapping = copy.deepcopy(network.mapping) if hasattr(network, 'mapping') else {}
        allPathDict = copy.deepcopy(network.allPathDict)
        sd_id = [list(G.adj._atlas[ps_id[j]].keys())[0] for j in range(jobs_num)]
        
        # Normalize inputs
        ina_candidates = get_ina_candidates(network)
        if len(ina_placement) == len(ina_candidates):
            # Expanded format - convert to compact
            vals = list(ina_placement)
            idx_sorted = sorted(range(len(vals)), key=lambda i: (vals[i], ina_candidates[i]), reverse=True)
            chosen_idx = idx_sorted[:ina_num]
            chosen_idx_sorted_by_id = sorted(chosen_idx, key=lambda i: ina_candidates[i])
            ina_id = [ina_candidates[i] for i in chosen_idx_sorted_by_id]
            
            # Convert routing to compact format
            y_full, y_d = jobs_routing
            max_workers_val = len(y_d[0]) if y_d else 0
            y_compact = [
                [[y_full[j][w][i] for i in chosen_idx_sorted_by_id] for w in range(max_workers_val)]
                for j in range(jobs_num)
            ]
            jobs_routing = [y_compact, y_d]
        else:
            ina_id = sorted(list(ina_placement))
        
        # Create model
        m = gp.Model()
        m.Params.LogToConsole = 0
        
        # Add variables
        gamma_j = m.addMVar(shape=(jobs_num,), vtype=GRB.CONTINUOUS)
        gamma_j_s_d = m.addMVar(shape=(jobs_num, ina_num), vtype=GRB.CONTINUOUS)
        gamma_j_w_s = m.addMVar(shape=(jobs_num, workers_num_align, ina_num), vtype=GRB.CONTINUOUS)
        gamma_j_w_d = m.addMVar(shape=(jobs_num, workers_num_align), vtype=GRB.CONTINUOUS)
        x_s = m.addMVar(shape=(ina_num,), vtype=GRB.BINARY)
        obj = m.addVar(vtype=GRB.CONTINUOUS)
        
        # Set objective
        m.setObjective(obj, GRB.MAXIMIZE)
        
        y_j_w_s = jobs_routing[0]
        y_j_w_d = jobs_routing[1]
        
        # Add constraints
        for j in range(jobs_num):
            for w in range(workers_num_align):
                for s in range(ina_num):
                    m.addConstr(gamma_j_w_s[j, w, s] <= ina_capacity * y_j_w_s[j][w][s])
                m.addConstr(gamma_j_w_d[j, w] <= ps_bandwidth * y_j_w_d[j][w])
        
        for j in range(jobs_num):
            for w in range(workers_num[j], workers_num_align):
                for s in range(ina_num):
                    m.addConstr(gamma_j_w_s[j, w, s] == 0)
                m.addConstr(gamma_j_w_d[j, w] == 0)
        
        m.addConstrs((gamma_j_w_d[j, w] + sum(gamma_j_w_s[j, w, _] for _ in range(ina_num))
                      >= gamma_j[j] for j in range(jobs_num) for w in range(workers_num[j])))
        
        m.addConstrs(obj <= gamma_j[j] / jobs_size[j] for j in range(jobs_num))
        
        m.addConstr(sum(x_s[i] for i in range(ina_num)) <= ina_num)
        m.addConstrs(x_s[i] == 1 for i in range(ina_num))
        
        for j in range(jobs_num):
            for w in range(workers_num_align):
                for s in range(ina_num):
                    m.addConstr(gamma_j_w_s[j, w, s] <= ina_capacity * y_j_w_s[j][w][s] * x_s[s])
        
        m.addConstrs((sum(gamma_j_w_s[j, w, i] for j in range(jobs_num) for w in range(workers_num[j])) 
                      <= ina_capacity * x_s[i] for i in range(ina_num)))
        
        m.addConstrs((gamma_j_w_s[j, w, i] <= gamma_j_s_d[j, i] 
                      for j in range(jobs_num) for w in range(workers_num[j]) for i in range(ina_num)))
        
        # Link capacity constraints
        for link_info in mapping.values():
            edge = link_info[0]
            link_capacity = link_info[1]
            
            link_loads = [0] * jobs_num
            for j in range(jobs_num):
                workers_to_ina, workers_to_ps, ina_to_ps = link_capacity_limitation(
                    j, edge, workers_id, ina_id, ps_id, allPathDict
                )
                
                load_w_to_ina = sum(gamma_j_w_s[j, idx[0], idx[1]] for idx in workers_to_ina)
                load_w_to_ps = sum(gamma_j_w_d[j, idx[0]] for idx in workers_to_ps)
                load_ina_to_ps = sum(gamma_j_s_d[j, idx[0]] for idx in ina_to_ps)
                
                link_loads[j] = load_w_to_ina + load_w_to_ps + load_ina_to_ps
                
                if edge == (sd_id[j], ps_id[j]) or edge == (ps_id[j], sd_id[j]):
                    link_capacity = ps_bandwidth
            
            m.addConstr(sum(link_loads) <= link_capacity)
        
        m.optimize()
        
        if m.status == GRB.OPTIMAL:
            makespan_result = 1 / obj.X
            return makespan_result
        else:
            if verbose:
                print(f"Optimization not optimal. Status: {m.status}")
            return float('inf')
            
    except (gp.GurobiError, ValueError, KeyError) as e:
        if verbose:
            print(f"Optimization error: {e}")
        return float('inf')

if __name__ == "__main__":
    evaluator = FixedINAEvaluation(
        topo_name='FatTree',
        ina_num_list=[3],
        jobs_num_list=[6],
        instances_num=1
    )
    results = evaluator.evaluate_program()
    print("Evaluation Results:", results)