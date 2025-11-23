from __future__ import annotations
from typing import List, Dict, Tuple, Any, Optional
import copy
import pyomo.environ as pyo
from MMBench.problem.problem_template.runtime.TemplateSolver import TemplateSolver

class ModelSolver(TemplateSolver):
    """
    LLMINA 问题的 MILP 求解器

    目标：最小化 makespan（最大完成时间）

    决策变量：
    - INA placement：在哪些交换机上部署 INA（预算 K）
    - Worker assignment：
        * 每个 worker 选择聚合点（INA 或 PS）
    - Rate allocation：
        * 每条流的传输速率（worker→INA / worker→PS / INA→PS）

    约束（概念层面）：
    - INA 部署预算
    - worker 唯一分配
    - INA 处理容量
    - 链路带宽
    - 各类速率一致性 / 流量守恒
    - makespan 约束
    """

    # ========= 初始化与预处理 =========

    def __init__(
        self,
        instance: dict,
        network,
        K: int,
        jobs_num: int,
        Cs: float = 750.0,
        Ps: float = 200.0,
        topo_name: str = "FatTree",
    ):
        """
        初始化 LLMINA MILP 求解器

        Args:
            instance:
                - workers_id: List[List[int]]    # [[job0_w0, job0_w1, ...], [job1_w0, ...]]
                - ps_id: List[int]               # 每个 job 的 PS 节点 ID
                - workers_num: List[int]         # 每个 job 的 worker 数量
                - jobs_size: List[float]         # 每个 job 的梯度数据量 m_j
                - tor_switch_workers: List[Dict[int,int]]  # ToR 下的 worker 分布（可选）

            network:
                - G: networkx.Graph
                - allPathDict: Dict[src][dst] -> List[int]  # 预计算路径（节点序列）
                - bandwidth_mapping: Dict[(u, v)] -> float   # 链路带宽 (Gbps)
                - 交换机 ID 集：
                    * FatTree: tors_id, aggrs_id, cores_id
                    * SpineLeaf: tors_id, spines_id
                    * 或通用 all_switches_id

            K: INA 部署预算（最多部署 K 个 INA 交换机）
            jobs_num: 并发任务数量
            Cs: INA 交换机处理容量 (Gbps)
            Ps: PS 端口带宽 (Gbps)
            topo_name: "FatTree" 或 "SpineLeaf" 或其他
        """
        self.problem_data = {
            "instance": instance,
            "network": network,
            "K": K,
            "jobs_num": jobs_num,
            "Cs": Cs,
            "Ps": Ps,
            "topo_name": topo_name,
        }

        self.model = None
        self.solution = None

        self._preprocess_data()

    def _preprocess_data(self) -> None:
        """
        数据预处理（重写父类钩子方法）：

        1. 深拷贝网络拓扑数据（避免污染原始 network）
        2. 确定 INA 候选交换机集合
        3. 提取每个 job 的 PS 相邻交换机（最后一跳带宽用 Ps）
        """
        network = self.problem_data["network"]
        jobs_num = self.problem_data["jobs_num"]
        ps_id = self.problem_data["instance"]["ps_id"]

        # 1) 深拷贝拓扑结构
        self.G = copy.deepcopy(network.G)
        self.allPathDict = copy.deepcopy(network.allPathDict)
        self.mapping = copy.deepcopy(network.mapping) if hasattr(network, "mapping") else {}

        # 2) INA 候选交换机
        self.ina_candidates: List[int] = self._get_ina_candidates()
        self.ina_candidates_num: int = len(self.ina_candidates)

        # 3) 每个 PS 的相邻交换机（用于“PS 最后一跳”带宽约束）
        self.sd_id: List[int] = [
            list(self.G.adj._atlas[ps_id[j]].keys())[0]  # PS 相邻的唯一交换机
            for j in range(jobs_num)
        ]

    # ========= 一些基础工具函数 =========

    def _get_ina_candidates(self) -> List[int]:
        """
        根据拓扑类型确定 INA 候选交换机集合。

        Returns:
            排序后的候选交换机 ID 列表。

        Raises:
            ValueError: 如果无法确定候选交换机集合。
        """
        network = self.problem_data["network"]
        topo_name = self.problem_data["topo_name"]

        # FatTree: ToR + Aggregation + Core
        if topo_name == "FatTree" or (
            hasattr(network, "tors_id")
            and hasattr(network, "aggrs_id")
            and hasattr(network, "cores_id")
        ):
            candidates = (
                list(network.tors_id)
                + list(network.aggrs_id)
                + list(network.cores_id)
            )

        # SpineLeaf: ToR(Leaf) + Spine
        elif topo_name == "SpineLeaf" or (
            hasattr(network, "tors_id") and hasattr(network, "spines_id")
        ):
            candidates = list(network.tors_id) + list(network.spines_id)

        # 通用：所有交换机
        elif hasattr(network, "all_switches_id"):
            candidates = list(network.all_switches_id)

        else:
            raise ValueError("无法从网络拓扑确定 INA 候选交换机集合")

        return sorted(candidates)

    def _get_path_links(self, src: int, dst: int) -> List[Tuple[int, int]]:
        """
        获取从 src 到 dst 路径上的所有链路 (u, v) 列表。
        """
        path = self.allPathDict[src][dst]
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    def _get_flows_on_link(
        self,
        edge: Tuple[int, int],
        src_set: List[int],
        dst_set: List[int],
    ) -> List[Tuple[int, int]]:
        """
        查找使用指定链路的所有流 (src_idx, dst_idx)，用于构造链路容量约束。

        Args:
            edge: 链路 (u, v)
            src_set: 源节点 ID 列表
            dst_set: 目标节点 ID 列表

        Returns:
            (src_idx, dst_idx) 对列表，索引是对 src_set / dst_set 的索引。
        """
        edge_reversed = edge[::-1]
        flows: List[Tuple[int, int]] = []

        for src_idx, src in enumerate(src_set):
            for dst_idx, dst in enumerate(dst_set):
                links = self._get_path_links(src, dst)
                if edge in links or edge_reversed in links:
                    flows.append((src_idx, dst_idx))

        return flows

    # ========= 建模主流程 =========

    def build_model(self) -> None:
        """
        构建完整的Pyomo MILP模型
        
        重写父类方法，创建LLMINA问题的优化模型：
        
        集合：
        - J: Jobs [0, jobs_num-1]
        - S: INA候选交换机 [0, ina_candidates_num-1]
        - W_align: Workers（对齐到最大数量）
        
        变量：
        - x_s: Binary - INA部署决策
        - y_jws: Binary - Worker到INA分配
        - y_jwd: Binary - Worker到PS分配
        - gamma_j: Continuous - Job有效速率
        - gamma_jws: Continuous - Worker到INA的速率
        - gamma_jwd: Continuous - Worker到PS的速率
        - gamma_js: Continuous - INA到PS的速率
        - alpha: Continuous - 逆makespan（优化目标）
        
        目标函数：
        - Maximize alpha (即 Minimize makespan = 1/alpha)
        
        约束：
        1. INA部署预算: Σx_s <= K
        2. Worker唯一分配: Σy_jws + y_jwd = 1
        3. INA可用性: y_jws <= x_s
        4. Dummy worker清零: gamma_jws[w>=workers_num[j]] = 0
        5. Big-M耦合: gamma_jws <= M*y_jws, gamma_jwd <= M*y_jwd
        6. INA处理容量: Σgamma_jws <= Cs*x_s
        7. Worker速率一致性: Σgamma_jws + gamma_jwd >= gamma_j
        8. INA出口速率一致性: gamma_js >= gamma_jws (for all w)
        9. INA流量守恒: gamma_js <= Σgamma_jws
        10. PS流入充分性: gamma_j <= Σgamma_js + Σgamma_jwd
        11. Makespan约束: alpha <= gamma_j / m_j
        12. 链路容量约束: Σ(flows on link) <= capacity
        """

        # 从problem_data中读取常用数据
        jobs_num = self.problem_data['jobs_num']
        workers_num = self.problem_data['instance']['workers_num']
        workers_num_align = max(self.problem_data['instance']['workers_num'])
        jobs_size = self.problem_data['instance']['jobs_size']
        workers_id = self.problem_data['instance']['workers_id']
        ps_id = self.problem_data['instance']['ps_id']
        K = self.problem_data['K']
        Cs = self.problem_data['Cs']
        Ps = self.problem_data['Ps']
        
        model = pyo.ConcreteModel(name="LLMINA_INA_Placement_and_Routing")
        
        # ================== 集合定义 ==================
        model.J = pyo.RangeSet(0, jobs_num - 1)
        model.S = pyo.RangeSet(0, self.ina_candidates_num - 1)
        model.W_align = pyo.RangeSet(0, workers_num_align - 1)
        
        # Per-job worker集合（用于仅针对实际worker的约束）
        model.W = {}
        for j in range(jobs_num):
            model.W[j] = pyo.RangeSet(0, workers_num[j] - 1)
        
        # ================== 变量定义 ==================
        
        # Binary: INA部署决策
        model.x_s = pyo.Var(model.S, domain=pyo.Binary, doc="INA部署在交换机s上")
        
        # Binary: Worker-to-INA分配
        model.y_jws = pyo.Var(
            model.J, model.W_align, model.S, 
            domain=pyo.Binary, 
            doc="Job j的Worker w使用INA交换机s"
        )
        
        # Binary: Worker-to-PS直连
        model.y_jwd = pyo.Var(
            model.J, model.W_align, 
            domain=pyo.Binary, 
            doc="Job j的Worker w直连到PS"
        )
        
        # Continuous: 速率变量
        model.gamma_j = pyo.Var(
            model.J, 
            domain=pyo.NonNegativeReals, 
            doc="Job j的有效速率"
        )
        
        model.gamma_jws = pyo.Var(
            model.J, model.W_align, model.S, 
            domain=pyo.NonNegativeReals, 
            doc="Worker w到INA交换机s的速率（Job j）"
        )
        
        model.gamma_jwd = pyo.Var(
            model.J, model.W_align, 
            domain=pyo.NonNegativeReals, 
            doc="Worker w到PS的速率（Job j）"
        )
        
        model.gamma_js = pyo.Var(
            model.J, model.S, 
            domain=pyo.NonNegativeReals, 
            doc="INA交换机s到PS的出口速率（Job j）"
        )
        
        # Auxiliary: alpha = 1/makespan（优化目标）
        model.alpha = pyo.Var(domain=pyo.NonNegativeReals, doc="逆makespan")
        
        # ================== 目标函数 ==================
        def obj_rule(m):
            # Maximize alpha = 1/makespan (即 minimize makespan)
            return m.alpha
        
        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
        
        # ================== 约束定义 ==================
        
        # 约束1: INA部署预算
        def ina_budget_rule(m):
            return sum(m.x_s[s] for s in m.S) <= K
        
        model.ina_budget = pyo.Constraint(rule=ina_budget_rule, doc="最多部署K个INA交换机")
        
        # 约束2: Worker聚合点唯一选择
        def worker_choice_rule(m, j, w):
            if w >= workers_num[j]:
                return pyo.Constraint.Skip  # 跳过dummy worker
            return (
                sum(m.y_jws[j, w, s] for s in m.S) + m.y_jwd[j, w] == 1
            )
        
        model.worker_choice = pyo.Constraint(
            model.J, model.W_align, 
            rule=worker_choice_rule, 
            doc="每个worker必须选择恰好一个聚合点"
        )
        
        # 约束3: INA可用性（只能使用已部署的INA）
        def ina_deployment_rule(m, j, w, s):
            return m.y_jws[j, w, s] <= m.x_s[s]
        
        model.ina_deployment = pyo.Constraint(
            model.J, model.W_align, model.S, 
            rule=ina_deployment_rule, 
            doc="Worker只能使用已部署INA的交换机"
        )
        
        # 约束4: Dummy worker清零
        def zero_dummy_workers_ina_rule(m, j, w, s):
            if w >= workers_num[j]:
                return m.gamma_jws[j, w, s] == 0
            return pyo.Constraint.Skip
        
        model.zero_dummy_ina = pyo.Constraint(
            model.J, model.W_align, model.S, 
            rule=zero_dummy_workers_ina_rule,
            doc="Dummy worker到INA的速率为0"
        )
        
        def zero_dummy_workers_ps_rule(m, j, w):
            if w >= workers_num[j]:
                return m.gamma_jwd[j, w] == 0
            return pyo.Constraint.Skip
        
        model.zero_dummy_ps = pyo.Constraint(
            model.J, model.W_align, 
            rule=zero_dummy_workers_ps_rule,
            doc="Dummy worker到PS的速率为0"
        )
        
        # 约束5: Big-M耦合（速率-分配绑定）
        M_ina = Cs  # INA速率上界
        def bigm_ina_rule(m, j, w, s):
            return m.gamma_jws[j, w, s] <= M_ina * m.y_jws[j, w, s]
        
        model.bigm_ina = pyo.Constraint(
            model.J, model.W_align, model.S, 
            rule=bigm_ina_rule, 
            doc="只有分配到INA时才有速率"
        )
        
        M_ps = Ps  # PS速率上界
        def bigm_ps_rule(m, j, w):
            return m.gamma_jwd[j, w] <= M_ps * m.y_jwd[j, w]
        
        model.bigm_ps = pyo.Constraint(
            model.J, model.W_align, 
            rule=bigm_ps_rule, 
            doc="只有分配到PS时才有速率"
        )
        
        # 约束6: INA交换机处理容量
        def ina_capacity_rule(m, s):
            return (
                sum(m.gamma_jws[j, w, s] 
                    for j in m.J 
                    for w in range(workers_num[j])) 
                <= Cs * m.x_s[s]
            )
        
        model.ina_capacity = pyo.Constraint(
            model.S, 
            rule=ina_capacity_rule, 
            doc="INA交换机处理容量限制"
        )
        
        # 约束7: Worker速率一致性
        def worker_rate_rule(m, j, w):
            if w >= workers_num[j]:
                return pyo.Constraint.Skip
            return (
                sum(m.gamma_jws[j, w, s] for s in m.S) + m.gamma_jwd[j, w] 
                >= m.gamma_j[j]
            )
        
        model.worker_rate = pyo.Constraint(
            model.J, model.W_align, 
            rule=worker_rate_rule, 
            doc="每个worker的速率必须满足job的有效速率"
        )
        
        # 约束8: INA出口速率一致性（聚合同步）
        # INA的出口速率不应低于任何worker的入口速率（聚合是同步操作）
        def egress_consistency_rule(m, j, w, s):
            if w >= workers_num[j]:
                return pyo.Constraint.Skip
            return m.gamma_js[j, s] >= m.gamma_jws[j, w, s]
        
        model.egress_consistency = pyo.Constraint(
            model.J, model.W_align, model.S, 
            rule=egress_consistency_rule, 
            doc="INA出口速率不低于任何worker的入口速率"
        )
        
        # 约束9: INA流量守恒（聚合不增加流量）
        def ina_flow_conservation_rule(m, j, s):
            return m.gamma_js[j, s] <= sum(m.gamma_jws[j, w, s] for w in range(workers_num[j]))
        
        model.ina_flow_conservation = pyo.Constraint(
            model.J, model.S,
            rule=ina_flow_conservation_rule,
            doc="INA出口速率不超过入口速率之和"
        )
        
        # 约束10: PS流入充分性
        def ps_inflow_rule(m, j):
            return m.gamma_j[j] <= (
                sum(m.gamma_js[j, s] for s in m.S) + 
                sum(m.gamma_jwd[j, w] for w in range(workers_num[j]))
            )
        
        model.ps_inflow = pyo.Constraint(
            model.J,
            rule=ps_inflow_rule,
            doc="Job速率不超过PS总流入速率"
        )
        
        # 约束11: Makespan约束
        def makespan_rule(m, j):
            # gamma_j / m_j >= alpha  =>  makespan = 1/alpha <= m_j / gamma_j
            return m.alpha <= m.gamma_j[j] / jobs_size[j]
        
        model.makespan = pyo.Constraint(
            model.J, 
            rule=makespan_rule, 
            doc="Makespan由最慢的job决定"
        )
        
        # 约束12: 网络链路容量
        # 预计算每条链路上的流
        link_flows = {}
        for link_id, link_info in self.mapping.items():
            edge = link_info[0]
            capacity = link_info[1]
            
            # 收集所有使用此链路的流（按job分类）
            flows_per_job = {}
            for j in range(jobs_num):
                w_to_s = self._get_flows_on_link(
                    edge, 
                    workers_id[j], 
                    self.ina_candidates
                )
                w_to_ps = self._get_flows_on_link(
                    edge, 
                    workers_id[j], 
                    [ps_id[j]]
                )
                s_to_ps = self._get_flows_on_link(
                    edge, 
                    self.ina_candidates, 
                    [ps_id[j]]
                )
                flows_per_job[j] = (w_to_s, w_to_ps, s_to_ps)
            
            link_flows[link_id] = (edge, capacity, flows_per_job)
        
        def link_capacity_rule(m, link_id):
            edge, base_capacity, flows_per_job = link_flows[link_id]
            
            total_load = 0
            for j in range(jobs_num):
                w_to_s, w_to_ps, s_to_ps = flows_per_job[j]
                
                # Worker-to-INA流
                for w_idx, s_idx in w_to_s:
                    total_load += m.gamma_jws[j, w_idx, s_idx]
                
                # Worker-to-PS流
                for w_idx, _ in w_to_ps:
                    total_load += m.gamma_jwd[j, w_idx]
                
                # INA-to-PS流
                for s_idx, _ in s_to_ps:
                    total_load += m.gamma_js[j, s_idx]
                
                # 检查是否为PS的最后一跳（使用PS带宽）
                if (edge == (self.sd_id[j], ps_id[j]) or 
                    edge == (ps_id[j], self.sd_id[j])):
                    capacity = Ps
                else:
                    capacity = base_capacity
            
            return total_load <= capacity
        
        model.link_capacity = pyo.Constraint(
            self.mapping.keys(), 
            rule=link_capacity_rule, 
            doc="网络链路容量限制"
        )
        
        self.model = model

    # ========= 求解与结果提取 =========

    def solve_with_pyomo(
        self,
        solver_name: str = "gurobi",
        time_limit: int = 300,
        mip_gap: float = 0.01,
        verbose: bool = False,
    ):
        """
        对接父类的通用求解流程：

        1. 调用 self.build_model()
        2. 调用父类 _solve_with_pyomo_core() 执行求解
        3. 提取解并组装成字典返回
        """
        solved = super()._solve_with_pyomo_core(
            solver_name=solver_name,
            time_limit=time_limit,
            mip_gap=mip_gap,
            verbose=verbose,
        )

        if not solved:
            return None

        m = self.model
        jobs_num = self.problem_data["jobs_num"]
        workers_num_align = max(self.problem_data["instance"]["workers_num"])

        # ----- makespan -----
        alpha_val = pyo.value(m.alpha)
        makespan = 1.0 / alpha_val if alpha_val > 1e-9 else float("inf")

        # ----- INA 部署 -----
        ina_binary = [int(round(pyo.value(m.x_s[s]))) for s in range(self.ina_candidates_num)]
        ina_placement_switches = [
            self.ina_candidates[s]
            for s in range(self.ina_candidates_num)
            if ina_binary[s] > 0.5
        ]

        # ----- worker 路由（离散部分） -----
        worker_to_ina = [
            [
                [int(round(pyo.value(m.y_jws[j, w, s]))) for s in range(self.ina_candidates_num)]
                for w in range(workers_num_align)
            ]
            for j in range(jobs_num)
        ]

        worker_to_ps = [
            [int(round(pyo.value(m.y_jwd[j, w]))) for w in range(workers_num_align)]
            for j in range(jobs_num)
        ]

        # ----- job 速率 -----
        job_rates = [pyo.value(m.gamma_j[j]) for j in range(jobs_num)]

        self.solution = {
            "ina_placement_switches": ina_placement_switches,
            "worker_to_ina": worker_to_ina,
            "worker_to_ps": worker_to_ps,
            "job_rates": job_rates,
            "makespan": makespan,
        }

        return self.solution       