from __future__ import annotations
from typing import List, Dict, Tuple, Any, Optional
import copy
import pyomo.environ as pyo
from MMBench.problem.problem_template.runtime.PyomoTemplateSolver import PyomoTemplateSolver

class PyomoSolver(PyomoTemplateSolver):
    """
    LLMINA问题的MILP求解器
    
    优化目标：最小化makespan（最大完成时间）
    
    决策变量：
    - INA placement: 在哪些交换机上部署INA（预算K）
    - Worker assignment: 每个worker选择哪个聚合点（INA或PS）
    - Rate allocation: 每条流的传输速率分配
    
    约束条件：
    - INA部署预算、Worker唯一分配、交换机处理容量、网络链路带宽、流速率一致性
    """
    
    def __init__(
        self,
        instance: dict,
        network,
        K: int,
        jobs_num: int,
        Cs: float = 750.0,
        Ps: float = 200.0,
        topo_name: str = 'FatTree',
    ):
        """
        初始化LLMINA MILP求解器
        
        Args:
            instance: 问题实例，包含：
                - workers_id: List[List[int]] - Worker节点ID
                  格式：[[job0_worker0, job0_worker1, ...], [job1_worker0, ...]]
                - ps_id: List[int] - 各job的PS节点ID
                - workers_num: List[int] - 各job的worker数量
                - jobs_size: List[float] - 各job的梯度数据量（m_j）
                - tor_switch_workers: List[Dict[int,int]] - 各ToR交换机下的worker分布
            
            network: 网络拓扑对象，包含：
                - G: networkx.Graph - 网络拓扑图
                - allPathDict: Dict[src][dst] -> List[int] - 预计算路径
                  格式：路径节点列表 [src, hop1, hop2, ..., dst]
                - bandwidth_mapping: Dict[(u,v)] -> float - 链路带宽容量(Gbps)
                - 交换机ID集合：tors_id, aggrs_id, cores_id (FatTree)
                                 或 tors_id, spines_id (SpineLeaf)
            
            K: INA部署预算（最多部署K个INA交换机）
            jobs_num: 并发任务数量
            Cs: INA交换机处理容量（Gbps）
            Ps: PS端口带宽（Gbps）
            topo_name: 拓扑类型 ('FatTree' 或 'SpineLeaf')
        """
        # 打包LLMINA问题数据
        problem_data = {
            # 核心问题参数
            'instance': instance,
            'network': network,
            'K': K,
            'jobs_num': jobs_num,
            'Cs': Cs,
            'Ps': Ps,
            'topo_name': topo_name,
        }

        self.problem_data = problem_data
    
    def _preprocess_data(self):
        """
        数据预处理：提取和验证网络拓扑数据
        
        重写父类钩子方法，实现LLMINA特定的数据预处理：
        1. 深拷贝网络拓扑数据（避免污染原始数据）
        2. 确定INA候选交换机集合
        3. 提取PS相邻交换机（用于最后一跳带宽约束）
        4. 验证数据完整性
        """
        # 从problem_data中读取数据
        network = self.problem_data['network']
        jobs_num = self.problem_data['jobs_num']
        ps_id = self.problem_data['instance']['ps_id']
        
        # 深拷贝网络拓扑（避免修改原始对象）
        self.G = copy.deepcopy(network.G)
        self.allPathDict = copy.deepcopy(network.allPathDict)
        self.mapping = copy.deepcopy(network.mapping) if hasattr(network, 'mapping') else {}
        
        # 确定INA候选交换机
        self.ina_candidates = self._get_ina_candidates()
        self.ina_candidates_num = len(self.ina_candidates)
        
        # 获取PS相邻交换机（用于最后一跳带宽约束）
        self.sd_id = [
            list(self.G.adj._atlas[ps_id[j]].keys())[0] 
            for j in range(jobs_num)
        ]
    
    def _get_ina_candidates(self) -> List[int]:
        """
        根据拓扑类型确定INA候选交换机集合
        
        Returns:
            排序后的候选交换机ID列表
        
        Raises:
            ValueError: 如果无法确定候选交换机
        """
        network = self.problem_data['network']
        topo_name = self.problem_data['topo_name']
        
        if topo_name == 'FatTree' or (
            hasattr(network, 'tors_id') and 
            hasattr(network, 'aggrs_id') and 
            hasattr(network, 'cores_id')
        ):
            # FatTree: ToR + Aggregation + Core
            candidates = (list(network.tors_id) + 
                         list(network.aggrs_id) + 
                         list(network.cores_id))
        elif topo_name == 'SpineLeaf' or (
            hasattr(network, 'tors_id') and 
            hasattr(network, 'spines_id')
        ):
            # SpineLeaf: ToR (Leaf) + Spine
            candidates = list(network.tors_id) + list(network.spines_id)
        elif hasattr(network, 'all_switches_id'):
            # 通用：所有交换机
            candidates = list(network.all_switches_id)
        else:
            raise ValueError("无法从网络拓扑确定INA候选交换机集合")
        
        return sorted(candidates)
    
    def _get_path_links(self, src: int, dst: int) -> List[Tuple[int, int]]:
        """
        获取从src到dst路径上的所有链路
        
        Args:
            src: 源节点ID
            dst: 目标节点ID
        
        Returns:
            链路列表 [(u1, v1), (u2, v2), ...]
        """
        path = self.allPathDict[src][dst]
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    
    def _get_flows_on_link(
        self, 
        edge: Tuple[int, int], 
        src_set: List[int], 
        dst_set: List[int]
    ) -> List[Tuple[int, int]]:
        """
        查找使用指定链路的所有流（src, dst）对
        
        Args:
            edge: 链路 (u, v)
            src_set: 源节点ID列表
            dst_set: 目标节点ID列表
        
        Returns:
            流索引对列表 [(src_idx, dst_idx), ...]
            其中 src_idx 是在 src_set 中的索引，dst_idx 是在 dst_set 中的索引
        """
        edge_reversed = edge[::-1]
        flows = []
        
        for src_idx, src in enumerate(src_set):
            for dst_idx, dst in enumerate(dst_set):
                links = self._get_path_links(src, dst)
                if edge in links or edge_reversed in links:
                    flows.append((src_idx, dst_idx))
        
        return flows
    
    def build_model(self):
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
        # 调用数据预处理
        self._preprocess_data()
        
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

    def solve_pyomo(self,
        solver_name: str = 'gurobi',
        time_limit: int = 300,
        mip_gap: float = 0.01,
        verbose: bool = False,):
        """
        父类中已经实现：
        1、self.build_model()
        2、pyomo_model.solve()
        3、return self.get_solution()
        """
        return super().solve(solver_name,time_limit,mip_gap,verbose)
    
    def get_solution(self) -> Optional[Dict[str, Any]]:
        m = self.model

        jobs_num = self.problem_data['jobs_num']
        workers_num_align = max(self.problem_data['instance']['workers_num'])

        # 提取alpha和makespan
        alpha_val = pyo.value(m.alpha)
        makespan = 1.0 / alpha_val if alpha_val > 1e-9 else float('inf')

        # 提取INA部署决策
        ina_placement = [
            int(round(pyo.value(m.x_s[s])))
            for s in range(self.ina_candidates_num)
        ]

        # 获取实际部署的交换机ID
        ina_placement_switches = [
            self.ina_candidates[s]
            for s in range(self.ina_candidates_num)
            if ina_placement[s] > 0.5
        ]

        # 提取路由决策（二值变量）
        worker_to_ina = [
            [
                [int(round(pyo.value(m.y_jws[j, w, s])))
                 for s in range(self.ina_candidates_num)]
                for w in range(workers_num_align)
            ]
            for j in range(jobs_num)
        ]

        worker_to_ps = [
            [int(round(pyo.value(m.y_jwd[j, w])))
             for w in range(workers_num_align)]
            for j in range(jobs_num)
        ]

        # 提取速率变量（连续）
        job_rates = [
            pyo.value(m.gamma_j[j])
            for j in range(jobs_num)
        ]

        # 构建完整解字典并缓存
        self.solution = {
            'ina_placement_switches': ina_placement_switches,
            'worker_to_ina': worker_to_ina,
            'worker_to_ps': worker_to_ps,
            'job_rates': job_rates,
            'makespan': makespan,
        }

        return self.solution