from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional, Type, Set
from collections import defaultdict
import pyomo.environ as pyo
import copy

# 假设这些模块存在于你的项目中
from .ModelSolver import ModelSolver
from .topo import FatTree, SpineLeaf
from .dataset import generate_dataset_job

class SolutionAnalyzer:
    """
    批量启发式解质量分析器。
    
    功能：
    1. 构造多种场景（不同的 jobs_num, ina_num, Cs 等）。
    2. 调用 Solver 求解。
    3. 对每个解进行深度约束分析（瓶颈识别、资源竞争分析）。
    4. 输出结构化的 JSON 数据，供反思智能体（Reflection Agent）使用。
    """

    _KNOWN_CONSTRAINT_COMPONENTS = [
        "ina_budget", "worker_choice", "ina_deployment", 
        "ina_capacity", "link_capacity", "makespan"
    ]

    def __init__(self, solver_class: Type, epsilon: float = 1e-5) -> None:
        self.solver_class = solver_class
        self.epsilon = epsilon
        
        # ===== 1. 定义固定的拓扑环境 =====
        # 这些参数决定了物理底座，通常在一次大的实验中保持不变
        self.k = 4  # FatTree 参数
        self.tops_num = 5
        self.tors_num = 10 
        self.hosts_num = 20
        self.basic_band = 100
        self.k1 = 0.5 
        self.k2 = 2
        
        self.topo_name = 'FatTree' # 或 'SpineLeaf'

        # 初始化物理网络对象
        if self.topo_name == 'FatTree':
            self.network = FatTree(self.k, self.basic_band, self.hosts_num, 'fixed', self.k1)
        elif self.topo_name == 'SpineLeaf':
            self.network = SpineLeaf(self.tops_num, self.tors_num, self.basic_band, self.hosts_num, 'fixed', self.k1)
        
        # 预计算节点角色（用于后续分析时的语义增强）
        self.node_roles = self._build_node_roles()

    def _build_node_roles(self) -> Dict[int, str]:
        """建立所有节点的角色映射表，方便 Log 输出"""
        roles = {}
        for nid in getattr(self.network, "tors_id", []): roles[nid] = "ToR"
        for nid in getattr(self.network, "aggrs_id", []): roles[nid] = "Aggr"
        for nid in getattr(self.network, "cores_id", []): roles[nid] = "Core"
        for nid in getattr(self.network, "spines_id", []): roles[nid] = "Spine"
        for nid in getattr(self.network, "all_workers_id", []): roles[nid] = "Server"
        return roles

    def _get_node_desc(self, node_id: int) -> str:
        """返回节点描述，如 ToR(5)"""
        return f"{self.node_roles.get(node_id, 'Node')}({node_id})"

    def run_batch_evaluation(self) -> Dict[str, Any]:
        """
        [主入口] 执行批量评估。
        
        Returns:
            Dict: 包含拓扑元数据和所有实例的分析结果列表。
            结构如下：
            {
                "topology_meta": { ... },
                "scenarios": [
                    {
                        "input_params": { "ina_num": 3, "jobs_num": 6, ... },
                        "solution_metrics": { "makespan": 10.5, "feasible": True, ... },
                        "bottleneck_analysis": { ... 详细瓶颈列表 ... }
                    },
                    ...
                ]
            }
        """
        # ===== 定义参数扫描空间 =====
        ina_num_list = [5]
        jobs_num_list = [5]
        Cs_list = [750.0]
        Ps = 200.0
        instance_num = 1

        # 结果容器
        report = {
            "topology_meta": {
                "name": self.topo_name,
                "hosts_per_tor": self.hosts_num,
                "basic_bandwidth": self.basic_band,
                "nodes_count": len(self.network.G.nodes)
            },
            "scenarios": []
        }

        # ===== 循环执行实例 =====
        for ina_budget in ina_num_list:
            for Cs in Cs_list:
                for jobs_num in jobs_num_list:
                    
                    # 生成数据集
                    dataset = generate_dataset_job(
                        jobs_num, self.network.all_workers_id, instance_num, self.network.hosts_num, True
                    )
                    
                    for name, instance in dataset.items():
                        print(f"Running evaluation: ina_budget={ina_budget}, Jobs={jobs_num}, Cs={Cs}...")
                        
                        # 1. 实例化 Solver
                        solver = self.solver_class(
                            instance,
                            network=self.network,
                            ina_budget=ina_budget,
                            jobs_num=jobs_num,
                            Cs=Cs,
                            Ps=Ps,
                            topo_name=self.topo_name
                        )
                        
                        # 2. 求解 (假设 solve 会填充 solver.solution 并在内部构建 Pyomo 模型)
                        solver.solve()
                        
                        # 3. 对该 Solver 实例进行独立分析
                        analysis_result = self._analyze_single_instance(solver)
                        
                        # 4. 封装单次场景报告
                        scenario_data = {
                            "input_params": {
                                "ina_budget": ina_budget,
                                "jobs_num": jobs_num,
                                "Cs": Cs,
                                "Ps": Ps,
                                "workers_distribution": instance.get("workers_num", [])
                            },
                            "analysis": analysis_result
                        }
                        
                        report["scenarios"].append(scenario_data)

        return report

    def _analyze_single_instance(self, solver: ModelSolver) -> Dict[str, Any]:
        """
        分析单个求解器实例的解质量和瓶颈。
        """
        # 确保 LP 子问题已求解 (获取连续变量)
        if solver.solution is None:
            return {"status": "No Solution Found"}

        # 尝试从 Solver 获取 LP 结果
        # 注意：这里假设 Solver 已经根据离散解固定了变量并求解了 makespan

        makespan = solver.get_makespan(solver.solution["ina_placement_switches"], solver.solution["worker_agg_id"])

        model = solver.model
        if model is None:
             return {"status": "Model Not Built"}

        # === 核心：约束与瓶颈分析 ===
        bottlenecks = self._detect_bottlenecks(solver, model)
        
        # 计算统计数据
        bottleneck_summary = defaultdict(int)
        for b in bottlenecks:
            bottleneck_summary[b['type']] += 1

        return {
            "status": "Feasible",
            "metrics": {
                "makespan": float(makespan) if makespan is not None else -1.0,
                # 如果 Solver 里有 LP 松弛界，可在此添加 gap
            },
            "bottleneck_counts": dict(bottleneck_summary),
            "bottleneck_details": bottlenecks  # 详细列表，供 Agent 推理
        }

    def _detect_bottlenecks(self, solver: ModelSolver, model: pyo.ConcreteModel) -> List[Dict[str, Any]]:
        """
        深入分析模型约束，找出所有的物理瓶颈。
        改进：合并物理链路的双向统计，避免重复报告同一条链路。
        """
        bottlenecks = []
        
        # 1. 预处理：构建流量映射 (Link -> Jobs)
        # link_usage 的 key 是规范化的 (u,v)，value 包含 forward/backward 负载
        link_usage, switch_usage = self._map_flow_to_resources(solver, model)
        
        # 用于物理链路去重: 存储 frozenset({u, v})
        processed_physical_links = set()
        
        # 2. 扫描链路容量约束 (Link Capacity)
        if hasattr(model, "link_capacity"):
            for index, con in model.link_capacity.items():
                body_val = pyo.value(con.body)
                limit_val = pyo.value(con.upper)
                
                # 判定当前方向是否饱和
                is_saturated = False
                if limit_val > 1e-6 and (limit_val - body_val) < max(self.epsilon * limit_val, 1e-4):
                    is_saturated = True
                
                # 如果当前方向饱和，或者我们需要报告接近饱和的链路，
                # 无论哪个方向触发了检查，我们都输出这条物理链路的完整状态。
                if is_saturated:
                    u, v = index
                    physical_link_key = frozenset({u, v})
                    
                    if physical_link_key in processed_physical_links:
                        continue # 该物理链路（的另一个方向）已经被处理过了
                    
                    processed_physical_links.add(physical_link_key)
                    
                    # --- 提取双向数据 ---
                    # 尝试在 link_usage 中找到规范化的 key
                    usage_info = None
                    canonical_u, canonical_v = u, v
                    
                    if (u, v) in link_usage:
                        usage_info = link_usage[(u, v)]
                        canonical_u, canonical_v = u, v
                    elif (v, u) in link_usage:
                        usage_info = link_usage[(v, u)]
                        canonical_u, canonical_v = v, u
                    else:
                        # 极其罕见的情况：约束里有这条边，但预处理没统计到流量（可能是空载但容量极小？）
                        usage_info = {"jobs": [], "forward": 0.0, "backward": 0.0}

                    # 区分是否为 PS 端口
                    ps_ids = set(solver.problem_data["instance"].get("ps_id", []))
                    is_ps_link = (u in ps_ids) or (v in ps_ids)
                    b_type = "PS_Port" if is_ps_link else "Network_Link"
                    
                    # 构建合并后的瓶颈对象
                    # 注意：load_gbps 记录的是触发瓶颈那个方向的负载，或者是双向总和，这里建议展示详细的双向结构
                    bottlenecks.append({
                        "type": b_type,
                        "resource_id": f"{canonical_u}-{canonical_v}", # 统一 ID
                        "source": self._get_node_desc(canonical_u),
                        "target": self._get_node_desc(canonical_v),
                        "capacity_gbps": float(limit_val), # 物理链路容量（假设双向对称或取了上限）
                        
                        # 详细的双向负载信息
                        "traffic_stats": {
                            "forward_dir": f"{self._get_node_desc(canonical_u)}->{self._get_node_desc(canonical_v)}",
                            "forward_load": float(usage_info["forward"]),
                            "forward_utilization": float(usage_info["forward"]) / float(limit_val) if limit_val > 0 else 0,
                            
                            "backward_dir": f"{self._get_node_desc(canonical_v)}->{self._get_node_desc(canonical_u)}",
                            "backward_load": float(usage_info["backward"]),
                            "backward_utilization": float(usage_info["backward"]) / float(limit_val) if limit_val > 0 else 0,
                        },
                        
                        # 汇总的 Job 列表（双向都在这）
                        "impacting_jobs": list(usage_info["jobs"])
                    })

        # 3. 扫描 INA 计算约束 (INA Capacity) - 逻辑保持不变
        if hasattr(model, "ina_capacity"):
            ina_candidates = getattr(solver, "ina_candidates", [])
            for s_idx in getattr(model, "S", []):
                if pyo.value(model.x_s[s_idx]) > 0.5:
                    con = model.ina_capacity[s_idx]
                    load_val = pyo.value(con.body)
                    limit_val = pyo.value(con.upper)
                    
                    if limit_val > 1e-6 and (limit_val - load_val) < max(self.epsilon * limit_val, 1e-4):
                        real_sw_id = ina_candidates[s_idx]
                        contributing_jobs = switch_usage.get(real_sw_id, set())
                        
                        bottlenecks.append({
                            "type": "INA_Compute",
                            "resource_id": str(real_sw_id),
                            "node_desc": self._get_node_desc(real_sw_id),
                            "load_gbps": float(load_val),
                            "capacity_gbps": float(limit_val),
                            "utilization": float(load_val / limit_val),
                            "impacting_jobs": list(contributing_jobs)
                        })

        return bottlenecks

    def _map_flow_to_resources(self, solver: ModelSolver, model: pyo.ConcreteModel):
        """
        辅助函数：遍历当前解的变量 (gamma)，统计每条边、每个交换机被哪些 Job 使用。
        """
        link_usage = {}  # (u,v) -> {jobs: set, forward: float, backward: float}
        switch_usage = {} # sw_id -> set(job_ids)
        
        pd = solver.problem_data
        jobs_num = pd["jobs_num"]
        workers_id = pd["instance"]["workers_id"]
        # 获取解中的离散决策
        worker_agg_id = solver.solution["worker_agg_id"] # dict: job -> worker_idx -> agg_node_id
        ps_id = pd["instance"]["ps_id"]
        ina_candidates = getattr(solver, "ina_candidates", [])
        ina_map = {uid: i for i, uid in enumerate(ina_candidates)}

        for j in range(jobs_num):
            curr_ps = ps_id[j]
            curr_workers = workers_id[j]
            
            # 1. Worker -> Agg 阶段
            for w_idx, w_id in enumerate(curr_workers):
                agg_node = worker_agg_id[j][w_idx]
                
                # 记录 INA 使用
                if agg_node != curr_ps:
                    if agg_node not in switch_usage: switch_usage[agg_node] = set()
                    switch_usage[agg_node].add(j)
                
                # 获取流量大小
                rate = 0.0
                if agg_node == curr_ps:
                    # worker直连PS
                    try: rate = pyo.value(model.gamma_jwd[j, w_idx])
                    except: rate = 0
                else:
                    # worker -> INA
                    s_idx = ina_map.get(agg_node)
                    if s_idx is not None:
                        try: rate = pyo.value(model.gamma_jws[j, w_idx, s_idx])
                        except: rate = 0
                
                if rate > 1e-6:
                    # 映射路径到物理边
                    try:
                        path = solver._get_path_links(w_id, agg_node) # 需要 Solver 提供路径查询方法
                        self._add_load_to_links(link_usage, path, rate, j, solver.bandwidth_mapping)
                    except Exception:
                        pass # 忽略路径查找失败

            # 2. Agg (INA) -> PS 阶段
            for s_idx, sw_id in enumerate(ina_candidates):
                if sw_id == curr_ps: continue
                
                try: rate = pyo.value(model.gamma_js[j, s_idx])
                except: rate = 0
                
                if rate > 1e-6:
                    try:
                        path = solver._get_path_links(sw_id, curr_ps)
                        self._add_load_to_links(link_usage, path, rate, j, solver.bandwidth_mapping)
                    except Exception:
                        pass

        return link_usage, switch_usage

    def _add_load_to_links(self, usage_dict, path, rate, job_id, bandwidth_map):
        """累加链路负载，处理无向图/双向边问题"""
        for u, v in path:
            # 规范化 key：始终使用 bandwidth_map 中的方向作为 key
            # 假设 bandwidth_map 只存了一个方向或者两个方向都有
            key = (u, v)
            is_forward = True
            
            if key not in bandwidth_map:
                if (v, u) in bandwidth_map:
                    key = (v, u)
                    is_forward = False
                else:
                    # 可能是 host-link 或者未定义的 link
                    continue 
            
            if key not in usage_dict:
                usage_dict[key] = {"jobs": set(), "forward": 0.0, "backward": 0.0}
            
            usage_dict[key]["jobs"].add(job_id)
            if is_forward:
                usage_dict[key]["forward"] += rate
            else:
                usage_dict[key]["backward"] += rate