from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional, Type, Set
from collections import defaultdict
import pyomo.environ as pyo
import copy

# 假设这些模块存在于你的项目中
from .ModelSolver import ModelSolver
from .topo import FatTree, SpineLeaf
from .dataset import generate_dataset_job
from .BaseSolve import RelaxSolve

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

    def _generate_topology_narrative(self) -> Dict[str, str]:
        """
        生成“人类可读”的拓扑说明书，替换原本的数字元数据。
        """
        return {
            "network_architecture": f"3-Tier {self.topo_name} Topology (Pod-based structure)",
            "physical_scale": f"Total {len(self.network.all_switches_id)} Switches serving {self.hosts_num} * {len(self.network.tors_id)} Servers.",
            "bandwidth_policy": "Strict Oversubscription (2:1). Traffic moving upward (to Core) faces 50 percent bandwidth reduction.",
            "critical_rule": "Cross-Pod traffic is EXPENSIVE (congested). Intra-Pod traffic is CHEAP (fast)."
        }
    
    def run_batch_evaluation(self) -> Dict[str, Any]:
        """
        [主入口] 执行批量评估。
        """
        # ===== 定义参数扫描空间 =====
        ina_num_list = [5]
        jobs_num_list = [5]
        Cs_list = [750.0]
        Ps = 200.0
        instance_num = 1

        # 结果容器
        report = {
            "topology_context": self._generate_topology_narrative(), # 注入世界观
            "evaluation_scenarios": []
        }

        # ===== 循环执行实例 =====
        for ina_budget in ina_num_list:
            for Cs in Cs_list:
                for jobs_num in jobs_num_list:
                    dataset = generate_dataset_job(jobs_num, self.network.all_workers_id, instance_num, self.network.hosts_num, True)
                    
                    for name, instance in dataset.items():
                        print(f"Running evaluation: ina_budget{ina_budget}, Jobs={jobs_num}, Cs={Cs}...")
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
                        # 2. 求解
                        solver.solve()
                        # 3. 构造 Base Solver 进行对比分析
                        base_solver = RelaxSolve(ina_budget, jobs_num, Cs, self.network, instance)
                        # 4. 对该 Solver 实例进行独立分析
                        analysis_result = self._analyze_single_instance(solver, base_solver)
                        
                        report["evaluation_scenarios"].append(analysis_result)
        return report

    def _analyze_single_instance(self, solver: ModelSolver, base_solver) -> Dict[str, Any]:
        """
        将单次求解结果转化为“场景故事”。
        """
        ina_budget = solver.problem_data["ina_budget"]
        jobs_num = solver.problem_data["jobs_num"]
        Cs = solver.problem_data["Cs"]
        Ps = solver.problem_data["Ps"]

        # 确保 LP 子问题已求解 (获取连续变量)
        if solver.solution is None:
             return {"status": "No Solution Found"}
        
        # 1. === 输入参数翻译 (Context Translation) ===
        # 将 K, Cs 等变量转化为具有物理意义的描述
        context_narrative = {
            "workload_description": f"{jobs_num} Concurrent DML Jobs (Moderate Load)",
            "acceleration_resource_budget": f"Allowed to deploy INA on {ina_budget} switches (approx {ina_budget/len(self.network.all_switches_id)*100:.1f}% of all switches).",
            "hardware_capability": {
                "switch_processing_power": f"{Cs} Gbps (Standard Programmable Switch)",
                "server_uplink_capacity": f"{Ps} Gbps (High-Performance NIC for PS)"
            },
            "job_spatial_distribution": self._analyze_job_locality_narrative(solver.problem_data["instance"])
        }

        # 获取评估算法的makespan
        makespan = solver.get_makespan(solver.solution["ina_placement_switches"], solver.solution["worker_agg_id"])
        # 获取对比算法的makespan
        base_Makespan = base_solver.base_solve()

        # 2. === 性能诊断 (Performance Diagnosis) ===
        perf_status = "Good" if makespan is not None and makespan < base_Makespan else "Poor - Optimization Needed"

        model = solver.model
        if model is None:
             return {"status": "Model Not Built"}

        # 3. === 瓶颈审计 (Resource Audit) ===
        link_usage, switch_usage, ps_usage = self._map_flow_to_resources(solver, model)
        raw_bottlenecks = self._detect_link_bottlenecks(solver, model, link_usage)
        raw_switch_usage = self._detect_switch_usage(solver, model, switch_usage)
        raw_ps_usage = self._detect_ps_usage(solver, model, ps_usage)
        
        # 语义化过滤：只保留“有趣”的发现
        semantic_bottlenecks = self._translate_bottlenecks(raw_bottlenecks)
        semantic_ina_usage = self._translate_ina_usage(raw_switch_usage)
        semantic_ps_usage = self._translate_ps_usage(raw_ps_usage)

        return {
            "scenario_narrative": context_narrative,
            "performance_outcome": {
                "iteration_makespan": f"{makespan:.2f} seconds" if makespan is not None else "N/A",
                "contrast_algorithm_makespan": f"{base_Makespan:.2f} seconds",
                "diagnosis": perf_status
            },
            "critical_issues_found": semantic_bottlenecks,
            "resource_utilization_report": semantic_ina_usage + semantic_ps_usage
        }

    def _detect_switch_usage(self, solver: ModelSolver, model: pyo.ConcreteModel, switch_usage) -> List[Dict[str, Any]]:
        # 3. 扫描 INA 计算约束 (INA Capacity)
        def get_usage(sw_id, switch_usage):
            usage = 0.0
            for job in switch_usage.get(sw_id).values():
                usage += job["load"]
            return usage

        all_switch_usage = []
        if hasattr(model, "ina_capacity"):
            ina_candidates = getattr(solver, "ina_candidates", [])
            Cs = float(solver.problem_data["Cs"])

            for s_idx in getattr(model, "S", []):
                # 只看已部署 INA
                x_val = pyo.value(model.x_s[s_idx])
                if x_val is None or x_val <= 0.5:
                    continue

                real_sw_id = ina_candidates[s_idx]
                node_desc = self._get_node_desc(real_sw_id)
                if real_sw_id in switch_usage:
                    utilization = float(get_usage(real_sw_id,switch_usage) / Cs) if Cs > 1e-9 else 0.0
                    all_switch_usage.append({
                        "ina_switch_id": str(real_sw_id),
                        "node_description": node_desc,
                        "usage": switch_usage[real_sw_id],
                        "capacity_gbps": Cs,
                        "utilization": utilization
                    })
                else:
                    all_switch_usage.append({
                        "ina_switch_id": str(real_sw_id),
                        "node_description": node_desc,
                        "usage": None,
                        "capacity_gbps": Cs,
                        "utilization": 0.0
                    })
        return all_switch_usage
    
    def _detect_link_bottlenecks(self, solver: ModelSolver, model: pyo.ConcreteModel, link_usage) -> List[Dict[str, Any]]:
        """
        深入分析模型约束，找出所有的物理瓶颈。
        改进：合并物理链路的双向统计，避免重复报告同一条链路。
        """
        bottlenecks = []

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
                    
                    # 语义化标签
                    semantic_label = "Unknown Link"
                    source_desc = self._get_node_desc(canonical_u)
                    target_desc = self._get_node_desc(canonical_v)
                    
                    if b_type == "PS_Port":
                        semantic_label = "PS Access Bottleneck (Last Mile Congestion)"
                    elif "Core" in source_desc or "Core" in target_desc:
                        semantic_label = "Core Layer Bottleneck (Cross-Pod Congestion)"
                    elif "Aggr" in source_desc and "ToR" in target_desc:
                        semantic_label = "Pod-Internal Link"

                    # 构建合并后的瓶颈对象
                    # 注意：load_gbps 记录的是触发瓶颈那个方向的负载，或者是双向总和，这里建议展示详细的双向结构
                    bottlenecks.append({
                        "type": b_type,
                        "semantic_label": semantic_label, # <--- 新增
                        "resource_id": f"{canonical_u}-{canonical_v}", # 统一 ID
                        "source": source_desc,
                        "target": target_desc,
                        "capacity_gbps": float(limit_val), # 物理链路容量（假设双向对称或取了上限）
                        
                        # 详细的双向负载信息
                        "traffic_stats": {
                            "forward_dir": f"{source_desc}->{target_desc}",
                            "forward_load": float(usage_info["forward"]),
                            "forward_utilization": float(usage_info["forward"]) / float(limit_val) if limit_val > 0 else 0,
                            
                            "backward_dir": f"{target_desc}->{source_desc}",
                            "backward_load": float(usage_info["backward"]),
                            "backward_utilization": float(usage_info["backward"]) / float(limit_val) if limit_val > 0 else 0,
                        },
                        
                        # 汇总的 Job 列表（双向都在这）
                        "impacting_jobs": list(usage_info["jobs"])
                    })

        return bottlenecks

    def _map_flow_to_resources(self, solver: ModelSolver, model: pyo.ConcreteModel):
        """
        遍历当前解的变量 (gamma)，统计每条边、每个 INA 交换机被哪些 Job 使用 & 负载。
        """
        link_usage = {}  # (u,v) -> {jobs: set, forward: float, backward: float}

        # INA 侧也做“链路容量同款”的独立统计
        switch_usage = {}  # ina_id -> {"jobs": {job_id: {"workers_count": int, "load": float}}}
        ps_usage = {}      # ps_id -> {"jobs": {job_id: {"workers_count": int, "load": float}}}

        pd = solver.problem_data
        jobs_num = pd["jobs_num"]
        workers_id = pd["instance"]["workers_id"]
        worker_agg_id = solver.solution["worker_agg_id"]
        ps_id = pd["instance"]["ps_id"]
        ina_candidates = getattr(solver, "ina_candidates", [])
        ina_map = {uid: i for i, uid in enumerate(ina_candidates)}

        for j in range(jobs_num):
            curr_ps = ps_id[j]
            curr_workers = workers_id[j]

            # 1. Worker -> Agg 阶段
            for w_idx, w_id in enumerate(curr_workers):
                agg_node = worker_agg_id[j][w_idx]

                rate = 0.0
                if agg_node == curr_ps:
                    # worker 直连 PS
                    try:
                        rate = pyo.value(model.gamma_jwd[j, w_idx]) or 0.0
                    except:
                        rate = 0.0
                    
                    # ===== 统计 PS 上的负载和 worker 数量 =====
                    if curr_ps not in ps_usage:
                        ps_usage[curr_ps] = {}
                    
                    if j not in ps_usage[curr_ps]:
                        ps_usage[curr_ps][j] = {"workers_count": 0, "load": 0.0}
                    
                    ps_usage[curr_ps][j]["workers_count"] += 1
                    ps_usage[curr_ps][j]["load"] += float(rate)

                else:
                    # worker -> INA
                    s_idx = ina_map.get(agg_node)
                    if s_idx is not None:
                        try:
                            rate = pyo.value(model.gamma_jws[j, w_idx, s_idx]) or 0.0
                        except:
                            rate = 0.0

                        # ===== 统计 INA 上的负载和 worker 数量 =====
                        if agg_node not in switch_usage:
                            switch_usage[agg_node] = {}

                        # 获取该 job 对应的 job_id
                        if j not in switch_usage[agg_node]:
                            switch_usage[agg_node][j] = {"workers_count": 0, "load": 0.0}

                        # 更新该 job 的 workers_count 和 load
                        switch_usage[agg_node][j]["workers_count"] += 1
                        switch_usage[agg_node][j]["load"] += float(rate)

                # 链路负载累计（原逻辑不变）
                if rate > 1e-6:
                    try:
                        path = solver._get_path_links(w_id, agg_node)
                        self._add_load_to_links(link_usage, path, float(rate), j, solver.bandwidth_mapping)
                    except Exception:
                        pass

            # 2. Agg (INA) -> PS 阶段（用于链路负载）
            for s_idx, sw_id in enumerate(ina_candidates):
                if sw_id == curr_ps:
                    continue

                try:
                    rate = pyo.value(model.gamma_js[j, s_idx]) or 0.0
                except:
                    rate = 0.0

                if rate > 1e-6:
                    try:
                        path = solver._get_path_links(sw_id, curr_ps)
                        self._add_load_to_links(link_usage, path, float(rate), j, solver.bandwidth_mapping)
                    except Exception:
                        pass

        return link_usage, switch_usage, ps_usage

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

    def _analyze_job_locality_narrative(self, instance) -> List[str]:
        """
        生成关于 Job 分布的自然语言描述。
        """
        descriptions = []
        for j, w_ids in enumerate(instance["workers_id"]):
            # 简化逻辑：假设我们能算出 Pod
            # 实际代码中需要调用 self._get_node_pod(w)
            pods = set([self._get_node_pod(w) for w in w_ids])
            if len(pods) == 1:
                desc = f"Job {j}: Localized (Ideal). All workers inside Pod {list(pods)[0]}."
            else:
                desc = f"Job {j}: Fragmented (Challenging). Spread across {len(pods)} Pods. Needs Core bandwidth."
            descriptions.append(desc)
        return descriptions

    def _get_node_pod(self, node_id: int) -> int:
        """辅助函数：根据 ID 判断节点所在的 Pod"""
        # FatTree 拓扑中，根据 ID 范围或连接关系推断 Pod
        # 这里使用简化的计算逻辑，假设 ID 是连续分配的，且结构规整
        # Pod ID = (ToR ID - offset) // (k / 2)
        if self.node_roles.get(node_id) == "ToR":
             try:
                 offset = self.network.hosts_num * len(self.network.tors_id) # 假设 ToR ID 从 host 之后开始
                 # 更稳健的方法是利用 tors_id 列表的索引
                 if node_id in self.network.tors_id:
                     idx = self.network.tors_id.index(node_id)
                     return idx // (self.k // 2)
             except:
                 pass
        
        # 如果是 Worker，找到它直连的 ToR
        # 假设拓扑图中已经有连接关系
        if self.node_roles.get(node_id) == "Server":
             neighbors = list(self.network.G.neighbors(node_id))
             for n in neighbors:
                 if self.node_roles.get(n) == "ToR":
                     if n in self.network.tors_id:
                         idx = self.network.tors_id.index(n)
                         return idx // (self.k // 2)
        
        return 0 # 默认 placeholder
    
    def _translate_bottlenecks(self, raw_bottlenecks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        将原始瓶颈数据翻译为“反思线索”。
        """
        insights = []
        for b in raw_bottlenecks:
            if isinstance(b, str): continue # 跳过说明文本
            
            # 提取关键信息，重组为句子
            # 检查是否有 'forward_utilization' 键，并确保它是数字
            try:
                u_pct = float(b['traffic_stats']['forward_utilization']) * 100
            except (KeyError, ValueError, TypeError):
                 u_pct = 0.0

            if u_pct < 90: continue # 忽略非严重瓶颈
            
            insight = {
                "location": b.get('semantic_label', 'Unknown Location'), 
                "specific_link": f"{b.get('source', 'Unknown')} --> {b.get('target', 'Unknown')}",
                "severity": "CRITICAL SATURATION (100 percent Full)",
                "root_cause_hint": ""
            }
            
            b_type = b.get('type', '')
            label = b.get('semantic_label', '')

            if "PS_Port" in b_type:
                insight["root_cause_hint"] = "Too many workers sending raw gradients directly to this PS. INA was not used effectively to reduce traffic volume."
            elif "Core" in label:
                insight["root_cause_hint"] = "Heavy Cross-Pod traffic detected. The algorithm failed to aggregate data within the local Pods before sending it up to the Core."
            
            insights.append(insight)
        
        if not insights:
            insights.append({"status": "Healthy", "message": "No specific links are fully saturated."})
            
        return insights

    def _translate_ina_usage(self, raw_switch_usage: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        翻译 INA 使用情况，重点指出“浪费”和“错配”。
        """
        audit = []
        for s in raw_switch_usage:
            if isinstance(s, str): continue
            
            util = s.get('utilization', 0.0)
            node_desc = s.get('node_description', 'Unknown Switch')
            
            if util < 0.01:
                audit.append({
                    "switch": node_desc,
                    "status": "WASTED BUDGET",
                    "observation": "INA feature enabled (costing budget) but processing ZERO traffic. Algorithm chose a poor location."
                })
            elif util > 0.99:
                audit.append({
                    "switch": node_desc,
                    "status": "OVERLOADED",
                    "observation": "Switch processing capacity exceeded. Algorithm assigned too many workers to this single aggregator."
                })
            else:
                audit.append({
                    "switch": node_desc,
                    "status": "Active",
                    "utilization": f"{util*100:.1f}%"
                })
        return audit

    def _detect_ps_usage(self, solver: ModelSolver, model: pyo.ConcreteModel, ps_usage) -> List[Dict[str, Any]]:
        # 统计 PS 负载 (PS Capacity)
        def get_usage(ps_id, ps_usage):
            usage = 0.0
            for job in ps_usage.get(ps_id).values():
                usage += job["load"]
            return usage

        all_ps_usage = []
        Ps = float(solver.problem_data["Ps"])

        # 遍历所有 PS (从 problem_data 中获取)
        all_ps_ids = set()
        for j_ps in solver.problem_data["instance"]["ps_id"]:
            all_ps_ids.add(j_ps)

        for ps_id in all_ps_ids:
            node_desc = self._get_node_desc(ps_id)
            if ps_id in ps_usage:
                utilization = float(get_usage(ps_id, ps_usage) / Ps) if Ps > 1e-9 else 0.0
                all_ps_usage.append({
                    "ps_id": str(ps_id),
                    "node_description": node_desc,
                    "usage": ps_usage[ps_id],
                    "capacity_gbps": Ps,
                    "utilization": utilization
                })
            else:
                 # 未使用的 PS
                all_ps_usage.append({
                    "ps_id": str(ps_id),
                    "node_description": node_desc,
                    "usage": None,
                    "capacity_gbps": Ps,
                    "utilization": 0.0
                })
        return all_ps_usage

    def _translate_ps_usage(self, raw_ps_usage: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        翻译 PS 使用情况
        """
        audit = []
        for s in raw_ps_usage:
            if isinstance(s, str): continue
            
            util = s.get('utilization', 0.0)
            node_desc = s.get('node_description', 'Unknown PS')
            
            # 统计 job 详情
            job_details = ""
            if s.get("usage"):
                for job_id, stats in s["usage"].items():
                    job_details += f"[Job {job_id}: {stats['workers_count']} workers] "
            
            if util > 0.99:
                audit.append({
                    "ps_node": node_desc,
                    "status": "OVERLOADED",
                    "observation": f"PS Bandwidth saturated. Handling: {job_details}"
                })
            elif util < 0.01:
                 audit.append({
                    "ps_node": node_desc,
                    "status": "IDLE",
                    "observation": "PS has almost no traffic."
                })
            else:
                audit.append({
                    "ps_node": node_desc,
                    "status": "Active",
                    "utilization": f"{util*100:.1f}%",
                    "details": job_details.strip()
                })
        return audit