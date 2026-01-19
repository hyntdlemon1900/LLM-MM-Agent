import copy
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random
import networkx as nx
from typing import List

class RelaxSolve:
    def __init__(self, ina_num, jobs_num, Cs, network, instance):
        self.ina_num = ina_num
        self.jobs_num = jobs_num
        self.Cs = Cs
        self.Ps = network.basic_band # Ps处的端口带宽单独定义
        self.network = network

        self.base_num = network.hosts_num
        self.G = copy.deepcopy(network.G)  # 拓扑
        self.bandwidth_mapping = copy.deepcopy(network.bandwidth_mapping)
        self.mapping = copy.deepcopy(network.mapping)
        self.S_id = copy.deepcopy(network.all_switches_id)
        self.allPathDict = copy.deepcopy(network.allPathDict)
        
        self.workers_id = instance['workers_id']
        self.ps_id = instance['ps_id']
        self.workers_num = instance['workers_num']
        self.jobs_size = instance['jobs_size']
        self.tor_switch_workers = instance['tor_switch_workers']
        self.sd_id = [list(self.G.adj._atlas[self.ps_id[j]].keys())[0] for j in range(self.jobs_num)]

        self.workers_num_align = max(self.workers_num)
        self.topo_name = 'FatTree' if hasattr(self.network, 'tors_id') and hasattr(self.network, 'aggrs_id') and hasattr(self.network, 'cores_id') else 'SpineLeaf' if hasattr(self.network, 'spines_id') else 'Unknown'

    def _get_ina_candidates(self):
        """
        Derive sorted INA candidate switch IDs from the topology.
        
        For FatTree: ToR + Aggregation + Core switches
        For SpineLeaf: ToR (leaf) + Spine switches
        Fallback: all switches
        """
        candidates = None
        # Prefer explicit role-based candidate sets if present on the network object
        try:
            # Check for FatTree topology
            if (hasattr(self.network, 'tors_id') and 
                hasattr(self.network, 'aggrs_id') and 
                hasattr(self.network, 'cores_id')):
                candidates = list(self.network.tors_id) + list(self.network.aggrs_id) + list(self.network.cores_id)
            # Check for SpineLeaf topology
            elif (hasattr(self.network, 'tors_id') and 
                  hasattr(self.network, 'spines_id')):
                candidates = list(self.network.tors_id) + list(self.network.spines_id)
        except Exception:
            candidates = None
        if not candidates:
            candidates = list(self.S_id)
        return sorted(candidates)

    def _normalize_expanded_inputs(self, ina_placement, jobs_routing):
        """Convert expanded x_s and expanded y to compact ina_id list and y for internal optimization."""
        ina_candidates = self._get_ina_candidates()

        # Determine selection from expanded x_s
        if isinstance(ina_placement, (list, tuple)) and len(ina_placement) == len(ina_candidates):
            vals = list(ina_placement)
        elif hasattr(ina_placement, 'shape'):
            vals = list(ina_placement.tolist())
        else:
            # Assume compact ID list already
            return ina_placement, jobs_routing

        # Rank by value then by candidate ID for stability
        idx_sorted = sorted(range(len(vals)), key=lambda i: (vals[i], ina_candidates[i]), reverse=True)
        chosen_idx = idx_sorted[: self.ina_num]
        chosen_idx_sorted_by_id = sorted(chosen_idx, key=lambda i: ina_candidates[i])
        ina_ids = [ina_candidates[i] for i in chosen_idx_sorted_by_id]

        # Expand routing expected: jobs_routing = [y_full (.., len(cands)), y_d]
        y_full, y_d = jobs_routing
        # Compute max_workers robustly
        if hasattr(y_d, 'shape'):
            max_workers = y_d.shape[1] if y_d.ndim >= 2 else (y_d.shape[0] if y_d.ndim == 1 else 0)
        else:
            max_workers = len(y_d[0]) if y_d else 0

        # Build compact y selecting chosen columns
        y_compact = [
            [
                [ y_full[j][w][i] for i in chosen_idx_sorted_by_id ]
                for w in range(max_workers)
            ]
            for j in range(self.jobs_num)
        ]
        return ina_ids, [y_compact, y_d]

    def gurobi_solve(self,ina_placement, jobs_routing):
        # Normalize inputs: accept compact or expanded
        if isinstance(ina_placement, (list, tuple)):
            # Could be compact IDs or expanded x_s
            if len(ina_placement) != self.ina_num:
                ina_ids, jobs_routing = self._normalize_expanded_inputs(ina_placement, jobs_routing)
            else:
                ina_ids = sorted(list(ina_placement))
        else:
            ina_ids, jobs_routing = self._normalize_expanded_inputs(ina_placement, jobs_routing)

        self.jobs_routing = jobs_routing
        self.ina_id = ina_ids  # 设置INA交换机ID列表

        # add model
        m = gp.Model()

        # disable output to a logfile
        m.Params.LogToConsole = 0

        # add variable
        gamma_j = m.addMVar(shape=(self.jobs_num,), vtype=GRB.CONTINUOUS)
        gamma_j_s_d = m.addMVar(shape=(self.jobs_num, self.ina_num), vtype=GRB.CONTINUOUS)
        gamma_j_w_s = m.addMVar(shape=(self.jobs_num, self.workers_num_align, self.ina_num), vtype=GRB.CONTINUOUS)
        gamma_j_w_d = m.addMVar(shape=(self.jobs_num, self.workers_num_align), vtype=GRB.CONTINUOUS)
        x_s = m.addMVar(shape=(self.ina_num,), vtype=GRB.BINARY)
        obj = m.addVar(vtype=GRB.CONTINUOUS)

        # Set objective
        m.setObjective(obj, GRB.MAXIMIZE)

        y_j_w_s = self.jobs_routing[0]
        y_j_w_d = self.jobs_routing[1]
        
        # 添加路由约束：将路由决策固定为给定值
        for j in range(self.jobs_num):
            for w in range(self.workers_num_align):
                for s in range(self.ina_num):
                    m.addConstr(gamma_j_w_s[j, w, s] <= self.Cs * y_j_w_s[j][w][s])
                m.addConstr(gamma_j_w_d[j, w] <= self.Ps * y_j_w_d[j][w])

        for j in range(self.jobs_num):
            for w in range(self.workers_num[j], self.workers_num_align):
                for s in range(self.ina_num):
                    m.addConstr(gamma_j_w_s[j, w, s] == 0)
                m.addConstr(gamma_j_w_d[j, w] == 0)

        m.addConstrs((gamma_j_w_d[j, w] + sum(gamma_j_w_s[j, w, _] for _ in range(self.ina_num))
                      >= gamma_j[j] for j in range(self.jobs_num) for w in range(self.workers_num[j])))

        m.addConstrs(obj <= gamma_j[j] / self.jobs_size[j] for j in range(self.jobs_num))

        m.addConstr(sum(x_s[i] for i in range(self.ina_num)) <= self.ina_num)

        m.addConstrs(x_s[i] == 1 for i in range(self.ina_num))
        
        for j in range(self.jobs_num):
            for w in range(self.workers_num_align):
                for s in range(self.ina_num):
                    m.addConstr(gamma_j_w_s[j, w, s] <= self.Cs * y_j_w_s[j][w][s] * x_s[s])
        
        m.addConstrs((sum(gamma_j_w_s[j, w, i] for j in range(self.jobs_num) for w in range(self.workers_num[j])) <= self.Cs * x_s[i] for i in range(self.ina_num)))

        m.addConstrs((gamma_j_w_s[j, w, i] <= gamma_j_s_d[j, i] for j in range(self.jobs_num) for w in range(self.workers_num[j]) for i in range(self.ina_num)))
        

        for v in self.mapping.values():
            e = v[0]  # 链路
            Ce = v[1]  # 链路容量

            le = [0 for _ in range(self.jobs_num)]
            for j in range(self.jobs_num):
                i_j_w_s, i_j_w_d, i_s_d = self.link_capacity_limitation_mjob(j, e)
                le[j] = sum(gamma_j_w_s[j, i11[0], i11[1]] for i11 in i_j_w_s) + sum(
                    gamma_j_w_d[j, i12[0]] for i12 in i_j_w_d) + sum(gamma_j_s_d[j, i2[0]] for i2 in i_s_d)
                if e == (self.sd_id[j], self.ps_id[j]) or e == (self.ps_id[j], self.sd_id[j]):
                    Ce = self.Ps
            m.addConstr(sum(le[j] for j in range(self.jobs_num)) <= Ce)

        m.optimize()

        obj = obj.X
        Jct = 1/obj
        gamma_j = gamma_j.X
        
        gamma_j_s_d = gamma_j_s_d.X
        gamma_j_w_s = gamma_j_w_s.X
        gamma_j_w_d = gamma_j_w_d.X
        x_s = x_s.X

        return Jct

    def priority(self, network: nx.Graph, jobs_num: int, job_sizes: List[int], switch_id: str, switch_label: int, workers_id: List[List[str]], ps_id_list: List[str]) -> float:
        """
        A higher score indicates the switch is a better candidate for in-network aggregation (INA) placement.

        Args:
            network (nx.Graph): The network topology (e.g., fat-tree), represented as a NetworkX graph.
            jobs_num (int): Total number of distributed training jobs (tenants).
            job_sizes (List[int]): Gradient size (in bytes) for each job. Length = jobs_num.
            switch_id (str): The ID of the switch being evaluated.
            switch_label(int): Type label of each aggregation point: 1 = Edge, 2 = Aggregation, 3 = Core
            workers_id (List[List[str]]): List of worker node ID lists, one per job. Length = jobs_num.
            ps_id_list (List[str]): List of parameter server node IDs, one per job. Length = jobs_num.
            
        Returns:
            float: INA priority score for the specified switch. Higher scores indicate better placement suitability.
        """
        score = 0.0
        total_worker_count = sum(len(workers) for workers in workers_id)
        total_job_traffic = sum(job_sizes)

        # Calculate distance to parameter servers
        for ps_id in ps_id_list:
            distance_to_ps = nx.shortest_path_length(network, source=switch_id, target=ps_id)
            score += (1 / (distance_to_ps + 1)) * 2  # Weight proximity to PS

        # Evaluate potential traffic savings from workers
        for job_index in range(jobs_num):
            for worker_id in workers_id[job_index]:
                distance_to_worker = nx.shortest_path_length(network, source=switch_id, target=worker_id)

                # Reward for closer worker distances based on job size
                score += (job_sizes[job_index] / (distance_to_worker + 1)) * 1.5  # Enhance score based on job sizes

        # Role-based adjustments for capacity and potential traffic handling
        if switch_label == 1:  # Edge switch
            score *= 3.0  # Heavy weighting for edge switches due to downlink capabilities
        elif switch_label == 2:  # Aggregation switch
            score *= 2.0  # Moderate weighting for aggregation switches
        # No adjustment for core switches

        # Penalize for high congestion based on utilized services
        if total_worker_count > 0:
            congestion_factor = len(ps_id_list) / total_worker_count  # Max jobs per worker
            score /= (1 + congestion_factor)  # Normalize score based on congestion

        # Normalize the score concerning total job sizes
        if total_job_traffic > 0:
            score /= total_job_traffic

        return score

    def get_inaid_placement(self, workers_id, network, ps_id_list, jobs_num, ina_num, job_sizes, callable_func: callable):   
        ina_candidates = self._get_ina_candidates()
        
        priority = {}

        if self.topo_name == 'FatTree':
            for switch_id in ina_candidates:
                if switch_id in network.tors_id:
                    label = 1
                elif switch_id in network.aggrs_id:
                    label = 2
                else:
                    label = 3
                # priority[switch_id] = callable_func(network.G, jobs_num, job_sizes, switch_id, workers_id, ps_id_list, label)
                priority[switch_id] = callable_func(network.G, jobs_num, job_sizes, switch_id, label, workers_id, ps_id_list)
        else:
            for switch_id in ina_candidates:
                priority[switch_id] = callable_func(network.G, jobs_num, job_sizes, switch_id, workers_id, ps_id_list)
        ina_id = []
        priority = sorted(priority.items(), key=lambda x: x[1], reverse=True)
        for i in range(ina_num):
            ina_id.append(priority[i][0])
        ina_id.sort()
        print("INA功能部署位置:",ina_id)
        return ina_id
    
    def base_solve(self):
        self.ina_id = self.get_inaid_placement(self.workers_id, self.network, self.ps_id, self.jobs_num, self.ina_num, self.jobs_size, self.priority)

        m = gp.Model()
        m.Params.LogToConsole = 0 # disable output to a logfile

        # add variable
        gamma_j = m.addMVar(shape=(self.jobs_num,), vtype=GRB.CONTINUOUS, name='gamma_j')
        gamma_j_s_d = m.addMVar(shape=(self.jobs_num, self.ina_num), vtype=GRB.CONTINUOUS, name="gamma_j_s_d") 
        gamma_j_w_s = m.addMVar(shape=(self.jobs_num, self.workers_num_align, self.ina_num), vtype=GRB.CONTINUOUS, name="gamma_j_w_s") 
        gamma_j_w_d = m.addMVar(shape=(self.jobs_num, self.workers_num_align), vtype=GRB.CONTINUOUS, name="gamma_j_w_d") 
        x_s = m.addMVar(shape=(self.ina_num,), vtype=GRB.BINARY, name='x_s')  # 实际上已经确定 约束中为其赋值
        y_j_w_s = m.addMVar(shape=(self.jobs_num, self.workers_num_align, self.ina_num), vtype=GRB.CONTINUOUS, name='y_j_w_s') 
        y_j_w_d = m.addMVar(shape=(self.jobs_num, self.workers_num_align), vtype=GRB.CONTINUOUS, name='y_j_w_d')
        obj = m.addVar(vtype=GRB.CONTINUOUS, name='obj')

        # Set objective
        m.setObjective(obj, GRB.MAXIMIZE) 
        # add constraints
        for j in range(self.jobs_num):
            for w in range(self.workers_num[j]):
                for s in range(self.ina_num):
                    m.addConstr(y_j_w_s[j, w, s] >= 0)
                    m.addConstr(y_j_w_s[j, w, s] <= 1)
                m.addConstr(y_j_w_d[j, w] >= 0)
                m.addConstr(y_j_w_d[j, w] <= 1)

        #additional
        for j in range(self.jobs_num):
            for w in range(self.workers_num[j], self.workers_num_align):
                for s in range(self.ina_num):
                    m.addConstr(y_j_w_s[j, w, s] == 0)
                    m.addConstr(gamma_j_w_s[j, w, s] == 0)
                m.addConstr(y_j_w_d[j, w] == 0)
                m.addConstr(gamma_j_w_d[j, w] == 0)

        # 12
        m.addConstrs((gamma_j_w_d[j, w] + sum(gamma_j_w_s[j, w, _] for _ in range(self.ina_num))
                      >= gamma_j[j] for j in range(self.jobs_num) for w in range(self.workers_num[j])), name='12')


        m.addConstrs(obj <= gamma_j[j] / self.jobs_size[j] for j in range(self.jobs_num))

        m.addConstrs(x_s[i] == 1 for i in range(self.ina_num))
        # 1
        m.addConstr(sum(x_s[i] for i in range(self.ina_num)) <= self.ina_num, name='1')
        # 2
        m.addConstrs((y_j_w_s[j, w, s] <= x_s[s] for j in range(self.jobs_num) for w in range(self.workers_num_align) for s in range(self.ina_num)), name='2')
        # 3
        m.addConstrs((sum(y_j_w_s[j, w, _] for _ in range(self.ina_num)) + y_j_w_d[j, w] == 1 for j in range(self.jobs_num) for w in range(self.workers_num[j])), name='3')
        # 5
        m.addConstrs((sum(gamma_j_w_s[j, w, i] for j in range(self.jobs_num) for w in range(self.workers_num[j])) <= self.Cs * x_s[i] for i in range(self.ina_num)), name='5')
        # 6
        m.addConstrs((gamma_j_w_s[j, w, s] <= self.Cs * y_j_w_s[j, w, s] for w in range(self.workers_num_align) for s in range(self.ina_num) for j in range(self.jobs_num)), name='6')
        m.addConstrs((gamma_j_w_d[j, w] <= self.Ps * y_j_w_d[j, w] for w in range(self.workers_num_align) for j in range(self.jobs_num)), name='6')
        # 7
        m.addConstrs((gamma_j_w_s[j, w, i] <= gamma_j_s_d[j, i] for j in range(self.jobs_num) for w in range(self.workers_num[j]) for i in range(self.ina_num)), name='7')
        # 8
        for v in self.mapping.values():
            e = v[0]  # 链路
            Ce = v[1]  # 链路容量

            le = [0 for _ in range(self.jobs_num)]
            for j in range(self.jobs_num):
                i_j_w_s, i_j_w_d, i_s_d = self.link_capacity_limitation_mjob(j, e) # job j的worker到switch和ps_j 以及 switch到ps_j经过链路e的（src,dst）列表
                le[j] = sum(gamma_j_w_s[j, i11[0], i11[1]] for i11 in i_j_w_s) + sum(
                    gamma_j_w_d[j][i12[0]] for i12 in i_j_w_d) + sum(gamma_j_s_d[j, i2[0]] for i2 in i_s_d) #job j在链路e上产生的流量
                if e == (self.sd_id[j], self.ps_id[j]) or e == (self.ps_id[j], self.sd_id[j]):
                        Ce = self.Ps
            m.addConstr(sum(le[j] for j in range(self.jobs_num)) <= Ce, name='8')
        
        m.setParam("OutputFlag", 0) # disable output to a logfile
        m.optimize()
        obj_relax_x = obj.X
        gamma_j_w_s_mid = gamma_j_w_s.X
        gamma_j_w_d_mid = gamma_j_w_d.X
        gamma_j_s_d_mid = gamma_j_s_d.X




        gamma_j_w = [] 
        for j in range(self.jobs_num):
            gamma_j_w.append([0 for _ in range(self.workers_num[j])])
            for w in range(self.workers_num[j]):
                gamma_j_w[j][w] = sum(gamma_j_w_s_mid[j, w, i] for i in range(self.ina_num)) + gamma_j_w_d_mid[j, w]

        choice = np.zeros((self.jobs_num, self.workers_num_align),dtype=int)
        p_j_w_s = np.zeros([self.jobs_num, self.workers_num_align, self.ina_num + 1])  # 多一维用于记录PS分到的梯度流

        for j in range(self.jobs_num): # 选择gamma最大的switch作为worker的聚合点
            for w in range(self.workers_num[j]):
                for s in range(self.ina_num):
                    if gamma_j_w[j][w] == 0:
                        # 如果分母为零，可以将结果设为0或其他合适的默认值
                        p_j_w_s[j][w][s] = 0
                        p_j_w_s[j][w][-1] = 0
                    else:
                        p_j_w_s[j][w][s] = gamma_j_w_s_mid[j][w][s] / gamma_j_w[j][w]
                p_j_w_s[j][w][-1] = gamma_j_w_d_mid[j, w] / gamma_j_w[j][w]
                p_j_w_s_max = max(p_j_w_s[j][w])
                max_index = [i for i in range(self.ina_num + 1) if p_j_w_s[j][w][i] == p_j_w_s_max]
                if len(max_index) == 1:
                    choice[j][w] = max_index[0]
                else:
                    choice[j][w] = random.choice(max_index)

        # 更新y到0 1变量
        constr_refs = []
        y_j_w_s_new = np.zeros([self.jobs_num, self.workers_num_align, self.ina_num])
        y_j_w_d_new = np.zeros([self.jobs_num, self.workers_num_align])

        for j in range(self.jobs_num):
            for w in range(self.workers_num[j]):
                if choice[j][w] != self.ina_num:
                    y_j_w_s_new[j][w][choice[j][w]] = 1
                else:
                    y_j_w_d_new[j][w] = 1

        for j in range(self.jobs_num):
            for w in range(self.workers_num_align):
                for s in range(self.ina_num):
                    constr_refs.append(m.addConstr(y_j_w_s[j, w, s] == y_j_w_s_new[j, w, s]))
                    #y_j_w_s[j, w, s].start = y_j_w_s_new[j, w, s]

            for w in range(self.workers_num_align):
                constr_refs.append(m.addConstr(y_j_w_d[j, w] == y_j_w_d_new[j, w]))
                #y_j_w_d[j, w].start = y_j_w_d_new[j, w]

        m.setParam("OutputFlag", 0) # disable output to a logfile
        m.optimize()

        obj_initial = obj.X
        Jct_initial = 1/obj_initial
        print("iiiiii",Jct_initial)
        gamma_j_w_s_new = gamma_j_w_s.X
        gamma_j_s_d_new = gamma_j_s_d.X
        gamma_j_w_d_new = gamma_j_w_d.X





        x = gamma_j_w_s_mid
        y = gamma_j_w_d_mid
        y_expanded = y[:, :, np.newaxis]
        
        tags_3d = np.concatenate((x, y_expanded), axis=2)
        num_jobs, max_workers_size, ap_num = tags_3d.shape

        tag_list = []
        for job in range(num_jobs):
            tag_sub_list = []
            tor_switch_workers = self.tor_switch_workers[job]
            count = 0
            
            for switch, num in tor_switch_workers.items():
                if num:
                    index_list = [count + index for index in range(num)]
                    tag_sub_list.append((tags_3d[job, index_list[0]], index_list))
                    count += num
            tag_list.append(tag_sub_list)
        
        bia = 0
        mark = []
        bestjct = 100000
        while True:
            y_j_w_s_val = np.zeros((num_jobs, max_workers_size, ap_num-1))
            y_j_w_d_val = np.zeros((num_jobs, max_workers_size))

            dis_list = []
            for job in range(num_jobs):
                job_tags = tag_list[job]

                alloc_list = []
                for tag, indices in job_tags:
                    count = len(indices)
                    tag = [i/sum(tag) for i in tag]

                    filtered = [(i, v) for i, v in enumerate(tag)]
                    filtered = sorted(filtered, key=lambda x: x[1], reverse=True)
                    # ###
                    # max_val = max(tag)
                    # filtered = [(i, v) for i, v in enumerate(tag) if max_val - v <= 0.25]
                    # filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:int(len(tag)*0.51)]

                    # 构造最终结果
                    selected = {i for i, _ in filtered}
                    filtered = [v if i in selected else 0 for i, v in enumerate(tag)]
                    filtered = [i/sum(filtered) for i in filtered]

                    raw = [p * count for p in filtered]

                    alloc = [int(x) for x in raw]
                    remainder = count - sum(alloc)
                    remainders = [(i, raw[i] - alloc[i]) for i in range(ap_num)]
                    remainders.sort(key=lambda x: -x[1])
                    for i in range(remainder):
                        alloc[remainders[i][0]] += 1

                    alloc_list.append(alloc)
                    # ###
                    # workers_allocated = 0    
                    # for index, num in enumerate(alloc):
                    #     for i in range(num):
                    #         if index != ap_num-1:
                    #             y_j_w_s_val[job, indices[workers_allocated], index] = 1
                    #         else:
                    #             y_j_w_d_val[job, indices[workers_allocated]] = 1
                    #         workers_allocated += 1

                #''' 微调alloc_list
                ga_all = sum(job_tags[0][0])
                ps_permit = int(self.Ps//ga_all)
                summed_alloc = np.sum(np.array(alloc_list), axis=0)
                nonzero_count = np.count_nonzero(summed_alloc[:-1])
                dis = summed_alloc[-1] + nonzero_count - ps_permit
                dis_list.append(dis)
                if job in mark:
                    dis -= bia
                if dis > 0:
                    for alloc in alloc_list:
                        nonzero_indices = [i for i, val in enumerate(alloc[:-1]) if val != 0]
                        if not nonzero_indices:
                            continue  # 无目标跳过
                        # Calculate the total of the non-zero values (excluding the last element)
                        total_nonzero = sum(alloc[i] for i in nonzero_indices)
                        # Calculate the allocation proportion for each non-zero element
                        proportions = [alloc[i] / total_nonzero for i in nonzero_indices]
                        # Distribute the remaining `alloc[-1]` proportionally
                        allocation = min(dis, alloc[-1])
                        raw = [p * allocation for p in proportions]
                        alloc_num_list = [int(x) for x in raw]
                        remainder = allocation - sum(alloc_num_list)
                        remainders = [(i, raw[i] - alloc_num_list[i]) for i in range(len(raw))]
                        remainders.sort(key=lambda x: -x[1])
                        for i in range(remainder):
                            alloc_num_list[remainders[i][0]] += 1

                        for i, idx in enumerate(nonzero_indices):
                            alloc[idx] += alloc_num_list[i]
                        alloc[-1] -= allocation
                        dis -= allocation
                            
                        # 如果 dis 减到 0，跳出循环
                        if dis <= 0:
                            break

                # If there is still remaining `dis`, we continue to allocate proportionally
                if dis > 0:
                    summed_alloc = np.sum(np.array(alloc_list), axis=0)
                    nonzero_indices = np.nonzero(summed_alloc)[0]
                    indices_of_min_k_nonzero = nonzero_indices[np.argsort(summed_alloc[nonzero_indices])[:dis]]
                    # rank_list = []
                    # for alloc in alloc_list:
                    #     rank_list.append(self.rank_nonzero(alloc))
                    # rank_score = np.array(self.nonzero_column_mean(rank_list))
                    # nonzero_indices = np.nonzero(rank_score)[0]
                    # indices_of_min_k_nonzero = nonzero_indices[np.argsort(-rank_score[nonzero_indices])[:dis]]
                        
                    for alloc in alloc_list:
                        nonzero_indices = [i for i, val in enumerate(alloc[:-1]) if val != 0]
                        if not nonzero_indices:
                            continue  # 无目标跳过

                        # 排除 `index_of_min_dis_nonzero` 的非零元素索引
                        remaining_nonzero_indices = [i for i in nonzero_indices if i not in indices_of_min_k_nonzero]

                        # 计算剩余非零元素的总和
                        total_nonzero = sum(alloc[i] for i in remaining_nonzero_indices)
                        
                        if total_nonzero == 0:  # 防止分母为0
                            continue

                        # 按比例计算分配比例
                        proportions = [alloc[i] / total_nonzero for i in remaining_nonzero_indices]
                        
                        allocation = sum(alloc[i] for i in indices_of_min_k_nonzero)
                        raw = [p * allocation for p in proportions]
                        alloc_num_list = [int(x) for x in raw]
                        remainder = allocation - sum(alloc_num_list)
                        remainders = [(i, raw[i] - alloc_num_list[i]) for i in range(len(raw))]
                        remainders.sort(key=lambda x: -x[1])
                        for i in range(remainder):
                            alloc_num_list[remainders[i][0]] += 1

                        for i, idx in enumerate(remaining_nonzero_indices):
                            alloc[idx] += alloc_num_list[i]
                        for idx in indices_of_min_k_nonzero:
                            alloc[idx] = 0


                workers_allocated = 0    
                for alloc in alloc_list:
                    for index, num in enumerate(alloc):
                        for i in range(num):
                            if index != ap_num-1:
                                y_j_w_s_val[job, workers_allocated, index] = 1
                            else:
                                y_j_w_d_val[job, workers_allocated] = 1
                            workers_allocated += 1
            #'''
            for c in constr_refs:
                m.remove(c)
            m.update()

            for j in range(self.jobs_num):
                for w in range(self.workers_num_align):
                    for s in range(self.ina_num):
                        constr_refs.append(m.addConstr(y_j_w_s[j, w, s] == y_j_w_s_val[j, w, s]))

                for w in range(self.workers_num_align):
                    constr_refs.append(m.addConstr(y_j_w_d[j, w] == y_j_w_d_val[j, w]))
            m.setParam("OutputFlag", 0)
            m.optimize()
            obj_val = obj.X
            Jct = 1/obj_val
            print('hhhhhh',Jct)
            gamma_j_new = gamma_j.X
            gamma_j_s_d_new = gamma_j_s_d.X
            gamma_j_w_s_new = gamma_j_w_s.X
            gamma_j_w_d_new = gamma_j_w_d.X
            x_s_new = x_s.X
            y_j_w_s_new = y_j_w_s.X
            y_j_w_d_new = y_j_w_d.X

            if Jct < bestjct:
                bestjct = Jct
                bestyjws = y_j_w_s_new
                bestyjwd = y_j_w_d_new
                bia += 1
                mark = [i for i, val in enumerate(dis_list) if val == max(dis_list)]
                gamma_j_val = gamma_j_new
                gamma_j_s_d_val = gamma_j_s_d_new
                gamma_j_w_s_val = gamma_j_w_s_new
                gamma_j_w_d_val = gamma_j_w_d_new
                x_s_val = x_s_new
                y_j_w_s_val = y_j_w_s_new
                y_j_w_d_val = y_j_w_d_new
            else:
                break
        return bestjct

    def links_set(self, src, dst):  # get src to dst links
        path = self.allPathDict[src][dst]
        links_set = []
        for i in range(len(path)-1):
            link = (path[i], path[i+1])
            links_set.append(link)
        return links_set

    def path_to_links(self, src_set, dst_set):  # 给一个源集合，目的集合，返回任一源到任一目的的所有链路
        path_to_links = dict()
        for src in src_set:
            for dst in dst_set:
                links_set = self.links_set(src, dst)
                path_to_links[(src, dst)] = links_set
        return path_to_links

    def l_i(self, e, src_set, dst_set):  # 把用到链路e的（w, s）挑出来，相加
        index_set = []
        path_to_links = self.path_to_links(src_set, dst_set)
        for key in path_to_links.keys():
            if e in path_to_links[key] or e[::-1] in path_to_links[key]:
            # if e in path_to_links[key] in path_to_links[key]:
                row = src_set.index(key[0])
                col = dst_set.index(key[1])
                index_set.append((row, col))
        return index_set #经过链路e的(src,dst)列表

    def link_capacity_limitation_mjob(self, j, e):
        i_j_w_s = self.l_i(e, self.workers_id[j], self.ina_id)
        i_j_w_d = self.l_i(e, self.workers_id[j], [self.ps_id[j]])
        i_s_d = self.l_i(e, self.ina_id, [self.ps_id[j]])

        return i_j_w_s, i_j_w_d, i_s_d