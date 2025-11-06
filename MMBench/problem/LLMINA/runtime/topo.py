# First networkx library is imported
# along with matplotlib
import networkx as nx
import random

# Defining a Class
class FatTree:
    def __init__(self, k, basic_band, w, band, k1):
        self.k = k  # k元拓扑
        self.basic_band = basic_band  # 底层带宽(host和tor交换机)
        self.G_content = []
        self.G = nx.Graph()
        self.all_workers_id = []
        self.tors_id = []
        self.aggrs_id = []
        self.cores_id = []
        self.mapping = dict()
        self.bandwidth_mapping = dict()  # key: (node1, node2), value: bandwidth
        self.all_nodes = 0  # 所有节点数
        self.all_switches_id = []
        self.band = band
        self.hosts_num = w
        self.create(w, k1)
        self.allPathDict = dict(nx.all_pairs_shortest_path(self.G))
        self.allPathDict_all = dict(nx.all_pairs_all_shortest_paths(self.G))


    # Driver code
    def create(self, w, k1): # w 每个tor交换机下挂载的worker数
        random.seed(1)
        # 正常 fattree
        band_worker_edge = self.basic_band  # 100
        # band_edge_aggr = band_worker_edge * w * k1
        # band_aggr_core = band_edge_aggr * self.k / 4
        band_edge_aggr = int(band_worker_edge * w * k1 * 0.5) 
        band_aggr_core = int(band_edge_aggr * 0.5)
        band = self.band

        # Calculating the pods, workers, core switches, aggregation switches
        CoreSwitch_Count = int((self.k / 2) ** 2)  # 核心层交换机/顶层 个数 16  9
        AggrSwitch_Count = int((self.k ** 2) / 2)  # 汇聚层交换机/二层 个数 32  18
        EdgeSwitch_Count = AggrSwitch_Count  # 接入层交换机/底层 个数 tor   32  18
        EdgeSwitch_Count_In_Pods = int(self.k // 2)  # 一个pod内的tor个数  4    3
        AggrSwitch_Count_In_Pods = EdgeSwitch_Count_In_Pods  # 一个 pod 里面的汇聚层交换机/二层 个数
        pods_Count = int(self.k)  # pod 的个数
        Workers_In_Tor = int(w)  # 一个tor有w个host
        Workers_In_Pod = Workers_In_Tor * EdgeSwitch_Count_In_Pods  # 一个pod内的worker数
        Workers_Count = pods_Count * Workers_In_Pod
        AllSwitch_Count = CoreSwitch_Count + EdgeSwitch_Count + AggrSwitch_Count  # 所有交换机个数
        Elements_Count = AllSwitch_Count + Workers_Count  # 所有节点个数 交换机 + worker
        self.all_workers_id = [i for i in range(0, Workers_Count)]
        self.tors_id = [i for i in range(Workers_Count, Workers_Count + EdgeSwitch_Count)]
        self.aggrs_id = [i for i in
                         range(Workers_Count + EdgeSwitch_Count, Workers_Count + EdgeSwitch_Count + AggrSwitch_Count)]
        self.cores_id = [i for i in range(Workers_Count + EdgeSwitch_Count + AggrSwitch_Count, Elements_Count)]
        self.all_nodes = Elements_Count
        self.all_switches_id = self.tors_id + self.aggrs_id + self.cores_id
        # Each Element has connection to itself, First we add those connections to output
        # counter is way that we address number of edge switches
        # counter starts at number of workers because 0 should be included in worker counts
        counter = Workers_Count

        # worker counter is a way to address the worker for mapping between edge switches and workers
        Worker_counter = 0
        # program goes stage by stage bottom up, first we create links between edge and workers
        index = 0  # link index
        for i in range(pods_Count):
            for j in range(EdgeSwitch_Count_In_Pods):
                for l in range(w):
                    self.G_content.append([Worker_counter, counter, band_worker_edge])
                    self.bandwidth_mapping[(Worker_counter, counter)] = band_worker_edge

                    self.mapping[index] = ((Worker_counter, counter), band_worker_edge)
                    index += 1
                    # band_worker_edge += 100  # test 异构
                    self.G_content.append([counter, Worker_counter, band_worker_edge])
                    self.bandwidth_mapping[(counter, Worker_counter)] = band_worker_edge
                    self.mapping[index] = ((counter, Worker_counter), band_worker_edge)
                    index += 1
                    # band_worker_edge += 100  # test 异构
                    Worker_counter += 1
                counter += 1

        # Handling connection between Aggregation switches and edge switches
        for i in range(pods_Count):
            for j in range(AggrSwitch_Count_In_Pods):
                for l in range(self.k // 2):
                    # 随机带宽
                    if band == 'rand':
                        band_edge_aggr_rand = random.randint(band_aggr_core, band_edge_aggr)
                    elif band == 'fixed':
                        band_edge_aggr_rand = band_edge_aggr
                    self.G_content.append([Worker_counter, counter, band_edge_aggr_rand])
                    self.bandwidth_mapping[(Worker_counter, counter)] = band_edge_aggr_rand
                    self.mapping[index] = ((Worker_counter, counter), band_edge_aggr_rand)
                    index += 1
                    self.G_content.append([counter, Worker_counter, band_edge_aggr_rand])
                    self.bandwidth_mapping[(counter, Worker_counter)] = band_edge_aggr_rand
                    self.mapping[index] = ((counter, Worker_counter), band_edge_aggr_rand)
                    index += 1
                    Worker_counter += 1
                Worker_counter -= self.k // 2
                counter += 1
            Worker_counter += self.k // 2

        # Implementing link between aggregation switches and core switches
        # we create core because when link reaches the end of cores we want to reset cores
        core = []
        for i in range(CoreSwitch_Count):
            core.append(counter)
            counter += 1
        p = 0
        for i in range(pods_Count):
            for j in range(AggrSwitch_Count_In_Pods):
                for l in range(self.k // 2):
                    counter = counter - self.k + 1
                    #随机带宽
                    if band == 'rand':
                        band_aggr_core_rand = random.randint(band_aggr_core, band_aggr_core)
                    elif band == 'fixed':
                        band_aggr_core_rand = band_aggr_core

                    self.G_content.append([Worker_counter, core[p], band_aggr_core_rand])
                    self.bandwidth_mapping[(Worker_counter, core[p])] = band_aggr_core_rand
                    self.mapping[index] = ((Worker_counter, core[p]), band_aggr_core_rand)
                    index += 1
                    self.G_content.append([core[p], Worker_counter, band_aggr_core_rand])
                    self.bandwidth_mapping[(core[p], Worker_counter)] = band_aggr_core_rand
                    self.mapping[index] = ((core[p], Worker_counter), band_aggr_core_rand)
                    index += 1
                    p += 1
                    if p == (self.k // 2) ** 2:
                        p = 0
                Worker_counter += 1
            p = p + 1
        self.G.add_weighted_edges_from(self.G_content)


class SpineLeaf:
    def __init__(self, tops_num, tors_num, basic_band, w, band, k1=1):
        self.band_worker_tor = basic_band
        self.band_tor_top = k1 * 20 * basic_band
        self.tops_num = tops_num
        self.tors_num = tors_num
        self.hosts_num = tors_num
        self.workers_num_in_tor = int(w)
        self.basic_band = basic_band  # 底层带宽
        self.G_content = []
        self.G = nx.Graph()
        self.all_workers_id = []
        self.tors_id = []
        self.spines_id = []
        self.mapping = dict() #key: edge_id, value: (node1, node2), bandwidth
        self.bandwidth_mapping = dict()  # key: (node1, node2), value: bandwidth
        self.workers_num = self.workers_num_in_tor * self.tors_num
        self.all_nodes = 0  # 所有节点数
        self.all_switches_id = []
        self.band = band
        self.create()
        #self.allPathDict = dict(nx.all_pairs_dijkstra_path(self.G))
        self.allPathDict = dict(nx.all_pairs_shortest_path(self.G))
        self.allPathDict_all = dict(nx.all_pairs_all_shortest_paths(self.G))


    def create(self):
        band_worker_tor = int(self.band_worker_tor)  # 字面意思
        band_tor_top = int(self.band_tor_top)  # 按照经典值给
        workers_num = self.workers_num_in_tor * self.tors_num
        band = self.band
        self.all_workers_id = [i for i in range(workers_num)]
        self.all_nodes = workers_num + self.tors_num + self.tops_num
        self.tors_id = [i for i in range(workers_num, workers_num + self.tors_num)]
        self.spines_id = [i for i in range(workers_num + self.tors_num, self.all_nodes)]
        self.all_switches_id = self.tors_id + self.spines_id
        index = 0  # 链路编号
        for i in range(self.tors_num):
            tor_id = self.tors_id[i]
            workers_id_in_tor = self.all_workers_id[i*self.workers_num_in_tor: (i+1)*self.workers_num_in_tor]
            for w in workers_id_in_tor:
                self.G_content.append([w, tor_id, band_worker_tor])
                self.bandwidth_mapping[(w, tor_id)] = band_worker_tor

                self.mapping[index] = ((w, tor_id), band_worker_tor)
                index += 1
                # band_worker_edge += 100  # test 异构
                self.G_content.append([tor_id, w, band_worker_tor])
                self.bandwidth_mapping[(tor_id, w)] = band_worker_tor
                self.mapping[index] = ((tor_id, w), band_worker_tor)
                index += 1
        # print('max',index)  288
        for i in range(self.tors_num):
            tor_id = self.tors_id[i]
            for top in self.spines_id:
                # band_tor_top_rand = random.random() * band_tor_top
                # 随机带宽
                if band == 'rand':
                    band_tor_top_rand = random.randint(band_worker_tor, band_tor_top)
                elif band == 'fixed':
                    band_tor_top_rand = band_tor_top

                self.G_content.append([tor_id, top, band_tor_top_rand])
                self.bandwidth_mapping[(tor_id, top)] = band_tor_top_rand

                self.mapping[index] = ((tor_id, top), band_tor_top_rand)
                # print(self.mapping[index])
                index += 1
                self.G_content.append([top, tor_id, band_tor_top_rand])
                self.bandwidth_mapping[(top, tor_id)] = band_tor_top_rand

                self.mapping[index] = ((top, tor_id), band_tor_top_rand)
                index += 1
        self.G.add_weighted_edges_from(self.G_content)

