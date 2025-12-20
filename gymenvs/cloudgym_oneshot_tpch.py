import numpy as np
import networkx as nx
from math import inf
import copy
import gym

from tensorboardX import SummaryWriter
import time

from playground.DAG.utils.runtime_calculation import total_energy, total_reliability
from playground.DAG.utils.get_irr import get_irrelevant

from datasets.TPCH_dag_data.dag_graph import DAGraph

def is_valid_topological_order(G, sequence):
    if not nx.is_directed_acyclic_graph(G):
        return False  # 
    node_to_order = {node: i for i, node in enumerate(sequence)}  #
    for u, v in G.edges():
        if node_to_order[u] > node_to_order[v]:  # 
            return False
    return True

NULL_ACTION =-1
max_node_num = 2000
max_edge_num = 12000
raw_node_attr_length = 2
raw_edge_attr_length = 1

def get_dict_from_networkx(dag):
    irr_pairs = get_irrelevant(dag)
    irr_pair_num = len(irr_pairs)

    xs, edge_indexs, edge_attrs, valid_masks,  irr_adjs= list(), [[],[]], list(), list(), list()
    node_num, edge_num = dag.number_of_nodes(), dag.number_of_edges()
    
    for node in dag.nodes:
        last_node = node
        task_feature = dag.nodes[node]['features']
        raw_v = [task_feature[0], task_feature[1]]
        xs.append(raw_v)
        irr_adjs.append(dag.nodes[node]['irr_adj'])
        valid_masks.append(1)
        for succ in dag.successors(node):
            edge_indexs[0].append(node)
            edge_indexs[1].append(succ)
            #edge_attr = dag[node][succ]['comm_ready']
            edge_attrs.append([1.0])
    
    
    ob_dict = {}
    xs = np.array(xs, dtype=np.float32)
    #xs = pp.scale(xs)  # 
    ob_dict['x'] = np.pad(xs, ((0,max_node_num - node_num), (0,0)))
    edge_indexs = np.array(edge_indexs, dtype=np.int64)
    ob_dict['edge_index'] = np.pad(edge_indexs, ((0,0), (0, max_edge_num - edge_num)))
    edge_attrs = np.array(edge_attrs, dtype=np.float32)
    ob_dict['edge_attr'] = np.pad(edge_attrs, ((0, max_edge_num - edge_num), (0, 0))) #
    valid_masks = np.array(valid_masks, dtype=np.int8)
    ob_dict['valid_mask'] = np.pad(valid_masks, (0, max_node_num - node_num))

    adj_mat = nx.to_numpy_array(dag.reverse(), nodelist=sorted(dag.nodes()))
    ob_dict['adj_mat'] = np.pad(adj_mat, ((0, max_node_num - node_num), (0, max_node_num - node_num)))

    irr_adjs = np.array(irr_adjs, dtype = np.int8)
    ob_dict['irr_adj'] = np.pad(irr_adjs, ((0, max_node_num - node_num), (0,max_node_num - node_num)))

    ob_dict['node_num'] = [node_num]
    ob_dict['edge_num'] = [edge_num]
    #ob_dict['irr_pair_num'] = [irr_pair_num]
    return ob_dict

class running_tpch:
    def __init__(self, merged_dag):
        self.now_time = 0
        self.now_resource = 1.0
        self.ready_nodes = set()
        self.running_nodes = set()
        self.finished_nodes = set()
                
        self.dag = copy.deepcopy(merged_dag)
        for node in self.dag.nodes:
            if self.dag.in_degree(node) == 0:
                self.dag.nodes[node]['is_ready'] = True   
                self.ready_nodes.add(node)
            else:
                self.dag.nodes[node]['is_ready'] = False   
            self.dag.nodes[node]['is_running'] = False
            self.dag.nodes[node]['is_finished'] = False
            self.dag.nodes[node]['start_time'] = 0
            
    def get_speedup_base(self):
        speedup_base = 0
        for node in self.dag.nodes:
            speedup_base += self.dag.nodes[node]['features'][0]
        return speedup_base

    def simulate(self, schedule_list):
        print(self.get_speedup_base())
        st = lambda x: self.dag.nodes[x]['start_time']
        ft = lambda x: self.dag.nodes[x]['start_time'] + self.dag.nodes[x]['features'][0]

        oft_list, oft = [0] * self.dag.number_of_nodes(), 0
        for node in schedule_list:
            duration = self.dag.nodes[node]['features'][0]
            resource = self.dag.nodes[node]['features'][1]
            
            while self.now_resource < resource or (self.dag.nodes[node]['is_ready'] is False):
                if not self.running_nodes:
                    break
                #    raise ValueError("total resource is not enough")
                first_finish_node = min(self.running_nodes, key=ft)
                first_finish_time = ft(first_finish_node)
                self.now_time = first_finish_time
                # 
                self.dag.nodes[first_finish_node]['is_running'] = False 
                self.dag.nodes[first_finish_node]['is_finished'] = True 
                self.running_nodes.remove(first_finish_node)
                self.finished_nodes.add(first_finish_node)
                #  
                self.now_resource = self.now_resource + self.dag.nodes[first_finish_node]['features'][1]
                # 
                for succ in self.dag.successors(first_finish_node):
                    is_ready = True
                    for pre_of_succ in self.dag.predecessors(succ):
                        if self.dag.nodes[pre_of_succ]['is_finished'] == False:
                            is_ready = False
                    if is_ready:
                        self.dag.nodes[succ]['is_ready'] = True
                        self.ready_nodes.add(succ)
            
            if self.now_resource >= resource:
                self.now_resource = self.now_resource - resource
                self.dag.nodes[node]['start_time'] = self.now_time
                self.dag.nodes[node]['is_ready'] = False
                self.dag.nodes[node]['is_running'] = True
                self.ready_nodes.remove(node)
                self.running_nodes.add(node)

                if ft(node) > oft:
                    oft = ft(node)
                oft_list[node] = -oft
                
        latest_finish_node = max(schedule_list, key=ft)
        makespan = ft(latest_finish_node)            

        #rew1: (end-lastend) / (end - start)
        #dense_rew.append(-(end-last_end)*1.0 / (end - start))
            
        # rew2: (end - ready) / (end - start)
        #dense_rew.append(-(end - ready_time)*1.0 / (end - start))
            
        #rew3: end
        #dense_rew = [-ft(node) for node in self.dag.nodes]
        #dense_rew.append(-end)

        #rew4: OFT
        #dense_rew = oft_list
        #print(dense_rew)
        
        #rew5: OFT improvement
        #dense_rew.append(-oftimp)

        # rew6 overall makespan
        #dense_rew = [-makespan for _ in self.dag.nodes]

        # rew7 mksp - st
        #dense_rew = [-(makespan - st(node)) for node in self.dag.nodes]
            
        #rew8 mksp - last
        #dense_rew.append(-(mksp - last_oft))

        #rew9 minstart when time reversed
        '''
        dense_rew = [0] * len(schedule_list)
        rev_schedule_list = schedule_list[::-1]
        minstart = st(rev_schedule_list[0])
        for node in rev_schedule_list:
            now_st = st(node)
            if now_st < minstart:
                minstart = now_st
            dense_rew[node] = - minstart
        '''
        #dense_rew.append(-minstart)

        #rew10 st
        dense_rew = [  st(node) for node in self.dag.nodes]
        #dense_rew = [ - st(node) for node in self.dag.nodes]
        
        return dense_rew, makespan
        
    
class CloudGym_oneshot_tpch(gym.Env):
    NULL_ACTION = NULL_ACTION
    max_node_num = max_node_num
    max_edge_num = max_edge_num
    raw_node_attr_length = raw_node_attr_length
    raw_edge_attr_length = raw_edge_attr_length
    
    metadata = {'render.modes': ['human']}
    def __init__(self, dag, workflow_index, dag_graph):
        self.dag = dag
        self.workflow_index = workflow_index
        
        self.dag_graph = dag_graph
        
        self.heur_mksp, self.st_list, self.ft_list = dag_graph.critical_path_scheduling(self.dag)
        #self.heur_mksp, self.st_list, self.ft_list = dag_graph.shortest_first_time(self.dag)
        self.dag = self.dag.reverse(copy=True)
        
        priority_list = [st for st in self.st_list]
        self.heur_list = np.argsort(priority_list)
        tpch_instance = running_tpch(merged_dag = self.dag)
        self.dense_rew_bsln, mksp = tpch_instance.simulate(self.heur_list)
        self.current_episode = 0
        #self.logpath = logpath
        #if self.logpath:
        #    self.summarywriter = SummaryWriter(log_dir = logpath)

        self.global_ob = None
        self.cum_rew = 0
        
        space_dict = {}
        space_dict['node_num'] = gym.spaces.Box(low = 0, high = max_node_num, shape = (1,), dtype=np.int64)  # 
        space_dict['edge_num'] = gym.spaces.Box(low = 0, high = max_edge_num, shape = (1,), dtype=np.int64) # 
        #space_dict['irr_pair_num'] = gym.spaces.Box(low = 0, high = max_edge_num, shape = (1,), dtype = np.int64) # 
        space_dict['x'] = gym.spaces.Box(low = 0, high = inf, shape = (max_node_num, raw_node_attr_length), dtype = np.float32)
        space_dict['edge_index'] = gym.spaces.Box(low = 0, high = max_edge_num, shape = (2, max_edge_num), dtype = np.int64)
        space_dict['edge_attr']  = gym.spaces.Box(low = 0, high = inf, shape = (max_edge_num, raw_edge_attr_length), dtype = np.float32)
        space_dict['valid_mask'] = gym.spaces.MultiBinary(max_node_num) # 
        
        space_dict['adj_mat'] = gym.spaces.Box(low = 0, high = 1, shape = (max_node_num, max_node_num), dtype = np.int8)
        #space_dict['irr_pair'] = gym.spaces.Box(low = 0, high = max_node_num, shape = (max_edge_num, 2), dtype = np.int64)
        space_dict['irr_adj'] = gym.spaces.Box(low = 0, high = 1, shape = (max_node_num, max_node_num), dtype = np.int8)
        
        self.observation_space = gym.spaces.Dict(space_dict)   
        self.action_space = gym.spaces.Box(low = -1e+8, high = 1e+8, shape = (max_node_num,), dtype = np.float32)

    def reset(self):
        ''' end the previous simulation'''
        if self.global_ob is None:
            self.global_ob = get_dict_from_networkx(self.dag)
        self.cum_rew = 0
        exit(0)
        return self.global_ob

    def step(self, action):
        tpch_instance = running_tpch(merged_dag = self.dag)
        priority_list = action[0 : self.dag.number_of_nodes()]
        #LS_topo = [t for t in nx.lexicographical_topological_sort(self.dag, key=lambda x: priority_list[x])]
        LS_topo = [t for t in nx.lexicographical_topological_sort(self.dag, key=lambda x: -priority_list[x])] # descending topo sort for rew_2 of one-sh
        
        dense_rew, mksp = tpch_instance.simulate(LS_topo)
        #dense_rew, mksp = tpch_instance.simulate(priority_list)
        is_done = True
        ''' 暂时只有单工作流'''
        if self.global_ob is None:
            self.global_ob = get_dict_from_networkx(self.dag)

        tag_scalar_dict = {
            'makespan': mksp,
            'energycost': 0,
            'reliability': 0
            }
        #self.summarywriter.add_scalars(main_tag = 'atag', tag_scalar_dict = tag_scalar_dict, global_step = self.current_episode)
        self.current_episode += 1

        reward = - mksp
        
        #heur rew baseline
        ''' heuristic as baseline'''
        dense_rew = [rew - b for rew, b in zip(dense_rew, self.dense_rew_bsln)]
        
        ''' average baseline'''
        #b = np.mean(dense_rew)
        #dense_rew = [rew - b for rew in dense_rew]
        
        ''' smoonthed average baseline'''
        '''
        alpha = 0.9
        rev_dense_rew = dense_rew[::-1]
        baselines = []
        baselines.append(rev_dense_rew[0])
        for i in range(1, len(dense_rew)):
            b = alpha * rev_dense_rew[i] + (1-alpha)*baselines[i-1]
            baselines.append(b)
        baselines = baselines[::-1]
        dense_rew =[rew - b for rew,b  in zip(dense_rew, baselines)]
        '''
        # 
        dense_rew = np.array(dense_rew, dtype=np.float32)
        dense_rew = np.pad(dense_rew, (0, max_node_num - dense_rew.shape[0]))
        
        return self.global_ob, reward, is_done, {'obj':[mksp, 0, 0], 'dense_rew':dense_rew}

    def render(self):
        pass




if __name__ == '__main__':
    
    from datasets.TPCH_dag_data.dag_generator import load_tpch_jobs
    dags = [50,100,150]
    for nd in dags:
        dags, roots, graphs = load_tpch_jobs(num_init_dags=nd,tid=0)
        merged_dag = nx.DiGraph()
        merged_dag.add_node(0, features=[0.0, 0.0])
        merged_dag.graph["features"] = [0.0, 0.0]


        for dag in dags:
            merged_dag.add_nodes_from(dag.nodes(data=True))
            merged_dag.add_edges_from(dag.edges(data=True))    
        
        dag_graph = DAGraph(resource_dim=1, feature_dim=2)
        

        env = CloudGym_oneshot_tpch(dag=merged_dag, workflow_index=1, dag_graph = dag_graph)
        print('TPC-h', nd, env.heur_mksp)

    
    #env.step([i for i in range(merged_dag.number_of_nodes())])
    #print(mksp)
