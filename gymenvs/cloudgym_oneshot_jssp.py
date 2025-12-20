import numpy as np
import networkx as nx
from math import inf
import heapq
from collections import defaultdict
import copy
import gym
import time

from tensorboardX import SummaryWriter

from playground.DAG.utils.runtime_calculation import total_energy, total_reliability
from playground.DAG.utils.get_irr import get_irrelevant, get_irrelevant_jssp

def cat_action_machinepri(action, machine_pri):
    joint_action = np.concatenate([action[..., np.newaxis], machine_pri], axis=-1)
    return joint_action

def seperate_joint_action(joint_action):
    action = joint_action[..., 0]
    machine_pri = joint_action[..., 1:]
    return action, machine_pri

def is_valid_topological_order(G, sequence):
    if not nx.is_directed_acyclic_graph(G):
        return False  
    node_to_order = {node: i for i, node in enumerate(sequence)}  
    for u, v in G.edges():
        if node_to_order[u] > node_to_order[v]:
            return False
    return True


NULL_ACTION =-1
max_node_num = 2000
max_edge_num = 2000
raw_node_attr_length = 2
raw_edge_attr_length = 1

def get_dict_from_networkx(dag):
    irr_pairs = get_irrelevant_jssp(dag)
    irr_pair_num = len(irr_pairs)

    xs, edge_indexs, edge_attrs, valid_masks,  irr_adjs= list(), [[],[]], list(), list(), list()
    node_num, edge_num = dag.number_of_nodes(), dag.number_of_edges()
    
    for node in dag.nodes:
        raw_v = [dag.nodes[node]['duration'], dag.nodes[node]['machine']]
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
    ob_dict['x'] = np.pad(xs, ((0,max_node_num - node_num), (0,0)))
    edge_indexs = np.array(edge_indexs, dtype=np.int64)
    ob_dict['edge_index'] = np.pad(edge_indexs, ((0,0), (0, max_edge_num - edge_num)))
    edge_attrs = np.array(edge_attrs, dtype=np.float32)
    ob_dict['edge_attr'] = np.pad(edge_attrs, ((0, max_edge_num - edge_num), (0, 0))) 
    valid_masks = np.array(valid_masks, dtype=np.int8)
    ob_dict['valid_mask'] = np.pad(valid_masks, (0, max_node_num - node_num))

    adj_mat = nx.to_numpy_array(dag, nodelist=sorted(dag.nodes()))
    ob_dict['adj_mat'] = np.pad(adj_mat, ((0, max_node_num - node_num), (0, max_node_num - node_num)))

    irr_adjs = np.array(irr_adjs, dtype = np.int8)
    ob_dict['irr_adj'] = np.pad(irr_adjs, ((0, max_node_num - node_num), (0,max_node_num - node_num)))

    ob_dict['node_num'] = [node_num]
    ob_dict['edge_num'] = [edge_num]
    #ob_dict['irr_pair_num'] = [irr_pair_num]
    return ob_dict

class running_jssp:
    def __init__(self, job_duration_map, job_operation_map):
        self.job_duration_map = job_duration_map
        self.job_operation_map = job_operation_map
        self.j_n = self.job_duration_map.shape[0]
        self.m_n = self.job_duration_map.shape[1]         

    def simulate(self, schedule_list):
        st_list = np.zeros(self.j_n * self.m_n) # 
        oft_list = np.zeros(self.j_n * self.m_n) # 
        ft_list = np.zeros(self.j_n * self.m_n) # 
        m_ft = np.zeros(self.m_n) # 
        if len(schedule_list) > self.j_n * self.m_n:
            schedule_list = schedule_list[1:-1]
        for node in schedule_list:        
            if node == self.j_n * self.m_n: #  entrance，skip
                st_list[node] = st
                print('eq')
                continue
            if node ==self.j_n * self.m_n + 1:
                print('eq')
                continue
            j = int(node / self.m_n)  #'''self.get_jo_by_index(node) '''
            o = node - self.m_n * j
            machine = self.job_operation_map[j][o]
            duration = self.job_duration_map[j][o]
            pred = node - 1 #'''self.get_pred_index(node)'''
            if o - 1 >= 0: # pred>=0:
                st = max(ft_list[pred], m_ft[machine])
            else:
                st = m_ft[machine]
            ft = st + duration    
            st_list[node] = st
            oft_list[node] = np.max(ft_list)
            ft_list[node] = ft 
            m_ft[machine] = ft
        mksp = np.max(ft_list)
        #rew1: (end-lastend) / (end - start)
        #dense_rew.append(-(end-last_end)*1.0 / (end - start))
            
        # rew2: (end - ready) / (end - start)
        #dense_rew.append(-(end - ready_time)*1.0 / (end - start))
            
        #rew3: end
        #dense_rew = ft_list

        #rew4: OFT
        #dense_rew = oft_list
        
        #rew5: OFT improvement
        #dense_rew.append(-oftimp)

        # rew6 overall makespan
        #dense_rew = [-mksp for _ in range(self.j_n * self.m_n)]

        # rew7 mksp - st
        #dense_rew = [-(mksp - st_list[node]) for node in range(self.j_n * self.m_n]
            
        #rew8 mksp - last
        #dense_rew.append(-(mksp - last_oft))

        #rew9 minstart when time reversed
        #dense_rew.append(-minstart)

        #rew10 st
        dense_rew = st_list
        return dense_rew, mksp        

def heuristic_jssp_scheduling(job_duration_map, job_operation_map):
    j_n, m_n = job_duration_map.shape
    job_operations = np.zeros(j_n, dtype=int)  # 
    job_completion_time = np.zeros(j_n)  # 
    machine_available_time = np.zeros(m_n)  # 

    st_list, ft_list = np.zeros(shape=(j_n, m_n), dtype=np.int64), np.zeros(shape=(j_n, m_n), dtype=np.int64)

    # 
    job_queue = []
    
    # 
    for j in range(j_n):
        op_idx = job_operations[j]  # 
        machine = job_operation_map[j, op_idx]
        time = job_duration_map[j, op_idx]
        heapq.heappush(job_queue, (time, j))

    # 
    while job_queue:
        _, job = heapq.heappop(job_queue)  # 
        op_idx = job_operations[job]  # 
        machine = job_operation_map[job, op_idx]
        time = job_duration_map[job, op_idx]

        # 
        start_time = max(job_completion_time[job], machine_available_time[machine])
        end_time = start_time + time

        # 
        st_list[job][op_idx] = start_time
        ft_list[job][op_idx] = end_time
        job_completion_time[job] = end_time
        machine_available_time[machine] = end_time

        # 
        job_operations[job] += 1
        if job_operations[job] < m_n:
            new_machine = job_operation_map[job, job_operations[job]]
            new_time = job_duration_map[job, job_operations[job]]
            heapq.heappush(job_queue, (new_time, job))

    mksp = np.max(ft_list)
    st_list = st_list.flatten()
    ft_list = ft_list.flatten()

    initial_list =np.argsort(st_list)
    return mksp, st_list, ft_list

def heuristic_with_pred_constrain(job_duration_map, job_operation_map, pred_constrain=None):
    j_n, m_n = job_duration_map.shape

    machine_available_time = defaultdict(int)
    job_op_ready_time = defaultdict(int)

    op_dependency = defaultdict(set)
    rev_dependency = defaultdict(list)

    if pred_constrain is None:
        pred_constrain = []

    # 
    for j in range(j_n):
        for o in range(1, m_n):
            op_dependency[(j, o)].add((j, o - 1))
            rev_dependency[(j, o - 1)].append((j, o))

    # 
    for (j1, o1), (j2, o2) in pred_constrain:
        op_dependency[(j2, o2)].add((j1, o1))
        rev_dependency[(j1, o1)].append((j2, o2))

    in_degree = {(j, o): len(op_dependency.get((j, o), [])) 
                 for j in range(j_n) for o in range(m_n)}

    ready_ops = [(job_duration_map[j][0], j, 0) 
                 for j in range(j_n) if in_degree[(j, 0)] == 0]
    heapq.heapify(ready_ops)

    schedule, st_list = [], []  # (job_id, op_id, machine_id, start, end)

    while ready_ops:
        _, j, o = heapq.heappop(ready_ops)
        m = job_operation_map[j][o]
        pt = job_duration_map[j][o]

        ready_time = max(job_op_ready_time[(j, o)], machine_available_time[m])
        start_time = ready_time
        end_time = start_time + pt

        machine_available_time[m] = end_time
        schedule.append((j, o, m, start_time, end_time))
        st_list.append(start_time)

        # 
        for (sj, so) in rev_dependency.get((j, o), []):
            in_degree[(sj, so)] -= 1
            job_op_ready_time[(sj, so)] = max(job_op_ready_time[(sj, so)], end_time)
            if in_degree[(sj, so)] == 0:
                heapq.heappush(ready_ops, (job_duration_map[sj][so], sj, so))
    mksp = max(st_list)
    return mksp   
    
class CloudGym_oneshot_jssp(gym.Env):
    NULL_ACTION = NULL_ACTION
    max_node_num = max_node_num
    max_edge_num = max_edge_num
    raw_node_attr_length = raw_node_attr_length
    raw_edge_attr_length = raw_edge_attr_length
    
    metadata = {'render.modes': ['human']}
    def __init__(self, jssp_instance, workflow_index):
        self.jssp_instance = jssp_instance
        self.job_duration_map = jssp_instance[0]
        self.job_operation_map = jssp_instance[1] - 1
        self.j_n = self.job_duration_map.shape[0]
        self.m_n = self.job_duration_map.shape[1]
        self.workflow_index = workflow_index

        self.dummy_start = self.j_n * self.m_n
        self.dummy_end = self.j_n * self.m_n + 1

        # 
        self.dag = nx.DiGraph()
        self.dag.add_node(self.dummy_start, duration=0, machine=0)
        self.dag.add_node(self.dummy_end, duration=0, machine=0)
        self.dag.add_edge(0, self.dummy_end)
        node = 0
        for j in range(self.j_n):
            for o in range(self.m_n):
                node = self.get_node_index_by_jo(j, o)

                self.dag.add_node(node, 
                                  duration=self.job_duration_map[j][o], 
                                  machine=self.job_operation_map[j][o])
                pred = self.get_pred_index(node)
                succ = self.get_succ_index(node)
                if pred: # is not None:
                    self.dag.add_edge(pred, node)
                else:
                    self.dag.add_edge(self.dummy_start, node)

                if succ is None:
                    self.dag.add_edge(node, self.dummy_end)
        self.heur_mksp, self.st_list, self.ft_list = heuristic_jssp_scheduling(self.job_duration_map, self.job_operation_map)
        priority_list = [st for st in self.st_list]
        self.heur_list = np.argsort(priority_list)
        jssp_running_instance = running_jssp(job_duration_map=self.job_duration_map, job_operation_map=self.job_operation_map)
        t1 = time.time()
        self.dense_rew_bsln, mksp_0 = jssp_running_instance.simulate(self.heur_list)
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
        self.action_space = gym.spaces.Box(low = -1e+20, high = 1e+20, shape = (max_node_num,), dtype = np.float32)

    def reset(self):
        ''' end the previous simulation'''
        if self.global_ob is None:
            self.global_ob = get_dict_from_networkx(self.dag)
        self.cum_rew = 0
        return self.global_ob

    def step(self, action):
        action, machine_pri = seperate_joint_action(action)
        priority_list = action[0 : self.dag.number_of_nodes()]
        LS_topo = [t for t in nx.lexicographical_topological_sort(self.dag, key=lambda x: priority_list[x])]
        #LS_topo = [t for t in nx.lexicographical_topological_sort(self.dag, key=lambda x: -priority_list[x])] # descending topo sort for rew_2 of one-shot

        jssp_running_instance = running_jssp(self.job_duration_map, self.job_operation_map)
        dense_rew, mksp = jssp_running_instance.simulate(LS_topo)
        
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
        
        #rew10 baseline
        dense_rew = [-rew + baseline for rew, baseline in zip(dense_rew, self.st_list)]
        
        dense_rew = np.array(dense_rew, dtype=np.float32)
        dense_rew = np.pad(dense_rew, (0, max_node_num - dense_rew.shape[0]))

        
        machine_available_mask = np.zeros((self.dag.number_of_nodes(), 4), dtype=bool)
        machine_chosen_mask = np.zeros((self.dag.number_of_nodes(), 4), dtype=bool)

        machine_available_mask = np.pad(machine_available_mask, ((0, max_node_num - machine_available_mask.shape[0]), (0, 0)))
        machine_chosen_mask = np.pad(machine_chosen_mask, ((0, max_node_num - machine_chosen_mask.shape[0]), (0, 0)))
        
        info = {'obj':[mksp, 0, -0], 
                'dense_rew':dense_rew, 
                'machine_available_mask':machine_available_mask, 
                'machine_chosen_mask': machine_chosen_mask
                }
        
        return self.global_ob, reward, is_done, info

    def render(self):
        pass

    def get_node_index_by_jo(self, j, o):
        return self.m_n * j + o

    def get_jo_by_node_index(self, node):
        j = int(node / self.m_n)
        o = node - self.m_n * j
        return j, o

    def get_pred_index(self, node):
        j, o = self.get_jo_by_node_index(node)
        if o <= 0:
            return None 
        return node - 1

    def get_succ_index(self, node):
        j, o = self.get_jo_by_node_index(node)
        if o >= self.m_n -1:
            return None
        return node + 1

if __name__ == '__main__':
    
    from datasets.JSSP_data.read_data import readdata
    for j in [20, 30]:
        for m in[10, 20]: 
            jssp_instance = readdata(j, m, 1, 42)[0]
            t1 = time.time()
            mksp, st_list, ft_list = heuristic_jssp_scheduling(jssp_instance[0], jssp_instance[1]-1)
            #env = CloudGym_oneshot_jssp(jssp_instance=jssp_instance, workflow_index=0)
            print(j,m,mksp,time.time()-t1)
    
    
    #env.step([i for i in range(200)])
    #mksp, dense_rew = dag_graph.ranker_based_scheduling(dag, [i for i in range(merged_dag.number_of_nodes())], resource_limit=600)
    #print(mksp, dense_rew)
