import numpy as np
from math import inf
import simpy
import gym
import os

from tensorboardX import SummaryWriter

from playground.DAG.utils.xml_reader import XMLReader
from playground.auxiliary.episode import Episode
from playground.DAG.utils.convertion import add_order_info_01
from playground.DAG.utils.runtime_calculation import total_energy, total_reliability

from core.cluster import Cluster
from core.scheduler import Scheduler
from core.broker import Broker
from core.simulation import Simulation

from heft import Heft_A, EFT, LINEAR_WEIGHTED_GREEDY, RANKED_MACHINE, GIVEN_MACHINE, CHEBYSHEV_WEIGHTED_GREEDY, NDsort_MACHINE, ALLOC_STR, PRIORIZED_MACHINE 

NULL_ACTION =-1
max_node_num = 2000
max_edge_num = 12000
machine_num = 4
raw_node_attr_length = 2 # + machine_num
raw_edge_attr_length = 1
#
def cat_action_machinepri(action, machine_pri):
    joint_action = np.concatenate([action[..., np.newaxis], machine_pri], axis=-1)
    return joint_action

def seperate_joint_action(joint_action):
    action = joint_action[..., 0]
    machine_pri = joint_action[..., 1:]
    return action, machine_pri
 
def get_dict_from_networkx(cluster, heft):
    nx_G = cluster.jobs[0].nx_G

    xs, edge_indexs, edge_attrs, valid_masks, irr_adjs= list(), [[],[]], list(), list(), list()
    node_num, edge_num = nx_G.number_of_nodes(), nx_G.number_of_edges()
    for node in range(node_num):
        task = nx_G.nodes[node]['task']
        task_config = task.task_config
        raw_v = [task_config.duration,
                    task_config.datasize
                ]
        #for x in task_config.machine_preference:
        #    raw_v.append(x)
        #for machine in cluster.machines:
        #    raw_v.append(machine.compute_capacity)
        #raw_v.append(heft.ranku(node))
        #raw_v.append(heft.rankd(node))
        xs.append(raw_v)
        irr_adjs.append(nx_G.nodes[node]['irr_adj'])
        valid_masks.append(1)
        for succ in nx_G.successors(node):
            edge_indexs[0].append(node)
            edge_indexs[1].append(succ)
            edge_attr = nx_G[node][succ]['comm_ready']
            edge_attrs.append([1.0] if edge_attr else [0.0])
    
    ob_dict = {}
    xs = np.array(xs, dtype=np.float32)
    #xs = pp.scale(xs)  # 
    ob_dict['x'] = np.pad(xs, ((0,max_node_num - node_num), (0,0)))
    edge_indexs = np.array(edge_indexs, dtype=np.int64)
    ob_dict['edge_index'] = np.pad(edge_indexs, ((0,0), (0, max_edge_num - edge_num)))
    edge_attrs = np.array(edge_attrs, dtype=np.float32)
    ob_dict['edge_attr'] = np.pad(edge_attrs, ((0, max_edge_num - edge_num), (0, 0))) 
    valid_masks = np.array(valid_masks, dtype=np.int8)
    ob_dict['valid_mask'] = np.pad(valid_masks, (0, max_node_num - node_num))

    import networkx as nx
    adj_mat = nx.to_numpy_array(nx_G, nodelist=sorted(nx_G.nodes()))
    ob_dict['adj_mat'] = np.pad(adj_mat, ((0, max_node_num - node_num), (0, max_node_num - node_num)))

    irr_adjs = np.array(irr_adjs, dtype = np.int8)
    ob_dict['irr_adj'] = np.pad(irr_adjs, ((0, max_node_num - node_num), (0,max_node_num - node_num)))

    ob_dict['node_num'] = [node_num]
    ob_dict['edge_num'] = [edge_num]
    #ob_dict['irr_pair_num'] = [irr_pair_num]
    return ob_dict
    

class CloudGym_oneshot(gym.Env):
    NULL_ACTION = NULL_ACTION
    max_node_num = max_node_num
    max_edge_num = max_edge_num
    machine_num = machine_num
    raw_node_attr_length = raw_node_attr_length
    raw_edge_attr_length = raw_edge_attr_length
    
    
    metadata = {'render.modes': ['human']}
    def __init__(self, machine_configs, task_configs, algorithm, event_file, logpath, workflow_index, weights):
        self.initial_state = machine_configs, task_configs, algorithm, event_file
        self.algorithm = algorithm
        self.weights = weights

        self.workflow_index = workflow_index
        
        self.env = simpy.Environment()
        self.cluster = Cluster()
        self.cluster.add_machines(machine_configs)
        task_broker = Episode.broker_cls(self.env, task_configs)
        self.scheduler = Scheduler(self.env, algorithm)
        self.simulation = Simulation(self.env, self.cluster, task_broker, self.scheduler, event_file)

        self.current_episode = 0
        self.logpath = logpath
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

        self.heft = Heft_A(self)
        self.LB_mksp = self.heft.get_makespan()

    def reset(self,):
        ''' end the previous simulation'''
        del self.env
        del self.simulation
        del self.cluster
        del self.scheduler
        
        self.env = simpy.Environment()
        machine_configs, task_configs, algorithm, event_file = self.initial_state
        self.cluster = Cluster()
        self.cluster.add_machines(machine_configs)
        task_broker = Episode.broker_cls(self.env, task_configs)
        self.scheduler = Scheduler(self.env, algorithm)
        self.simulation = Simulation(self.env, self.cluster, task_broker, self.scheduler, event_file)
        if self.global_ob is None:
            self.global_ob = get_dict_from_networkx(self.cluster, self.heft)
        #self.current_episode = 0
        self.cum_rew = 0
        return self.global_ob

    def step(self, action):
        action, machine_pri = seperate_joint_action(action)
        LS_topo, C_energy, C_reliab = self.scheduler.schedule_according_to_LS(action)
        self.heft.ranked_jobs = LS_topo
        self.heft.eval = True #True
        self.heft.set_alloc_strategy(EFT) 
        #self.heft.set_alloc_strategy(PRIORIZED_MACHINE) 
        
        self.heft.set_machine_pri(machine_pri)
        orders, dense_rew, C_mksp = self.heft.get_makespan()
        
        machine_available_mask = self.heft.machine_available_mask
        machine_chosen_mask = self.heft.machine_chosen_mask
        C_energy, C_reliab = self.heft.energy, self.heft.reliability

        available_counts = machine_available_mask.sum(axis=-1)
        chosen_counts = machine_chosen_mask.sum(axis=-1)       
        #LB_mksp, LB_energy, UB_reliab = self.LB_mksp, self.cluster.LB_energy, self.cluster.UB_reliability
        #(C_mksp, LB_mksp, C_energy, LB_energy, C_reliab, UB_reliab)
        W_mksp, W_energy, W_reliab = self.weights[0], self.weights[1], self.weights[2]
        is_done = True
        if self.global_ob is None:
            self.global_ob = get_dict_from_networkx(self.cluster, self.heft)

        tag_scalar_dict = {
            'makespan': C_mksp,
            'energycost': C_energy,
            'reliability': C_reliab
            }
        #self.summarywriter.add_scalars(main_tag = 'atag', tag_scalar_dict = tag_scalar_dict, global_step = self.current_episode)
        self.current_episode += 1
        if self.current_episode % 100 == 0:
            figdir = self.logpath + '/gantt'
            if figdir and not os.path.exists(figdir):
                os.makedirs(figdir, exist_ok=True)
            figpath = self.logpath + '/gantt/' + str(self.current_episode) + '.png'
            #self.heft.draw_gantt(orders, figpath)

        '''Utility function: weighted chebyshev scalarization '''
        reward = - W_mksp * C_mksp - W_energy * C_energy + W_reliab * C_reliab
        #reward = - max([W_mksp * abs(C_mksp - LB_mksp), W_energy * abs(C_energy - LB_energy), W_reliab * abs(C_reliab - UB_reliab)])
        # 填充dense_rew至定长
        dense_rew = np.array(dense_rew, dtype=np.float32)
        dense_rew = np.pad(dense_rew, (0, max_node_num - dense_rew.shape[0]))
        # 填充machine mask至定长
        machine_available_mask = np.pad(machine_available_mask, ((0, max_node_num - machine_available_mask.shape[0]), (0, 0)))
        machine_chosen_mask = np.pad(machine_chosen_mask, ((0, max_node_num - machine_chosen_mask.shape[0]), (0, 0)))
        
        info = {'obj':[C_mksp, C_energy, -C_reliab], 
                'dense_rew':dense_rew, 
                'machine_available_mask':machine_available_mask, 
                'machine_chosen_mask': machine_chosen_mask
                }
        return self.global_ob, reward, is_done, info

    def render(self):
        pass  


''' for test'''
'''
from core.machine import MachineConfig
from playground.DAG.algorithm.heuristics.test_algorithm import TestAlgorithm
from core.job import Job
import torch
from torch_geometric.data import Data

if __name__ == '__main__':
    
    machine_configs = [MachineConfig(i, 1, 1, i, (i+1) * 0.1, (i+1)*5, 0.00001*(i*i+ 2)) for i in range(4)]
    dataset_name = 'MONTAGE'
    dataset_size = '50'   
    jobs_xml = '../RLHe/database_sci/raw/%s/%s.n.%s.0.dax' % (dataset_name, dataset_name, dataset_size)
    xml_reader = XMLReader(jobs_xml)
    jobs_configs = xml_reader.generate(0, 1)
    env = CloudGym_oneshot(machine_configs, jobs_configs, None, None, './tensorlogs/test')
    env.step([i for i in range(52)])
'''