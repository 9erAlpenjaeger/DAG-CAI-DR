import numpy as np
from math import inf
import simpy
import gym
from sklearn import preprocessing as pp

from tensorboardX import SummaryWriter

from playground.DAG.model.dagnn import DAGNN

from playground.DAG.utils.xml_reader import XMLReader
from playground.auxiliary.episode import Episode
from playground.DAG.utils.convertion import add_order_info_01
from playground.DAG.utils.runtime_calculation import total_energy, total_reliability

from core.cluster import Cluster
from core.scheduler import Scheduler
from core.broker import Broker
from core.simulation import Simulation


NULL_ACTION = -1
max_node_num = 1050
max_edge_num = 12000
raw_node_attr_length = 14
raw_edge_attr_length = 1


def get_dict_from_networkx(cluster, now_time):
    nx_G = cluster.jobs[0].nx_G
    lft = cluster.estimated_lft()
    
    xs, edge_indexs, edge_attrs, valid_masks = list(), [[],[]], list(), list()
    node_num, edge_num = nx_G.number_of_nodes(), nx_G.number_of_edges()
    for node in range(node_num):
        task = nx_G.nodes[node]['task']
        task_config = task.task_config
        task_instance = task.task_instances[0]
        raw_v = [task_config.duration,
                    task_config.datasize,
                    #len(task.parents),
                    #len(task.childs),
                    int(task.schedulable and task.has_waiting_task_instances),
                    int(task.scheduled),
                    int(task.started),
                    int(task.finished),
                    task.started_timestamp if task.started_timestamp is not None else 0,
                    task.finished_timestamp if task.finished_timestamp is not None else 9999999,
                    now_time,
                    lft,
                ]
        for machine in cluster.machines:
            raw_v.append(machine.compute_capacity)
        xs.append(raw_v)
        #print('raw len', len(raw_v))
        if task.has_waiting_task_instances and task.schedulable:
            valid_masks.append(1)
        else:
            valid_masks.append(0)
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
    ob_dict['edge_attr'] = np.pad(edge_attrs, ((0, max_edge_num - edge_num), (0, 0))) #
    valid_masks = np.array(valid_masks, dtype=np.int8)
    ob_dict['valid_mask'] = np.pad(valid_masks, (0, max_node_num - node_num))
    ob_dict['node_num'] = node_num
    ob_dict['edge_num'] = edge_num
    return ob_dict
    

class CloudGym(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, machine_configs, task_configs, algorithm, event_file, logpath, weights, greedy_rule):
        self.initial_state = machine_configs, task_configs, algorithm, event_file
        self.algorithm = algorithm
        
        self.env = simpy.Environment()
        self.cluster = Cluster()
        self.cluster.add_machines(machine_configs)
        task_broker = Episode.broker_cls(self.env, task_configs)
        self.scheduler = Scheduler(self.env, algorithm)
        self.simulation = Simulation(self.env, self.cluster, task_broker, self.scheduler, event_file)

        self.simulation.run()

        self.current_episode = 0
        self.logpath = logpath
        if self.logpath:
            self.summarywriter = SummaryWriter(log_dir = logpath)
        
        self.cum_rew = 0
        self.weights = weights
        self.greedy_rule = greedy_rule
        
        space_dict = {}
        space_dict['node_num'] = gym.spaces.Box(low = 0, high = max_node_num, shape = (1,), dtype=np.int64)  # 
        space_dict['edge_num'] = gym.spaces.Box(low = 0, high = max_edge_num, shape = (1,), dtype=np.int64) # 
        space_dict['x'] = gym.spaces.Box(low = 0, high = inf, shape = (max_node_num, raw_node_attr_length), dtype = np.float32)
        space_dict['edge_index'] = gym.spaces.Box(low = 0, high = max_edge_num, shape = (2, max_edge_num), dtype = np.int64)
        space_dict['edge_attr']  = gym.spaces.Box(low = 0, high = inf, shape = (max_edge_num, raw_edge_attr_length), dtype = np.float32)
        space_dict['valid_mask'] = gym.spaces.MultiBinary(max_node_num) # 
        self.observation_space = gym.spaces.Dict(space_dict)   
        self.action_space = gym.spaces.Discrete(max_node_num)
        #self.action_space = gym.spaces.Box(low=NULL_ACTION, high = max_node_num, shape=(1,), dtype=np.int64)

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
        self.simulation.run()
        ob = get_dict_from_networkx(self.cluster, self.env.now)

        #for key in ob:
        #    print(key, ob[key].shape)
        return ob

    def step(self, action):
        self.cluster.current_action = None if action == NULL_ACTION else action
        #t1 = self.cluster.estimated_lft()
        #print('action is ', action)
        reward = self.scheduler.make_decision_from_action(self.weights, self.greedy_rule)
        #t2 = self.cluster.estimated_lft()
        #reward = t1-t2
        while True:
            if self.simulation.finished:
                break
            schedulable_task_num = len(self.cluster.schedulable_tasks_which_has_waiting_instance)
            if schedulable_task_num == 0:
                self.env.run(until = self.env.now + 1)
            else:
                break

        is_done = self.simulation.finished
        if is_done and self.logpath:
            makespan = self.env.now
            reliability = total_reliability(self.cluster)
            energycost = total_energy(self.cluster)
            tag_scalar_dict = {
                'makespan': makespan,
                'reliability': reliability,
                'energycost': energycost
                }
            self.summarywriter.add_scalars(main_tag = 'atag', tag_scalar_dict = tag_scalar_dict, global_step = self.current_episode)
            self.current_episode += 1
        else:
            reward = 0

        ob = get_dict_from_networkx(self.cluster, self.env.now)
        return ob, reward, is_done, {}


    def render(self):
        pass
