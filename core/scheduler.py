from playground.DAG.utils.runtime_calculation import computation_time, communication_time, reliability, total_reliability, total_energy, energy
from playground.DAG.utils.get_tasks import get_task_by_index
import networkx as nx
from core.occurence import Occurence

from datetime import datetime
from random import randint
import numpy as np

EFT = 0
LINEAR = 1
CHEBY = 2

WEIGHTS = [0.1, 0.0001, 0.8999]


class Scheduler(object):
    def __init__(self, occurence_monitor, algorithm):
        self.occurence_monitor = occurence_monitor
        self.algorithm = algorithm
        self.simulation = None
        self.cluster = None
        self.destroyed = False
        self.valid_pairs = {}

        self.cnt = datetime.now() - datetime.now() 

    def attach(self, simulation):
        self.simulation = simulation
        self.cluster = simulation.cluster

    def schedule_according_to_LS(self, action):
        def is_valid_topological_order(G, sequence):
            if not nx.is_directed_acyclic_graph(G):
                return False  # 
            node_to_order = {node: i for i, node in enumerate(sequence)}  # 
            for u, v in G.edges():
                if node_to_order[u] > node_to_order[v]: 
                    return False
            return True
        priority_list = action[0:self.cluster.jobs[0].job_size]
        DG = self.cluster.jobs[0].nx_G
        LS_topo = [t for t in nx.lexicographical_topological_sort(DG, key=lambda x: priority_list[x])]
        #LS_topo = [t for t in nx.lexicographical_topological_sort(DG, key=lambda x: -priority_list[x])] # descending topo sort for rew_2 of one-shot
        return LS_topo, 0, 0
        
        while True:
            for task_index, i in zip(LS, range(job.job_size)):
                if LS_scheduled_mask[i] == 0 and task_index in [task.task_index for task in job.ready_tasks]:
                    task = get_task_by_index(job, task_index)
                    for c in task.childs:
                        c._ready = True
                        for p in c.parents:
                            if p.ready == False:
                                c._ready = False
                                break
                    LS_scheduled_mask[i] = 1
                    LS_topo.append(task_index)
                    cnt += 1
                    break
            if cnt == job.job_size:
                return LS_topo, 0, 0

    ''' for non-gym environment '''
    def make_decision(self):
        # non-eft
        while True:
            task = self.algorithm(self.cluster, self.occurence_monitor.now)
            if task is None:
                break
            else:
                eft = 99999999

                eft_machine = None
                for machine in self.cluster.machines:
                    estimated_st = task.task_instances[0].estimated_started_timestamp_for_unscheduled(machine)
                    estimated_ft = estimated_st + computation_time(task, machine)
                    if estimated_ft < eft:
                        eft = estimated_ft
                        eft_machine = machine
                task.schedule_task_instance(eft_machine)
                a = task.task_instances[0].estimated_started_timestamp_for_scheduled

    def make_decision_from_action(self, weights, greedy_rule):
        # non-eft
        if self.cluster.current_action is not None:
            tasks = self.cluster.schedulable_tasks_which_has_waiting_instance
            task = tasks[self.cluster.current_action]
            #task = tasks[-1]
            ''' for test'''
            machine = self.cluster.machines[randint(0,3)]
            est = task.task_instances[0].estimated_started_timestamp_for_unscheduled(machine)
            eft = est + computation_time(task, machine)
            task.schedule_task_instance(machine) 
            #print('Task:', task.task_index, -eft)
            return -eft
        
            mx_rew = -9999999
            mx_rew_machine = None
            
            lft = -999999999
            for machine in self.cluster.machines:
                eft = machine.estimated_lft() 
                if eft > lft:
                    lft = eft
            
            estimated_st = task.task_instances[0].estimated_started_timestamp_for_unscheduled(machine)
            estimated_ft = lambda machine: estimated_st + computation_time(task, machine)
            
            old_energy = total_energy(self.cluster)
            estimated_energy = lambda machine: old_energy + energy(task, machine)
            
            old_reliability = total_reliability(self.cluster)
            estimated_reliability = lambda machine: old_reliability * reliability(task, machine)
            
            if greedy_rule == 'EFT':
                machine = min(self.cluster.machines, key=estimated_ft)
                rew = estimated_ft(machine)
            elif greedy_rule == 'LINEAR':
                obj = lambda machine: weights[0]*estimated_ft(machine) + \
                                      weights[1]*estimated_energy(machine) - \
                                      weights[2]*estimated_reliability(machine)
                machine = min(self.cluster.machines, key=obj)
                rew = obj(machine)
            elif greedy_rule == 'CHEBY':
                min_ft = min([estimated_ft(machine) for machine in self.cluster.machines])
                min_eg = min([estimated_energy(machine) for machine in self.cluster.machines])
                max_rb = max([estimated_reliability(machine) for machine in self.cluster.machines])
                weighted_obj = lambda machine: max([weights[0] * abs(estimated_ft(machine) - min_ft), 
                                                weights[1] * abs(estimated_energy(machine) - min_eg), 
                                                weights[2] * abs(estimated_reliability(machine) - max_rb)]) 
                machine = min(self.cluster.machines, key=weighted_obj)     
                rew = weighted_obj(machine)           
            rew = -rew
            task.schedule_task_instance(machine)    
            print('Task:', task.task_index, rew)
            return rew
        else:
            return 0

    def make_decision_from_task_machine_action(self):
        if self.cluster.current_action is not None:
            action_task = self.cluster.current_action[0]
            tasks = self.cluster.schedulable_tasks_which_has_waiting_instance
            task = tasks[self.cluster.current_action]
            
            action_machine = self.cluster.current_action[1]
            machine = self.cluster.machines[action_machine]
            
            lft = -999999999
            for machine in self.cluster.machines:
                eft = machine.estimated_lft() 
                if eft > lft:
                    lft = eft
            estimated_st = task.task_instances[0].estimated_started_timestamp_for_unscheduled(machine)
            estimated_ft = estimated_st + computation_time(task, machine)

            delta_makespan = estimated_ft - lft
            if delta_makespan < 0:
                delta_makespan = 0

            rew = -delta_makespan
            return rew
        else:
            return 0        
    def index_topo_trans(self, index):
        schedulable_task_ids = [task.id for task in self.cluster.schedulable_tasks_which_has_waiting_instance]
        
        for i in range(0, len(schedulable_task_ids)):
            if schedulable_task_ids[i] == index:
                schedulable_index = i
        return schedulable_index
    
    def run(self):
        self.machine_queue_checking()


    def machine_queue_checking(self):
        
        tt = datetime.now()
        for machine in self.cluster.machines:
            if len(machine.task_instances_queue) > 0:
                current_task_instance = machine.task_instances_queue[0]
                while len(machine.running_task_instances) == 0 and current_task_instance.task.comm_recv_ready:
                    #print('Now start executing:', current_task_instance.task.task_index, 'at:', self.occurence_monitor.now)
                    
                    current_task_instance.execute()
                    
                    if len(machine.task_instances_queue) > 0:
                        current_task_instance = machine.task_instances_queue[0]
                    else:
                        break       
        
        
        if not self.simulation.finished:
            machine_queue_checking_occurence = Occurence(
                    trigger_time=self.occurence_monitor.now+1,
                    action=self.machine_queue_checking
                )
            self.occurence_monitor.add_occurence(machine_queue_checking_occurence) 
        else:
            self.destroyed = True
            machine = self.cluster.machines[0]
        self.cnt = self.cnt + datetime.now() - tt
