from operator import attrgetter
import numpy as np
import xml.dom.minidom
import networkx as nx
from random import randint
from core.job import JobConfig, TaskConfig
from core.machine import MachineConfig
from playground.DAG.utils.feature_synthesize import father_task_indices
import numpy as np

class XMLReader(object):
    def __init__(self, filenames, machine_num, variance):
        self.CNT = 0
        self.job_configs = []
        self.machine_num = machine_num
        self.variance = variance
        for filename in filenames:
            job_config = self.read_from_file(filename=filename)
            self.job_configs.append(job_config)

        self.job_configs.sort(key=attrgetter('submit_time'))
        for i in range(len(self.job_configs)):
            self.job_configs[i].id = i

    def read_from_file(self, filename):
        G, start_node, end_node = single_DAG_reading(filename)
        submit_time = self.CNT 
        self.CNT += 5000
        task_configs = []
        for n in G.nodes():
            '''默认值'''
            cpu, memory, disk, instances_num= 1, 1, 1, 1
            duration = G.nodes[n]['x'][0]
            datasize = G.nodes[n]['x'][1]
            parent_indices, child_indices = [], []
            for pre in G.predecessors(n):
                parent_indices.append(pre)
            for suc in G.successors(n):
                child_indices.append(suc)
            machine_preference = generate_random_preference(num_p=self.machine_num, variance=self.variance)
            task_config = TaskConfig(n, instances_num, cpu, memory, disk, duration, datasize, parent_indices, machine_preference)
            task_config.child_indices = child_indices
            task_configs.append(task_config)

        job_config = JobConfig(idx = 0,
                               submit_time=submit_time,
                               task_configs=task_configs,
                               start_node=start_node,
                               end_node=end_node)
        return job_config
                    
    def generate(self, offset, number):
        number = number if offset + number < len(self.job_configs) else len(self.job_configs) - offset
        ret = self.job_configs[offset: offset + number]
        the_first_job_config = ret[0]
        submit_time_base = the_first_job_config.submit_time

        tasks_number = 0
        task_instances_numbers = []
        task_instances_durations = []
        task_instances_datasizes = []
        task_instances_cpu = []
        task_instances_memory = []
        for job_config in ret:
            tasks_number += len(job_config.task_configs)
            for task_config in job_config.task_configs:
                task_instances_numbers.append(task_config.instances_number)
                task_instances_durations.extend([task_config.duration] * int(task_config.instances_number))
                task_instances_datasizes.extend([task_config.datasize] * int(task_config.instances_number))
                task_instances_cpu.extend([task_config.cpu] * int(task_config.instances_number))
                task_instances_memory.extend([task_config.memory] * int(task_config.instances_number))
        return ret

def single_DAG_reading(filename):
    G = nx.DiGraph()
    dom = xml.dom.minidom.parse(filename)
    root = dom.documentElement
    tasks = root.getElementsByTagName('job')
    id_index = {}
    current_index = 1
    for task in tasks:
        if task.hasAttribute('id'):
            task_id = task.getAttribute('id')
            #task_id = convert_id(task_id)
            id_index[task_id] = current_index
        if task.hasAttribute('runtime'):
            task_duration = task.getAttribute('runtime')
            task_duration = float(task_duration) + 1
        # get the size of the output data
        temp_root = task
        useses = temp_root.getElementsByTagName('uses')
        for uses in useses:
            link_type = uses.getAttribute('link')
            size = uses.getAttribute('size')
            size = int(size)
            if link_type == 'output': 
                break
        G.add_node(current_index)
        G.nodes[current_index]['x'] = [task_duration, size]
        current_index += 1
    childs = root.getElementsByTagName('child')
    for child in childs:
        child_id = child.getAttribute('ref')
        child_index = id_index[child_id]
        temp_root = child
        parents = temp_root.getElementsByTagName('parent')

        for parent in parents:
            parent_id = parent.getAttribute('ref')
            parent_index = id_index[parent_id]
            G.add_edge(parent_index, child_index)
    start_pes_node = 0
    node_indices = [n for n in G.nodes]
    max_node_indice = np.max(node_indices)
    end_pes_node = max_node_indice + 1 
    G.add_node(start_pes_node)
    G.add_node(end_pes_node)
    G.nodes[start_pes_node]['x'] = [0, 0]
    G.nodes[end_pes_node]['x']   = [0, 0]
    start_sizes = []
    for n in G.nodes:
        if n != start_pes_node and n != end_pes_node:
            pre_num = 0
            suc_num = 0
            for pre in G.predecessors(n):
                pre_num += 1
            for suc in G.successors(n):
                suc_num += 1
            if pre_num == 0:
                G.add_edge(start_pes_node, n)
                start_sizes.append(G.nodes[n]['x'][1])
            if suc_num == 0:
                G.add_edge(n, end_pes_node)
    G.nodes[start_pes_node]['x'] = [0, np.mean(start_sizes)]
    return G, start_pes_node, end_pes_node

def convert_id(task_id):
    task_id = task_id[2:]
    task_id = task_id.lstrip('0')
    if task_id == '':
        task_id = '0'
    task_id = int(task_id) + 1
    return task_id