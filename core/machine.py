from enum import Enum
from playground.DAG.utils.runtime_calculation import communication_time, computation_time

class MachineConfig(object):
    idx = 0

    def __init__(self, id, cpu_capacity, memory_capacity, disk_capacity, compute_capacity, energy_cost, fault_rate, cpu=None, memory=None, disk=None):
        self.id = id
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.disk_capacity = disk_capacity
        
        self.compute_capacity = compute_capacity
        
        self.energy_cost = energy_cost
        self.fault_rate = fault_rate

        self.cpu = cpu_capacity if cpu is None else cpu
        self.memory = memory_capacity if memory is None else memory
        self.disk = disk_capacity if disk is None else disk

        


class MachineDoor(Enum):
    TASK_IN = 0
    TASK_OUT = 1
    NULL = 3


class Machine(object):
    def __init__(self, machine_config):
        self.id = machine_config.id
        self.cpu_capacity = machine_config.cpu_capacity
        self.memory_capacity = machine_config.memory_capacity
        self.disk_capacity = machine_config.disk_capacity
        self.compute_capacity = machine_config.compute_capacity
        
        self.energy_cost = machine_config.energy_cost
        self.fault_rate = machine_config.fault_rate
        
        self.cpu = machine_config.cpu
        self.memory = machine_config.memory
        self.disk = machine_config.disk

        self.cluster = None
        self.task_instances = []
        self.finished_task_instances = []
        self.task_instances_queue = []
        self.machine_door = MachineDoor.NULL

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        if isinstance(other, Machine):
            return self.id == other.id
        else:
            return False


    def reset(self):
        self.task_instances = []
        self.task_instances_queue = []        

    def run_task_instance(self, task_instance):
        self.cpu -= task_instance.cpu
        self.memory -= task_instance.memory
        self.disk -= task_instance.disk
        self.task_instances.append(task_instance)
        self.machine_door = MachineDoor.TASK_IN

    def stop_task_instance(self, task_instance):
        self.cpu += task_instance.cpu
        self.memory += task_instance.memory
        self.disk += task_instance.disk
        self.machine_door = MachineDoor.TASK_OUT
        self.pop_task_instance()
        self.finished_task_instances.append(task_instance)

    def add_task_instance(self, task_instance):
        self.task_instances_queue.append(task_instance)

    def pop_task_instance(self):
        self.task_instances_queue.pop(0)


    ''' '''
    def estimated_lft(self):
        if len(self.task_instances_queue) == 0:
            if len(self.finished_task_instances) > 0:
                return self.finished_task_instances[-1].finished_timestamp
            else:
                return 0
        last_task_instance = self.task_instances_queue[-1]
        lst = last_task_instance.estimated_started_timestamp_for_scheduled
        lft = lst + computation_time(last_task_instance.task, self)
        return lft

    @property
    def running_task_instances(self):
        ls = []
        for task_instance in self.task_instances:
            if task_instance.started and not task_instance.finished:
                ls.append(task_instance)
        return ls

    @property
    def occupied(self):
        if len(self.running_task_instances) == 0:
            return False
        else:
            return True

    def attach(self, cluster):
        self.cluster = cluster

    def accommodate(self, task):
        return self.cpu >= task.task_config.cpu and \
               self.memory >= task.task_config.memory and \
               self.disk >= task.task_config.disk

    @property
    def feature(self):
        return [self.cpu, self.memory, self.disk]

    @property
    def capacity(self):
        return [self.cpu_capacity, self.memory_capacity, self.disk_capacity]

    @property
    def state(self):
        return {
            'id': self.id,
            'cpu_capacity': self.cpu_capacity,
            'memory_capacity': self.memory_capacity,
            'disk_capacity': self.disk_capacity,
            'cpu': self.cpu / self.cpu_capacity,
            'memory': self.memory / self.memory_capacity,
            'disk': self.disk / self.disk_capacity,
            'running_task_instances': len(self.running_task_instances),
            'finished_task_instances': len(self.finished_task_instances)
        }

    def __eq__(self, other):
        return isinstance(other, Machine) and other.id == self.id
