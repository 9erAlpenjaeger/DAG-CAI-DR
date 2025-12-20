from core.machine import Machine
import numpy as np
import networkx as nx
from playground.DAG.utils.runtime_calculation import communication_time, computation_time, energy, reliability
from playground.DAG.utils.get_tasks import get_task_by_index

class Cluster(object):
    def __init__(self):
        self.machines = []
        self.jobs = []
        self.current_action = None

        self._UB_makespan = None
        self._LB_makespan = None
        self._UB_energy = None
        self._LB_energy = None
        self._UB_reliability = None
        self._LB_reliability = None

    def comp_time(self, task, machine):
        #if task.task_index == task.job.start_node:
        #    return 0
        #print(task.task_config.duration, machine.compute_capacity)
        if task.task_index == task.job.start_node or task.task_index == task.job.end_node:
            return 0
        return int(task.task_config.duration / machine.compute_capacity) + 1

    def comm_time(self, machine1, machine2, task1, task2):
        return 0
        if machine1 == machine2:
            return 0#int(task1.task_config.datasize / (self.bandwidth(machine1, machine2))) + 1#0
        else:
            #print(int(task1.task_config.datasize / (self.bandwidth(machine1, machine2))) + 1)
            return int(task1.task_config.datasize / (self.bandwidth(machine1, machine2))) + 1

    def bandwidth(self, machine1, machine2):
        return 1000

    def estimated_lft(self):
        machines_lft = [machine.estimated_lft() for machine in self.machines]
        return np.max(machines_lft)

    @property 
    # 这个其实应该用heft来算
    # 改写成所有job的makespan，对于特定job，arrival_time作为其lower bound
    def LB_makespan(self):
        if self._LB_makespan is None:
            machine_fastest = max(self.machines, key=lambda x: x.compute_capacity)
            job = self.jobs[0]
            DG = self.jobs[0].nx_G
            for n in DG.nodes():
                task = get_task_by_index(job, n)
                t = computation_time(task, machine_fastest)
                for s in DG.successors(n):
                    DG.add_edges_from([(n,s,{'mintime': t})])
            cp = nx.dag_longest_path(DG, weight='mintime')
            self._LB_makespan = 0
            for n in cp:
                task = get_task_by_index(job, n)
                self._LB_makespan += computation_time(task, machine_fastest)
        return self._LB_makespan + 1e-8
        
    @property
    def UB_makespan(self):
        if self._UB_makespan is None:
            machine_fastest = min(self.machines, key=lambda x: x.compute_capacity)
            self._UB_makespan = 0
            for job in self.jobs:
                for task in job.tasks:
                    self._UB_makespan += computation_time(task, machine_fastest)
        return self._UB_makespan + 1e-8

    @property
    def LB_energy(self):
        if self._LB_energy is None:
            self._LB_energy = 0
            machine_most_energysaving = min(self.machines, key=lambda x: (x.energy_cost/x.compute_capacity))
            for job in self.jobs:
                for task in job.tasks:            
                    self._LB_energy += energy(task, machine_most_energysaving)
        return self._LB_energy + 1e-8        
    
    @property    
    def UB_energy(self):
        if self._UB_energy is None:
            self._UB_energy = 0
            machine_most_energysaving = max(self.machines, key=lambda x: (x.energy_cost/x.compute_capacity))
            for job in self.jobs:
                for task in job.tasks:
                    self._UB_energy += energy(task, machine_most_energysaving)
        return self._UB_energy + 1e-8

    @property
    def LB_reliability(self):
        if self._LB_reliability is None:
            self._LB_reliability = 1.0
            machine_most_reliable = max(self.machines, key=lambda x: (x.fault_rate/x.compute_capacity))
            for job in self.jobs:
                for task in job.tasks:
                    self._LB_reliability *= reliability(task, machine_most_reliable)
        return self._LB_reliability + 1e-8

    @property
    def UB_reliability(self):
        if self._UB_reliability is None:
            self._UB_reliability = 1.0
            machine_most_reliable = min(self.machines, key=lambda x: (x.fault_rate/x.compute_capacity))
            for job in self.jobs:
                for task in job.tasks:
                    self._UB_reliability *= reliability(task, machine_most_reliable)
        return self._UB_reliability + 1e-8

    @property 
    def unoccupied_machines(self):
        ls = []
        for machine in self.machines:
            if not machine.occupied:
                ls.append(machine)
        return ls
    
    @property
    def unfinished_jobs(self):
        ls = []
        for job in self.jobs:
            if not job.finished:
                ls.append(job)
        return ls

    @property
    def tasks(self):
        ls = []
        for job in self.jobs:
            ls.extend(job.tasks)
        return ls

    @property
    def unscheduled_tasks(self):
        ls = []
        for job in self.jobs:
            ls.extend(job.unscheduled_tasks)
        return ls

    @property
    def scheduled_tasks(self):
        ls = []
        for job in self.jobs:
            ls.extend(job.scheduled_tasks)
        return ls

    @property
    def started_tasks(self):
        ls = []
        for job in self.jobs:
            ls.extend(job.started_tasks)
        return ls
    
    @property
    def unstarted_tasks(self):
        ls = []
        for job in self.jobs:
            ls.extend(job.unstarted_tasks)
        return ls
    
    @property
    def unfinished_tasks(self):
        ls = []
        for job in self.jobs:
            ls.extend(job.unfinished_tasks)
        return ls

    @property
    def unfinished_tasks_total_duration(self):
        durations = [task.task_config.duration for task in self.unfinished_tasks]
        return np.sum(durations)

    @property
    def ready_unfinished_tasks(self):
        ls = []
        for job in self.jobs:
            ls.extend(job.ready_unfinished_tasks)
        return ls

    @property
    def comm_recv_ready_tasks(self):
        ls = []
        for job in self.jobs:
            ls.extend(job.comm_recv_ready_tasks)
        return ls

    @property
    def comm_recv_ready_unfinished_tasks(self):
        ls = []
        for job in self.jobs:
            ls.extend(job.comm_recv_ready_unfinished_tasks)
        return ls    

    @property
    def schedulable_unfinished_tasks(self):
        ls = []
        for job in self.jobs:
            ls.extend(job.schedulable_unfinished_tasks)
        return ls

    @property
    def tasks_which_has_waiting_instance(self):
        ls = []
        for job in self.jobs:
            ls.extend(job.tasks_which_has_waiting_instance)
        return ls

    @property
    def ready_tasks_which_has_waiting_instance(self):
        ls = []
        for job in self.jobs:
            ls.extend(job.ready_tasks_which_has_waiting_instance)
        return ls

    @property
    def comm_recv_ready_tasks_which_has_waiting_instance(self):
        ls = []
        for job in self.jobs:
            ls.extend(job.comm_recv_ready_tasks_which_has_waiting_instance)
        return ls

    @property
    def schedulable_tasks_which_has_waiting_instance(self):
        ls = []
        for job in self.jobs:
            ls.extend(job.schedulable_tasks_which_has_waiting_instance)
        return ls

    @property
    def has_schedulable_tasks_which_has_waiting_instance(self):
        for job in self.jobs:
            # 需要只对unfinished的job生效
            for task in job.unscheduled_tasks:
                if task.schedulable and task.has_waiting_task_instances:
                    return True
        return False
    
    @property
    def finished_jobs(self):
        ls = []
        for job in self.jobs:
            if job.finished:
                ls.append(job)
        return ls

    @property
    def finished_tasks(self):
        ls = []
        for job in self.jobs:
            ls.extend(job.finished_tasks)
        return ls

    @property
    def finished(self):
        if len(self.jobs) == 0:
            return False
        for job in self.jobs:
            if not job.finished:
                return False
        return True

    @property
    def running_task_instances(self):
        task_instances = []
        for machine in self.machines:
            task_instances.extend(machine.running_task_instances)
        return task_instances

    def add_machines(self, machine_configs):
        for machine_config in machine_configs:
            machine = Machine(machine_config)
            self.machines.append(machine)
            machine.attach(self)

    def add_job(self, job):
        self.jobs.append(job)
        job.attach(self)

    @property
    def cpu(self):
        return sum([machine.cpu for machine in self.machines])

    @property
    def memory(self):
        return sum([machine.memory for machine in self.machines])

    @property
    def disk(self):
        return sum([machine.disk for machine in self.machines])

    @property
    def cpu_capacity(self):
        return sum([machine.cpu_capacity for machine in self.machines])

    @property
    def memory_capacity(self):
        return sum([machine.memory_capacity for machine in self.machines])

    @property
    def disk_capacity(self):
        return sum([machine.disk_capacity for machine in self.machines])

    @property
    def state(self):
        return {
            'arrived_jobs': len(self.jobs),
            'unfinished_jobs': len(self.unfinished_jobs),
            'finished_jobs': len(self.finished_jobs),
            'unfinished_tasks': len(self.unfinished_tasks),
            'finished_tasks': len(self.finished_tasks),
            'running_task_instances': len(self.running_task_instances),
            'machine_states': [machine.state for machine in self.machines],
            'cpu': self.cpu / self.cpu_capacity,
            'memory': self.memory / self.memory_capacity,
            'disk': self.disk / self.disk_capacity,
        }
