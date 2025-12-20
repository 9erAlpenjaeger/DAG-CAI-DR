from playground.DAG.utils.runtime_calculation import communication_time, computation_time
from core.occurence import Occurence

from itertools import chain
from abc import ABC, abstractmethod

class Event():
    def __init__(self, task, machine, st, ft):
        self.task = task
        self.machine = machine 
        self.st = st
        self.ft = ft

    def __lt__(self, other):
        return self.st < other.st

    def __eq__(self, other):
        if self.task == other.task:
            if self.machine == other.machine:
                return True
            else:
                raise AssertionError('Check the Event class in base_list_algorithm code...')
                return False
        else:
            return False

class BaseListAlgorithm(ABC):
    def __init__(self, cluster, occurence_monitor):
        self.simulation = None
        self.cluster = cluster
        self.occurence_monitor = occurence_monitor

        self.tindex = 0 # 
        self.task_list = [] #
        self.tasks_events = {}
        for machine in self.cluster.machines:
            self.machine_events[machine] = []
        # 待补充
        self.reliability = 1.0 
        self.energy = 0.0

    def attach(self, simulation):
        self.simulation = simulation

    def start_time(self, task, machine): # 
        duration = self.cluster.comp_time(task, machine) #
        if len(task.parents) == 0:
            estimated_ct = max(task.job.job_config.submit_time, self.occurence_monitor.now)  # 
        else:
            ct = lambda pre: \
                        max(self.tasks_events[pre].ft, self.occurence_monitor.now) + \
                        self.cluster.comm_time(machine1 = pre.assigned_machine, machine2 = machine, task1 = pre, task2 = task)
            estimated_ct = max([ct(pre) for pre in task.parents])
        estimated_st = self.find_first_gap(machine, estimated_ct, duration)
        return estimated_st

    def find_first_gap(self, machine, estimated_ct, duration):
        if estimated_ct < self.occurence_monitor.now:
            raise AssertionError('Estimated communication time is not calculated correctly...')
        if len(self.machine_events[machine]) == 0:
            earlist_st = estimated_ct
            return earlist_st
        else:
            events_ = chain([Event(task=None, machine=None, st=None, ft=self.occurence_monitor.now)], \
                                    sorted(self.machine_events[machine])) 
            events_ = list(events_)
            for i in range(len(events_) - 1):
                e1 = events_[i]
                e2 = events_[i + 1]
                earliest_st = max(e1.ft, estimated_ct)
                if e2.st - earliest_st > duration:
                    return earliest_st
            # if gap not found
            e = events_[-1]
            earliest_st = max(e.ft, estimated_ct)
            return earliest_st

    def provisionable(self, task):
        for pre in task.parents:
            if not pre.started:
                return False 
        return True

    def has_finished_parent(self, task):
        if len(task.parents) == 0:
            return True
        for pre in task.parents:
            if pre.finished:
                return True
        return False
        
    def provision_machine(self, task, machine):
        # 如果task == job.start_node呢
        estimated_st = self.start_time(task, machine)
        estimated_ft = estimated_st + self.cluster.comp_time(task, machine)
        if task.task_index == task.job.end_node:
            print('endnode', estimated_st, estimated_ft)

        task.assigned_machine = machine
        event = Event(task=task, machine=machine, st=estimated_st, ft=estimated_ft)
        self.tasks_events[task] = event
        self.machine_events[machine].append(event) 
        for pre in task.parents:
            if pre.finished or self.tasks_events[pre].ft <= self.occurence_monitor.now: # 
                ct = self.cluster.comm_time(
                                            machine1 = pre.assigned_machine,
                                            machine2 = task.assigned_machine,
                                            task1 = pre,
                                            task2 = task)
                start_transmission_occurence = Occurence(
                                            trigger_time= self.occurence_monitor.now + ct,
                                            otype = Occurence.FINISH,
                                            action = pre.task_instances[0].transmission,
                                            dest = task)
                self.occurence_monitor.add_occurence(start_transmission_occurence)

    def start_operations(self, task):
        event = self.tasks_events[task]
        machine = event.machine
        duration = self.cluster.comp_time(task, machine)
        event.st = self.occurence_monitor.now 
        event.ft = event.st + duration

    def finish_operations(self, task):
        event = self.tasks_events[task]
        #event.ft = self.occurence_monitor.now
        machine = event.machine 
        self.machine_events[machine].remove(event)
        self.tasks_events.pop(task)

    def find_first_unstarted_in_list(self):
        while True:
            if self.tindex >= len(self.task_list) or (not self.task_list[self.tindex].started):
                break 
            else:
                self.tindex += 1

    @abstractmethod
    def submit_operations(self):
        pass
        
    def provision_operations(self):
        if self.tindex >= len(self.task_list):
            return
        task = self.task_list[self.tindex]
        while self.provisionable(task): #
            if task not in self.tasks_events:
                machine = self.machine_choosing(task)
                self.provision_machine(task=task, machine=machine)
            self.tindex += 1
            self.find_first_unstarted_in_list()   
            if self.tindex >= len(self.task_list):
                return
            task = self.task_list[self.tindex]   

    def task_start_checking(self):
        if len([job for job in self.cluster.jobs if not job.finished]):
            self.provision_operations()
            for task in self.task_list:
                if not task.started:
                    if task.comm_recv_ready and task in self.tasks_events:
                        event = self.tasks_events[task]
                        if event.machine != task.assigned_machine:
                            raise AssertionError('Machine not provisoned correctly, check the provision algorithms in base_list_algorithm code...')
                        if not task.assigned_machine.occupied:
                            if task.task_index == task.job.start_node:
                                print(task.id, 'start at', self.occurence_monitor.now)                    
                            task.task_instances[0].execute()
        if not self.simulation.finished:
            task_start_checking_occurence = Occurence(
                    trigger_time = self.occurence_monitor.now + 1,
                    otype = Occurence.PROVISION,
                    action = self.task_start_checking 
                )
            self.occurence_monitor.add_occurence(task_start_checking_occurence)
        

    @abstractmethod
    def generate_list(self):
        pass

    @abstractmethod
    def machine_choosing(self, task): 
        pass