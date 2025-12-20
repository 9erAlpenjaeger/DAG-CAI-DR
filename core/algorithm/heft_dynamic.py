from core.algorithm.base_list_algorithm import BaseListAlgorithm

import numpy as np
from overrides import overrides

class HeftDynamic(BaseListAlgorithm):
    def __init__(self, cluster, occurence_monitor):
        super().__init__(cluster, occurence_monitor)
        self.ranku_dict = {}
        self.rankd_dict = {}

    def wbar(self, task):
        ''' Average computation cost'''
        et = np.mean([self.cluster.comp_time(task, machine) for machine in self.cluster.machines])
        return et

    def cbar(self, task1, task2):
        ''' Average communication cost'''
        return 0
        mn = len(self.cluster.machines)
        if mn == 1:
            return 0
        else:
            cts = []
            for machine1 in self.cluster.machines:
                for machine2 in self.cluster.machines:
                    ct = self.cluster.comm_time(machine1, machine2, task1, task2)
                    cts.append(1.0*ct)
            return np.mean(cts)

    def ranku(self, task):
        if task in self.ranku_dict:
            return self.ranku_dict[task]
        else:
            ranku_task = self.wbar(task)
            if len(task.childs):
                ranku_task += max(self.cbar(task, taskj) + self.ranku(taskj)\
                                    for taskj in task.childs)
            self.ranku_dict[task] = ranku_task
            return self.ranku_dict[task]

    def rankd(self,task):
        if task in self.rankd_dict:
            return self.rankd_dict[task]
        else:
            rankd_task = self.wbar(task)
            if len(task.parents):
                rankd_task += max(self.cbar(task, taskj) + self.rankd(taskj)\
                                    for taskj in task.parents)
            self.rankd_dict[task] = rankd_task
        return self.rankd_dict[task]

    @overrides
    def submit_operations(self):
        self.tasks_events = {task: event for task, event in self.tasks_events.items() if (not task.job.finished) and self.has_finished_parent(task)}
        for machine in self.cluster.machines:
            for event in self.machine_events[machine]:
                self.machine_events[machine] = [event for event in self.machine_events[machine] \
                                                if (not event.task.job.finished) and self.has_finished_parent(event.task)]
        self.generate_list()

    @overrides
    def generate_list(self):
        # according to rankup sorting rule
        self.ranku_dict = {} # reset the ranku list
        self.task_list.extend(list(reversed(sorted([task for task in self.cluster.tasks if not task.job.started], key = self.ranku)))) # 
        self.tindex = 0
        self.find_first_unstarted_in_list()

    @overrides
    def machine_choosing(self, task):
        # according to Earliest Finish Time greedy rule
        ft = lambda machine: self.start_time(task, machine) + self.cluster.comp_time(task, machine)
        
        #print([self.cluster.comp_time(task, machine) for machine in self.cluster.machines], min(self.cluster.machines, key=ft).id)
        machine = min(self.cluster.machines, key = ft)
        #min_machine = min([ft(machine)for machine in self.cluster.machines])
        #machines = [m for m in self.cluster.machines if ft(m) == min_machine]
        #machine = np.random.choice(machines)
        return machine