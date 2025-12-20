from core.config import *
from core.occurence import Occurence
import numpy as np
import networkx as nx

from playground.DAG.utils.runtime_calculation import communication_time, computation_time
from playground.DAG.utils.get_irr import get_irrelevant
from playground.DAG.utils.get_tasks import get_task_by_index

from datetime import datetime

class Task(object):
    def __init__(self, occurence_monitor, job, task_config):
        self.occurence_monitor = occurence_monitor
        self.job = job
        self.job_id = self.job.id
        self.task_index = task_config.task_index
        self.task_config = task_config
        self.instances_number = task_config.instances_number
        self.scheduled = False
        self._ready = False
        self._comm_recv_ready = False
        self._schedulable = False
        self._started = False
        self._finished = False
        self._parents = None
        self._childs = None
        self.assigned_machine = None
        self.data_location = set()

        self.task_instances = []
        task_instance_config = TaskInstanceConfig(task_config)
        for task_instance_index in range(int(self.task_config.instances_number)):
            self.task_instances.append(TaskInstance(self.occurence_monitor, self, task_instance_index, task_instance_config))
        self.next_instance_pointer = 0

    def reset(self):
        self.scheduled = False
        self._ready = False
        self._comm_recv_ready = False
        self._schedulable = False
        self._started = False
        self._finished = False
        self._parents = None
        self._childs = None
        self.assigned_machine = None
        self.data_location = set()
    
    @property
    def id(self):
        return int(self.job_id * 10000 + self.task_index)

    @property
    def nx_G(self):
        return self.job.nx_G

    @property
    def parents(self):
        if self._parents is None:
            parent_indices = self.nx_G.predecessors(self.task_index)
            self._parents = [get_task_by_index(self.job, parent_index) for parent_index in parent_indices]
        return self._parents
    
    @property
    def childs(self):
        if self._childs is None:
            child_indices = self.nx_G.successors(self.task_index)
            self._childs = [get_task_by_index(self.job, child_index) for child_index in child_indices]
        return self._childs

    @property
    def ready(self):
        if not self._ready:
            for p in self.parents:
                if not p.finished:
                    return False
            self._ready = True
        return self._ready
            
    @property
    def ready_and_unscheduled(self):
        return self.ready and (not self.scheduled)

    @property
    def comm_recv_ready(self):
        if not self._comm_recv_ready:
            if self.task_index == self.job.start_node:
                self._comm_recv_ready = True
                return self._comm_recv_ready

            if self.assigned_machine is None:
                return False
            for p in self.parents:
                if not self.assigned_machine in p.data_location:
                    return False
            self._comm_recv_ready = True
        return self._comm_recv_ready
    
    @property
    def schedulable(self):
        #t0:
        if not self._schedulable:
            if len(self.parents) == 0:
                self._schedulable = True
                return self._schedulable
            one_pre_started = False
            for p in self.parents:
                if not p.scheduled:
                    return False
                if p.started:
                    one_pre_started = True
                    #break
            self._schedulable = one_pre_started
        return self._schedulable
        #t1:
        '''
        if not self._schedulable:
            for p in self.parents:
                if not p.started:
                    return False
            self._schedulable = True
        return self._schedulable
        '''
    @property
    def running_task_instances(self):
        ls = []
        for task_instance in self.task_instances:
            if task_instance.scheduled and not task_instance.finished:
                ls.append(task_instance)
        return ls

    @property
    def finished_task_instances(self):
        ls = []
        for task_instance in self.task_instances:
            if task_instance.finished:
                ls.append(task_instance)
        return ls

    # the most heavy
    def schedule_task_instance(self, machine):
        self.assigned_machine = machine
        self.task_instances[self.next_instance_pointer].schedule(machine)
        self.next_instance_pointer += 1

    @property
    def started(self):
        if self._started:
            return True
        for task_instance in self.task_instances:
            if task_instance.started:
                self._started = True
                return True
        return False
    '''
    @property
    def scheduled(self):
        for task_instance in self.task_instances:
            if task_instance.scheduled:
                return True
        return False
    '''
    @property
    def waiting_task_instances_number(self):
        return self.instances_number - self.next_instance_pointer

    @property
    def has_waiting_task_instances(self):
        return self.instances_number > self.next_instance_pointer

    @property
    def finished(self):
        """
        A task is finished only if it has no waiting task instances and no running task instances.
        :return: bool
        """
        if self._finished:
            return True
        else:
            for task_instance in self.task_instances:
                if task_instance.finished:
                    self._finished = True  
                    return True
            return False
        '''
        else:
            if self.has_waiting_task_instances:
                return False
            elif len(self.running_task_instances) != 0:
                return False
            else:
                self._finished = True
                return True
        '''

    @property
    def started_timestamp(self):
        t = None
        for task_instance in self.task_instances:
            if task_instance.started_timestamp is not None:
                if (t is None) or (t > task_instance.started_timestamp):
                    t = task_instance.started_timestamp
        return t

    @property
    def finished_timestamp(self):
        if not self.finished:
            return None
        t = None
        for task_instance in self.task_instances:
            if (t is None) or (t < task_instance.finished_timestamp):
                t = task_instance.finished_timestamp
        return t

    def __hash__(self):
        #print(dir(self.job))
        return int(self.job_id * 10000 + self.task_index)

    def __eq__(self, other):
        if isinstance(other, Task):
            return (self.job_id == other.job_id) and (self.task_index == other.task_index) 
        else:
            return False

    def __lt__(self, other):
        # 仅作为排序依据
        if isinstance(other, Task):
            if self.job_id < other.job_id:
                return True 
            elif self.job_id > other.job_id:
                return False
            else:
                return self.task_index < other.task_index
        return False

    def __str__(self):
        return str(self.job_id) + '_' + str(self.task_index)
    '''  '''



class Job(object):
    task_cls = Task

    def __init__(self, occurence_monitor, job_config):
        self.occurence_monitor = occurence_monitor
        self.job_config = job_config
        self.id = job_config.id

        self.cluster = None

        self.tasks_map = {}
        self._tasks = None

        self.nx_G = nx.DiGraph()

        self.start_node = self.job_config.start_node
        self.end_node = self.job_config.end_node

        self.cnt1, self.cnt2 = datetime.now() - datetime.now(), datetime.now() - datetime.now()
        
        ''' build the networkx DAG of the workflow'''
        '''  '''
        for task_config in job_config.task_configs:
            task_index = task_config.task_index
            self.nx_G.add_node(task_index)
            task = Job.task_cls(occurence_monitor, self, task_config)
            self.nx_G.nodes[task_index]['task'] = task
        for task_config in job_config.task_configs:
            task_index = task_config.task_index
            '''  '''
            for parent_index in task_config.parent_indices:
                self.nx_G.add_edge(parent_index, task_index)
                '''  '''
                self.nx_G[parent_index][task_index]['comm_ready'] = False
        self.job_size = self.nx_G.number_of_nodes()

        self.irrelevant_pair = get_irrelevant(self.nx_G)
            
        for task_config in job_config.task_configs:
            task_index = task_config.task_index
            self.tasks_map[task_index] = Job.task_cls(occurence_monitor, self, task_config)

        self.unscheduled_tasks = set(self.tasks)
        self.scheduled_tasks = set()

        self.unstarted_tasks = set(self.tasks)
        self.started_tasks = set()

        self.unfinished_tasks = set(self.tasks)
        self.finished_tasks = set()


    def attach(self, cluster):
        self.cluster = cluster
                        
    '''  '''
    @property
    def tasks(self):
        if self._tasks is None:
            self._tasks = []
            for task_index in self.nx_G.nodes():
                task = get_task_by_index(self, task_index)
                self._tasks.append(task)
            
            #self._tasks = list(sorted(self._tasks))
        return self._tasks

    '''
    @property
    def unfinished_tasks(self):
        ls = []
        for task in self.tasks:
            if not task.finished:
                ls.append(task)
        return ls
    '''

    @property
    def ready_tasks(self):
        ls = []
        for task in self.tasks:
            if task.ready:
                ls.append(task)
        return ls

    @property
    def schedulable_unfinished_tasks(self):
        ls = []
        for task in self.tasks:
            if not task.finished and task.schedulable:
                ls.append(task)
        return ls
    '''
    @property
    def unstarted_tasks(self):
        ls = []
        for task in self.tasks:
            if not task.started:
                ls.append(task)
        return ls
    '''
    
    @property
    def schedulable_unstarted_tasks(self):
        ls = []
        for task in self.tasks:
            if not task.started and task.schedulable:
                ls.append(task)
        return ls

    @property
    def ready_and_unscheduled_tasks(self):
        ls = []
        for task in self.tasks:
            if task.ready_and_unscheduled:
                ls.append(task)
        return ls
           
    @property
    def ready_unfinished_tasks(self):
        ls = []
        for task in self.tasks:
            if not task.finished and task.ready:
                ls.append(task)
        return ls

    @property
    def comm_recv_ready_tasks(self):
        ls = []
        for task in self.tasks:
            if task.comm_recv_ready:
                ls.append(task)
        return ls

    @property
    def comm_recv_ready_unfinished_tasks(self):
        ls = []
        for task in self.tasks:
            if not task.finished and task.comm_recv_ready:
                ls.append(task)
        return ls        
    '''
    @property
    def unscheduled_tasks(self):
        return [task for task in self.tasks if not task.scheduled]
        ls = []
        for task in self.tasks:
            if not task.scheduled:
                ls.append(task)
        return ls
    
    @property
    def scheduled_tasks(self):
        ls = []
        for task in self.tasks:
            if task.scheduled:
                ls.append(task)
        return ls
    '''

    @property
    def tasks_which_has_waiting_instance(self):
        ls = []
        for task in self.tasks:
            if task.has_waiting_task_instances:
                ls.append(task)
        return ls

    @property
    def ready_tasks_which_has_waiting_instance(self):
        ls = []
        for task in self.unscheduled_tasks:
            if task.has_waiting_task_instances and task.ready:
                ls.append(task)
        return ls
    
    @property
    def comm_recv_ready_tasks_which_has_waiting_instance(self):
        ls = []
        for task in self.unscheduled_tasks:
            if task.has_waiting_task_instances and task.comm_recv_ready:
                ls.append(task)
        return ls
    
    @property
    def schedulable_tasks_which_has_waiting_instance(self):
        #return [task for task in self.tasks if task.has_waiting_task_instances and task.schedulable]
        ls = []
        for task in self.unscheduled_tasks:
            if task.has_waiting_task_instances and task.schedulable:
                ls.append(task)
        return ls        

    @property
    def running_tasks(self):
        ls = []
        for task in self.tasks:
            if task.started and not task.finished:
                ls.append(task)
        return ls

    '''
    @property
    def finished_tasks(self):
        ls = []
        for task in self.tasks:
            if task.finished:
                ls.append(task)
        return ls
    '''

    @property
    def scheduled(self):
        for task in self.tasks:
            if task.started:
                return True
        return False

    @property
    def started(self):
        return self.tasks[self.start_node].started
    '''
        for task in self.tasks:
            if task.started:
                return True
        return False
    '''

    @property
    def finished(self):
        end_task = get_task_by_index(self, self.end_node)
        return end_task.finished

    @property
    def started_timestamp(self):
        t = None
        for task in self.tasks:
            if task.started_timestamp is not None:
                if (t is None) or (t > task.started_timestamp):
                    t = task.started_timestamp
        return t

    @property
    def finished_timestamp(self):
        if not self.finished:
            return None
        t = None
        for task in self.tasks:
            if (t is None) or (t < task.finished_timestamp):
                t = task.finished_timestamp
        return t


class TaskInstance(object):
    def __init__(self, occurence_monitor, task, task_instance_index, task_instance_config):
        self.occurence_monitor = occurence_monitor
        self.task = task
        self.task_instance_index = task_instance_index
        self.config = task_instance_config
        self.cpu = task_instance_config.cpu
        self.memory = task_instance_config.memory
        self.disk = task_instance_config.disk
        self.duration = task_instance_config.duration
        self.datasize = task_instance_config.datasize

        self.machine = None
        self.process = None
        self.new = True

        self.started = False
        self.scheduled = False
        self.finished = False
        self.started_timestamp = None
        self.finished_timestamp = None
        self._estimated_started_timestamp_for_scheduled = None
        self._parents_comm_ft = None

    @property
    def id(self):
        return str(self.task.id) + '-' + str(self.task_instance_index)

    def parents_comm_finished_timestamp(self, machine):
        parents_comm_ft_ls = []
        for p in self.task.parents:
            if p.scheduled == False:
                raise ValueError("tasks with unscheduled parents are not permitted to call property function: 'parents_comm_finished_timestamp'")
    
            p_instance = p.task_instances[0]
            if p.finished:
                parent_ft = p_instance.finished_timestamp
            else:
                parent_st = p_instance.estimated_started_timestamp_for_scheduled
                parent_ft = parent_st + computation_time(p, p_instance.machine)
            parent_comm_time = communication_time(p, p_instance.machine, machine)
            parent_comm_ft = parent_ft + parent_comm_time
            parents_comm_ft_ls.append(parent_comm_ft)
        if len(parents_comm_ft_ls) > 0:
            return np.max(parents_comm_ft_ls)
        else:
            return self.task.job.job_config.submit_time + 1
            
    @property
    def estimated_started_timestamp_for_scheduled(self):
        if self.task.scheduled == False:
            raise ValueError("unscheduled tasks are not permitted to call property function: 'estimated_started_timestamp_for_scheduled'")
        if self._estimated_started_timestamp_for_scheduled is None:
            if self.started:
                self._estimated_started_timestamp_for_scheduled = self.started_timestamp
            else:  
                parents_comm_ft = self.parents_comm_finished_timestamp(self.machine)
                self._estimated_started_timestamp_for_scheduled = parents_comm_ft
                if len(self.machine.task_instances_queue) > 1:
                    last_task_instance = self.machine.task_instances_queue[-2]
                    last_estimated_st = last_task_instance.estimated_started_timestamp_for_scheduled
                    last_estimated_ft = last_estimated_st + computation_time(last_task_instance.task, self.machine)
                    if last_estimated_ft > self._estimated_started_timestamp_for_scheduled:
                        self._estimated_started_timestamp_for_scheduled = last_estimated_ft
        return self._estimated_started_timestamp_for_scheduled

    def estimated_started_timestamp_for_unscheduled(self, machine):
        if self.task.scheduled == True:
            raise ValueError("scheduled tasks are not permitted to call property function: 'estimated_started_timestamp_for_unscheduled'")
        if self.started:
            estimated_st = self.started_timestamp
        else:  
            parents_comm_ft = self.parents_comm_finished_timestamp(machine)
            estimated_st = parents_comm_ft 
            if len(machine.task_instances_queue) > 0:
                last_task_instance = machine.task_instances_queue[-1] 
                last_estimated_st = last_task_instance.estimated_started_timestamp_for_scheduled
                last_estimated_ft = last_estimated_st + computation_time(last_task_instance.task, machine)
                if last_estimated_ft > estimated_st:
                    estimated_st = last_estimated_ft
        return estimated_st       

    def do_work(self):
        ''' start executing'''
        self.started = True
        self.task.job.started_tasks.add(self.task)
        self.task.job.unstarted_tasks.remove(self.task)
        real_runtime = self.task.job.cluster.comp_time(self.task, self.machine)
        estimated_finish_time = real_runtime + self.occurence_monitor.now 
        if real_runtime != 0:
            task_finish_occurence = Occurence(
                                trigger_time=estimated_finish_time,
                                otype = Occurence.FINISH,
                                action=self.finish_action
                                )
            self.occurence_monitor.add_occurence(task_finish_occurence)
        else:
            self.finish_action()
        '''
        if real_runtime > 0:
            yield self.env.timeout(real_runtime)
        self.finished_timestamp = self.env.now
        self.finished = True
        self.machine.stop_task_instance(self)
        for child in self.task.childs:
            yield self.env.process(self.transmission(child))
        '''

    def finish_action(self):
        self.finished_timestamp = self.occurence_monitor.now
        self.finished = True
        self.machine.stop_task_instance(self)
        self.task.job.finished_tasks.add(self.task)
        self.task.job.unfinished_tasks.remove(self.task)
        self.task.data_location.add(self.task.assigned_machine) #   
        for child in self.task.childs: 
            if child.assigned_machine is not None:
                ct = self.task.job.cluster.comm_time(
                                            machine1 = self.task.data_location,
                                            machine2 = child.assigned_machine,
                                            task1 = self.task,
                                            task2 = child)
                print(ct)
                start_transmission_occurence = Occurence(
                                            trigger_time= self.occurence_monitor.now + ct,
                                            otype = Occurence.FINISH,
                                            action = self.transmission,
                                            dest = child)
                self.occurence_monitor.add_occurence(start_transmission_occurence)

    def transmission(self, dest): 
        task_index = self.task.task_index
        dest_index = dest.task_index
        self.task.nx_G[task_index][dest_index]['comm_ready'] = True
        self.task.data_location.add(dest.assigned_machine)

    def schedule(self, machine):
        # HINT: not called in list-based scheduling
        self.scheduled = True
        self.task.scheduled = True
        self.task.job.scheduled_tasks.add(self.task)
        self.task.job.unscheduled_tasks.remove(self.task)
        # self.started = True
        self.machine = machine
        self.machine.add_task_instance(self)
        
   
                
    def execute(self):    
        self.machine = self.task.assigned_machine
        self.machine.add_task_instance(self)
        self.machine.run_task_instance(self)
        self.started_timestamp = self.occurence_monitor.now
        # 开始了就要把
        self.do_work()
        #yield self.env.process(self.do_work())
