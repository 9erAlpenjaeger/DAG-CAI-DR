from operator import attrgetter
import pandas as pd
import numpy as np

from core.job import JobConfig, TaskConfig
from core.machine import MachineConfig
from playground.DAG.utils.feature_synthesize import father_task_indices


class CSVReader(object):
    def __init__(self, filename):
        self.filename = filename
        df = pd.read_csv(self.filename)

        df.job_id = df.job_id.astype(dtype=int)
        df.instances_num = df.instances_num.astype(dtype=int)

        job_task_map = {}
        job_submit_time_map = {}

        edge_list = []
        
        for i in range(len(df)):
            series = df.iloc[i]
            job_id = series.job_id
            if series.parent_id == 'None':
                task_id, parent_indices = int(series.task_id), []
            else:
                task_id, parent_indices = int(series.task_id), [int(parent_index) for parent_index in series.parent_id.split('_')]
                for parent_index in series.parent_id.split('_'):
                    edge_list.append([int(parent_index),int(series.task_id)])
            # series_task_id = str(int(series.task_id))
            # task_id, parent_indices = father_task_indices(series_task_id, '0')

            cpu = series.cpu
            memory = series.memory
            disk = series.disk
            duration = series.duration
            datasize = series.datasize
            submit_time = series.submit_time
            instances_num = series.instances_num

            task_configs = job_task_map.setdefault(job_id, [])
            task_configs.append(TaskConfig(task_id, instances_num, cpu, memory, disk, duration, datasize, parent_indices))
            job_submit_time_map[job_id] = submit_time

        for task_config in task_configs:
            child_indices = []
            for edge in edge_list:
                if task_config.task_index == edge[0]:
                    child_indices.append(edge[1])
            task_config.child_indices = child_indices

        
        job_configs = []
        for job_id, task_configs in job_task_map.items():
            job_configs.append(JobConfig(job_id, job_submit_time_map[job_id], task_configs))
        job_configs.sort(key=attrgetter('submit_time'))

        self.job_configs = job_configs

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
            job_config.submit_time -= submit_time_base
            tasks_number += len(job_config.task_configs)
            for task_config in job_config.task_configs:
                task_instances_numbers.append(task_config.instances_number)
                task_instances_durations.extend([task_config.duration] * int(task_config.instances_number))
                task_instances_datasizes.extend([task_config.datasize] * int(task_config.instances_number))
                task_instances_cpu.extend([task_config.cpu] * int(task_config.instances_number))
                task_instances_memory.extend([task_config.memory] * int(task_config.instances_number))
        '''
        print('Jobs number: ', len(ret))
        print('Tasks number:', tasks_number)

        print('Task instances number mean: ', np.mean(task_instances_numbers))
        print('Task instances number std', np.std(task_instances_numbers))

        print('Task instances cpu mean: ', np.mean(task_instances_cpu))
        print('Task instances cpu std: ', np.std(task_instances_cpu))

        print('Task instances memory mean: ', np.mean(task_instances_memory))
        print('Task instances memory std: ', np.std(task_instances_memory))

        print('Task instances duration mean: ', np.mean(task_instances_durations))
        print('Task instances duration std: ', np.std(task_instances_durations))
        
        print('Task instances datasize mean: ', np.mean(task_instances_datasizes))
        print('Task instances datasize std: ', np.std(task_instances_datasizes))
        '''
        return ret

class MachineReader(object):
    def __init__(self, filename):
        df = pd.read_csv(self.filename)
        self.machine_configs = []
        for i in range(len(df)):
            series = df.iloc[i]
            machine_id = int(series.machine_id)
            cpu_capacity = int(series.cpu)
            memory_capacity = int(series.memory)
            disk_capacity = int(series.disk)
            compute_capacity = int(series.compute_capacity)
            machine_config = MachineConfig(machine_id, cpu_capacity, memory_capacity, disk_capacity, compute_capacity)
            self.machine_configs.append(machine_config)