class TaskInstanceConfig(object):
    def __init__(self, task_config):
        self.cpu = task_config.cpu
        self.memory = task_config.memory
        self.disk = task_config.disk
        self.duration = task_config.duration
        self.datasize = task_config.datasize


class TaskConfig(object):
    def __init__(self, task_index, instances_number, cpu, memory, disk, duration, datasize, parent_indices, machine_preference):
        self.task_index = task_index
        self.instances_number = instances_number
        self.cpu = cpu
        self.memory = memory
        self.disk = disk
        self.duration = duration
        self.datasize = datasize
        self.parent_indices = parent_indices
        self.machine_preference = machine_preference
        self.child_indices = []
        


class JobConfig(object):
    def __init__(self, idx, submit_time, task_configs, start_node, end_node):
        self.submit_time = submit_time
        self.task_configs = task_configs
        self.id = idx
        self.start_node = start_node
        self.end_node = end_node
