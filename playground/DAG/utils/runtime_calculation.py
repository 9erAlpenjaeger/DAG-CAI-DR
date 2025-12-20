import math

def computation_time(task, machine):
    #print(int(task.task_config.duration / machine.compute_capacity) + 1)
    return int(task.task_config.duration / machine.compute_capacity) + 1

def preferred_computation_time(task, machine):
    machine_id = machine.id
    return task.task_config.duration / task.task_config.machine_preference[machine_id] + 0.1

def communication_time(task, machine_send, machine_recv):
    #return 0
    if machine_send == machine_recv:
        return 0
    return int(task.task_config.datasize / (1000)) + 1

def energy(task, machine):
    return computation_time(task, machine) * machine.energy_cost

def total_energy(cluster):
    t_e = 0.0
    for task in cluster.scheduled_tasks:
        machine = task.task_instances[0].machine
        t_e = t_e + energy(task, machine)
    return t_e  

def reliability(task, machine):
    #print(computation_time(task, machine), machine.fault_rate)
    #print(math.exp(-computation_time(task, machine) * machine.fault_rate))
    return math.exp(-computation_time(task, machine) * machine.fault_rate)

def total_reliability(cluster):
    t_r = 1.0
    for task in cluster.scheduled_tasks:
        machine = task.task_instances[0].machine
        t_r = t_r * reliability(task, machine)
    return t_r