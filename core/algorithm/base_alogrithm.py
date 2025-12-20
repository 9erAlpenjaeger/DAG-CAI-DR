from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):

    def __init__(self, cluster, occurence_monitor):
        self.cluster = cluster
        self.occurence_monitor = occurence_monitor

    @abstractmethod
    def task_to_schedule(self):
        pass
        
    @abstractmethod
    def task_to_execute(self):
        pass

    @abstractmethod
    def machine_to_provision(self):
        pass

class BaseTaskSequencing():
    def __init__(self):
        pass

class BaseMachineProvisioning():
    def __init__(self):
        pass

class BaseTaskExecuting():
    def __init__(self):
        pass

