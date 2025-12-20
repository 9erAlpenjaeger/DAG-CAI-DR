from core.monitor import Monitor
from core.occurence import OccurenceMonitor

from datetime import datetime

class Simulation(object):
    def __init__(self, occurence_monitor, cluster, task_broker, algorithm, event_file):
        self.occurence_monitor = occurence_monitor
        self.cluster = cluster
        self.task_broker = task_broker
        self.algorithm = algorithm
        self.event_file = event_file
        if event_file is not None:
            self.monitor = Monitor(self)

        self.task_broker.attach(self)
        self.algorithm.attach(self)

        self.cnt = datetime.now() - datetime.now() 

    def run(self):
        #print('run')
        # Starting monitor process before task_broker process
        # and scheduler process is necessary for log records integrity.
        if self.event_file is not None:
            self.monitor.run()
        self.task_broker.run()
        self.algorithm.task_start_checking()

    @property
    def finished(self):
        is_finished = self.task_broker.destroyed \
               and self.cluster.finished #len(self.cluster.unfinished_tasks) == 0
        return is_finished
