from core.job import Job
from core.occurence import Occurence


class Broker(object):
    job_cls = Job

    def __init__(self, occurence_monitor, job_configs):
        self.occurence_monitor = occurence_monitor
        self.simulation = None
        self.cluster = None
        self.destroyed = False
        self.job_configs = job_configs

    def attach(self, simulation):
        self.simulation = simulation
        self.cluster = simulation.cluster

        ''' for static scheduling only'''
        
        for job_config in self.job_configs:
            job = Broker.job_cls(self.occurence_monitor, job_config)
                # 问题出在这
            self.cluster.add_job(job)
        

        

    def run(self):
        for i in range(len(self.job_configs)):
            job_config = self.job_configs[i]
            last_flag = i>=(len(self.job_configs)-1)
            assert job_config.submit_time >= self.occurence_monitor.now
            new_job_occurence = Occurence(
                            trigger_time=job_config.submit_time,
                            otype = Occurence.ARRIVAL,
                            action=self.adding_a_new_job,
                            job=Broker.job_cls(self.occurence_monitor, job_config),
                            last_flag = last_flag
                        )
            self.occurence_monitor.add_occurence(new_job_occurence)
        #self.destroyed = True

    def adding_a_new_job(self, job, last_flag):
        self.cluster.add_job(job)
        print('Workflow',job.id, 'is submitted at timepoint:', self.occurence_monitor.now)
        #self.simulation.algorithm.generate_list()
        self.simulation.algorithm.submit_operations()
        if not self.destroyed:
            if last_flag:
                self.destroyed = True
        #print(self.destroyed)


    def poisson_generator(self, inverval):
        pass
