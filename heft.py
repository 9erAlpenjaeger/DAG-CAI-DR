

from functools import partial
from collections import namedtuple
from itertools import chain
import time
import numpy as np
import os
from copy import deepcopy

from gymenvs.omcloudgym import OmCloudGym
#from gymenvs.cloudgym_oneshot import CloudGym_oneshot
from core.machine import MachineConfig
from playground.DAG.utils.xml_reader import XMLReader
from playground.DAG.utils.json_reader import JSONReader



from playground.DAG.utils.get_tasks import get_task_by_index
from playground.DAG.utils.runtime_calculation import computation_time, preferred_computation_time, communication_time, reliability, energy
Event = namedtuple('Event', 'job start end')
WCT = namedtuple('WCT', 'ready start lastend end oft oftimp lastoft minstart') 

EFT = 0
LINEAR_WEIGHTED_GREEDY = 1
RANKED_MACHINE = 2
GIVEN_MACHINE = 3
CHEBYSHEV_WEIGHTED_GREEDY = 4
NDsort_MACHINE = 5
PRIORIZED_MACHINE = 6
ALLOC_STR = ['EFT', 'LINEAR_WEIGHTED_GREEDY', 'RANKED_MACHINE', 'GIVEN_MACHINE', 'CHEBYSHEV_WEIGHTED_GREEDY', 'NDsort_MACHINE']

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield root + '/' + f
            
class Heft_A():
    def __init__(self, env):
        self.env = env
        self.ranked_jobs = None
        self.job_WCT_match = None
        self.job_machine_match = None
        self.machine_priority = None
        self.weights = self.env.weights
        self.load_dag()
        
        orders, jobson = self.schedule()
        dense_rew, mksp = self.makespan(orders=orders)
        self.baseline_ranked_jobs = deepcopy(self.ranked_jobs)
        self.baseline_dense_rew = dense_rew
        
        self.machine_available_mask = np.zeros((self.num_n, self.num_p), dtype=bool)
        self.machine_chosen_mask = np.zeros((self.num_n, self.num_p), dtype=bool)
    
    def load_dag(self):
        self.eval = False
        self.alloc_strategy = None
        self.reliability = 1.0
        self.energy = 0.0
        self.job = self.env.cluster.jobs[0]
        self.G_T = self.job.nx_G
        self.machines = self.env.cluster.machines
        self.num_n = self.G_T.number_of_nodes()
        self.num_p = len(self.machines)
        
        self.prec, self.succ = {}, {}
        self.timer = time.time()
        self.ranku_list = np.zeros([self.num_n, 2]) # (if_ranked, rank_value)
        self.rankd_list = np.zeros([self.num_n, 2]) # (if_ranked, rank_value)
        for n in self.G_T.nodes():
            prec_list = []
            for prec in self.G_T.predecessors(n):
                prec_list.append(prec)
            self.prec[n] = prec_list
        for n in self.G_T.nodes():
            succ_list = []
            for succ in self.G_T.successors(n):
                succ_list.append(succ)
            self.succ[n] = succ_list       
 
        self.commcost_list = np.zeros([self.num_n, self.num_n, self.num_p, self.num_p])
        for ni in self.G_T.nodes():
            for nj in self.G_T.successors(ni):
                for ai in range(self.num_p):
                    for aj in range(self.num_p):
                        self.commcost_list[ni][nj][ai][aj] = 0

    def set_alloc_strategy(self, alloc_strategy):
        self.alloc_strategy = alloc_strategy

    def set_machine_pri(self, machine_pri):
        self.machine_priority = machine_pri
            
    def compcost(self, job, agent):
        task = get_task_by_index(self.job, job)
        machine = self.machines[agent]
        comptime = computation_time(task, machine)
        #comptime = preferred_computation_time(task, machine)
        return comptime

    def commcost(self, job1, job2, agent1, agent2):
        return 0
        task = get_task_by_index(self.job, job1)
        machine1 = self.machines[agent1]
        machine2 = self.machines[agent2]
        commtime = communication_time(task, machine1, machine2)
        return commtime

    def wbar(self, ni):
        """ Average computation cost """
        comptimes = [self.compcost(ni, agent) for agent in range(self.num_p)]
        return 1. * sum(comptimes) / self.num_p

    def cbar(self, ni, nj):
        """ Average communication cost """
        #return 0
        if self.num_p == 1:
            return 0
        summed_c = 1.0 * self.commcost(ni, nj, 0, 1) * (self.num_p - 1) / self.num_p
        return summed_c

    def rankud(self, ni):
        return self.ranku(ni) + self.rankd(ni)

    def ranku(self, ni):
        """ Rank of job
        [1]. http://en.wikipedia.org/wiki/Heterogeneous_Earliest_Finish_Time
        """
        if self.ranku_list[ni][0] == 1:
            ranku_ni = self.ranku_list[ni][1]
        elif ni in self.succ and self.succ[ni]:
            try:
                ranku_ni = self.wbar(ni) + max(self.cbar(ni, nj) + self.ranku(nj) for nj in self.succ[ni])
            except:
                exit(0)
            self.ranku_list[ni][0] = 1
            self.ranku_list[ni][1] = ranku_ni
        else:
            ranku_ni = self.wbar(ni)
            self.ranku_list[ni][0] = 1
            self.ranku_list[ni][1] = ranku_ni
        return ranku_ni

    def rankd(self, ni):
        if self.rankd_list[ni][0] == 1:
            rankd_ni = self.rankd_list[ni][1]
        elif ni in self.succ and self.succ[ni]:
            rankd_ni = self.wbar(ni) + max(self.cbar(ni, nj) + self.rankd(nj) for nj in self.succ[ni])
            self.rankd_list[ni][0] = 1
            self.rankd_list[ni][1] = rankd_ni
        else:
            rankd_ni = self.wbar(ni)
            self.rankd_list[ni][0] = 1
            self.rankd_list[ni][1] = rankd_ni
        return rankd_ni

    def endtime(self, job, events):
        """ Endtime of job in list of events """
        for e in events:
            if e.job == job:
                return e.end

    def find_first_gap(self, agent_orders, desired_start_time, duration):
        """Find the first gap in an agent's list of jobs
        The gap must be after `desired_start_time` and of length at least
        `duration`.
        """
        # No jobs: can fit it in whenever the job is ready to run
        if (agent_orders is None) or (len(agent_orders)) == 0:
            return desired_start_time

        # Try to fit it in between each pair of Events, but first prepend a
        # dummy Event which ends at time 0 to check for gaps before any real
        # Event starts.
        a = chain([Event(None,None,0)], agent_orders[:-1])
        for e1, e2 in zip(a, agent_orders):
            earliest_start = max(desired_start_time, e1.end)
            if e2.start - earliest_start > duration:
                return earliest_start

        # No gaps found: put it at the end, or whenever the task is ready
        return max(agent_orders[-1].end, desired_start_time)

    def comm_ready_time(self, job, orders, jobson, agent):
        if job in self.prec and self.prec[job]:
            comm_ready = max([self.endtime(p, orders[jobson[p]])
                        + self.commcost(p, job, agent, jobson[p]) for p in self.prec[job]])
        else:
            comm_ready = 0
        return comm_ready        

    def start_time(self, job, orders, jobson, agent):
        """ Earliest time that job can be executed on agent """
        duration = self.compcost(job, agent)
        comm_ready = self.comm_ready_time(job, orders, jobson, agent)
        return self.find_first_gap(orders[agent], comm_ready, duration)

    def allocate(self, job, orders, jobson):
        """ Allocate job to the machine with earliest finish time. Operates in place"""
        task = get_task_by_index(self.job, job)
        
        st = partial(self.start_time, job, orders, jobson)
        ft = lambda agent: st(agent) + self.compcost(job, agent)            
        eg = lambda agent: self.energy + energy(task, self.machines[agent])
        rb = lambda agent: self.reliability * reliability(task, self.machines[agent])
        if self.eval == False:
            if self.alloc_strategy == GIVEN_MACHINE:            
                agent = self.job_machine_match[job]
            else:
                agent = min(orders.keys(), key=ft)
        elif self.alloc_strategy == EFT:
            agent = min(orders.keys(), key=ft)
        start, end = st(agent), ft(agent)
        machine = self.machines[agent]
        #self.available_machine_mask_for_job[]
        self.reliability = self.reliability * reliability(task, machine)
        self.energy = self.energy + energy(task, machine)
        
        orders[agent].append(Event(job, start, end))
        orders[agent] = sorted(orders[agent], key=lambda e: e.start)
        self.job_WCT_match[job] = WCT(0, start, 0, end, 0, 0, 0, 0)

        jobson[job] = agent

    def makespan(self, orders):
        """ Finish time of last job """
        for i in range(len(self.ranked_jobs)):
            job = self.ranked_jobs[i]
            start = self.job_WCT_match[job][1]
            end = self.job_WCT_match[job][3]
            if i != 0:
                last_job = self.ranked_jobs[i-1]
                last_end = self.job_WCT_match[last_job][3]
                if end > oft:
                    oftimp = end - oft
                    oft = end
                else:
                    oftimp = 0
                last_oft = self.job_WCT_match[last_job][4]
            else:
                last_end = 0
                oft = end
                oftimp = 0
                last_oft = 0
                
            if len(self.prec[job]) == 0:
                ready_time = 0
            else:
                ready_time = max(self.job_WCT_match[j][-1] for j in self.prec[job])
            self.job_WCT_match[job] = WCT(ready_time, start, last_end, end, oft, oftimp, last_oft, 0)
        dense_rew = []

        mksp = max(v[-1].end for v in orders.values() if v)
        for i in range(len(self.ranked_jobs)):
            index = len(self.ranked_jobs) - i - 1
            job = self.ranked_jobs[index]
            start = self.job_WCT_match[job][1]
            if i == 0:
                minstart = start
            elif start < minstart:
                minstart = start 
            wct = self.job_WCT_match[job]
            self.job_WCT_match[job] = WCT(wct[0], wct[1], wct[2], wct[3], wct[4], wct[5], wct[6], minstart)
        
        for job in range(self.num_n):
            wct = self.job_WCT_match[job]
            ready_time, start, last_end, end, oft, oftimp, last_oft, minstart = wct[0], wct[1], wct[2], wct[3], wct[4], wct[5], wct[6], wct[7]

            dense_rew.append(-end) # negdenserew
        
        dense_rew = np.array(dense_rew)
        return dense_rew, mksp

    def schedule(self):
        """ Schedule computation dag onto worker agents
        inputs:
        succ - DAG of tasks {a: (b, c)} where b, and c follow a
        agents - set of agents that can perform work
        compcost - function :: job, agent -> runtime
        commcost - function :: j1, j2, a1, a2 -> communication time
        """
        self.energy = 0.0
        self.reliability = 1.0
        agents = range(self.num_p)
        # rank = partial(self.ranku, agents=agents)
        if not self.eval:
            jobs = set(self.succ.keys()) | set(x for xx in self.succ.values() for x in xx)
            #self.ranku_list = self.ranku_list = np.zeros([self.num_n, 2])
            # re_calculate ranku ifafeaaf
            jobs = sorted(jobs, key=self.ranku)
            self.ranked_jobs = list(reversed(jobs))

        orders = {agent: [] for agent in agents}
        self.job_WCT_match = {job: [] for job in self.ranked_jobs}
        jobson = dict()
        for job in self.ranked_jobs:
            self.allocate(job, orders, jobson)
        return orders, jobson

    def get_makespan(self):
        self.machine_available_mask = np.zeros((self.num_n, self.num_p), dtype=bool)
        self.machine_chosen_mask = np.zeros((self.num_n, self.num_p), dtype=bool)
        
        orders, jobson = self.schedule()
        dense_rew, mksp = self.makespan(orders=orders)
        for job in range(self.num_n):
            dense_rew[job] = dense_rew[job] - self.baseline_dense_rew[job]
        
        return orders, dense_rew, mksp

    def get_CFT(self):
        orders, jobson = self.schedule()
        FT = [v[-1].end for v in orders.values() if v]
        return 0, max(FT)
    
    def get_speedup_base(self):
        speedup_base = 0
        agent = 3
        for n in self.G_T.nodes():
            speedup_base += self.compcost(n, agent)
        return speedup_base
            
    def draw_gantt(self, orders, savepath = None, job_labels=True, cmap='tab20', figsize=(10, 5)):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.get_cmap(cmap, self.num_n)

        for machine in range(self.num_p):
            agent = machine
            for event in orders[machine]:
                job = event[0]
                start = self.job_WCT_match[job][1]
                finish = self.job_WCT_match[job][3]
                ax.barh(agent, finish - start, left=start, height=0.6,
                        color=colors(job), edgecolor='black')
                if job_labels:
                    ax.text((start + finish)/2, agent, f'J{job}',
                            va='center', ha='center', color='white', fontsize=8)
        ax.set_yticks(range(self.num_p))
        ax.set_yticklabels([f'M{m}' for m in range(self.num_p)])
        ax.set_xlabel("Time")
        ax.set_ylabel("Machine")
        ax.set_title("Gantt Chart (Job Scheduling)")
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.invert_yaxis()  
        plt.tight_layout()
        #plt.show()
        if savepath:
            plt.savefig(savepath, bbox_inches='tight', dpi=300)
        plt.close()

