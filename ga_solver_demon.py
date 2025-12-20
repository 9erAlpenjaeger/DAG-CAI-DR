from deap import base, creator, tools
import copy
import random
import numpy as np
import networkx as nx
import multiprocessing
from pathos.pools import ProcessPool
from multiprocessing import Process, Queue
import threading
import math
import time
from tensorboardX import SummaryWriter

from scipy.stats import kendalltau

from gymenvs.cloudgym_oneshot import CloudGym_oneshot
from core.machine import MachineConfig
from playground.DAG.utils.xml_reader import XMLReader
from heft import LINEAR_WEIGHTED_GREEDY 

creator.create('FitnessMin', base.Fitness, weights=(1.0,)) 
creator.create('Makespan', float)
creator.create('Reliability', float)
creator.create('Energy', float)
creator.create('Individual', list, fitness = creator.FitnessMin, makespan = creator.Makespan, reliability = creator.Reliability, energy = creator.Energy) 
#creator.create('')

class ga_solver_demon:
    def __init__(self, env, dataset_name, dataset_size, workflow_index, size_of_population = 128, cx_pr = 0.15, mu_pr = 0.3, gen_num = 200):
        self.env = copy.deepcopy(env)

        self.dataset_name =dataset_name
        self.dataset_size =dataset_size
        self.workflow_index=workflow_index
        
        self.nx_G = self.env.cluster.jobs[0].nx_G
        self.task_num = self.nx_G.number_of_nodes()
        self.size_of_population = size_of_population
        self.cx_pr = cx_pr
        self.mu_pr = mu_pr
        self.gen_num = gen_num   
        return
    
    def generate_toolbox_for_problem(self):
        toolbox = base.Toolbox()
        pool = multiprocessing.Pool()
        #pool = ProcessPool()
        toolbox.register('map', pool.map)
        toolbox.register("seq", random.randint, 0, 1000)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.seq, n = self.task_num)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", self.topo_cx)
        toolbox.register("mutate", self.topo_mu)
        toolbox.register("select", tools.selTournament, tournsize=2)
        toolbox.register("clone", copy.deepcopy)
        self.toolbox = toolbox

    def topo_ind_initialization(self, old_ind):
        nx_G_c = self.nx_G.copy(as_view=True)
        index = 0
        for n in nx_G_c.nodes():
            nx_G_c.nodes[n]['indegree'] = 0
            for _ in nx_G_c.predecessors(n):
                nx_G_c.nodes[n]['indegree'] += 1
        ind = copy.deepcopy(old_ind)
        for _ in range(0, nx_G_c.number_of_nodes()):
            zero_indegrees = []
            for n in nx_G_c.nodes():
                if nx_G_c.nodes[n]['indegree'] == 0:
                    zero_indegrees.append(n)
            task = np.random.choice(zero_indegrees)
            ''' task index '''
            ind[index] = task
            index += 1
            nx_G_c.nodes[task]['indegree'] = -1
            for succ in nx_G_c.successors(task):
                nx_G_c.nodes[succ]['indegree'] -= 1
        return ind

    def init_pop(self, pop):
        ind_index = 0
        process, queue = list(), list()
        cnt = 0
        for ind in pop: 
            q = Queue()
            p = Process(target=self.topo_ind_initialization, args=(ind, ind_index, q))
            process.append(p)
            queue.append(q)
            ind_index += 1
        for p in process:
            p.start() 
        
        for p in process:
            p.join()
        for q in queue:
            ret = q.get()
            ind, ind_index = ret[0], ret[1]
            pop[ind_index] = ind             

    def topo_cx(self, ind1, ind2):
        inda = self.toolbox.clone(ind1)
        indb = self.toolbox.clone(ind2)
        
        size = min(len(ind1), len(ind2))
        cxpoint = np.random.randint(1, size-1)
        ind1[0:cxpoint], ind2[0:cxpoint] = ind2[0:cxpoint], ind1[0:cxpoint]
        pointerA = cxpoint
        pointerB = cxpoint
        for i in range(0, size):
            Ta, Tb = inda[i], indb[i]
            if pointerA >= size or pointerB >= size:
                print(ind1, ind1.fitness.values)
                exit()
            if Ta not in ind1[0:cxpoint]:
                ind1[pointerA] = Ta
                pointerA += 1
            if Tb not in ind2[0:cxpoint]:
                ind2[pointerB] = Tb
                pointerB += 1
        return ind1, ind2

    def topo_mu(self, ind):
        size = len(ind)
        mupoint = np.random.randint(1, size-1)
        start = mupoint
        end = mupoint
        task_index = ind[mupoint]
        while start >=0 and ind[start] not in self.nx_G.predecessors(task_index):
            start -= 1
        while end < size and ind[end] not in self.nx_G.successors(task_index):
            end += 1
        if start+1 > end-1:
            print(ind, ind.values.fitness)
            exit()
        if start+1 == end-1:
            newpoint = start+1
        else:
            newpoint = np.random.randint(start+1 , end-1)  
        #ind[mupoint], ind[newpoint] = ind[newpoint], ind[mupoint]
        if newpoint < mupoint:
            ind.pop(mupoint)
            ind.insert(newpoint, task_index)
        elif newpoint > mupoint:
            ind.insert(newpoint, task_index)
            ind.pop(mupoint)
        return ind

    def evaluate(self, ind, ind_index, q, gen):
        LS_topo = ind
        self.env.heft.ranked_jobs = LS_topo
        self.env.heft.eval = True
        self.env.heft.set_alloc_strategy(LINEAR_WEIGHTED_GREEDY)
        
        dense_rew, C_mksp = self.env.heft.get_makespan()
        C_energy, C_reliab = self.env.heft.energy, self.env.heft.reliability
        
        q.put([C_mksp, C_energy, C_reliab, ind_index])

    def eval_pop(self, pop, gen):
        ind_index = 0
        process, queue = list(), list()
        for ind in pop: 
            q = Queue()
            p = Process(target=self.evaluate, args=(ind, ind_index, q, gen))
            process.append(p)
            queue.append(q)
            ind_index += 1
        for p in process:
            p.start() 
        
        for p in process:
            p.join()
        for q in queue:
            ret = q.get()
            makespan, energy, relia, ind_index = ret[0], ret[1], ret[2], ret[3]
            pop[ind_index].fitness.values = 0 * relia - 0 * energy - makespan,  
            pop[ind_index].makespan.values = makespan
            pop[ind_index].energy.values = energy
            pop[ind_index].reliability.values = relia
    
    def solve(self):    
        log = SummaryWriter(log_dir = './tensorGA/test4 %s-%s-%drounds-cxpr%f-mupr%f' %(self.dataset_name, self.dataset_size, self.gen_num, self.cx_pr, self.mu_pr))
        self.generate_toolbox_for_problem()
        ind = self.toolbox.individual()
        pop = self.toolbox.population(n = self.size_of_population)

        for i in range(len(pop)):
            pop[i] = self.topo_ind_initialization(pop[i])

        self.eval_pop(pop, 0)

        for gen in range(self.gen_num):
            selectedTour = self.toolbox.select(pop, self.size_of_population)
            selectedInd = list(map(self.toolbox.clone, selectedTour))
            # 过去存在最优且可行解即保留下来
            for child1,child2 in zip(selectedInd[::2],selectedInd[1::2]):
                if random.random() < self.cx_pr:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values 
                    del child2.fitness.values
            for mutant in selectedInd:
                if random.random() < self.mu_pr:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
                    
            invalid_ind = [ind for ind in selectedInd if not ind.fitness.valid]

            self.eval_pop(invalid_ind, gen)

            pop[:] = selectedInd
                
            a, m, r, e = list(), list(), list(), list()
            for index in range(len(pop)):
                ind = pop[index]
                a.append(ind.fitness.values[0])
                m.append(ind.makespan.values)
                r.append(ind.reliability.values)
                e.append(ind.energy.values)
            
            tag_scalar_dict = { #'best_feasible_fitness '+pen_coe_str:None,
                                   'mean_fit ': np.mean(a),
                                   'min_fit ': np.min(a),
                                   'max_fit ': np.max(a),
                                   'std_fit ': np.std(a),
                                   'mean_makespan ': np.mean(m),
                                   'min_makespan ': np.min(m),
                                   'max_makespan ': np.max(m),
                                   'std_makespan ': np.std(m),

            }                                   
            log.add_scalars(main_tag = 'atag', tag_scalar_dict = tag_scalar_dict, global_step = gen)

            #print(np.min(m), min_cnt, np.std(m), kld_sum)
        demo_actions = np.zeros((self.size_of_population, self.task_num))
        for i, ind in enumerate(pop):
            demo_actions[i][ind] = np.arange(self.task_num)
        np.save('./demonstrations/Pegasus/%s/%s-%d.npy' %(self.dataset_name, 
                                                                         self.dataset_size, 
                                                                         self.workflow_index), 
                demo_actions)
