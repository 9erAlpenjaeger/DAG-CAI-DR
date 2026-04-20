import argparse
import setproctitle
import time
from datetime import datetime
import networkx as nx

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from gymenvs.cloudgym_oneshot_tpch import CloudGym_oneshot_tpch
from datasets.TPCH_dag_data.dag_generator import load_tpch_jobs
from datasets.TPCH_dag_data.dag_graph import DAGraph
from RL_models import list_oneshot

import torch

def parse_args():
    parse = argparse.ArgumentParser(description='initialize DRL parameters for TPC-H DAG scheduling dataset')
    parse.add_argument('--is_dw', type=str)
    parse.add_argument('--attn_type', type=str)
    parse.add_argument('--tpch_size', type=int)
    parse.add_argument('--dag_number', type=int)
    parse.add_argument('--n_steps', type=int)
    parse.add_argument('--GNN_model', type=str)
    parse.add_argument('--gpu', type=int, default=1)
    args = parse.parse_args()
    return args

def get_demo(tpch_size, dag_number):
    demo = np.load('./demonstrations/TPCH/%d-%d.npy' %(tpch_size, 
                                                                                dag_number))
    max_node_num = CloudGym_oneshot_tpch.max_node_num
        # padding operation
    B, n = demo.shape
    padded_demo = np.zeros((B, max_node_num), dtype = demo.dtype)
    padded_demo[:, :n] = demo
    return padded_demo 

def make_env(dag_graph, tpch_size, tid):
    def _init():
        dags, roots, graphs = load_tpch_jobs(num_init_dags=tpch_size, tid=tid)
        merged_dag = nx.DiGraph()
        merged_dag.add_node(0, features=[0.0, 0.0])
        merged_dag.graph["features"] = [0.0, 0.0]

        for dag in dags:
            merged_dag.add_nodes_from(dag.nodes(data=True))
            merged_dag.add_edges_from(dag.edges(data=True))  
        for root in roots:
            merged_dag.add_edge(root, 0)

        env = CloudGym_oneshot_tpch(dag=merged_dag, workflow_index=tid, dag_graph = dag_graph)
        return env
    return _init


def generate_envs(dag_graph, tpch_size, n_steps, GNN, dag_num, dag_num_for_eval, use_demonstrations):
    if tpch_size == 'ALL':
        tpch_sizes = [50, 100, 150]
    else:
        tpch_sizes = [tpch_size]
    
    env_list = list()
    for tpch_size in tpch_sizes:
        env_list.extend([make_env(dag_graph, tpch_size, i) 
                        for i in range(0, dag_num)])
    n_env = SubprocVecEnv(env_list)

    use_env_for_eval = (dag_num_for_eval != 0)
    if use_env_for_eval:
        env_for_eval_list = list()
        for tpch_size in tpch_sizes:
            env_list.extend([make_env(dag_graph, tpch_size, i) 
                            for i in range(0, dag_num)])
        n_env_for_eval = SubprocVecEnv(env_for_eval_list)
    else:
        n_env_for_eval = None

    return n_env, n_env_for_eval, None

        
if __name__ == "__main__":
    #rewtype = 'ihnorm_d_rew_10_1'
    rewtype = 'TPCH-cai'
    #rewtype = 'test'
    setproctitle.setproctitle(rewtype)
    torch.autograd.set_detect_anomaly(True)
    args = parse_args()
    
    if args.is_dw == 'dw':
        is_dw = True
    else:
        is_dw = False
    attn_type = args.attn_type
    
    tpch_size = args.tpch_size
    dag_number = args.dag_number
    n_steps = args.n_steps
    GNN = args.GNN_model
    gpu = args.gpu
    batch_size = 50
    n_epochs = 50
    gamma = 0.99
    c_logits = 0.01
    NormAdv = 1
    
    #workflow_num = 1
    dag_num_for_eval = 0

    use_env_for_eval = (dag_num_for_eval != 0)
    use_demonstrations = False

    tensorboard_log = './tensorlogs/GAM/ONESHOT-SO/TPCH/cai-%s/%s/%s-ev-%s/%s-%s-nsteps%d--%s' %(
                                                                                                         is_dw,
                                                                                                         rewtype,
                                                                                                         attn_type,
                                                                                                         tpch_size, 
                                                                                                         dag_number, 
                                                                                                         GNN,
                                                                                                         n_steps,
                                                                                                         time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))    
    dag_graph = DAGraph(resource_dim=1, feature_dim=2)

    n_env, n_env_for_eval, demo_buffer = generate_envs(dag_graph, tpch_size, n_steps, GNN, dag_number, dag_num_for_eval, use_demonstrations)

    #exit(0)
    policy_kwargs = dict(
        features_extractor_class=list_oneshot.FeatureEmbedding,
        features_extractor_kwargs=dict( in_channels = -1, agg = 'attn_h', gnn = GNN),
        ptrnet_kwargs = {'attn_type': attn_type}
    )

    model = list_oneshot.ReinforceOneshot(
                policy = list_oneshot.CustomActorCritic, 
                env = n_env, 
                use_env_for_eval = use_env_for_eval,
                env_for_eval = n_env_for_eval,
                policy_kwargs = policy_kwargs, 
                n_steps =n_steps, 
                #batch_size = batch_size,
                #n_epochs=n_epochs,
                device = 'cuda:%d' %(gpu), 
                learning_rate = 5e-4, # set small
                c_logits = c_logits,
                is_dw = is_dw,
                tensorboard_log = tensorboard_log,
                reference_points=0,
                use_demonstrations = use_demonstrations,
                demonstrations = demo_buffer
    )
    print(datetime.now(), 'A')
    model.learn(8000, tb_log_name="ex1")    
    print(datetime.now(), 'B')
