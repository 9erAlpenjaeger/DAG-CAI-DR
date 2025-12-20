import argparse
import setproctitle
import time
from datetime import datetime
import networkx as nx

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from gymenvs.cloudgym_oneshot_jssp import CloudGym_oneshot_jssp
from datasets.JSSP_data.read_data import readdata
from RL_models import list_oneshot

import torch

def parse_args():
    parse = argparse.ArgumentParser(description='initialize DRL parameters for TPC-H DAG scheduling dataset')
    parse.add_argument('--j_size', type=int)
    parse.add_argument('--m_size', type=int)
    parse.add_argument('--dag_number', type=int)
    parse.add_argument('--seed', type=int)
    parse.add_argument('--n_steps', type=int)
    parse.add_argument('--GNN_model', type=str)
    #parse.add_argument('--is_dw', type=str)
    parse.add_argument('--attn_type', type=str)
    parse.add_argument('--gpu', type=int, default=1)
    args = parse.parse_args()
    return args
        
if __name__ == "__main__":
    #rewtype = 'ihnorm_d_rew_10_1'
    rewtype = 'MLE'
    #rewtype = "test"
    setproctitle.setproctitle(rewtype)
    torch.autograd.set_detect_anomaly(True)
    args = parse_args()
    
    j_size = args.j_size
    m_size = args.m_size
    dag_number = args.dag_number
    seed = args.seed
    n_steps = args.n_steps
    GNN = args.GNN_model
    attn_type = args.attn_type
    gpu = args.gpu
    is_dw = False
        
    batch_size = 50
    n_epochs = 50
    gamma = 0.99
    c_logits = 0.01
    NormAdv = 1
    
    agg =  'attn_h'

    tensorboard_log = './tensorlogs/ONESHOT-SO/JSSP_new/%s/%s/-%d-%d/%s' %(
                                                                                                         is_dw,
                                                                                                         attn_type,
                                                                                                         j_size, 
                                                                                                         m_size,
                                                                                                         time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))    
    def make_env(jssp_instance, workflow_inex):
        def _init():
            env = CloudGym_oneshot_jssp(jssp_instance=jssp_instance, workflow_index=0)
            return env
        return _init
    jssp_instances = readdata(j_size, m_size, dag_number, seed)
    n_env = SubprocVecEnv([make_env(jssp_instance, workflow_index) \
                           for jssp_instance, workflow_index \
                            in zip(jssp_instances, range(dag_number))])

    policy_kwargs = dict(
        features_extractor_class=list_oneshot.FeatureEmbedding,
        features_extractor_kwargs=dict( in_channels = -1, agg = agg, gnn = GNN),
        ptrnet_kwargs = {'attn_type': attn_type}
    )

    model = list_oneshot.ReinforceOneshot(
                policy=list_oneshot.CustomActorCritic, 
                env=n_env, 
                use_env_for_eval = False,
                env_for_eval=None,
                policy_kwargs = policy_kwargs, 
                n_steps =n_steps, 
                #device = 'cpu',
                device = 'cuda:%d' %(gpu),
                learning_rate = 5e-4, # set small
                c_logits = 0.01,
                is_dw = is_dw,
                tensorboard_log = tensorboard_log,
                reference_points=0,
                use_demonstrations = False,
                demonstrations = None
    )
    print(datetime.now(), 'A')
    model.learn(3000, tb_log_name="ex1")    
    print(datetime.now(), 'B')