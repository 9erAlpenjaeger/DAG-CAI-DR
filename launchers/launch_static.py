import argparse
import setproctitle
import time
import numpy as np
from datetime import datetime

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from core.machine import MachineConfig
from playground.DAG.utils.xml_reader import XMLReader
from playground.DAG.utils.json_reader import JSONReader
from gymenvs.cloudgym_oneshot import CloudGym_oneshot
from RL_models import list_oneshot


def parse_args():
    parse = argparse.ArgumentParser(description='initialize DRL parameters')
    parse.add_argument('--is_dw', type=str)
    parse.add_argument('--attn_type', type=str)
    parse.add_argument('--param_type', type=str, default="CCR")
    parse.add_argument('--param_value', type=float, default="1")
    parse.add_argument('--dataset_name', type=str, default="SIPHT")
    parse.add_argument('--dataset_size', type=str, default="100")
    parse.add_argument('--n_steps', type=int, default=8)
    parse.add_argument('--GNN_model', type=str, default='GAT')
    parse.add_argument('--gpu', type=int, default=1)
    args = parse.parse_args()
    return args


def get_demo(dataset_name, dataset_size, workflow_index):
    demo = np.load('./demonstrations/Pegasus/%s/%s-%d.npy' %(dataset_name, 
                                                                                dataset_size, 
                                                                                workflow_index))
    max_node_num = CloudGym_oneshot.max_node_num
        # padding operation
    B, n = demo.shape
    padded_demo = np.zeros((B, max_node_num), dtype = demo.dtype)
    padded_demo[:, :n] = demo
    return padded_demo 

###################################################################################################################################################################

def make_env(dataset_name, dataset_size, machine_num, variance, seed, logpath, workflow_index):
    def _init():
        machine_configs = [MachineConfig(i, 1, 1, i, 1 , 1, 1) for i in range(machine_num)]
        weights = [1.0, 0.0, 0.0]
        jobs_xml = '.dataset/Pegasus/Pegasus_data/raw/%s/%s.n.%s.%d.dax' % (dataset_name, dataset_name, dataset_size, workflow_index)
        np.random.seed(seed)
        xml_reader = XMLReader([jobs_xml], machine_num, variance)
        
        jobs_configs = xml_reader.generate(0, 1) 
        env = CloudGym_oneshot(machine_configs, jobs_configs, None, None, logpath, workflow_index, weights)
        return env
    return _init


def generate_envs(dataset_name, dataset_size, machine_num, variance, seed, n_steps, GNN, workflow_num, workflow_num_for_eval, use_demonstrations, logpath):
    if dataset_name == 'ALL':
        dataset_names = ['SIPHT', 'LIGO', 'GENOME']
    else:
        dataset_names = [dataset_name]
    
    env_list = list()
    for dataset_name in dataset_names:
        env_list.extend([make_env(dataset_name, dataset_size, machine_num, variance, seed, logpath, 20 - workflow_index) 
                        for workflow_index in range(0, workflow_num)])
    n_env = SubprocVecEnv(env_list)

    use_env_for_eval = (workflow_num_for_eval != 0)
    if use_env_for_eval:
        env_for_eval_list = list()
        for dataset_name in dataset_names:
            env_for_eval_list.extend([make_env(dataset_name, dataset_size, machine_num, variance, seed, workflow_index) 
                            for workflow_index in range(workflow_num, workflow_num + workflow_num_for_eval)])
        n_env_for_eval = SubprocVecEnv(env_for_eval_list)
    else:
        n_env_for_eval = None

    if use_demonstrations:
        demo_list = list()
        for dataset_name in dataset_names:
            demo_list.extend([get_demo(dataset_name, dataset_size, workflow_index) 
                              for workflow_index in range(workflow_num)])
        demo_action = collect_demo(demo_list)
        demo_buffer = DemonstrationBuffer(demo_action)
    else:
        demo_buffer = None

    return n_env, n_env_for_eval, demo_buffer

###################################################################################################################################################################

if __name__ == "__main__":
    rewtype = 'MLE'
    #rewtype = "test"
    setproctitle.setproctitle(rewtype)
    
    args = parse_args()

    param_type = args.param_type
    param_value = args.param_value
    wf_num=20
    seed=42
    
    dataset_name = args.dataset_name
    dataset_size = args.dataset_size 
    n_steps = args.n_steps
    GNN = args.GNN_model
    attn_type = args.attn_type
    gpu = args.gpu
    if args.is_dw == 'dw':
        is_dw = True
    else:
        is_dw = False
        
    workflow_num = 20
    workflow_num_for_eval = 0

    use_env_for_eval = (workflow_num_for_eval != 0)
    use_demonstrations = False

    use_pegasus = True
    if use_pegasus:
        tensorboard_log = './tensorlogs/%s/%s/%s-ev-%s-nonorm/%s-%s-nsteps%d--%s' %(
                    is_dw,
                    rewtype,
                    attn_type,
                    dataset_name, 
                    dataset_size, 
                    GNN,
                    n_steps,
                    time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))    

        machine_num = CloudGym_oneshot.machine_num
        variance = 10
        
        n_env, n_env_for_eval, demo_buffer = generate_envs(dataset_name,
                                                        dataset_size, 
                                                        machine_num,
                                                        variance,
                                                        seed,
                                                        n_steps, 
                                                        GNN, 
                                                        workflow_num, 
                                                        workflow_num_for_eval, 
                                                        use_demonstrations,
                                                        tensorboard_log
                                                        )
    else:
        exit(0)
    
    reference_points = list()
    
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
                #device= 'cpu',
                device = 'cuda:%d' %(gpu), 
                learning_rate = 5e-4, # set small
                c_logits = 0.01,
                is_dw = is_dw,
                tensorboard_log = tensorboard_log,
                reference_points=reference_points,
                use_demonstrations = use_demonstrations,
                demonstrations = demo_buffer
    )
    model.learn(100, tb_log_name="run_0")    