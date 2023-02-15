import argparse
import collections.abc
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import ray
import time
import traceback

from pathlib import Path
from ray.rllib.models import ModelCatalog
from ray.tune.logger import NoopLogger
from ray.tune.registry import register_env
from ray.util.multiprocessing import Pool

#modified environment
from environments.coverage3mod import CoverageEnv
#original environment
#from environments.coverage3 import CoverageEnv
#from environments.coverage3mod import CoverageEnvExplAdv as CoverageEnv


from environments.path_planning import PathPlanningEnv
from models.adversarial import AdversarialModel
from trainers.multiagent_ppo2 import MultiPPOTrainer
from trainers.random_heuristic import RandomHeuristicTrainer

#from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing

from trainers.hom_multi_action_dist import TorchHomogeneousMultiActionDistribution

import imageio
import string
import random
import decimal
from copy import deepcopy

import pickle

        
import torch
import torch.nn as nn

def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


# VAE with one stochastic layer z
class VAE(nn.Module):

    def __init__(self, args, d, h_num, scaled=True):
        super(VAE, self).__init__()
        self.dim = 128
        self.Nz = 10
        self.hid_num = h_num
        self.output_type = 'gaussian'
        self.decoder_type = 'gaussian'
        self.scaled_mean = scaled
        self.fc1 = nn.Linear(d, h_num)
        self.fc2_mu = nn.Linear(h_num, self.Nz)
        self.fc2_sigma = nn.Linear(h_num, self.Nz)
        self.fc3 = nn.Linear(self.Nz, h_num)
        if self.decoder_type == 'gaussian':
            self.fc4_mu = nn.Linear(h_num, d)
            self.fc4_sigma = nn.Linear(h_num, d)
        else:
            self.fc4 = nn.Linear(h_num, d)

    def forward(self, x):
        x = x.view(-1, self.dim)
        x = torch.tanh(self.fc1(x))
        mu_z = self.fc2_mu(x)
        log_sigma_z = self.fc2_sigma(x)
        eps = torch.randn_like(mu_z)
        x = mu_z + torch.exp(log_sigma_z) * eps
        x = torch.tanh(self.fc3(x))
        if self.output_type == 'gaussian':
            if self.scaled_mean:
                mu = torch.sigmoid(self.fc4_mu(x))
            else:
                mu = self.fc4_mu(x)
            log_sigma = self.fc4_sigma(x)
            return mu, mu_z, log_sigma, log_sigma_z
        else:
            x = self.fc4(x)
            return x, mu_z, '_', log_sigma_z



def run_trial(trainer_class=MultiPPOTrainer, checkpoint_path=None, trial=0, cfg_update={}, render=False,scaler=1.0):
    try:
        t0 = time.time()
        cfg = {'env_config': {}, 'model': {}}
        if checkpoint_path is not None:
            # We might want to run policies that are not loaded from a checkpoint
            # (e.g. the random policy) and therefore need this to be optional
            with open(Path(checkpoint_path).parent/"params.json") as json_file:
                cfg = json.load(json_file)

        if 'evaluation_config' in cfg:
            # overwrite the environment config with evaluation one if it exists
            cfg = update_dict(cfg, cfg['evaluation_config'])

        cfg = update_dict(cfg, cfg_update)

        trainer = trainer_class(
            env=cfg['env'],
            logger_creator=lambda config: NoopLogger(config, ""),
            config={
                "framework": "torch",
                "seed": trial,
                "num_workers": 0,
                "env_config": cfg['env_config'],
                "model": cfg['model']
            }
        )
        
        
        if checkpoint_path is not None:
            checkpoint_file = Path(checkpoint_path)/('checkpoint-'+os.path.basename(checkpoint_path).split('_')[-1])
            trainer.restore(str(checkpoint_file))
            

        envs = {'coverage': CoverageEnv, 'path_planning': PathPlanningEnv}
        env = envs[cfg['env']](cfg['env_config'])
        env.seed(trial)
        obs = env.reset()
        
        #use the efm of each agent, show it to the vae. If the efm norm value is < 13, then accept input. trust = True
        # else if > 13, then set trust to False
            
            
        action_size = 6

        
#         def update_trust2(obs,obs_actions):
#             for i in env.teams.keys():
#                 for r in range(len(env.teams[i])):
#                     for t in range(action_size):
#                         if i == 0:
#                             pass
#                         else:
#                             if t==0:
#                                 #change idx
#                                 idx1 = 0
#                                 idx2 = t
#                             else:
#                                 idx1 = 1
#                                 idx2 = t-1
#                             #print(obs_actions,r+1,t)
#                             if trust_evaluator_action(obs,r+1,t, list(obs_actions)[t]):
#                                 env.teams[1][r].can_be_trusted[idx1][idx2] = True
#                             else:
#                                 env.teams[1][r].can_be_trusted[idx1][idx2] = False #those that have value of False are not added to  
#             return
        
        results = []
        
#         for r in range(len(env.teams[1])):
#             for j in range(0,6):
#                     env.teams[1][r].trust_history[j] = {0:0,1:0}
#                     env.teams[1][r].trust_history_belief[j] = 1.0 #start with a trust belief of 50/50
        
        
        
        ts_cnt = [0]*3
        ts_cnt2=[0]*3
        
        ts_tpcnt1 = [0]*3
        ts_tpcnt2=[0]*3
        #render = True
        images = []
        #stdscalar = 1.0
        agent_data = []
        #save_gso = True
        #save_gso = False
        #save_model = True
        
        #uniform_weight = 1/15# 1/20 #1/345


        


        
        
        def load_vae():
            loadpath = "vaeb/Auto-Encoding_Variational_Bayes-main/save/vae__dataset__decoder_.pth"
            #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            net = VAE( args = None, d=128, h_num=20)
            #net.to(device)
            optimizer = torch.optim.Adagrad(net.parameters())
            
            #model = TheModelClass(*args, **kwargs)
            net.load_state_dict(torch.load(loadpath))
            #net.eval()
            return net
        
        vae_model = load_vae()
        
        
        for i in range(cfg['env_config']['max_episode_len']):
            
            #set the individual to always trust the information of the cooperative team
            env.teams[0][0].consensus = [[True],[True]*5]
            
            actions = []
            
            all_obs = []
            

                
            trainer.compute_action2(obs) #record the deterministic outputs
            
            model_params =  trainer.get_policy().model
            shared_f =model_params.shared_feature.numpy()
            efm =model_params.extract_feature_map.numpy()
            
            for r in range(0,6):
                #vae_model.forward(efm[:,:,r]) is forwarding agent r through the vae
                #then we should update every agent other than r to trust/not trust if mu_squared <13
                mu, mu_z, log_sigma, log_sigma_z = vae_model.forward(torch.Tensor(efm[:,:,r]))
                mu_squared = torch.norm(mu - torch.Tensor(efm[:,:,r]),dim=1)
                if mu_squared < 13:
                    for j in range(0,5):    
                        if r == 0:
                            env.teams[1][j].consensus[0][r] = True
                        else:
                            env.teams[1][j].consensus[1][r-1] = True
                else:
                    for j in range(0,5):    
                        if r == 0:
                            env.teams[1][j].consensus[0][r] = False
                        else:
                            env.teams[1][j].consensus[1][r-1] = False
            

            ### coop agents do not know how to behave in the presence of self-interested communication
           
            
            for j in range(0,1): #append self-interested agent
                agent_gso = env.compute_gso2(agent_id=j)
                agent_obs = env.ustate.copy()
                agent_obs['gso'] = agent_gso
                
                
                #compute the action given the specific gso
                agent_action = list(trainer.compute_action2(agent_obs))[j]
                
                
                actions.append(agent_action)
                all_obs.append(agent_obs)
                
            
            for j in range(1,4): #only the first 3 coop agents are part of this env
                env.false_belief_coop = False
                env.true_belief_adv = False
                
                agent_gso = env.compute_gso2(agent_id=j)
                agent_obs = env.ustate.copy()
                agent_obs['gso'] = agent_gso
                
                
                #compute the action given the specific gso
                agent_action = list(trainer.compute_action2(agent_obs))[j]
                
                
                actions.append(agent_action)
                all_obs.append(agent_obs)
                
                if env.false_belief_coop ==True:
                    ts_cnt[j-1]+=1
                else:
                    ts_tpcnt1[j-1]+=1
                    
                if env.true_belief_adv==True:
                    ts_cnt2[j-1]+=1
                else:
                    ts_tpcnt2[j-1]+=1
                    

    
            for j in range(4,6):
                agent_gso = env.compute_gso2(agent_id=j)
                agent_obs = env.ustate.copy()
                agent_obs['gso'] = agent_gso
                
                
                #compute the action given the specific gso
                agent_action = list(trainer.compute_action2(agent_obs))[j]
                
                
                actions.append(agent_action)
                all_obs.append(agent_obs)
                
                
                
                
                
                
           
                    
            actions = tuple(actions)
            
            obs, reward, done, info = env.step(actions)

            
            
            for j, reward in enumerate(list(info['rewards'].values())):
                results.append({
                    'step': i,
                    'agent': j,
                    'trial': trial,
                    'reward': reward
                    ,'ts_coop_false_trust_coop': deepcopy(ts_cnt)
                    ,'ts_coop_true_trust_adv': deepcopy(ts_cnt2)
                    ,'ts_coop_true_trust_coop': deepcopy(ts_tpcnt1)
                    ,'ts_coop_false_trust_adv': deepcopy(ts_tpcnt2)
                    ,'consensus': -1 if j==0 else deepcopy(env.teams[1][j-1].consensus)
                })
                
  

        print("Done", time.time() - t0)
        
            
    except Exception as e:
        print(e, traceback.format_exc())
        raise
    df = pd.DataFrame(results)
    return df

def path_to_hash(path):
    path_split = path.split('/')
    checkpoint_number_string = path_split[-1].split('_')[-1]
    path_hash = path_split[-2].split('_')[-2]
    return path_hash + '-' + checkpoint_number_string

def serve_config(checkpoint_path, trials, cfg_change={}, trainer=MultiPPOTrainer,scaler=1.0):
    with Pool() as p:
        results = pd.concat(p.starmap(run_trial, [(trainer, checkpoint_path, t, cfg_change,False,scaler) for t in range(trials)]))
    return results



def initialize():
    ray.init()
    register_env("coverage", lambda config: CoverageEnv(config))
    #register_env("path_planning", lambda config: PathPlanningEnv(config))
    ModelCatalog.register_custom_model("adversarial", AdversarialModel)
    ModelCatalog.register_custom_action_dist("hom_multi_action", TorchHomogeneousMultiActionDistribution)

def eval_nocomm(env_config_func, prefix):
    trials = 100
    checkpoint = "../../../ray_results/MultiPPO_2021-10-11_10-54-46/MultiPPO_coverage_2d1d6_00000/checkpoint_007500"
    out_path ="vaeb/vae_out"
    initialize()
    results = []
    #w_eval = [True,False]
    wo_eval = [True]
    #j = 1.0
    
    for i in wo_eval:
        cfg_change={'env_config': env_config_func(i)} #communicate = True
        df = serve_config(checkpoint, trials, cfg_change=cfg_change, trainer=MultiPPOTrainer,scaler=1.0)
        df['comm'] = i
        results.append(df)

    with open(Path(checkpoint).parent/"params.json") as json_file:
        cfg = json.load(json_file)
        if 'evaluation_config' in cfg:
            update_dict(cfg, cfg['evaluation_config'])

    df = pd.concat(results)
    df.attrs = cfg
    filename = prefix + "-" + path_to_hash(checkpoint) + "_vae.pkl"
    df.to_pickle(Path(out_path)/filename)
    
    

    


def eval_nocomm_adv(mode=0):
    # all cooperative agents can still communicate, but adversarial communication is switched
    
    if mode==0:
        eval_nocomm(lambda comm: {
            'disabled_teams_comms': [not comm, False], # en/disable comms for adv and always enabled for coop
            'disabled_teams_step': [False, False] # both teams operating
        }, "eval_adv")

    
 

def serve():
    checkpoint = "../../../ray_results/MultiPPO_2021-10-11_10-54-46/MultiPPO_coverage_2d1d6_00000/checkpoint_007500"
    initialize()
    run_trial(checkpoint_path=checkpoint, trial=0, render=False)
    
if __name__ == '__main__':
    ##initialize()
    #eval_nocomm_coop()
    
    #eval_nocomm_adv(mode=1)
    eval_nocomm_adv(mode=0)
    
    #serve()
    
    exit()
    

