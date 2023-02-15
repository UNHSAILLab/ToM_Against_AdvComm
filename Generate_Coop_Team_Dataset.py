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
from models.adversarial3 import AdversarialModel
from trainers.multiagent_ppo2 import MultiPPOTrainer
from trainers.random_heuristic import RandomHeuristicTrainer

#from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing

from trainers.hom_multi_action_dist import TorchHomogeneousMultiActionDistribution

import imageio
import string
import random

from copy import deepcopy

import pickle







def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d



def run_trial(trainer_class=MultiPPOTrainer, checkpoint_path=None, trial=0, cfg_update={}, render=False,stdscalar=1.0):
    #dataset = []
    
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
        
        
        
        
        images = []
        
        
        results = []
        
        defmode = 1
        
        
        
        
        
        for i in range(cfg['env_config']['max_episode_len']):

            mod_obs = obs.copy()
            mod_obs['agents'] = list(mod_obs['agents'])
            mod_obs['agents'][5] = mod_obs['agents'][0]
            mod_obs['agents'] = tuple(mod_obs['agents'])
            
            
            #compute_action() uses a stochastic PPO policy
            #compute_action2() uses a deterministic PPO policy
            
            #to extract data to build a bayesian belief, we need data to build the prior.
            #However, we need stochastic actions to have more of a variety of states otherwise
            #it will be rigid with noise
            
            #therefore: we extract (obs?shared_features?,action?)
            #or (use the prilimiary concept of all agents except agent0 are trustworthy, obs)
            #(T/NT, obs)

                
            actions = trainer.compute_action2(obs) #record the deterministic outputs
            
            model_params =  trainer.get_policy().model
            shared_f =model_params.shared_feature.numpy()
            efm =model_params.extract_feature_map.numpy()
                            
                    
            
#             if not withadv:
#                 for a in range(0,len(actions)):
#                     if a == 0:
#                         pass
#                     else:
#                         dataset.append(deepcopy(efm[:,:,a]))
    
                    
            
            
            
            obs, reward, done, info = env.step(actions)
            

            if render:
                env.render2().savefig(os.path.join(tmp_path,str(i)+".png"))
                
                

                
            for j, reward in enumerate(list(info['rewards'].values())):
                results.append({
                    'step': i,
                    'agent': j,
                    'trial': trial,
                    'reward': reward,
                    'dataset': deepcopy(efm[:,:,j]),
                    'action': deepcopy(list(actions)[j])
                    
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

def serve_config(checkpoint_path, trials, cfg_change={}, trainer=MultiPPOTrainer,stdscalar=1.0):
    with Pool() as p:
        results = pd.concat(p.starmap(run_trial, [(trainer, checkpoint_path, t, cfg_change,False,stdscalar) for t in range(trials)]))
    return results



def initialize():
    ray.init()
    register_env("coverage", lambda config: CoverageEnv(config))
    #register_env("path_planning", lambda config: PathPlanningEnv(config))
    ModelCatalog.register_custom_model("adversarial", AdversarialModel)
    ModelCatalog.register_custom_action_dist("hom_multi_action", TorchHomogeneousMultiActionDistribution)

def eval_nocomm(env_config_func, prefix):
    
    
    trials = 100
    #checkpoint = "../../../ray_results/MultiPPO_2021-10-11_10-54-46/MultiPPO_coverage_2d1d6_00000/checkpoint_007500"
    checkpoint = "../../../ray_results/MultiPPO_2022-04-12_11-07-33/MultiPPO_coverage_4799f_00000/checkpoint_015000" #readapt adv
    #checkpoint = "../../../ray_results/MultiPPO_2022-04-07_11-22-07/MultiPPO_coverage_7cc4c_00000/checkpoint_011240" #readapt coop
    
    #out_path ="../../../ray_results/MultiPPO_2021-10-11_10-54-46/MultiPPO_coverage_2d1d6_00000/gaussian"
    out_path ="../../../ray_results/MultiPPO_2022-04-12_11-07-33/MultiPPO_coverage_4799f_00000/gaussian" #readapt adv
    #out_path ="../../../ray_results/MultiPPO_2022-04-07_11-22-07/MultiPPO_coverage_7cc4c_00000/gaussian" #readapt coop
    
    
    initialize()
    results = []
    #wo_eval = [True,False]
    #wo_eval = [True]
    wo_eval = [False]
    
    for i in wo_eval:
        cfg_change={'env_config': env_config_func(i)} #communicate = True
        df = serve_config(checkpoint, trials, cfg_change=cfg_change, trainer=MultiPPOTrainer)
        df['comm'] = i
        results.append(df)

    with open(Path(checkpoint).parent/"params.json") as json_file:
        cfg = json.load(json_file)
        if 'evaluation_config' in cfg:
            update_dict(cfg, cfg['evaluation_config'])

    df = pd.concat(results)
    df.attrs = cfg
    if withadv:
        filename = prefix + "-" + path_to_hash(checkpoint) + "_dataset_adv_with_label.pkl"
    else:
        filename = prefix + "-" + path_to_hash(checkpoint) + "_dataset_with_label.pkl"
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
    
    
withadv = False
#withadv = True
    
if __name__ == '__main__':
    #initialize()
    #eval_nocomm_coop()
    
    #eval_nocomm_adv(mode=1)
    eval_nocomm_adv(mode=0)
    
    #serve()
    
    exit()
    

