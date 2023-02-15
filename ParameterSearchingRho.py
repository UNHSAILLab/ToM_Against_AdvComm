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

def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d



def run_trial(trainer_class=MultiPPOTrainer, checkpoint_path=None, trial=0, cfg_update={}, render=False,stdscalar=1.0):
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
        
        if render:
            render_path =  "../../../ray_results/MultiPPO_2021-10-11_10-54-46/MultiPPO_coverage_2d1d6_00000/render"
            tmp_path = os.path.join(render_path,"tmp")
            try:
                os.mkdir(tmp_path)
                print("/tmp created")
            except:
                os.rmdir(tmp_path)
                os.mkdir(tmp_path)

        def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
            return ''.join(random.choice(chars) for _ in range(size))
        
        
            
        def trust_evaluator_action(obs,evaluator_agent_idx,evaluated_agent_idx,observed_action):
            #with compute_actions2 which is set to be deterministic; we do not have to worry about stochastic actions!
            
            if evaluator_agent_idx == evaluated_agent_idx:
                return True
            obs_copy = obs.copy()
            
            obs_copy['agents'] = list(obs_copy['agents'])
            obs_copy['agents'][evaluator_agent_idx] = obs_copy['agents'][evaluated_agent_idx]
            obs_copy['agents'] = tuple(obs_copy['agents'])
            evaluator_action = list(trainer.compute_action2(obs_copy))[evaluator_agent_idx]
            

            if evaluator_action == observed_action:
                return True
            else:
                return False
            
            
        action_size = 6

        
        def update_trust2(obs,obs_actions):
            for i in env.teams.keys():
                for r in range(len(env.teams[i])):
                    for t in range(action_size):
                        if i == 0:
                            pass
                        else:
                            # t is evaulated
                            # env.teams[i][r] is evaluator
                            if t==0:
                                #change idx
                                idx1 = 0
                                idx2 = t
                            else:
                                idx1 = 1
                                idx2 = t-1
                            #print(obs_actions,r+1,t)
                            if trust_evaluator_action(obs,r+1,t, list(obs_actions)[t]):
                                env.teams[1][r].can_be_trusted[idx1][idx2] = True
                            else:
                                env.teams[1][r].can_be_trusted[idx1][idx2] = False #those that have value of False are not added to  
            return
        
        
        def ofd(mean,std,val,stdscalar=1.0):
            #out of distribution detection
            if val < mean:
                if mean - std*stdscalar <= val:
                    return False
                else:
                    return True
            elif val > mean:
                if mean+std*stdscalar >= val:
                    return False
                else:
                    return True
            if val == mean:
                return False
            
            
        
        
        

        results = []
        
        for r in range(len(env.teams[1])):
            for j in range(0,6):
                    env.teams[1][r].trust_history[j] = {0:0,1:0}
        
        
        
        ts_cnt = 0
        ts_cnt2=0
        #render = True
        images = []
        #stdscalar = 1.0
        
        for i in range(cfg['env_config']['max_episode_len']):
            
            #set the individual to always trust the information of the cooperative team
            env.teams[0][0].can_be_trusted = [[True],[True]*5]
            
            actions = []
            
            all_obs = []
            
            #start tallying the beliefs    
            for r in range(len(env.teams[1])):
                for j in range(0,6):
                    if j == 0:
                        belief = env.teams[1][r].can_be_trusted[0][j]
                    else:
                        belief = env.teams[1][r].can_be_trusted[1][j-1]
                        
                    if belief:
                        env.teams[1][r].trust_history[j][1]+=1
                    else:
                        env.teams[1][r].trust_history[j][0]+=1
                        

            ##now determine the consensus for each agent's history
            for r in range(len(env.teams[1])):
                true_dist = []
                false_dist = []
               
            
                for j in range(0,6):
                    true_dist.append( env.teams[1][r].trust_history[j][1] )
                    false_dist.append( env.teams[1][r].trust_history[j][0] ) 
                
                true_dist = np.array(true_dist)
                false_dist = np.array(false_dist)
                true_dist_std = true_dist.std()
                false_dist_std = false_dist.std()
                true_dist_mean = true_dist.mean()
                false_dist_mean = false_dist.mean()
                
                for j in range(0,6):
                    max_belief1 = not ofd(true_dist_mean,true_dist_std, env.teams[1][r].trust_history[j][1],stdscalar=stdscalar)
                    max_belief2 = not ofd(false_dist_mean,false_dist_std, env.teams[1][r].trust_history[j][0],stdscalar=stdscalar)
                    

                    if max_belief1 == True & max_belief2 == True:
                        max_belief_bool = True
                    else:
                        max_belief_bool = False
                    if j==0:
                        env.teams[1][r].consensus[0][0] = max_belief_bool
                    else:
                        env.teams[1][r].consensus[1][j-1] = max_belief_bool

            ### coop agents do not know how to behave in the presence of self-interested communication
            
            env.false_belief_coop = False
            env.true_belief_adv = False
            
            for j in range(0,6):
                agent_gso = env.compute_gso2(agent_id=j)
                agent_obs = env.ustate.copy()
                agent_obs['gso'] = agent_gso
                
                #compute the action given the specific gso
                agent_action = list(trainer.compute_action2(agent_obs))[j]
                actions.append(agent_action)
                all_obs.append(agent_obs)
                
            if env.false_belief_coop:
                ts_cnt+=1
            if env.true_belief_adv:
                ts_cnt2+=1
                    
            actions = tuple(actions)
            
            obs, reward, done, info = env.step(actions)
            
            for k in env.teams.keys():
                for r in range(len(env.teams[k])):
                    for t in range(action_size):
                        if k == 0:
                            pass
                        else:
                            # t is evaulated
                            # env.teams[i][r] is evaluator
                            if t==0:
                                #change idx
                                idx1 = 0
                                idx2 = t
                            else:
                                idx1 = 1
                                idx2 = t-1
                            if trust_evaluator_action(all_obs[t],r+1,t, list(actions)[t]):
                                env.teams[1][r].can_be_trusted[idx1][idx2] = True
                            else:
                                env.teams[1][r].can_be_trusted[idx1][idx2] = False #those that have value of False are not added 

            if render:
                env.render().savefig(os.path.join(tmp_path,str(i)+".png"))
                
            for j, reward in enumerate(list(info['rewards'].values())):
                results.append({
                    'step': i,
                    'agent': j,
                    'trial': trial,
                    'reward': reward
                    ,'ts_coop_false_trust_coop': ts_cnt
                    ,'ts_coop_true_trust_adv': ts_cnt2
                    ,'std_scalar':stdscalar
                })
                
        print("Timesteps that had a false trust belief for coop:",ts_cnt)
        print("Timesteps that had a true trust belief in adv for coop:",ts_cnt2)        

        print("Done", time.time() - t0)
        
        if render:  
            gif_id = id_generator(size=10) 
            assert len(os.listdir(tmp_path)) == 345 #sanity check
            
            with imageio.get_writer(os.path.join(render_path,gif_id+".gif"), mode='I') as writer:
                for num in range(len(os.listdir(tmp_path))):
                    image = imageio.imread(os.path.join(tmp_path,str(num)+".png"))
                    writer.append_data(image)
                writer.close()
                    
            print("gif rendered:{}.gif".format(gif_id))
            
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
    checkpoint = "../../../ray_results/MultiPPO_2021-10-11_10-54-46/MultiPPO_coverage_2d1d6_00000/checkpoint_007500"
    out_path ="../../../ray_results/MultiPPO_2021-10-11_10-54-46/MultiPPO_coverage_2d1d6_00000/r"
    initialize()
    results = []
    #w_eval = [True,False]
    wo_eval = [True]
    
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
    filename = prefix + "-" + path_to_hash(checkpoint) + ".pkl"
    df.to_pickle(Path(out_path)/filename)
    
    
def eval_nocomm2(env_config_func, prefix):
    trials = 100
    checkpoint = "../../../ray_results/MultiPPO_2021-10-11_10-54-46/MultiPPO_coverage_2d1d6_00000/checkpoint_007500"
    out_path ="../../../ray_results/MultiPPO_2021-10-11_10-54-46/MultiPPO_coverage_2d1d6_00000/r"
    initialize()
    results = []
    
    for j in [0.0,0.5,1.0,1.5,2.0,2.5,3.0]:
        for i in [True]:
            cfg_change={'env_config': env_config_func(i)} #communicate = True
            df = serve_config(checkpoint, trials, cfg_change=cfg_change, trainer=MultiPPOTrainer,stdscalar=j)
            df['comm'] = i
            results.append(df)

        with open(Path(checkpoint).parent/"params.json") as json_file:
            cfg = json.load(json_file)
            if 'evaluation_config' in cfg:
                update_dict(cfg, cfg['evaluation_config'])

    df = pd.concat(results)
    df.attrs = cfg
    filename = prefix + "-" + path_to_hash(checkpoint) + "_trustscoring.pkl"
    df.to_pickle(Path(out_path)/filename)

    


def eval_nocomm_adv(mode=0):
    # all cooperative agents can still communicate, but adversarial communication is switched
    
    if mode==0:
        eval_nocomm(lambda comm: {
            'disabled_teams_comms': [not comm, False], # en/disable comms for adv and always enabled for coop
            'disabled_teams_step': [False, False] # both teams operating
        }, "eval_adv")
    
    if mode==1:
        eval_nocomm2(lambda comm: {
            'disabled_teams_comms': [not comm, False], # en/disable comms for adv and always enabled for coop
            'disabled_teams_step': [False, False] # both teams operating
        }, "eval_adv")
    
    
 

def serve():
    checkpoint = "../../../ray_results/MultiPPO_2021-10-11_10-54-46/MultiPPO_coverage_2d1d6_00000/checkpoint_007500"
    initialize()
    run_trial(checkpoint_path=checkpoint, trial=0, render=True)
    
if __name__ == '__main__':
    #initialize()
    #eval_nocomm_coop()
    
    eval_nocomm_adv(mode=1)
    #eval_nocomm_adv(mode=0)
    
    #serve()
    
    exit()
    

