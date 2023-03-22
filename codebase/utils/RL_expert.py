''' Added by Prasanth Suresh on 03/18/23 '''

import os
import argparse
import time 
# import sys
# import torch
# from tqdm import tqdm
import gym
import numpy as np
import pickle
path = os.path.dirname (os.path.realpath (__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(path, os.pardir))

# sys.path.append(PACKAGE_PATH)
import shutup; shutup.please()
from stable_baselines3.ppo import PPO
from stable_baselines3.common.env_util import make_vec_env

class Record():
    def __init__(self, env_id, n_envs=4, device='cpu', seed=1024):
        # Parallel environments
        self.env_name = env_id
        self.save_path = f"{PACKAGE_PATH}/saved_model/PPO_{self.env_name}"
        self.seed = seed
        self.device = device
        self.env = make_vec_env(self.env_name, n_envs=n_envs)
        # self.env = gym.make(self.env_name)
        
    def train(self):
        model = PPO("MlpPolicy", self.env, verbose=1, device=self.device)
        model.learn(total_timesteps=250000)
        model.save(self.save_path)
        del model # remove to demonstrate saving and loading
        
    def check_pickled_file(self):
        from pathlib import Path
        file_path = Path(PACKAGE_PATH).parent
        with open(f'{file_path}/assets/InvertedPendulum_expert_paths.pickle', 'rb') as handle:
            print("Pickle file...")
            contents = pickle.load(handle)
            print(contents)
    
    def record(self, num_steps=10000, save_env_id=''):        
        model = PPO.load(self.save_path, device=self.device)
        self.env.seed(int(time.time()))
        d = {}
        obs_list = []
        act_list = []
        obs = self.env.reset()
        dones = [False]
        while not all(dones):   
            action, _states = model.predict(obs)
            obs_list.append(obs.squeeze())
            act_list.append(action[0])
            # self.env.render()
            obs, rewards, dones, info = self.env.step(action)
            
        d = {'obs': np.array(obs_list), 'acs': np.array(act_list)}        
        from pathlib import Path
        file_path = Path(PACKAGE_PATH).parent
        with open(f'{file_path}/assets/InvertedPendulum_expert_paths.pickle', 'wb') as handle:
            print("Pickle dumping...")
            pickle.dump([d], handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO forward reinforcement learning')
    parser.add_argument('--env', type=str, default='InvertedPendulum-v2', help='Provide the env')
    # parser.add_argument('--training_epochs', type=int, default=20, help='Total training epochs')
    args = parser.parse_args()

    env_id = args.env
    # load_env_id = save_env_id = env_id.replace(":", "_")

    ppo = Record(env_id, n_envs=1)
    # ppo.train()
    ppo.record()
    # ppo.check_pickled_file()