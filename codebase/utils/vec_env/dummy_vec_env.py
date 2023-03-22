import contextlib
import numpy as np
from gym import spaces
from . import VecEnv
from .util import copy_obs_dict, dict_to_obs, obs_space_info

class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), self.env.observation_space, self.env.action_space)
        obs_space = self.env.observation_space 
        self.keys, shapes, dtypes = obs_space_info(obs_space)       
        self.buf_obs = { k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys }
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews  = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None

    def step_async(self, actions):
        listify = True
        with contextlib.suppress(TypeError):
            if len(actions) == self.num_envs:
                listify = False
        if not listify:
            self.actions = actions
        else:
            assert (
                self.num_envs == 1
            ), f"actions {actions} is either not a list or has a wrong size - cannot match to {self.num_envs} environments"
            self.actions = [actions]

    def step_wait(self):
        for e in range(self.num_envs):
            action = self.actions[e]
            if isinstance(self.envs[e].action_space, spaces.Discrete):
                action = int(action)

            obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(action)
            if self.buf_dones[e]:
                obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.buf_infos.copy())

    def reset(self):
        for e in range(self.num_envs):
            obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return self._obs_from_buf()

    def close(self):
        return

    def _save_obs(self, e, obs):
        for k in self.keys:
            self.buf_obs[k][e] = obs if k is None else obs[k]

    def _obs_from_buf(self):
        return dict_to_obs(copy_obs_dict(self.buf_obs))
    
    def expert_ac(self):
        expert_ac = np.array(self.env.get_curr_ac()) if hasattr(self.env, 'get_curr_ac') else None
        return expert_ac

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]
        
