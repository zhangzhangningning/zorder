import gym
import os
from gym import spaces
# import pygame
import numpy as np
import subprocess


class SelColEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):  # params
        # table涉及的列传入进来，0表示workload不涉及该列，1表示workload涉及该列
        # 比如（1,1,1）workload涉及表中的三列
        # self.aaa = params['']
        ColShow = (1,1,1,1)
        workload = (2,2,1,1,2)
        self.ColShow = np.array(ColShow)
        self.workload = np.array(workload)
        # 用来最后计算reward的
        self.length = len(self.ColShow)
        self.SelCol = np.array([0] * self.length)
        self.idx = 0
        self.next_state = np.zeros(self.length)
        # action应该是选择的列，
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict(
            {
                "next_col": spaces.Box(low = 0, high = 1, shape = (self.length,), dtype = int),
                "workload": spaces.Box(low = 0, high = 10, shape=(5,), dtype = int),
            }
        )

    def reset(self, seed=None, options=None):
        # super().reset(seed=seed)
        self.idx = 0
        self.SelCol = np.array([0]*self.length)
        self.next_state = np.zeros(self.length)
        self.next_state[self.idx + 1] = 1
        observation = self._get_obs()
        return observation

    def step(self, action):
        if self.idx + 1 == len(self.ColShow):
            if np.all(self.SelCol == 0):
                done = True
                reward = self.get_reward('/home/ning/my_spark/share/CoWorkAlg/skip_files.txt')
            else:
                # 列选完了，该计算reward了，假设为1
                self.save_col('/home/ning/my_spark/share/CoWorkAlg/ColSelect.txt')
                # self.get_zfile()
                # self.execu_spark()
                done = True
                reward = self.get_reward('/home/ning/my_spark/share/CoWorkAlg/skip_files.txt')
        else:
            done = False
            if action == 0:
                self.SelCol[self.idx] = 0
            else:
                self.SelCol[self.idx] = 1
            self.idx += 1
            self.next_state = np.zeros(self.length)
            self.next_state[self.idx] = 1
            reward = 0
            # observation = self._get_obs()
            # info = {}
        return self._get_obs(), reward, done ,{}

    def _get_obs(self):
        return {"next_col":self.next_state,"workload":self.workload}

    def save_col(self,filename):
        col_string = np.array2string(self.SelCol,separator= ',')
        with open (filename,'a') as f:
            f.write(col_string)
        with open (filename, 'a') as f:
            f.write('\n')
    
    def get_reward(self,filename):
        with open(filename,'r') as f:
            lines = f.readlines()
            last_line = lines[-1]
        reward = int(last_line)
        return reward
    def execu_spark(self):
        # cmd = "cd / && spark-submit /home/ning/my_spark/share/CoWorkAlg/execu_query.py"
        subprocess.run(['docker', 'exec','-it', 'my_spark-spark-1', '/opt/bitnami/python/bin/python','/opt/share/CoWorkAlg/execu_query.py'])
    
    def get_zfile(self):
        os.system('python /home/ning/zorder/GetZorder/GetZFile.py')

