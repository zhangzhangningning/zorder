import gym
import os
from gym import spaces
# import pygame
import numpy as np
import subprocess
import random
import shutil
import pickle
import time


class SelColEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self):  # params
        # table涉及的列传入进来，0表示workload不涉及该列，1表示workload涉及该列
        # 比如（1,1,1）workload涉及表中的三列
        # self.aaa = params['']
        ColShow = [1] * 11
        # workload = [1, 1, 2, 1, 1, 3, 2, 1, 5, 4, 10, 4, 2, 1, 3, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 4, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 3, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1]
        workload = [13, 25, 17, 20, 15, 10]
        self.rand_num = 0
        self.best_reward = -100
        self.best_actions = []
        # self.define_col_num = 6
        # self.parent_path = '/home/ning/zorder/Actions_Rewards/'
        # self.mkfile()
        self.init_reward_selected_cols()
        self.done_col_reward = {}
        self.ColShow = np.array(ColShow)
        self.workload = np.array(workload)
        self.workload_length = len(self.workload)
        # 用来最后计算reward的
        self.length = len(self.ColShow)
        self.SelCol = np.array([0] * self.length)
        self.idx = 0
        self.next_state = np.zeros(self.length)
        # action应该是选择的列，
        self.action_space = spaces.Discrete(2)
        # self.action_space = np.array()
        # self.observation_space = spaces.Box(low = 0, high = 1, shape = (self.length,))

        self.observation_space = spaces.Dict(
            {
                "next_col": spaces.Box(low = 0, high = 1, shape = (self.length,), dtype = int),
                # "workload": spaces.Box(low = 0, high = 100, shape=(self.workload_length,), dtype = int),
            }
        )

    def reset(self, seed=None, options=None):
        # super().reset(seed=seed)
        self.idx = 0
        self.SelCol = np.array([0]*self.length)
        self.next_state = np.zeros(self.length)
        self.next_state[self.idx] = 1
        observation = self._get_obs()
        # observation = self.next_state
        return observation

    def step(self, action):
        if self.idx + 1 == len(self.ColShow):
            done = True
            if (self.SelCol.any() == 0):
                reward = -100
                self.save_col('/home/ning/zorder/New_agg_result/select_cols.txt')
            else:
                # select_cols_num = np.count_nonzero(self.SelCol == 1)
                # while (np.count_nonzero(self.SelCol == 1) < self.define_col_num):
                #     self.rand_num = random.randint(0,100)
                #     self.SelCol[self.rand_num % len(self.ColShow)] = 1
                # while (np.count_nonzero(self.SelCol == 1) > self.define_col_num):
                #     self.rand_num = random.randint(0,100)
                #     self.SelCol[self.rand_num % len(self.ColShow)] = 0
                self.done_col_reward = self.Get_done_reward()
                if self.done_col_reward:
                    self.best_actions,self.best_reward = self.Get_best_reward_action()
                cases = len(self.done_col_reward)
                if cases >= 1:
                    self.save_col('/home/ning/zorder/New_agg_result/select_cols.txt')
                    if str(self.SelCol) in self.done_col_reward.keys():
                        reward = self.done_col_reward[str(self.SelCol)]
                    else:
                        self.execu_predicted_files()
                        reward = self.Get_reward()
                        if str(self.SelCol) in self.done_col_reward.keys():
                            pass
                        else:
                            self.done_col_reward[str(self.SelCol)] = reward
                else:
                    while str(self.SelCol) in self.done_col_reward.keys():
                        random1 = np.random.randint(0,100)
                        random2 = np.random.randint(0,100)
                        self.SelCol[random1 % self.length] = random2 % 2
                    self.save_col('/home/ning/zorder/New_agg_result/select_cols.txt')
                    self.execu_predicted_files()
                    reward = self.Get_reward()
                    if str(self.SelCol) in self.done_col_reward.keys():
                            pass
                    else:
                        self.done_col_reward[str(self.SelCol)] = reward
                self.save_done_reward()      
                reward = str(reward)
                reward = reward.strip('\n')
                reward = float(reward)
                if reward > float(self.best_reward):
                    self.best_reward = reward
                    self.best_actions = self.SelCol.copy()
            self.save_rewards('/home/ning/zorder/New_agg_result/reward.txt',reward)
        else:
            if self.best_reward > -100:
                self.rand_num = random.randint(0,10)
                if self.rand_num > 3:
                    action = self.best_actions[self.idx]
            done = False
            if action == 0:
                self.SelCol[self.idx] = 0
            else:
                self.SelCol[self.idx] = 1
            self.idx += 1
            self.next_state = self.SelCol.copy()
            self.next_state[self.idx] = 1
            reward = -110
        reward = float(reward)
        if reward == self.best_reward:
            # print(reward)
            # print(self.best_actions)
            reward = -reward
        return self._get_obs(), reward, done ,{}
        # return self.next_state, reward, done ,{} 

    def _get_obs(self):
        return {"next_col":self.next_state}
        # return {"next_col":self.next_state,"workload":self.workload}
        # return self.next_state

    def save_col(self,filename):
        # self.SelCol = [0,1,1,0]
        col_string = np.array2string(self.SelCol,separator= ',')
        with open (filename,'a') as f:
            f.write(col_string)
        with open (filename, 'a') as f:
            f.write('\n')
    
    # def get_reward(self):
    #     Sel_Col = str(self.SelCol)
    #     return self.rewards[Sel_Col]
    
    def save_rewards(self,filename,reward):
        reward = str(reward)
        reward = reward.strip('\n')
        with open(filename,'a') as f:
            f.write(reward)
            f.write(" ")
            f.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            f.write('\n')


    # def execu_spark(self):
    #     # cmd = "cd / && spark-submit /home/ning/my_spark/share/CoWorkAlg/execu_query.py"
    #     subprocess.run(['docker', 'exec','-it', 'my_spark-spark-1', '/opt/bitnami/python/bin/python','/opt/share/CoWorkAlg/execu_query.py'])
    def Get_reward(self):
        with open('/home/ning/zorder/New_agg_result/done_reward.txt','r') as f:
            lines = f.readlines()
        reward = lines[-1]
        return reward
    def execu_predicted_files(self):
        # os.system('/bin/bash  /home/ning/zorderlearn/py38/bin/activate') 
        os.chdir('/home/ning/zorderlearn/ValidateScanFileNumbers/')
        # os.system("python /home/ning/zorderlearn/ValidateScanFileNumbers/dim_reduction.py --dataset=tpch --glob='tpch_sample-Dim4-2.3*' --num-queries=2000 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --input-encoding=embed --output-encoding=embed")
        os.system("python /home/ning/zorderlearn/ValidateScanFileNumbers/dim_reduction.py --dataset=dmv --glob='Dmv_sample-Dim4-3.2*' --num-queries=2000 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --input-encoding=embed --output-encoding=embed")

    def Get_Single_min_ratio(self):
        with open('/home/ning/zorder/Actions_Rewards/single_min_select_ratio.txt','r') as f:
            lines = f.readlines()
        reward = lines[-1]
        return reward
    def mkfile(self):
        open(self.parent_path + '/' + 'rewards.txt','w')
        open(self.parent_path + '/' + 'Select_cols.txt','w')
        open(self.parent_path + '/' + 'workload_erows_ratio.txt','w')
        self.dir_path = self.parent_path + str(self.define_col_num)
        folder = os.path.exists(self.dir_path)
        if folder:
            shutil.rmtree(self.dir_path)
        os.mkdir(self.dir_path)
        # open(self.dir_path + '/' + 'rewards.txt','w')
        # open(self.dir_path + '/' + 'Select_cols.txt','w')
        # open(self.dir_path + '/' + 'workload_erows_ratio.txt','w')
        # return self.dir_path
    def save_done_reward(self):
        with open('/home/ning/zorder/New_agg_result/done_epsiode.pkl','wb') as f:
            pickle.dump(self.done_col_reward,f)
    def Get_done_reward(self):
        with open('/home/ning/zorder/New_agg_result/done_epsiode.pkl','rb') as f:
            self.done_col_reward = pickle.load(f)
            return self.done_col_reward
    def init_reward_selected_cols(self):
        with open('/home/ning/zorder/New_agg_result/reward.txt','w') as f:
            f.write('')
        with open('/home/ning/zorder/New_agg_result/select_cols.txt','w') as f:
            f.write('')
    def Get_best_reward_action(self):
        done_col_reward = list(self.done_col_reward.items())
        best_action = done_col_reward[0][0]
        best_reward = float(str(done_col_reward[0][1]).strip('\n'))
        for i in range(len(done_col_reward)):
            temp_reward = float(str(done_col_reward[i][1]).strip('\n'))
            if temp_reward > best_reward:
                best_reward = temp_reward
                best_action = done_col_reward[i][0]
        best_action = ','.join(str(best_action).split())
        best_action = eval(best_action)
        best_action = np.array(best_action)
        print(type(best_action))
        print(best_reward)
        return best_action,best_reward