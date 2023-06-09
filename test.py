import gym
import zorder
from stable_baselines3 import DQN 
from stable_baselines3 import A2C
env = gym.make('zorder/SelCol-v0')


class QLearning(A2C):
    def __init__(self, policy, env, verbose=0, **kwargs):
        super(QLearning, self).__init__(policy=policy,
                                        env=env,
                                        verbose=verbose,
                                        **kwargs)


model = QLearning('MultiInputPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

