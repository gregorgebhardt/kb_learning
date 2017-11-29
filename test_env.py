import gym
import kb_learning.envs

import numpy as np

env = gym.make('Kilobots-QuadPushingEnv_w{:03}_kb{}-v0'.format(int(.0 * 100), 15))

env.render()
for i in range(120):
    env.step(env.action_space.sample())
