import numpy as np
import gym
from kb_learning.envs import register_object_env

env_id = register_object_env(weight=.0, num_kilobots=15, object_shape='quad', object_width=.15, object_height=.15,
                             light_type='linear', light_radius=.3)

print(env_id)

env = gym.make(env_id)

for i in range(500):
    env.render()
    env.step(env.action_space.sample())