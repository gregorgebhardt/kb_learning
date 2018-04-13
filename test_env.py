import numpy as np
import gym
from kb_learning.envs import register_object_env, SARSSampler
from kb_learning.kernel import KilobotEnvKernel
from kb_learning.ac_reps.spwgp import SparseWeightedGP

weight = .0
num_kilobots = 15
object_shape = 'quad'
object_width = .15
object_height = .15
light_type = 'linear'
light_radius = .3


env_id = register_object_env(weight=weight, num_kilobots=num_kilobots, object_shape=object_shape,
                             object_width=object_width, object_height=object_height,
                             light_type=light_type, light_radius=light_radius)

print(env_id)

env = gym.make(env_id)

# for i in range(500):
#     env.render()
#     env.step(env.action_space.sample())

sampler = SARSSampler(register_object_env, num_episodes=1, num_steps_per_episode=1, weight=weight,
                      num_kilobots=num_kilobots, object_shape=object_shape,
                      object_width=object_width, object_height=object_height,
                      light_type=light_type, light_radius=light_radius)
kernel = KilobotEnvKernel(30)
policy = SparseWeightedGP(kernel, output_dim=1)

sars = sampler(policy, 1, 1)