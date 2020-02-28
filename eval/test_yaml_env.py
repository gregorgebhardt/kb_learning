import random

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

from kb_learning.envs import NormalizeActionWrapper
from kb_learning.envs._multi_object_env import MultiObjectDirectControlEnv
from kb_learning.policy_networks.swarm_policy import SwarmPolicyNetwork
from kb_learning.tools.trpo_tools import ActWrapper, traj_segment_generator_ma

env_config_yaml = '''
!EvalEnv
width: 1.
height: 1.
resolution: 608

objects:
    - !ObjectConf
      idx: 0
      shape: square
      width: .08
      height: .08
      init: random
      
# objects:
#     - !ObjectConf
#       idx: 0
#       shape: c_shape
#       width: .1
#       height: .1
#       init: random
#     - !ObjectConf
#       idx: 0
#       shape: t_shape
#       width: .1
#       height: .1
#       init: random

# objects:
#     - !ObjectConf
#       idx: 0
#       shape: l_shape
#       width: .1
#       height: .1
#       init: random
#     - !ObjectConf
#       idx: 0
#       shape: l_shape
#       width: .1
#       height: .1
#       init: random

kilobots: !KilobotsConf
    num: 10
    mean: random
    std: .03
'''


def main():
    env_config = yaml.load(env_config_yaml)
    env_config.objects = [env_config.objects[0]] * 8
    env = MultiObjectDirectControlEnv(configuration=env_config, agent_reward=True, swarm_reward=True,
                                      agent_type='SimpleVelocityControlKilobot',
                                      done_after_steps=512,
                                      reward_function='object_clustering_amp')
    env.video_path = 'video_out'
    # env.render_mode = 'array'
    wrapped_env = NormalizeActionWrapper(env)

    for t in range(500):
        wrapped_env.render()
        wrapped_env.step(wrapped_env.action_space.sample())


if __name__ == '__main__':
    main()
    # eval_beta_values()
