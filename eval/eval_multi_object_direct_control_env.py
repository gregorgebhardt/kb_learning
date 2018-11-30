import os
import random
import tempfile
import zipfile

import cloudpickle
import gym
from baselines.common import tf_util
import matplotlib.pyplot as plt

from kb_learning.envs import MultiObjectTargetAreaDirectControlEnv, NormalizeActionWrapper
import yaml
import numpy as np

from kb_learning.envs._multi_object_env import MultiObjectDirectControlEnv
from kb_learning.policy_networks.mlp_policy import MlpPolicyNetwork
from kb_learning.policy_networks.swarm_policy import SwarmPolicyNetwork
from kb_learning.tools.trpo_tools import ActWrapper, traj_segment_generator_ma, add_vtarg_and_adv_ma

env_config_yaml = '''
!EvalEnv
width: 1.0
height: 1.0
resolution: 608

# objects:
#     - !ObjectConf
#       idx: 0
#       shape: square
#       width: .1
#       height: .1
#       init: random
      
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

objects:
    - !ObjectConf
      idx: 0
      shape: l_shape
      width: .1
      height: .1
      init: random
    - !ObjectConf
      idx: 0
      shape: l_shape
      width: .1
      height: .1
      init: random

kilobots: !KilobotsConf
    num: 10
    mean: random
    std: .03
'''


def main():
    env_config = yaml.load(env_config_yaml)
    # env_config.objects = [env_config.objects[0]] * 4
    env = MultiObjectDirectControlEnv(configuration=env_config, agent_reward=True, swarm_reward=True,
                                      agent_type='SimpleVelocityControlKilobot',
                                      done_after_steps=512,
                                      reward_function='moving_objects')
    env.video_path = 'video_out'
    # env.render_mode = 'array'
    wrapped_env = NormalizeActionWrapper(env)

    def policy_fn(name, **policy_params):
        return SwarmPolicyNetwork(name=name, **policy_params)
        # return MlpPolicyNetwork(name=name, **policy_params)

    # policy_params = dict(ob_space=env.observation_space,
    #                      ac_space=wrapped_env.action_space,
    #                      env_config=env_config,
    #                      # we observe all other agents with (r, sin(a), cos(a), sin(th), cos(th), lin_vel, rot_vel)
    #                      agent_dims=env.kilobots_observation_space.shape[0] // env.num_kilobots,
    #                      num_agent_observations=env_config.kilobots.num,
    #                      # we observe all objects with (r, sin(a), cos(a), sin(th), cos(th), valid_indicator)
    #                      object_dims=env.object_observation_space.shape[0],
    #                      num_object_observations=len(env_config.objects),
    #                      swarm_net_size=(64,),
    #                      swarm_net_type='softmax',
    #                      objects_net_size=(64,),
    #                      objects_net_type='softmax',
    #                      extra_net_size=(64,),
    #                      concat_net_size=(64,))
    # pi = policy_fn('pi', **policy_params)
    # tf_util.initialize()

    pi = ActWrapper.load('policies/nn_based/trpo_ma_LL/assembly/mean_max/policy.pkl', policy_fn)

    np.random.seed(0)
    random.seed(0)
    seg_gen = traj_segment_generator_ma(pi, wrapped_env, 512, False, render=True)

    for _ in range(4):
        seg = seg_gen.__next__()
        # add_vtarg_and_adv_ma(seg, gamma=0.99, lam=.95)
        # # seg['rew'] /= seg['rew'].std()
        # a_target = seg['adv']
        # a_target = (a_target - a_target.mean()) / a_target.std()
        #
        # print('steps: {} episode return: {}'.format(seg['ep_lens'], seg['ep_rets']))
        # plt.plot(seg['rew'], label='reward')
        # plt.plot(seg['vpred'], label='v-pred')
        # plt.plot(a_target, label='a_target')
        # plt.plot(seg['tdlamret'], label='td_lam_ret')
        # plt.legend()
        # plt.show()


if __name__ == '__main__':
    main()
