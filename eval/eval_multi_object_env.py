import cloudpickle
import gym
from baselines.common import tf_util
from kb_learning.envs import NormalizeActionWrapper
import yaml
import numpy as np
from kb_learning.envs._multi_object_env import MultiObjectTargetAreaEnv, MultiObjectTargetPoseEnv
from kb_learning.policy_networks.mlp_policy import MlpPolicyNetwork
from kb_learning.policy_networks.swarm_policy import SwarmPolicyNetwork
import tensorflow as tf

from kb_learning.tools.ppo_tools import traj_segment_generator, add_vtarg_and_adv
from kb_learning.tools.trpo_tools import ActWrapper

import matplotlib.pyplot as plt

env_config_yaml = ''' #!yaml
!EvalEnv
width: 1.2
height: 1.2
resolution: 600

objects:
    - !ObjectConf
      idx: 0
      shape: square
      width: .15
      height: .15
      init: random
      target: !TargetPose
        pose: random
        accuracy: [.02, .02, .05]
        periodic: True
        frequency: 4
    # - !ObjectConf
    #   idx: 1
    #   shape: square
    #   width: .15
    #   height: .15
    #   init: random
    # - !ObjectConf
    #   idx: 2
    #   shape: square
    #   width: .15
    #   height: .15
    #   init: random
    # - !ObjectConf
    #   idx: 3
    #   shape: square
    #   width: .15
    #   height: .15
    #   init: random

light: !LightConf
    type: momentum
    radius: .2
    init: object

# light: !LightConf
#     type: composite
#     init: random
#     components:
#         - !LightConf
#           type: momentum
#           radius: .1
#           init: object
#         - !LightConf
#           type: momentum
#           radius: .1
#           init: random

kilobots: !KilobotsConf
    num: 10
    mean: light
    std: .03
'''


def main():
    env_config = yaml.load(env_config_yaml)
    env = MultiObjectTargetPoseEnv(configuration=env_config, done_after_steps=64)
    # env = MultiObjectTargetAreaEnv(configuration=env_config, done_after_steps=256)
    wrapped_env = NormalizeActionWrapper(env)

    obs = env.reset()

    def policy_fn(name, **policy_params):
        # return SwarmPolicyNetwork(name=name, **policy_params)
        return MlpPolicyNetwork(name=name, **policy_params)

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

    pi = ActWrapper.load('policies/nn_based/trpo/1obj/policy.pkl', policy_fn)

    seg_gen = traj_segment_generator(pi, wrapped_env, 256, True, render=True)

    steps = 0
    ep_reward = .0
    for _ in range(4):
        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma=0.99, lam=.95)
        # seg['rew'] /= seg['rew'].std()
        a_target = seg['adv']
        a_target = (a_target - a_target.mean()) / a_target.std()

        print('steps: {} episode return: {}'.format(seg['ep_lens'], seg['ep_rets']))
        plt.plot(seg['rew'], label='reward')
        plt.plot(seg['vpred'], label='v-pred')
        plt.plot(a_target, label='a_target')
        plt.plot(seg['tdlamret'], label='td_lam_ret')
        plt.legend()
        plt.show()

        # m_pos = env._screen.get_mouse_position()
        # print(m_pos)
        # l_pos = env.get_light().get_state()
        # obs[:], reward, dones, infos = env.step(np.array(m_pos) - l_pos)

        # obs[:], reward, dones, infos = wrapped_env.step(wrapped_env.action_space.sample())


if __name__ == '__main__':
    main()
