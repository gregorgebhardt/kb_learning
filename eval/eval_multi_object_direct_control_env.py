import os
import tempfile
import zipfile

import cloudpickle
import gym
from baselines.common import tf_util

from kb_learning.envs import MultiObjectDirectControlEnv, NormalizeActionWrapper
import yaml
import numpy as np

from kb_learning.policy_networks.trpo_policy import SwarmPolicy
from kb_learning.tools.trpo_tools import ActWrapper

env_config_yaml = '''
!EvalEnv
width: 1.0
height: 1.0
resolution: 600

objects:
    - !ObjectConf
      idx: 0
      shape: square
      width: .05
      height: .05
      init: random

kilobots: !KilobotsConf
    num: 10
    mean: random
    std: .03
'''


def main():
    env_config = yaml.load(env_config_yaml)
    env_config.objects = [env_config.objects[0]] * 10
    env = MultiObjectDirectControlEnv(configuration=env_config, agent_reward=True, swarm_reward=False,
                                      done_after_steps=500,
                                      reward_function='object_cleanup_sparse')
    wrapped_env = NormalizeActionWrapper(env)

    obs = env.reset()

    def policy_fn(name, ob_space, ac_space, env_config, agent_dims, object_dims, swarm_net_size, swarm_net_type,
                  objects_net_size, objects_net_type, extra_net_size, concat_net_size):
        return SwarmPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                           num_agent_observations=env_config.kilobots.num - 1, agent_obs_dims=agent_dims,
                           num_object_observations=len(env_config.objects), object_obs_dims=object_dims,
                           swarm_net_size=swarm_net_size, swarm_net_type=swarm_net_type,
                           objects_net_size=objects_net_size, objects_net_type=objects_net_type,
                           extra_net_size=extra_net_size, concat_net_size=concat_net_size)

    pi = ActWrapper.load('policies/nn_based/trpo_ma/clean_up_sparse/10obj/policy.pkl', policy_fn)

    steps = 0
    ep_reward = .0
    for _ in range(2000):
        env.render()

        ac, vpred = pi.act(obs, False)
        obs, reward, dones, infos = wrapped_env.step(ac)

        # obs, reward, dones, infos = wrapped_env.step(wrapped_env.action_space.sample())
        # obs, reward, dones, infos = wrapped_env.step(None)
        steps += 1
        ep_reward += reward

        # print(reward)

        if dones is True:
            print('steps: {} episode reward: {}'.format(steps, ep_reward/steps))
            steps = 0
            ep_reward = .0
            env.reset()


if __name__ == '__main__':
    main()
