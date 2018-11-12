import os
import tempfile
import zipfile

import cloudpickle
import gym
from baselines.common import tf_util

from kb_learning.envs import MultiObjectDirectControlEnv, NormalizeActionWrapper
import yaml
import numpy as np

from kb_learning.policy_networks.trpo_policy import MlpPolicy
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
      width: .15
      height: .15
      init: random
    - !ObjectConf
      idx: 1
      shape: square
      width: .15
      height: .15
      init: random

kilobots: !KilobotsConf
    num: 10
    mean: random
    std: .03
'''


def main():
    env_config = yaml.load(env_config_yaml)
    env = MultiObjectDirectControlEnv(configuration=env_config, done_after_steps=200)
    wrapped_env = NormalizeActionWrapper(env)

    obs = env.reset()

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                         # we observe all other agents with (r, sin(a), cos(a), sin(th), cos(th), lin_vel, rot_vel)
                         num_agent_observations=env.num_kilobots - 1, agent_obs_dims=7,
                         # we observe all objects with (r, sin(a), cos(a), sin(th), cos(th))
                         num_object_observations=len(env.objects), object_obs_dims=6,
                         swarm_net_size=[64],
                         obj_net_size=[64],
                         extra_net_size=[64],
                         concat_net_size=[64])
    pi = policy_fn("pi", env.observation_space, env.kilobots[0].action_space)

    act_params = {
        'name':     "pi",
        'ob_space': env.observation_space,
        'ac_space': env.kilobots[0].action_space,
    }

    pi = ActWrapper(pi, act_params)

    sess = tf_util.get_session()
    policy_path = 'policies/nn_based/trpo_ma/2obj/policy.pkl'
    with open(policy_path, "rb") as f:
        model_data, act_params = cloudpickle.load(f)

    with tempfile.TemporaryDirectory() as td:
        arc_path = os.path.join(td, "packed.zip")
        with open(arc_path, "wb") as f:
            f.write(model_data)

        zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
        pi.load_state(os.path.join(td, "model"))

    steps = 0
    ep_reward = .0
    for _ in range(2000):
        env.render()

        ac, vpred = pi.act(obs, False)

        obs, reward, dones, infos = wrapped_env.step(ac)
        # obs[:], reward, dones, infos = env.step(env.action_space.sample())
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
