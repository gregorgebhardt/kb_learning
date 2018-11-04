import cloudpickle
import gym
from kb_learning.envs import MultiObjectDirectControlEnv, NormalizeActionWrapper
import yaml
import numpy as np


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
    # with open('policies/nn_based/ppo_mo/objs_2/make_model.pkl', 'rb') as fh:
    #     make_model = cloudpickle.load(fh)
    #
    # model = make_model()
    # model.load('policies/nn_based/ppo_mo/objs_2/model_parameters')

    env_config = yaml.load(env_config_yaml)
    # env = NormalizeActionWrapper(MultiObjectEnv(env_config, done_after_steps=200))
    env = MultiObjectDirectControlEnv(configuration=env_config, done_after_steps=200)

    obs = env.reset()
    # states = model.initial_state

    steps = 0
    ep_reward = .0
    for _ in range(2000):
        env.render()

        # actions, values, states, neglogpacs = model.eval_step(obs, S=states, M=dones)
        # actions = actions.reshape((2,))

        # m_pos = env._screen.get_mouse_position()
        # print(m_pos)
        # kb_pos = np.array([kb.get_position() for kb in env.get_kilobots()])

        # obs[:], reward, dones, infos = env.step(np.array(m_pos) - kb_pos)
        obs[:], reward, dones, infos = env.step(env.action_space.sample())
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
