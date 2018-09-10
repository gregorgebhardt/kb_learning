import numpy as np

from kb_learning.envs import ObjectRelativeEnv, register_object_relative_env, register_object_absolute_env

import gym

# env_id = register_object_absolute_env(10, 'corner_quad', .15, .15, 'circular', .2)
env_id = register_object_relative_env(.5, 10, 'quad', .15, .15, 'circular', .2)
env: ObjectRelativeEnv = gym.make(env_id)

n_episodes = 10

_reward = 0
for j in range(n_episodes):
    obs = env.reset()
    last_pose = np.array(env.get_objects()[0].get_pose())
    pose_diff = []
    env.render()
    ep_r = 0
    while True:
        m_pos = env._screen.get_mouse_position()
        # print(m_pos)
        l_pos = env.get_light().get_state()
        obs, r, done, info = env.step(np.array(m_pos) - l_pos)
        pose_diff.append(env.get_objects()[0].get_pose() - last_pose)

        last_pose = np.array(env.get_objects()[0].get_pose())
        ep_r += r
        if done:
            print('done!')
            break

    print(ep_r)
    _reward += ep_r

print('mean reward/episode: {}'.format(_reward/n_episodes))
