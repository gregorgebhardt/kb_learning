import numpy as np
import gym
from kb_learning.envs import register_object_env
from pose_control import get_pose_controller

def main():
    env_id = register_object_env(entry_point='kb_learning.envs:PoseControlEnv', num_kilobots=10,
                                 object_shape='corner-quad', object_width=.05, object_height=.3,
                                 light_type='circular', light_radius=.2)
    env = gym.make(env_id)

    obs = env.reset()
    controller = get_pose_controller(goal_pose=env.get_desired_pose(), k_p=.01, k_t=.08, C=.05)

    for _ in range(2000):
        obj_pose = obs[-3:]
        light_pos = obs[-5:-3]
        swarm_pos = obs[:-5]
        swarm_pos = swarm_pos.reshape(-1, 2).mean(axis=0)

        action = controller(swarm_pos, light_pos, obj_pose)
        action = np.minimum(action, env.action_space.high / 2)
        action = np.maximum(action, env.action_space.low / 2)
        # print(action)

        obs[:], rewards, done, info = env.step(action)

        if done is True:
            env.reset()

        env.render()


if __name__ == '__main__':
    main()


