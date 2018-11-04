import cloudpickle
import gym
from kb_learning.envs import register_object_env, NormalizeActionWrapper


def main():
    with open('policies/nn_based/ppo/absolute_env/make_model.pkl', 'rb') as fh:
        make_model = cloudpickle.load(fh)

    model = make_model()
    model.load('policies/nn_based/ppo/absolute_env/model_parameters')

    env_id = register_object_env(entry_point='kb_learning.envs:ObjectAbsoluteEnv', num_kilobots=10,
                                 object_shape='corner_quad', object_width=.15, object_height=.15,
                                 light_type='circular', light_radius=.15)
    env = NormalizeActionWrapper(gym.make(env_id))

    obs = env.reset()
    states = model.initial_state
    dones = False

    for _ in range(2000):
        actions, values, states, neglogpacs = model.eval_step(obs, S=states, M=dones)

        obs[:], rewards, dones, infos = env.step(actions[0])

        if dones is True:
            env.reset()

        env.render()


if __name__ == '__main__':
    main()
