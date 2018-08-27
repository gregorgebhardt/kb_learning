import cloudpickle
import gym
from kb_learning.envs import register_object_env, NormalizeActionWrapper


def main():
    with open('make_model.pkl', 'rb') as fh:
        make_model = cloudpickle.load(fh)

    model = make_model()

    model.load('00400')

    env_id = register_object_env(1., 10, 'quad', .15, .15, 'circular', .15)
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
