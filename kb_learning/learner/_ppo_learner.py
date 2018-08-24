import baselines.common.tf_util as U
import gym
from baselines import logger
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.bench.monitor import Monitor
from baselines.ppo2.ppo2 import learn

from kb_learning.envs import register_object_env, NormalizeActionWrapper

logger.configure(format_strs=['stdout'])


def train(num_envs):
    sess = U.single_threaded_session()

    with sess:
        # create env here
        # TODO add wrapper for gym to reset in case object close to env boundaries or num time steps too large
        env_id = register_object_env(.0, 15, 'quad', .15, .15, 'circular', .3)

        def _make_env(i):
            def _make_env_i():
                import numpy as np
                np.random.seed(1234 * i + 7809)
                env = Monitor(NormalizeActionWrapper(gym.make(env_id)), 'env_{}_monitor.csv'.format(i), True)
                return env

            return _make_env_i

        envs = [_make_env(i) for i in range(num_envs)]

        # wrap env in SubprocVecEnv
        vec_env = SubprocVecEnv(envs)

        # def policy_fn(name):
        #     return mlp_policy.MlpPolicy(name=name, ob_space=vec_env.observation_space, ac_space=vec_env.action_space,
        #         hid_size=256, num_hid_layers=3)

        model = learn(network='mlp', env=vec_env, total_timesteps=1000000, nsteps=500, seed=1234, log_interval=5)

        vec_env.close()


if __name__ == '__main__':
    train(num_envs=20)
