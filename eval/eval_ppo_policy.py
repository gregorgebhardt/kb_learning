import cloudpickle
import gym
from baselines.common import tf_util
from baselines.common.policies import build_policy
import tensorflow as tf

from kb_learning.envs import register_object_env, NormalizeActionWrapper
from kb_learning.policy_networks import swarm_policy_network


def main():
    with open('policies/nn_based/ppo/absolute_env/make_model.pkl', 'rb') as fh:
        make_model = cloudpickle.load(fh)

    model = make_model()
    model.load('policies/nn_based/ppo/absolute_env/model_parameters')

    env_id = register_object_env(entry_point='kb_learning.envs:ObjectAbsoluteEnv', num_kilobots=10,
                                 object_shape='corner_quad', object_width=.15, object_height=.15,
                                 light_type='circular', light_radius=.2)
    env = NormalizeActionWrapper(gym.make(env_id))

    # network = swarm_policy_network(num_agents=10, light_dims=2, num_objects=1, object_dims=4)
    # policy = build_policy(env.observation_space, env.action_space, network)
    # sess = tf_util.get_session()
    #
    # with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE) as ppo_scope:
    #     pi = policy(1, 1, sess)
    # states = tf.constant(0)
    #
    # tf_util.initialize()
    #
    # tf_util.load_variables('policies/nn_based/ppo/absolute_env/model_parameters', sess=sess)

    states = model.initial_state
    obs = env.reset()
    dones = False

    for _ in range(2000):
        actions, values, states, neglogpacs = pi(obs, S=states, M=dones)

        obs[:], rewards, dones, infos = env.step(actions[0])

        if dones is True:
            env.reset()

        env.render()


if __name__ == '__main__':
    main()
