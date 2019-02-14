import random

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

from kb_learning.envs import NormalizeActionWrapper
from kb_learning.envs._multi_object_env import MultiObjectDirectControlEnv
from kb_learning.policy_networks.swarm_policy import SwarmPolicyNetwork
from kb_learning.tools.trpo_tools import ActWrapper, traj_segment_generator_ma

env_config_yaml = '''
!EvalEnv
width: 1.
height: 1.
resolution: 608

objects:
    - !ObjectConf
      idx: 0
      shape: square
      width: .08
      height: .08
      init: random
      
# objects:
#     - !ObjectConf
#       idx: 0
#       shape: c_shape
#       width: .1
#       height: .1
#       init: random
#     - !ObjectConf
#       idx: 0
#       shape: t_shape
#       width: .1
#       height: .1
#       init: random

# objects:
#     - !ObjectConf
#       idx: 0
#       shape: l_shape
#       width: .1
#       height: .1
#       init: random
#     - !ObjectConf
#       idx: 0
#       shape: l_shape
#       width: .1
#       height: .1
#       init: random

kilobots: !KilobotsConf
    num: 10
    mean: random
    std: .03
'''


def main():
    env_config = yaml.load(env_config_yaml)
    env_config.objects = [env_config.objects[0]] * 8
    env = MultiObjectDirectControlEnv(configuration=env_config, agent_reward=True, swarm_reward=True,
                                      agent_type='SimpleVelocityControlKilobot',
                                      done_after_steps=512,
                                      reward_function='object_clustering_amp')
    env.video_path = 'video_out'
    # env.render_mode = 'array'
    wrapped_env = NormalizeActionWrapper(env)

    def policy_fn(name, **policy_params):
        return SwarmPolicyNetwork(name=name, **policy_params)
        # return MlpPolicyNetwork(name=name, **policy_params)

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

    update_params = dict(
        ob_space=env.observation_space,
        num_agent_observations=env.num_kilobots - 1,
        num_object_observations=len(env_config.objects),
    )
    policy_path = 'policies/nn_based/trpo_ma/object_sorting3/mean_max/policy.pkl'
    pi = ActWrapper.load(policy_path, policy_fn, update_params=update_params)

    # get betas
    import tensorflow as tf
    all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    beta_variables = list(filter(lambda v: 'beta' in v.name, all_variables))

    np.random.seed(0)
    random.seed(0)
    seg_gen = traj_segment_generator_ma(pi, wrapped_env, 512, False, render=True)

    for _ in range(4):
        seg = seg_gen.__next__()
        # add_vtarg_and_adv_ma(seg, gamma=0.99, lam=.95)
        # # seg['rew'] /= seg['rew'].std()
        # a_target = seg['adv']
        # a_target = (a_target - a_target.mean()) / a_target.std()
        #
        # print('steps: {} episode return: {}'.format(seg['ep_lens'], seg['ep_rets']))
        # plt.plot(seg['rew'], label='reward')
        # plt.plot(seg['vpred'], label='v-pred')
        # plt.plot(a_target, label='a_target')
        # plt.plot(seg['tdlamret'], label='td_lam_ret')
        # plt.legend()
        # plt.show()


def eval_beta_values():
    env_config = yaml.load(env_config_yaml)
    env_config.objects = [env_config.objects[0]] * 4
    env = MultiObjectDirectControlEnv(configuration=env_config, agent_reward=True, swarm_reward=True,
                                      agent_type='SimpleVelocityControlKilobot',
                                      done_after_steps=512,
                                      reward_function='object_clustering_amp')

    def policy_fn(name, **policy_params):
        return SwarmPolicyNetwork(name=name, **policy_params)

    update_params = dict(
        ob_space=env.observation_space,
        num_agent_observations=env.num_kilobots - 1,
        num_object_observations=len(env_config.objects),
    )
    # policy_path = 'policies/nn_based/trpo_ma/moving_objects/softmax_softmax/{}/policy.pkl'
    policy_path = 'policies/nn_based/trpo_ma/object_sorting_amp2/softmax_softmax/{}/policy.pkl'

    f = plt.figure()
    gs = gridspec.GridSpec(4, 1)
    f2 = plt.figure()
    gs2 = gridspec.GridSpec(2, 4)

    initial_betas = None

    for i, (it, of) in enumerate(zip(['it_0000', 'it_0049', 'it_0149', 'it_0249'], [-.33, -.11, .11, .33])):

        with tf.Graph().as_default():
            pi = ActWrapper.load(policy_path.format(it), policy_fn, update_params=update_params)
            # get betas
            all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            beta_variables = list(filter(lambda v: 'beta' in v.name, all_variables))
            # kernel_variables = list(filter(lambda v: v.name in ['pi/pol/fc1/kernel:0', 'pi/vf/fc1/kernel:0'], all_variables))

            sess = tf.get_default_session()

            # ax_1 = f2.add_subplot(gs2[0, i])
            # ax_2 = f2.add_subplot(gs2[1, i])
            # ax_1.name = kernel_variables[0].name
            # ax_1.matshow(sess.run(kernel_variables[0]))
            # ax_2.name = kernel_variables[1].name
            # ax_2.matshow(sess.run(kernel_variables[1]))

            if initial_betas is None:
                initial_betas = [sess.run(b) for b in beta_variables]

            print(it)
            for b, init_b, s in zip(beta_variables, initial_betas, gs):
                ax = f.add_subplot(s)
                b = ax.bar(np.arange(64)+of, sess.run(b) - init_b, width=.22, label=it)

            ax.legend()

            # plt.savefig()

    plt.show(block=True)


if __name__ == '__main__':
    main()
    # eval_beta_values()
