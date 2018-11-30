import gym
import numpy as np
import tensorflow as tf
from baselines.common.distributions import make_pdtype
from baselines.common import tf_util

from kb_learning.policy_networks.max_embedding import MaxEmbedding
from kb_learning.policy_networks.mean_embedding import MeanEmbedding
from kb_learning.policy_networks.softmax_embedding import SoftMaxEmbedding
from kb_learning.policy_networks.multi_layer_perceptron import MultiLayerPerceptron


def feature_network(*, swarm_input_layer, swarm_net_size, swarm_net_type, num_agent_observations, agent_obs_dims,
                    objects_input_layer, objects_net_size, objects_net_type, num_object_observations, object_obs_dims,
                    extra_input_layer, extra_net_size, concat_net_size):
    with tf.variable_scope('swarm_net'):
        if swarm_net_type == 'mean':
            # mean embedding for the swarm
            swarm_layer = MeanEmbedding(swarm_input_layer, swarm_net_size, num_agent_observations,
                                        agent_obs_dims).out
        elif swarm_net_type == 'max':
            # max embedding for the swarm
            swarm_layer = MaxEmbedding(swarm_input_layer, swarm_net_size, num_agent_observations,
                                       agent_obs_dims).out
        elif swarm_net_type == 'softmax':
            swarm_layer = SoftMaxEmbedding(swarm_input_layer, swarm_net_size, num_agent_observations,
                                           agent_obs_dims).out
        elif swarm_net_type == 'mlp':
            swarm_layer = MultiLayerPerceptron(swarm_input_layer, swarm_net_size).out
        else:
            swarm_layer = swarm_input_layer

    with tf.variable_scope('obj_net'):
        if objects_net_type == 'mean':
            # mean embedding for the objects
            objects_layer = MeanEmbedding(objects_input_layer, objects_net_size, num_object_observations,
                                          object_obs_dims, last_as_valid=True).out
        elif objects_net_type == 'max':
            # max embedding for the objects
            objects_layer = MaxEmbedding(objects_input_layer, objects_net_size, num_object_observations,
                                         object_obs_dims, last_as_valid=True).out
        elif swarm_net_type == 'softmax':
            objects_layer = SoftMaxEmbedding(objects_input_layer, objects_net_size, num_object_observations,
                                             object_obs_dims, last_as_valid=True).out
        elif objects_net_type == 'mlp':
            objects_layer = MultiLayerPerceptron(objects_input_layer, objects_net_size).out
        else:
            objects_layer = objects_input_layer

    with tf.variable_scope('extra_net'):
        if len(extra_net_size) > 0 and extra_input_layer.shape[1] > 0:
            # mlp for additional inputs
            extra_layer = MultiLayerPerceptron(extra_input_layer, extra_net_size,
                                               activation=tf.nn.leaky_relu).out
        else:
            extra_layer = extra_input_layer

    # tf.debugging.assert_all_finite(swarm_layer, 'non-finite values in swarm_layer')
    # tf.debugging.assert_all_finite(objects_layer, 'non-finite values in objects_layer')
    # tf.debugging.assert_all_finite(extra_layer, 'non-finite values in extra_layer')

    # concat mean embeddings and extra inputs
    concat_layer = tf.concat([swarm_layer, objects_layer, extra_layer], axis=1)

    # mlp after concatenation
    return MultiLayerPerceptron(concat_layer, concat_net_size, activation=tf.nn.leaky_relu).out


def split_input_tensor(X, *, num_agent_observations, agent_obs_dims, num_object_observations, object_obs_dims):
    size_kilobot_obs = num_agent_observations * agent_obs_dims
    size_objects_obs = num_object_observations * object_obs_dims
    # the first columns are the observations of the kilobots
    swarm_input_layer = tf.slice(X, [0, 0], [-1, size_kilobot_obs], name='kb_slice')
    # followed by the observations of the objects
    objects_input_layer = tf.slice(X, [0, size_kilobot_obs], [-1, size_objects_obs], name='ob_slice')
    # the extra dims can contain information about the agent, the target area or the light source
    extra_input_layer = tf.slice(X, [0, size_kilobot_obs + size_objects_obs], [-1, -1], name='ex_slice')

    return swarm_input_layer, objects_input_layer, extra_input_layer


class SwarmPolicyNetwork(object):
    recurrent = False

    def __init__(self, name, ob_space, ac_space,
                 num_agent_observations=10, agent_dims=2,
                 num_object_observations=1, object_dims=4,
                 swarm_net_size=(64,), swarm_net_type='mean',
                 objects_net_size=(64,), objects_net_type='mean',
                 extra_net_size=(64,),
                 concat_net_size=(64,),
                 pd_bias_init=None,
                 # pd_bias_init=(.1, .0, .0, -.5),
                 gaussian_fixed_var=False,
                 weight_sharing=False,
                 **kwargs):

        self.layer_norm = False
        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope().name

            assert isinstance(ob_space, gym.spaces.Box), 'observation space is not a Box'

            self.pdtype = make_pdtype(ac_space)

            # the observation is usually a tensor with shape num_agents x observation_dims
            ob = tf_util.get_placeholder(name="ob", dtype=tf.float32, shape=(None, ob_space.shape[-1]))

            # split the observation tensor into agent observations, object observations and additional observations
            swarm_input_layer, objects_input_layer, extra_input_layer = split_input_tensor(ob,
                                                                                           num_agent_observations=num_agent_observations, agent_obs_dims=agent_dims,
                                                                                           num_object_observations=num_object_observations, object_obs_dims=object_dims)

            with tf.variable_scope('vf'):
                # construct the feature network for the value function
                feature_layer = feature_network(swarm_input_layer=swarm_input_layer, swarm_net_size=swarm_net_size,
                                                swarm_net_type=swarm_net_type,
                                                num_agent_observations=num_agent_observations,
                                                agent_obs_dims=agent_dims, objects_input_layer=objects_input_layer,
                                                objects_net_size=objects_net_size, objects_net_type=objects_net_type,
                                                num_object_observations=num_object_observations,
                                                object_obs_dims=object_dims, extra_input_layer=extra_input_layer,
                                                extra_net_size=extra_net_size, concat_net_size=concat_net_size)

                self.vpred = tf.layers.dense(feature_layer, 1, name='final',
                                             kernel_initializer=tf_util.normc_initializer(.1),
                                             # bias_initializer=tf.constant_initializer(-1)
                                             )[:, 0]

            # policy network
            with tf.variable_scope('pol'):
                if not weight_sharing:
                    # construct the feature network for the policy
                    feature_layer = feature_network(swarm_input_layer=swarm_input_layer, swarm_net_size=swarm_net_size,
                                                    swarm_net_type=swarm_net_type,
                                                    num_agent_observations=num_agent_observations,
                                                    agent_obs_dims=agent_dims, objects_input_layer=objects_input_layer,
                                                    objects_net_size=objects_net_size, objects_net_type=objects_net_type,
                                                    num_object_observations=num_object_observations,
                                                    object_obs_dims=object_dims, extra_input_layer=extra_input_layer,
                                                    extra_net_size=extra_net_size, concat_net_size=concat_net_size)

                if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                    mean = tf.layers.dense(feature_layer, self.pdtype.param_shape()[0] // 2, name='final',
                                           kernel_initializer=tf_util.normc_initializer(0.01))
                    logstd = tf.get_variable(name="logstd", shape=[1, self.pdtype.param_shape()[0] // 2],
                                             initializer=tf.zeros_initializer())
                    pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
                else:
                    pd_dims = self.pdtype.param_shape()[0]
                    if pd_bias_init is None:
                        pd_bias_init = np.zeros(ac_space.shape[0] * 2)
                    pd_bias_init = np.asarray(pd_bias_init)
                    pdparam = tf.layers.dense(feature_layer, pd_dims, name='final',
                                              kernel_initializer=tf_util.normc_initializer(0.01),
                                              bias_initializer=tf.constant_initializer(pd_bias_init))

        self.pd = self.pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = tf_util.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = tf_util.function([ob, stochastic], [ac, self.vpred])
        # self._me_v = tf_util.function([ob], [me_v.me_out])
        # self._me_pi = tf_util.function([ob], [me_pi.me_out])

    def act(self, ob, stochastic=False):
        ac1, vpred1 = self._act(ob, stochastic)
        return ac1, vpred1

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    @staticmethod
    def get_initial_state():
        return []


def swarm_policy_network(*, num_agents, agent_dims, num_objects, object_dims,
                         swarm_net_size=(64,), swarm_net_type='mean',
                         objects_net_size=(64,), objects_net_type='mean',
                         extra_net_size=(64,),
                         concat_net_size=(256,)):

    def policy_function(X):
        swarm_input_layer, objects_input_layer, extra_input_layer = split_input_tensor(X,
            num_agent_observations=num_agents, agent_obs_dims=2, num_object_observations=num_objects,
            object_obs_dims=object_dims)

        feature_layer = feature_network(swarm_input_layer=swarm_input_layer, swarm_net_size=swarm_net_size,
                                        swarm_net_type=swarm_net_type, num_agent_observations=num_agents,
                                        agent_obs_dims=agent_dims, objects_input_layer=objects_input_layer,
                                        objects_net_size=objects_net_size, objects_net_type=objects_net_type,
                                        num_object_observations=num_objects, object_obs_dims=object_dims,
                                        extra_input_layer=extra_input_layer, extra_net_size=extra_net_size,
                                        concat_net_size=concat_net_size)

        return feature_layer, None

    return policy_function
