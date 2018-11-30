import gym
import tensorflow as tf
import numpy as np
from baselines.common import tf_util
from baselines.common.distributions import make_pdtype
from kb_learning.policy_networks.multi_layer_perceptron import MultiLayerPerceptron


class MlpPolicyNetwork(object):
    def __init__(self, name, ob_space, ac_space, feature_net_size=(100, 50, 25), pd_bias_init=None,
                 weight_sharing=False, **kwargs):

        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope().name

            assert isinstance(ob_space, gym.spaces.Box), 'observation space is not a Box'

            self.pdtype = make_pdtype(ac_space)

            # the observation is usually a tensor with shape num_agents x observation_dims
            ob = tf_util.get_placeholder(name="ob", dtype=tf.float32, shape=(None, ob_space.shape[-1]))

            with tf.variable_scope('vf'):
                feature_layer = MultiLayerPerceptron(X=ob, hidden_sizes=feature_net_size, activation=tf.nn.tanh).out

                self.vpred = tf.layers.dense(feature_layer, 1, name='final',
                                             kernel_initializer=tf_util.normc_initializer(1.))[:, 0]

            with tf.variable_scope('pol'):
                if not weight_sharing:
                    feature_layer = MultiLayerPerceptron(X=ob, hidden_sizes=feature_net_size, activation=tf.nn.tanh).out

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
