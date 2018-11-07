import gym
import tensorflow as tf
import tensorflow.contrib as tfc
from baselines.common.distributions import make_pdtype
from baselines.common import tf_util


class MeanEmbedding:
    def __init__(self, input_ph, hidden_sizes, nr_obs, dim_obs, layer_norm=False):

        num_layers = len(hidden_sizes)
        # put each observation into a single row
        reshaped_input = tf.reshape(input_ph, shape=(-1, int(dim_obs)))

        # data_input_layer = tf.slice(reshaped_input, [0, 0], [-1, dim_obs - 2])
        # valid_input_layer = tf.slice(reshaped_input, [0, dim_obs - 2], [-1, 1])
        # valid_indices = tf.where(tf.cast(valid_input_layer, dtype=tf.bool))[:, 0:1]
        # valid_data = tf.gather_nd(data_input_layer, valid_indices)

        last_out = reshaped_input

        if num_layers > 0:
            for i in range(num_layers):
                last_out = tf.layers.dense(last_out, hidden_sizes[i], name="fc%i" % (i + 1),
                                           kernel_initializer=tf_util.normc_initializer(1.0))
                if layer_norm:
                    last_out = tfc.layers.layer_norm(last_out)
                last_out = tf.nn.relu(last_out)

            # fc_out = last_out

            # last_out_scatter = tf.scatter_nd(valid_indices, fc_out,
            #                                  shape=tf.cast(
            #                                      [tf.shape(data_input_layer)[0], tf.shape(fc_out)[1]],
            #                                      tf.int64))

            # reshape output so that we have the observations of one time step in one row
            reshaped_output = tf.reshape(last_out, shape=(-1, nr_obs, hidden_sizes[-1]))

        else:
            reshaped_output = tf.reshape(last_out, shape=(-1, nr_obs, dim_obs))

        # reshaped_nr_obs_var = tf.reshape(valid_input_layer, shape=(-1, nr_obs, 1))

        # n = tf.maximum(tf.reduce_sum(reshaped_nr_obs_var, axis=1, name="nr_agents_test"), 1)

        # sum along rows and divide by nr_obs to compute mean
        last_out_sum = tf.reduce_sum(reshaped_output, axis=1)
        last_out_mean = tf.divide(last_out_sum, nr_obs)

        self.me_out = last_out_mean


class MlpPolicy(object):
    recurrent = False

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self.layer_norm = False
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space,
              num_observed_kilobots=10, kilobot_dims=2,
              num_observed_objects=1, object_dims=4,
              hid_size=(64,), kb_me_size=(64,), ob_me_size=(64,),
              gaussian_fixed_var=False):
        # kilobot_dims = 6
        # object_dims = 4

        assert isinstance(ob_space, gym.spaces.Box), 'observation space is not a Box'

        self.pdtype = make_pdtype(ac_space)

        # the observation is usually a tensor with shape num_agents x observation_dims
        ob = tf_util.get_placeholder(name="ob", dtype=tf.float32, shape=(None, ob_space.shape[-1]))

        size_kilobot_obs = num_observed_kilobots * kilobot_dims
        size_objects_obs = num_observed_objects * object_dims
        # the first columns are the observations of the kilobots
        kilobots_input_layer = tf.slice(ob, [0, 0], [-1, size_kilobot_obs], name='kb_slice')
        # followed by the observations of the objects
        objects_input_layer = tf.slice(ob, [0, size_kilobot_obs], [-1, size_objects_obs], name = 'ob_slice')
        # the extra dims can contain information about the agent, the target area or the light source
        extra_input_layer = tf.slice(ob, [0, size_kilobot_obs + size_objects_obs], [-1, -1], name='ex_slice')

        with tf.variable_scope('vf'):
            with tf.variable_scope('me_kb'):
                # mean embedding for the swarm
                me_kb_layer = MeanEmbedding(kilobots_input_layer, kb_me_size, num_observed_kilobots, kilobot_dims)
            with tf.variable_scope('me_ob'):
                # mean embedding for the objects
                me_ob_layer = MeanEmbedding(objects_input_layer, ob_me_size, num_observed_objects, object_dims)

            # concat mean embeddings and extra inputs
            last_out = tf.concat([me_kb_layer.me_out, me_ob_layer.me_out, extra_input_layer], axis=1)

            # mlp after concatenation
            for i, _size in enumerate(hid_size):
                last_out = tf.layers.dense(last_out, _size, name="fc%i" % (i + 1),
                                           kernel_initializer=tf_util.normc_initializer(1.0))
                if self.layer_norm:
                    last_out = tfc.layers.layer_norm(last_out)
                last_out = tf.nn.relu(last_out)

            self.vpred = tf.layers.dense(last_out, 1, name='final',
                                         kernel_initializer=tf_util.normc_initializer(1.0))[:, 0]

        with tf.variable_scope('pol'):
            with tf.variable_scope('me_kb'):
                me_kb_layer = MeanEmbedding(kilobots_input_layer, kb_me_size, num_observed_kilobots, kilobot_dims)
            with tf.variable_scope('me_ob'):
                me_ob_layer = MeanEmbedding(objects_input_layer, ob_me_size, num_observed_objects, object_dims)
            last_out = tf.concat([me_kb_layer.me_out, me_ob_layer.me_out, extra_input_layer],
                                 axis=1)
            for i, _size in enumerate(hid_size):
                last_out = tf.layers.dense(last_out, _size, name="fc%i" % (i + 1),
                                           kernel_initializer=tf_util.normc_initializer(1.0))
                if self.layer_norm:
                    last_out = tfc.layers.layer_norm(last_out)
                last_out = tf.nn.relu(last_out)

            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(last_out, self.pdtype.param_shape()[0] // 2, name='final',
                                       kernel_initializer=tf_util.normc_initializer(1.))
                logstd = tf.get_variable(name="logstd", shape=[1, self.pdtype.param_shape()[0] // 2],
                                         initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, self.pdtype.param_shape()[0], name='final',
                                          kernel_initializer=tf_util.normc_initializer(1.))

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
