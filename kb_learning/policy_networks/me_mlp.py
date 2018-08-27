import baselines.common.tf_util as U
from kb_learning.policy_networks import mean_embedding as me
import numpy as np
import tensorflow as tf


def me_mlp(num_me_inputs, dim_me_inputs, layers=(2, 2), num_hidden=((128, 128), (128, 128)),
           activation=tf.nn.leaky_relu):
    """
    MLP mean embedding followed by another MLP to be used in a policy / q-function approximator

    Parameters:
    ----------

    layers: tuple(int, ...)          number of fully-connected layers of mean embedding and fc layers (default: (2, 2))

    num_hidden: tuple(tuple(int,..), ...)   size of fully-connected layers in mean embedding and fc layers (default: ((64, 64), (64, 64)))

    activation:                     activation function (default: tf.nn.relu)

    Returns:
    -------

    function that builds MLP mean embedding network with a given input tensor / placeholder
    """
    def network_fn(X):
        num_me_layers = layers[0]
        num_mlp_layers = layers[1]
        num_me_hidden = num_hidden[0]
        num_mlp_hidden = num_hidden[1]
        num_kilobots = num_me_inputs  # env._num_kilobots
        kb_single_state = dim_me_inputs
        kilobot_states = num_kilobots * kb_single_state  # env._kilobot_space.shape
        # env_states = env.observation_space.shape - kilobot_states

        kilobot_states_input_layer = tf.slice(X, [0, 0], [-1, kilobot_states])
        env_states_input_layer = tf.slice(X, [0, kilobot_states], [-1, -1])

        with tf.variable_scope('me'):
            mean_embedding = me.MeanEmbedding(kilobot_states_input_layer, num_me_layers, num_me_hidden,
                                              num_kilobots, kb_single_state)

        h = tf.concat([mean_embedding.me_out, env_states_input_layer], axis=1)
        for i in range(num_mlp_layers):
            h = tf.layers.dense(h, num_mlp_hidden[i], name="fc%i" % (i + 1),
                                kernel_initializer=U.normc_initializer(1.0), activation=activation)

        return h, None

    return network_fn
