import baselines.common.tf_util as U
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc


class SoftMaxEmbedding:
    def __init__(self, input_ph, hidden_sizes, nr_obs, dim_obs, activation=tf.nn.leaky_relu,
                 last_as_valid=False):
        assert len(hidden_sizes) > 0
        beta = tf.get_variable(name='beta', shape=hidden_sizes[-1],
                               initializer=tf.random_uniform_initializer(maxval=10))

        reshaped_input = tf.reshape(input_ph, shape=(-1, int(dim_obs)))
        if last_as_valid:
            layer_input = tf.slice(reshaped_input, [0, 0], [-1, int(dim_obs) - 1])
            layer_valid = tf.slice(reshaped_input, [0, int(dim_obs) - 1], [-1, 1])
        else:
            layer_input = reshaped_input
            layer_valid = None

        last_out = layer_input

        for i, layer_size in enumerate(hidden_sizes):
            last_out = tf.layers.dense(last_out, layer_size, name="fc%i" % (i + 1),
                                       activation=activation,
                                       kernel_initializer=U.normc_initializer(1.0))

        phi = tf.reshape(last_out, shape=(-1, nr_obs, hidden_sizes[-1]))

        # compute weights w = exp(φ * β)
        # 1. take product of features φ and temperature β
        phi_beta = phi * beta
        # 2. find max of φ * β along each feature dimension
        Z = tf.reduce_max(phi_beta, axis=1, keepdims=True)
        # 3. subtract Z and take exp
        w = tf.exp(phi_beta - Z)

        if last_as_valid:
            # set weights to zero for invalid observations
            layer_valid = tf.reshape(layer_valid, shape=(-1, nr_obs, 1))
            w = tf.multiply(w, layer_valid)

        # compute ∑ φ * w / ∑ w
        softmax_features = tf.reduce_sum(phi * w, axis=1) / tf.reduce_sum(w, axis=1)
        softmax_features = tf.where(tf.is_nan(softmax_features), tf.zeros_like(softmax_features), softmax_features)

        self.out = softmax_features
