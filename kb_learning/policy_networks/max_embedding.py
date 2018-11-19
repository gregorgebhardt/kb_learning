import tensorflow as tf
import numpy as np

import baselines.common.tf_util as U


class MaxEmbedding:
    def __init__(self, input_ph, hidden_sizes, nr_obs, dim_obs, activation=tf.nn.relu,
                 last_as_valid=False):

        reshaped_input = tf.reshape(input_ph, shape=(-1, int(dim_obs)))
        if last_as_valid:
            layer_input = tf.slice(reshaped_input, [0, 0], [-1, int(dim_obs)-1])
            layer_valid = tf.slice(reshaped_input, [0, int(dim_obs) - 1], [-1, 1])
        else:
            layer_input = reshaped_input
            layer_valid = None

        last_out = layer_input

        if len(hidden_sizes) > 0:
            for i, layer_size in enumerate(hidden_sizes):
                last_out = tf.layers.dense(last_out, layer_size, name="fc%i" % (i + 1),
                                           activation=activation,
                                           kernel_initializer=U.normc_initializer(1.0))

            if last_as_valid:
                last_out = tf.multiply(last_out, layer_valid)

            reshaped_output = tf.reshape(last_out, shape=(-1, nr_obs, hidden_sizes[-1]))
        else:
            if last_as_valid:
                last_out = tf.multiply(last_out, layer_valid)
            reshaped_output = tf.reshape(last_out, shape=(-1, nr_obs, dim_obs))

        last_out_mean = tf.reduce_max(reshaped_output, axis=1)

        self.out = last_out_mean
