import baselines.common.tf_util as U
import numpy as np
import tensorflow as tf


class SoftMaxEmbedding:
    def __init__(self, input_ph, hidden_sizes, nr_obs, dim_obs, activation=tf.nn.relu,
                 last_as_valid=False):
        assert len(hidden_sizes) > 0
        alpha = tf.Variable(np.ones(hidden_sizes[-1]) * .5, dtype=tf.float32)

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

        # compute weights w = exp(phi * alpha)
        w = tf.exp(last_out * alpha)

        if last_as_valid:
            # set weights to zero for invalid observations
            w = tf.multiply(w, layer_valid)

        reshaped_output = tf.reshape(last_out, shape=(-1, nr_obs, hidden_sizes[-1]))
        reshaped_w = tf.reshape(w, shape=(-1, nr_obs, hidden_sizes[-1]))

        # compute ∑ phi * w / ∑ w (w is zero for invalid observations)
        softmax_features = tf.reduce_sum(reshaped_output * reshaped_w, axis=1) / tf.reduce_sum(reshaped_w, axis=1)
        # softmax_features = tf.where(tf.is_nan(softmax_features), tf.zeros_like(softmax_features), softmax_features)

        self.out = softmax_features
