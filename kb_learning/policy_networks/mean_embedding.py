import tensorflow as tf
import tensorflow.contrib as tfc

import baselines.common.tf_util as U


class MeanEmbedding:
    def __init__(self, input_ph, hidden_sizes, nr_obs, dim_obs, activation=tf.nn.leaky_relu,
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
                reshaped_output = tf.reshape(last_out, shape=(-1, nr_obs, dim_obs - 1))
            else:
                reshaped_output = tf.reshape(last_out, shape=(-1, nr_obs, dim_obs))

        if last_as_valid:
            reshaped_valid = tf.reshape(layer_valid, shape=(-1, nr_obs))
            num_obs_objects = tf.reduce_sum(reshaped_valid)
            last_out_sum = tf.reduce_sum(reshaped_output, axis=1)
            with tf.control_dependencies([tf.assert_greater(num_obs_objects, .0)]):
                last_out_mean = tf.divide(last_out_sum, num_obs_objects)
        else:
            last_out_mean = tf.reduce_mean(reshaped_output, axis=1)


        self.out = last_out_mean
