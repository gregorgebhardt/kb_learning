import tensorflow as tf
import tensorflow.contrib as tfc

import baselines.common.tf_util as U


class MeanEmbedding:
    def __init__(self, input_ph, hidden_sizes, nr_obs, dim_obs, activation=tf.nn.leaky_relu, layer_norm=False):

        reshaped_input = tf.reshape(input_ph, shape=(-1, int(dim_obs)))

        last_out = reshaped_input

        if len(hidden_sizes) > 0:
            for i, layer_size in enumerate(hidden_sizes):
                last_out = tf.layers.dense(last_out, layer_size, name="fc%i" % (i + 1),
                                           kernel_initializer=U.normc_initializer(1.0))
                if layer_norm:
                    last_out = tfc.layers.layer_norm(last_out)
                    # tf.summary.scalar(last_out.name + '_norm', last_out)

                last_out = activation(last_out)
                # tf.summary.histogram(last_out.name + '_hist', last_out)

            fc_out = last_out

            reshaped_output = tf.reshape(fc_out, shape=(-1, nr_obs, hidden_sizes[-1]))

        else:
            reshaped_output = tf.reshape(reshaped_input, shape=(-1, nr_obs, dim_obs - 2))

        last_out_mean = tf.reduce_mean(reshaped_output, axis=1)
        # tf.summary.histogram(last_out_mean.name + '_hist', last_out_mean)

        self.out = last_out_mean
