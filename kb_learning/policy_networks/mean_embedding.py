import baselines.common.tf_util as U
import tensorflow as tf
import tensorflow.contrib as tfc


class MeanEmbedding:
    def __init__(self, input_ph, num_layers, hidden_sizes, nr_obs, dim_obs, layer_norm=False):

        reshaped_input = tf.reshape(input_ph, shape=(-1, int(dim_obs)))

        last_out = reshaped_input

        if num_layers > 0:
            for i in range(num_layers):
                last_out = tf.layers.dense(last_out, hidden_sizes[i], name="fc%i" % (i + 1),
                                           kernel_initializer=U.normc_initializer(1.0))
                if layer_norm:
                    last_out = tfc.layers.layer_norm(last_out)
                last_out = tf.nn.relu(last_out)

            fc_out = last_out

            reshaped_output = tf.reshape(fc_out, shape=(-1, nr_obs, hidden_sizes[-1]))

        else:
            reshaped_output = tf.reshape(reshaped_input, shape=(-1, nr_obs, dim_obs - 2))

        last_out_mean = tf.reduce_mean(reshaped_output, axis=1)

        self.me_out = last_out_mean
