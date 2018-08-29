import tensorflow as tf
from baselines.common.models import fc
import numpy as np


def mlp(size=(64, 64), activation=tf.nn.leaky_relu):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """
    def network_fn(X):
        h = tf.layers.flatten(X)
        for i, s in enumerate(size):
            h = activation(fc(h, 'mlp_fc{}'.format(i), nh=s, init_scale=np.sqrt(2)))
        return h, None

    return network_fn