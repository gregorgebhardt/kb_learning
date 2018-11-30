import tensorflow as tf
import tensorflow.contrib as tfc


class MultiLayerPerceptron:
    def __init__(self, X, hidden_sizes=(64, 64), activation=tf.nn.leaky_relu):

        h = tf.layers.flatten(X)

        if activation == tf.nn.tanh:
            initializer = tfc.layers.xavier_initializer()
        else:
            initializer = tf.initializers.orthogonal(1.0)

        for i, s in enumerate(hidden_sizes):
            h = tf.layers.dense(h, s, name="fc%i" % (i + 1), activation=activation,
                                kernel_initializer=initializer)

        self.out = h


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
    if activation == tf.nn.tanh:
        initializer = tfc.layers.xavier_initializer()
    else:
        initializer = tf.initializers.orthogonal(1.0)

    def network_fn(X):
        h = tf.layers.flatten(X)
        for i, s in enumerate(size):
            h = tf.layers.dense(h, s, name="fc%i" % (i + 1), activation=activation,
                                kernel_initializer=initializer)
        return h, None

    return network_fn
