import tensorflow as tf

from .mean_embedding import MeanEmbedding
from .multi_layer_perceptron import MultiLayerPerceptron


def swarm_policy_network(num_agents, light_dims, num_objects, object_dims, swarm_network_size=(128, 128),
                         light_network_size=(128, 128), objects_network_size=(128, 128),
                         concat_network_size=(128, 128), activation=tf.nn.leaky_relu):

    def network_fn(X):
        swarm_input_layer = tf.slice(X, [0, 0], [-1, num_agents * 2])
        light_input_layer = tf.slice(X, [0, num_agents * 2], [-1, light_dims])
        objects_input_layer = tf.slice(X, [0, num_agents * 2 + light_dims], [-1, num_objects * object_dims])

        with tf.variable_scope('kb_net'):
            swarm_network = MeanEmbedding(swarm_input_layer, swarm_network_size, num_agents, 2,
                                          activation=activation)

        with tf.variable_scope('li_net'):
            light_network = MultiLayerPerceptron(light_input_layer, light_network_size, activation=activation)

        with tf.variable_scope('obj_net'):
            objects_network = MeanEmbedding(objects_input_layer, objects_network_size, num_objects, object_dims,
                                            activation=activation)

        h = tf.concat([swarm_network.out, light_network.out, objects_network.out], axis=1)

        h = MultiLayerPerceptron(h, concat_network_size, activation=activation).out

        return h, None

    return network_fn
