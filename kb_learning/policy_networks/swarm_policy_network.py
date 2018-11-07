import tensorflow as tf

from .mean_embedding import MeanEmbedding
from .multi_layer_perceptron import MultiLayerPerceptron


def swarm_policy_network(*, num_agents, num_objects, object_dims, extra_dims,
                         swarm_network_size=(128, 128), swarm_network_type='me',
                         objects_network_size=(128, 128), objects_network_type='me',
                         exta_network_size=(128, 128),
                         concat_network_size=(128, 128), activation=tf.nn.leaky_relu):

    def network_fn(X):
        # the input tensor with shape num_observations x

        swarm_input_layer = tf.slice(X, name='kb_slice', begin=[0, 0], size=[-1, num_agents * 2])
        objects_input_layer = tf.slice(X, name='ob_slice', begin=[0, num_agents * 2],
                                       size=[-1, num_objects * object_dims])
        extra_input_layer = tf.slice(X, name='ex_slice', begin=[0, num_agents * 2 + num_objects * object_dims],
                                     size=[-1, extra_dims])

        with tf.variable_scope('kb_net'):
            if swarm_network_type == 'me':
                swarm_network = MeanEmbedding(swarm_input_layer, swarm_network_size, num_agents, 2,
                                              activation=activation).out
            else:
                swarm_network = MultiLayerPerceptron(swarm_input_layer, swarm_network_size, activation=activation).out

        if objects_network_size:
            with tf.variable_scope('obj_net'):
                if objects_network_type == 'me':
                    objects_network = MeanEmbedding(objects_input_layer, objects_network_size, num_objects, object_dims,
                                                    activation=activation).out
                else:
                    objects_network = MultiLayerPerceptron(objects_input_layer, objects_network_size,
                                                           activation=activation).out
        else:
            objects_network = objects_input_layer

        if exta_network_size:
            with tf.variable_scope('ex_net'):
                extra_network = MultiLayerPerceptron(extra_input_layer, exta_network_size, activation=activation).out
        else:
            extra_network = extra_input_layer

        concat_layer = tf.concat([swarm_network, objects_network, extra_network], axis=1)

        h = MultiLayerPerceptron(concat_layer, concat_network_size, activation=activation).out

        return h, None

    return network_fn
