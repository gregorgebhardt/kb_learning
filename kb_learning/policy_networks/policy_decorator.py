import tensorflow as tf


def compute_mean(network_function, num_kilobots):
    kilobot_states = 2*num_kilobots

    def _decorated_network_function(X):
        kilobot_states_input_layer = tf.slice(X, [0, 0], [-1, kilobot_states])
        env_states_input_layer = tf.slice(X, [0, kilobot_states], [-1, -1])

        X = tf.reduce_mean(tf.reshape(kilobot_states_input_layer, (kilobot_states_input_layer.shape[0], -1, 2)), axis=1)

        X = tf.concat([X, env_states_input_layer], axis=1)

        return network_function(X)

    return _decorated_network_function


def compute_mean_var(network_function, num_kilobots):
    kilobot_states = 2 * num_kilobots

    def _decorated_network_function(X):
        kilobot_states_input_layer = tf.slice(X, [0, 0], [-1, kilobot_states])
        env_states_input_layer = tf.slice(X, [0, kilobot_states], [-1, -1])

        X = tf.reshape(kilobot_states_input_layer, (kilobot_states_input_layer.shape[0], -1, 2))
        X_mean, X_var = tf.nn.moments(X, axes=[1])
        X = tf.concat([X_mean, X_var, env_states_input_layer], axis=1)

        return network_function(X)

    return _decorated_network_function
