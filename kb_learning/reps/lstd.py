import numpy as np
from ..tools import np_chunks


class LeastSquaresTemporalDifference:
    def __init__(self):
        self.discount_factor = 0.98
        self.lstd_regularization_factor = 1e-8
        self.lstd_projection_regularization_factor = 1e-6

    def learnLSTD(self, phi, phi_next, reward):
        """compute the parameters theta of an approximation to the Q-function Q(s,a) = phi(s,a) * theta

        :param phi: the state-action features for samples s, a
        :param phi_next: the next state action features for samples s', a'
        :param reward: the reward for taking action a in state s
        :return: the parameters theta of the Q-function approximation
        """
        _A = phi.T.dot(phi - self.discount_factor * phi_next)
        _b = (phi * reward).sum(axis=0).T

        _I = np.eye(phi.shape[1])

        _C = np.linalg.solve(phi.T.dot(phi) + self.lstd_regularization_factor * _I, phi.T).T
        _X = _C.dot(_A + self.lstd_regularization_factor * _I)
        _y = _C.dot(_b)

        return np.linalg.solve(_X.T.dot(_X) + self.lstd_projection_regularization_factor * _I, _X.T.dot(_y))


class LeastSquarsTemporalDifferenceOptimized:
    def __init__(self, discount_factor=0.98, regularization_factor=1e-8, projection_regularization_factor=1e-6):
        self.discount_factor = discount_factor
        self.regularization_factor = regularization_factor
        self.projection_regularization_factor = projection_regularization_factor

    def learn_q_function(self, feature_mapping, feature_dim, policy, num_policy_samples,
                         states, actions, rewards, next_states, chunk_size=1):
        # compute some kernel matrices iteratively to save memory
        K = np.zeros((feature_dim, feature_dim))
        K_next = self.regularization_factor * np.eye(feature_dim)
        K_inv = np.eye(feature_dim) / self.regularization_factor
        b = np.zeros((feature_dim, 1))

        for s, a, r, s_ in zip(np_chunks(states, chunk_size), np_chunks(actions, chunk_size),
                               np_chunks(rewards, chunk_size), np_chunks(next_states, chunk_size)):
            phi = feature_mapping(s, a)
            # TODO adapt this function to process chunks of data
            phi_next = np.array([feature_mapping(s_, policy(s_)) for _ in range(num_policy_samples)]).mean(0)
            # if chunk_size is 1:
            #     phi_outer = np.outer(phi, phi)
            #     phi_outer_next = np.outer(phi, phi_next)
            # else:
            phi_outer = phi.T.dot(phi)
            phi_outer_next = phi.T.dot(phi_next)

            K += phi_outer
            K_next += self.discount_factor * phi_outer_next
            # K_inv -= K_inv.dot(phi_outer).dot(K_inv) / (1 + phi.dot(K_inv).dot(phi.T))
            K_inv -= K_inv.dot(phi.T).dot(np.linalg.inv(np.eye(phi.shape[0]) + phi.dot(K_inv).dot(phi.T))).dot(
                phi).dot(K_inv)

            b += phi.T.dot(r)

        A = (K - K_next + self.regularization_factor * np.eye(feature_dim)).dot(K_inv)

        return np.ravel(np.linalg.solve(A.dot(K).dot(A.T) + self.projection_regularization_factor * np.eye(feature_dim),
                                        A.dot(K).dot(K_inv).dot(b)))
