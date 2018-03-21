from typing import Tuple
import numpy as np
import scipy.linalg

import logging
logger = logging.getLogger('kb_learning.gp')


class SparseGPPolicy:
    def __init__(self, kernel, action_bounds: Tuple[np.ndarray, np.ndarray]=None):
        # TODO add documentation
        """

        :param kernel:
        :param action_bounds:
        """
        self.gp_prior_variance = 0.01
        self.gp_prior_mean = None
        self.gp_noise_variance = 1e-6
        self.gp_min_variance = 0.0005
        self.gp_cholesky_regularizer = 1e-9

        self.Q_Km = None
        self.alpha = None
        self.trained = False

        self.sparse_states = None
        self.k_cholesky = None
        self.kernel = kernel
        self.action_bounds = action_bounds
        self.action_bounds_enforce = False

    @property
    def action_dim(self):
        if self.action_bounds:
            return self.action_bounds[0].shape[0]

    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)

    def train(self, states, actions, weights, sparse_states):
        """

        :param states: N x d_states
        :param actions: N x d_actions
        :param weights: N
        :param sparse_states: M x d_states
        :return:
        """
        if actions.ndim == 1:
            actions = actions.reshape((-1, 1))

        self.sparse_states = sparse_states

        weights /= weights.max()

        # kernel matrix on subset of samples
        K_m = self.gp_prior_variance * self.kernel(sparse_states)
        K_mn = self.gp_prior_variance * self.kernel(sparse_states, states)

        # fix cholesky with regularizer
        reg_I = self.gp_cholesky_regularizer * np.eye(K_m.shape[0])
        while True:
            try:
                K_m_c = np.linalg.cholesky(K_m), True
                logger.debug('regularization for cholesky: {}'.format(reg_I[0, 0]))
                break
            except np.linalg.LinAlgError:
                K_m += reg_I
                reg_I *= 2
        else:
            raise Exception("SparseGPPolicy: Cholesky decomposition failed")

        L = self.gp_prior_variance * self.kernel.diag(states) \
            - np.sum(K_mn * scipy.linalg.cho_solve(K_m_c, K_mn), axis=0).squeeze() \
            + self.gp_noise_variance * (1 / weights)
        L = np.diag(1 / L)

        Q = K_m + K_mn.dot(L).dot(K_mn.T)
        if self.gp_prior_mean:
            actions -= self.gp_prior_mean(states)

        self.alpha = np.linalg.solve(Q, K_mn).dot(L).dot(actions)

        self.Q_Km = np.linalg.pinv(K_m) - np.linalg.pinv(Q)

        self.trained = True

    def get_mean_action(self, states, return_k=False):
        if self.trained:
            k = self.gp_prior_variance * self.kernel(states, self.sparse_states)
            mean_action = k.dot(self.alpha)
        else:
            k = None
            mean_action = np.zeros((states.shape[0], self.action_dim))

        if self.gp_prior_mean:
            mean_action += self.gp_prior_mean(states)

        if return_k:
            return mean_action, k

        return mean_action

    def get_mean_sigma_action(self, states):
        mean_action, k = self.get_mean_action(states, True)

        if self.trained:
            sigma_sqr = self.gp_prior_variance * self.kernel.diag(states) - np.sum(k.T * self.Q_Km.dot(k.T), axis=0) \
                        + self.gp_noise_variance
        else:
            sigma_sqr = self.gp_prior_variance + self.gp_noise_variance

        if np.isscalar(sigma_sqr):  # single number
            sigma_sqr = np.array([sigma_sqr])

        sigma_sqr[sigma_sqr < self.gp_min_variance] = self.gp_min_variance

        sigma_action = np.sqrt(sigma_sqr)

        return mean_action, sigma_action

    def sample_actions(self, states):
        if states.ndim == 1:
            states = np.array([states])

        gp_mean, gp_sigma = self.get_mean_sigma_action(states)

        if gp_mean.ndim > 1:
            action_samples = np.random.normal(gp_mean, gp_sigma[:, None])
        else:
            action_samples = np.random.normal(gp_mean, gp_sigma)

        if self.action_bounds_enforce:
            # check samples against bounds from action space
            action_samples = np.minimum(action_samples, self.action_bounds[1])
            action_samples = np.maximum(action_samples, self.action_bounds[0])

        return np.reshape(action_samples, (-1, self.action_dim))

    def __call__(self, states):
        return self.sample_actions(states)
