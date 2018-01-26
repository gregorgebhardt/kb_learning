from typing import Tuple
import numpy as np
import scipy.linalg

from sklearn.gaussian_process.kernels import Kernel
from gym import Space

import logging
logger = logging.getLogger(__name__)


class SparseGPPolicy:
    def __init__(self, kernel: Kernel, action_bounds: Tuple[np.ndarray, np.ndarray]):
        # TODO add documentation
        """

        :param kernel:
        :param action_space: gym.Space
        """
        self.gp_prior_variance = 0.01
        self.gp_regularizer = 1e-9
        self.gp_noise_variance = 1e-6

        self.gp_min_variance = 0.0005
        self.use_gp_bug = False

        self.Q_Km = None
        self.alpha = None
        self.trained = False

        self.sparse_states = None
        self.k_cholesky = None
        self.kernel = kernel
        # TODO fix this: don't use the action space here, use bounds
        self.action_bounds = action_bounds
        self.action_dim = action_bounds[0].shape[0]

    def _get_random_actions(self, num_samples=1):
        # sample random numbers in [0.0, 1.0)
        samples = np.random.random((num_samples, self.action_dim))
        # multiply by range
        samples *= self.action_bounds[1] - self.action_bounds[0]
        # add lower bound to the samples
        samples += self.action_bounds[0]

        return samples

    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)

    def train(self, states, actions, weights, sparse_states):
        """

        :param state: N x d_states
        :param actions: N x d_actions
        :param weights: N
        :param sparse_states: M x d_states
        :return:
        """

        self.sparse_states = sparse_states

        weights /= weights.max()

        # kernel matrix on subset of samples
        K_m = self.gp_prior_variance * self.kernel(sparse_states)
        K_mn = self.gp_prior_variance * self.kernel(sparse_states, states)

        # fix cholesky with regularizer
        reg_I = self.gp_regularizer * np.eye(K_m.shape[0])
        while True:
            try:
                # K_m_c = scipy.linalg.cholesky(K_m, lower=True), True
                K_m_c = np.linalg.cholesky(K_m), True
                logger.info('regularization for chol: {}'.format(reg_I[0, 0]))
                break
            except np.linalg.LinAlgError:
                K_m += reg_I
                reg_I *= 2
        else:
            raise Exception("SparseGPPolicy: Cholesky decomposition failed")

        L = self.kernel.diag(states) \
            - np.sum(K_mn * scipy.linalg.cho_solve(K_m_c, K_mn), axis=0).squeeze() \
            + self.gp_noise_variance
        L = np.diag((1 / L) * weights)

        Q = K_m + K_mn.dot(L).dot(K_mn.T)
        self.alpha = np.linalg.solve(Q, K_mn).dot(L).dot(actions)

        self.Q_Km = np.linalg.pinv(K_m) - np.linalg.pinv(Q)

        self.trained = True

    def get_random_action(self):
        return self._get_random_actions()

    def get_mean_action(self, S):
        if not self.trained:
            if len(S.shape) == 1:
                return self.get_random_action()
            else:
                return self._get_random_actions(S.shape[0])

        k = self.kernel(S, self.sparse_states)
        return k.T.dot(self.alpha)

    def sample_actions(self, states):
        if not self.trained:
            if states.ndim > 1:
                return self._get_random_actions(states.shape[0])
            else:
                return self._get_random_actions(1)

        action_dim = self.alpha.shape[1]

        k = self.kernel(states, self.sparse_states) * self.gp_prior_variance

        gp_mean = k.dot(self.alpha)

        gp_sigma = self.gp_prior_variance * self.kernel.diag(states) \
            - np.sum(k.T * self.Q_Km.dot(k.T), axis=0) + self.gp_noise_variance

        if gp_sigma.shape == ():  # single number
            gp_sigma = np.array([gp_sigma])

        gp_sigma[gp_sigma < 0] = 0

        gp_sigma = np.tile(np.sqrt(gp_sigma)[:, np.newaxis], (1, action_dim))

        if self.use_gp_bug:
            gp_sigma += np.sqrt(self.gp_regularizer)
        else:
            gp_sigma = np.sqrt(np.square(gp_sigma) + self.gp_regularizer)
        gp_sigma[gp_sigma < self.gp_min_variance] = self.gp_min_variance

        norm_variates = np.random.normal(0.0, 1.0, (states.shape[0], action_dim))
        action_samples = norm_variates * gp_sigma + gp_mean

        # check samples against bounds from action space
        action_samples = np.minimum(action_samples, self.action_bounds[1])
        action_samples = np.maximum(action_samples, self.action_bounds[0])

        return action_samples

    def __call__(self, states):
        return self.sample_actions(states)
