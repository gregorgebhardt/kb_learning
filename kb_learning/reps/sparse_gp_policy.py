import numpy as np
import scipy.linalg

from sklearn.gaussian_process.kernels import Kernel
from gym import Space

import logging
logger = logging.getLogger(__name__)


class SparseGPPolicy:
    def __init__(self, kernel: Kernel, action_space: Space):
        # TODO add documentation
        """

        :param kernel:
        :param action_space: gym.Space
        """
        self.gp_prior_variance = 0.001
        self.gp_regularizer = 1e-9
        self.gp_noise_variance = 1e-6

        self.gp_min_variance = 0.005
        self.use_gp_bug = False

        self.Q_Km = None
        self.alpha = None
        self.trained = False

        self.sparse_states = None
        self.k_cholesky = None
        self.kernel = kernel
        # TODO fix this: don't use the action space here, use bounds
        self.action_space = action_space

    def _get_random_actions(self, num_samples=1):
        samples = np.array([*(self.action_space.sample() for _ in range(num_samples))])

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
        # sparse_weights = weights[sparse_index]
        # K_m = K_m_c[0].dot(np.diag(sparse_weights)).dot(K_m_c[0].T)

        L = self.kernel.diag(states) \
            - np.sum(K_mn * scipy.linalg.cho_solve(K_m_c, K_mn), axis=0).squeeze() \
            + self.gp_noise_variance
        L = np.diag((1 / L) * weights)

        Q = K_m + K_mn.dot(L).dot(K_mn.T)
        self.alpha = np.linalg.solve(Q, K_mn).dot(L).dot(actions)

        self.Q_Km = np.linalg.pinv(K_m) - np.linalg.pinv(Q)

        # k = self.gp_prior_variance * self.kernel(sparse_states, sparse_states)

        # for i in range(100):
        #     try:
        #         self.k_cholesky = scipy.linalg.cholesky(k + np.eye(k.shape[0]) * self.gp_regularizer * 2**i)
        #         break
        #     except scipy.linalg.LinAlgError:
        #         continue
        # else:
        #     raise Exception("SparseGPPolicy: Cholesky decomposition failed")
        #
        # kernel_vectors = self.gp_prior_variance * self.kernel(state, sparse_states)
        #
        # _regularizer = 0
        # while True:
        #     try:
        #         k_cholesky_inv = np.linalg.pinv(self.k_cholesky + _regularizer * np.eye(self.k_cholesky.shape[0]))
        #         k_cholesky_t_inv = np.linalg.pinv(self.k_cholesky.T + _regularizer * np.eye(self.k_cholesky.shape[0]))
        #         break
        #     except scipy.linalg.LinAlgError:
        #         if _regularizer == 0:
        #             _regularizer = 1e-10
        #         else:
        #             _regularizer *= 2
        #
        # feature_vectors = kernel_vectors.dot(k_cholesky_inv).dot(k_cholesky_t_inv)
        # feature_vectors_w = feature_vectors * weights[:, None]
        #
        # x = feature_vectors_w.T.dot(feature_vectors)
        # x += np.eye(feature_vectors.shape[1]) * self.SparseGPInducingOutputRegularization
        # y = np.linalg.solve(x, feature_vectors_w.T).dot(actions)
        #
        # self.alpha = np.linalg.solve(self.k_cholesky, np.linalg.solve(self.k_cholesky.T, y))

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

        # temp = np.linalg.solve(self.k_cholesky.T, k_.T)
        # temp = np.square(temp.T)
        # gp_sigma = temp.sum(1)
        #
        # kernel_self = self.gp_prior_variance * self.kernel.diag(states)
        # gp_sigma = kernel_self.squeeze() - gp_sigma.squeeze()

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
        action_samples = np.minimum(action_samples, self.action_space.high)
        action_samples = np.maximum(action_samples, self.action_space.low)

        return action_samples

    def __call__(self, states):
        return self.sample_actions(states)
