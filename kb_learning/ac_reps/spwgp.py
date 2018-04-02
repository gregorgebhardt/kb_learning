from typing import Tuple
import numpy as np
from scipy import linalg

import logging
logger = logging.getLogger('kb_learning.gp')


class SparseWeightedGP:
    def __init__(self, kernel, output_bounds: Tuple[np.ndarray, np.ndarray]=None):
        # TODO add documentation
        """

        :param kernel:
        :param output_bounds:
        """
        self.gp_prior_variance = 0.01
        self.gp_prior_mean = None
        self.gp_noise_variance = 1e-6
        self.gp_min_variance = 0.0005
        self.gp_cholesky_regularizer = 1e-9

        self.Q_Km = None
        self.alpha = None
        self.trained = False

        self.sparse_inputs = None
        self.k_cholesky = None
        self.kernel = kernel
        self.output_bounds = output_bounds
        self.output_bounds_enforce = False

    @property
    def output_dim(self):
        if self.output_bounds:
            return self.output_bounds[0].shape[0]

    def train(self, inputs, outputs, weights, sparse_inputs):
        """
        :param inputs: N x d_input
        :param outputs: N x d_output
        :param weights: N
        :param sparse_inputs: M x d_output
        :return:
        """
        if outputs.ndim == 1:
            outputs = outputs.reshape((-1, 1))

        self.sparse_inputs = sparse_inputs

        weights /= weights.max()

        # kernel matrix on subset of samples
        K_m = self.gp_prior_variance * self.kernel(sparse_inputs)
        K_mn = self.gp_prior_variance * self.kernel(sparse_inputs, inputs)

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

        L = self.gp_prior_variance * self.kernel.diag(inputs) \
            - np.sum(K_mn * linalg.cho_solve(K_m_c, K_mn), axis=0).squeeze() \
            + self.gp_noise_variance * (1 / weights)
        L = 1 / L

        Q = K_m + (K_mn * L).dot(K_mn.T)
        if self.gp_prior_mean:
            outputs -= self.gp_prior_mean(inputs)

        self.alpha = (np.linalg.solve(Q, K_mn) * L).dot(outputs)

        self.Q_Km = np.linalg.pinv(K_m) - np.linalg.pinv(Q)

        self.trained = True

    def get_mean(self, inputs, return_k=False):
        if self.trained:
            k = self.gp_prior_variance * self.kernel(inputs, self.sparse_inputs)
            mean = k.dot(self.alpha)
        else:
            k = None
            mean = np.zeros((inputs.shape[0], self.output_dim))

        if self.gp_prior_mean:
            mean += self.gp_prior_mean(inputs)

        if return_k:
            return mean, k

        return mean

    def get_mean_sigma(self, inputs):
        mean, k = self.get_mean(inputs, True)

        if self.trained:
            sigma_sqr = self.gp_prior_variance * self.kernel.diag(inputs) - np.sum(k.T * self.Q_Km.dot(k.T), axis=0) \
                        + self.gp_noise_variance
        else:
            sigma_sqr = self.gp_prior_variance + self.gp_noise_variance

        if np.isscalar(sigma_sqr):  # single number
            sigma_sqr = np.array([sigma_sqr])

        sigma_sqr[sigma_sqr < self.gp_min_variance] = self.gp_min_variance

        sigma = np.sqrt(sigma_sqr)

        return mean, sigma

    def sample(self, inputs):
        if inputs.ndim == 1:
            inputs = np.array([inputs])

        gp_mean, gp_sigma = self.get_mean_sigma(inputs)

        if gp_mean.ndim > 1:
            samples = np.random.normal(gp_mean, np.sqrt(gp_sigma[:, None]))
        else:
            samples = np.random.normal(gp_mean, np.sqrt(gp_sigma))

        if self.output_bounds_enforce:
            # check samples against bounds from action space
            samples = np.minimum(samples, self.output_bounds[1])
            samples = np.maximum(samples, self.output_bounds[0])

        return np.reshape(samples, (-1, self.output_dim))

    def __call__(self, inputs):
        return self.sample(inputs)
