import numpy as np
import scipy.linalg

from sklearn.gaussian_process.kernels import Kernel
from gym import Space

# def fromSerializableDict(d):
#     return SparseGPPolicy.fromSerializableDict(d)


class SparseGPPolicy:
    def __init__(self, kernel: Kernel, action_space: Space, gp_min_variance=1.0, gp_regularizer=0.05):
        # TODO add documentation
        """

        :param kernel:
        :param action_space: gym.Space
        :param gp_min_variance:
        :param gp_regularizer:
        """
        self.gp_prior_variance = 0.001
        self.gp_regularizer = gp_regularizer  # TODO toolbox / NLopt
        self.SparseGPInducingOutputRegularization = 1e-6

        self.gp_min_variance = gp_min_variance
        self.use_gp_bug = False

        self.alpha = None
        self.trained = False

        self.sparse_states = None
        self.k_cholesky = None
        self.kernel = kernel
        self.action_space = action_space

    def _get_random_actions(self, num_samples=1):
        samples = np.array([*(self.action_space.sample() for _ in range(num_samples))])

        return samples

    def train(self, state, actions, weights, sparse_states):
        """

        :param state: N x d_states
        :param actions: N x d_actions
        :param weights: N
        :param sparse_states: M x d_states
        :return:
        """

        self.sparse_states = sparse_states

        # kernel matrix on subset of samples
        k = self.gp_prior_variance * self.kernel(sparse_states, sparse_states)

        weights /= weights.max()

        for i in range(100):
            try:
                self.k_cholesky = scipy.linalg.cholesky(k + np.eye(k.shape[0]) * self.gp_regularizer * 2**i)
                break
            except scipy.linalg.LinAlgError:
                continue
        else:
            raise Exception("SparseGPPolicy: Cholesky decomposition failed")

        kernel_vectors = self.gp_prior_variance * self.kernel(state, sparse_states)

        _regularizer = 0
        while True:
            try:
                k_cholesky_inv = np.linalg.pinv(self.k_cholesky + _regularizer * np.eye(self.k_cholesky.shape[0]))
                k_cholesky_t_inv = np.linalg.pinv(self.k_cholesky.T + _regularizer * np.eye(self.k_cholesky.shape[0]))
                break
            except scipy.linalg.LinAlgError:
                if _regularizer == 0:
                    _regularizer = 1e-10
                else:
                    _regularizer *= 2

        feature_vectors = kernel_vectors.dot(k_cholesky_inv).dot(k_cholesky_t_inv)
        feature_vectors_w = feature_vectors * weights[:, None]

        x = feature_vectors_w.T.dot(feature_vectors)
        x += np.eye(feature_vectors.shape[1]) * self.SparseGPInducingOutputRegularization
        y = np.linalg.solve(x, feature_vectors_w.T).dot(actions)

        self.alpha = np.linalg.solve(self.k_cholesky, np.linalg.solve(self.k_cholesky.T, y))

        self.trained = True

    def get_random_action(self):
        return self._get_random_actions()

    def get_mean_action(self, S):
        if not self.trained:
            if len(S.shape) == 1:
                return self.get_random_action()
            else:
                return self._get_random_actions(S.shape[0])

        k_ = self.gp_prior_variance * self.kernel(S, self.sparse_states)
        return k_.dot(self.alpha)

    def sample_actions(self, states, low_memory=False):
        if not self.trained:
            if states.ndim > 1:
                return self._get_random_actions(states.shape[0])
            else:
                return self._get_random_actions(1)

        action_dim = self.alpha.shape[1]

        if low_memory:
            # TODO check against IP implementation
            k_ = self.gp_prior_variance * self.kernel(states, self.sparse_states)
        else:
            k_ = self.gp_prior_variance * self.kernel(states, self.sparse_states)

        gp_mean = k_.dot(self.alpha)

        temp = np.linalg.solve(self.k_cholesky.T, k_.T)
        temp = np.square(temp.T)
        gp_sigma = temp.sum(1)

        kernel_self = self.gp_prior_variance * self.kernel.diag(states)
        gp_sigma = kernel_self.squeeze() - gp_sigma.squeeze()

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
