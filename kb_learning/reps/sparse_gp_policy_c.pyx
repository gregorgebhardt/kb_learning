import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float_t DTYPE_t

import scipy.linalg

# def fromSerializableDict(d):
#     return SparseGPPolicy.fromSerializableDict(d)


cdef class SparseGPPolicy:
    cdef double gp_prior_variance
    cdef double gp_regularizer
    cdef double gp_inducing_output_regularization
    cdef double gp_min_variance
    cdef np.ndarray alpha
    cdef unsigned trained
    cdef np.ndarray sparse_states
    cdef np.ndarray k_cholesky
    cdef kernel
    cdef action_space

    def __init__(self, kernel, action_space, gp_min_variance=1.0, gp_regularizer=0.05):
        # TODO add documentation
        """

        :param kernel:
        :param action_space: gym.Space
        :param gp_min_variance:
        :param gp_regularizer:
        """
        self.gp_prior_variance = 0.001
        self.gp_regularizer = gp_regularizer  # TODO toolbox / NLopt
        self.gp_inducing_output_regularization = 1e-6

        self.gp_min_variance = gp_min_variance
        # self.use_gp_bug = False

        self.alpha = None
        self.trained = False

        self.sparse_states = None
        self.k_cholesky = None
        self.kernel = kernel
        self.action_space = action_space

    cdef _get_random_actions(self, num_samples=1):
        cdef list samples = []

        for i in range(num_samples):
            samples.append(self.action_space.sample())

        return np.array(samples)

    cpdef train(self, np.ndarray states, np.ndarray actions, np.ndarray weights, np.ndarray sparse_states):
        self.sparse_states = sparse_states

        # kernel matrix on subset of samples
        cdef np.ndarray gram_matrix = self.gp_prior_variance * self.kernel(sparse_states, sparse_states)
        # cdef np.ndarray I = np.eye(sparse_states.shape[0])

        weights /= weights.max()
        cdef np.ndarray I_w = np.diag(weights)

        cdef int i
        for i in range(100):
            try:
                self.k_cholesky = scipy.linalg.cho_factor(gram_matrix + I_w * self.gp_regularizer * 2**i)
                break
            except scipy.linalg.LinAlgError:
                continue
        else:
            raise Exception("SparseGPPolicy: Cholesky decomposition failed")

        cdef np.ndarray kernel_vectors = self.gp_prior_variance * self.kernel(states, sparse_states)
        cdef np.ndarray feature_vectors = scipy.linalg.cho_solve(self.k_cholesky, kernel_vectors)
        cdef np.ndarray feature_vectors_w = feature_vectors * weights.T

        cdef np.ndarray x = feature_vectors_w.T.dot(feature_vectors)
        x += np.eye(feature_vectors.shape[1]) * self.gp_inducing_output_regularization
        cdef np.ndarray y = np.linalg.solve(x, feature_vectors_w.T.dot(actions))

        self.alpha = scipy.linalg.cho_solve(self.k_cholesky, y)

        self.trained = True

    def get_random_action(self):
        return self._get_random_actions()

    cpdef get_mean_action(self, np.ndarray states):
        if not self.trained:
            if states.ndim == 1:
                return self.get_random_action()
            else:
                return self._get_random_actions(states.shape[0])

        cdef np.ndarray k_ = self.gp_prior_variance * self.kernel(states, self.sparse_states)
        return k_.dot(self.alpha)

    cdef sample_actions(self, np.ndarray states, unsigned low_memory=False):
        if not self.trained:
            if states.ndim > 1:
                return self._get_random_actions(states.shape[0])
            else:
                return self._get_random_actions(1)

        cdef int action_dim = self.alpha.shape[1]

        cdef np.ndarray k_
        if low_memory:
            # TODO check against IP implementation
            k_ = self.gp_prior_variance * self.kernel(states, self.sparse_states)
        else:
            k_ = self.gp_prior_variance * self.kernel(states, self.sparse_states)

        cdef np.ndarray gp_mean = k_.dot(self.alpha)

        cdef np.ndarray temp = np.linalg.solve(self.k_cholesky.T, k_.T)
        temp = np.square(temp.T)
        cdef np.ndarray gp_sigma = temp.sum(1)

        cdef np.ndarray kernel_self = self.gp_prior_variance * self.kernel.diag(states)
        gp_sigma = np.array([kernel_self.squeeze() - gp_sigma.squeeze()])

        # if gp_sigma.shape == ():  # single number
        #     gp_sigma = np.array([gp_sigma])

        gp_sigma[gp_sigma < 0] = 0

        gp_sigma = np.tile(np.sqrt(gp_sigma)[:, np.newaxis], (1, action_dim))

        gp_sigma = np.sqrt(np.square(gp_sigma) + self.gp_regularizer)
        gp_sigma[gp_sigma < self.gp_min_variance] = self.gp_min_variance

        cdef np.ndarray action_samples = np.random.normal(0.0, 1.0, (states.shape[0], action_dim))
        action_samples *= gp_sigma
        action_samples += gp_mean

        # check samples against bounds from action space
        action_samples = np.minimum(action_samples, self.action_space.high)
        action_samples = np.maximum(action_samples, self.action_space.low)

        return action_samples

    def __call__(self, np.ndarray states):
        return self.sample_actions(states)
