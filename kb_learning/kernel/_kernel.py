import numpy as np

from sklearn.metrics.pairwise import pairwise_distances

from . import KilobotSwarmKernel, ExponentialQuadraticKernel

# from ._kilobot_kernel_numba import KilobotKernel

# class Kernel:
#     # Evaluates kernel for all elements
#     # Compute gram matrix of the form
#     #  -----------------------------
#     #  | k(x₁,y₁) | k(x₁,y₂) | ... |
#     #  -----------------------------
#     #  | k(x₂,y₁) | k(x₂,y₂) | ... |
#     #  -----------------------------
#     #  | ...      | ...      | ... |
#     #  -----------------------------
#     # if y=None, K(x,x) is computed
#     def get_gram_matrix(self, a, b=None):
#         pass
#
#     def get_gram_matrix_multi(self, a, b=None):
#         pass
#
#     # Returns the diagonal of the gram matrix
#     # which means the kernel is evaluated between every data point and itself
#     def get_gram_diag(self, data):
#         diag = np.zeros(data.shape[0])
#
#         for i in range(data.shape[0]):
#             diag[i] = self.get_gram_matrix(data[i, :])
#
#
# class ExponentialQuadraticKernel(Kernel):
#     def __init__(self, normalized=0):
#         self.normalized = normalized
#         self.bandwidth = None
#
#     def get_gram_matrix(self, a, b=None):
#         # assert a.dtype == DTYPE
#
#         if np.isscalar(self.bandwidth):
#             q = np.eye(a.shape[1]) / (self.bandwidth ** 2)
#         else:
#             assert (a.shape[1] == self.bandwidth.shape[0])
#             q = np.diag((np.ones(self.bandwidth.shape[0]) / (self.bandwidth ** 2)))
#
#         aq = a.dot(q)
#         aq_a = np.sum(aq * a, axis=1)
#         if b is None:
#             sqdist = aq_a[:, np.newaxis] + aq_a - 2 * aq.dot(a.T)
#         else:
#             bq_b = np.sum(b.dot(q) * b, axis=1)
#             # Equivalent to MATLAB bsxfun(@plus, ..)
#             sqdist = aq_a[:, np.newaxis] + bq_b - 2 * aq.dot(b.T)
#         K = np.exp(-0.5 * sqdist)
#
#         if self.normalized:
#             K = K / np.sqrt(np.prod(self.bandwidth) ** 2 * (2 * np.pi) ** a.shape[1])
#
#         return K
#
#     def get_gram_matrix_multi(self, a, b=None):
#         """
#
#         :param a: q x n x d matrix of kilobot positions
#         :param b: r x m x d matrix of kilobot positions
#         :return:  q x r x n x m matrix if a and b are given, q x n x n if only a is given
#         """
#         # assert a.dtype == DTYPE
#
#         # if np.isscalar(self.bandwidth):
#         #     q = np.eye(a.shape[-1]) / (self.bandwidth ** 2)
#         # else:
#         #     # assert (a.shape[-1] == self.bandwidth.shape[0])
#         #     q = np.diag((np.ones(self.bandwidth.shape[0]) / (self.bandwidth ** 2)))
#         bw = np.array(1/ (self.bandwidth**2), ndmin=3)
#
#         aq = a * bw
#         aq_a = np.sum(aq * a, axis=-1)
#         if b is None:
#             sq_dist = aq_a[:, None, :] + aq_a[:, :, None]
#             sq_dist -= 2 * np.einsum('qid,qjd->qij', aq, a)
#         else:
#             # assert b.dtype == DTYPE
#             bq_b = np.sum((b * bw) * b, axis=-1)
#             sq_dist = aq_a[:, None, :, None] + bq_b[None, :, None, :]
#             sq_dist -= 2 * np.einsum('qid,rjd->qrij', aq, b)
#         K = np.exp(-0.5 * sq_dist)
#
#         return K
#
#     def get_gram_diag(self, data):
#         return np.ones(data.shape[0])
#
#     def get_derivation_param(self, data):
#         """
#
#         :param data: n x d
#         :return: p x n x n, where p is the number of bandwidth parameters
#         """
#         gram_matrix = self.get_gram_matrix(data, data)
#         gradient_matrices = np.zeros([self.bandwidth.shape[0], gram_matrix.shape[0], gram_matrix.shape[1]])
#
#         for dim in range(self.bandwidth.shape[0]):
#             sqdist = (data[:, dim, np.newaxis] - data[:, dim]) ** 2
#             gradient_matrices[dim, :, :] = gram_matrix * (sqdist / (self.bandwidth[dim] ** 3))
#
#         return gradient_matrices, gram_matrix
#
#     def get_derivation_data(self, ref_data, cur_data):
#         """
#
#         :param ref_data: n x d
#         :param cur_data: m x d
#         :return: m x d x n
#         """
#         gram_matrix = self.get_gram_matrix(ref_data, cur_data)
#         q = np.diag((np.ones(self.bandwidth.shape[0]) / (self.bandwidth ** 2)))
#
#         scaleddist = -2 * (cur_data.dot(q) - ref_data.dot(q))
#
#         return gram_matrix.T[np.newaxis, ...] * scaleddist
#
#
# class KilobotKernel:
#     def __init__(self, bandwidth_factor=1.):
#         self._kernel_func = ExponentialQuadraticKernel()
#         self.bandwidth_factor = bandwidth_factor
#
#     def _compute_kb_distance(self, k1, k2):
#         """Computes the kb distance matrix between any configuration in k1 and any configuration in k2.
#
#         :param k1: q x 2*d1 matrix of q configurations with each d1 kilobots
#         :param k2: r x 2*d2 matrix of r configurations with each d2 kilobots
#         :return: q x r matrix with the distances between the configurations in k1 and k2
#         """
#
#         # number of samples in k1
#         q = k1.shape[0]
#
#         # number of samples in k2
#         r = k2.shape[0]
#
#         # number of kilobots in k1
#         num_kb_1 = k1.shape[1] // 2
#         # number of kilobots in k2
#         num_kb_2 = k2.shape[1] // 2
#
#         # reshape matrices
#         # cdef k1_reshaped = np.c_[k1.flat[0::2], k1.flat[1::2]]
#         # cdef k2_reshaped = np.c_[k2.flat[0::2], k2.flat[1::2]]
#
#         k1_reshaped = k1.reshape(q, num_kb_1, 2)
#         k2_reshaped = k2.reshape(r, num_kb_2, 2)
#
#         k_nm = np.empty((q, r))
#         k_n = self._kernel_func.get_gram_matrix_multi(k1_reshaped).sum(axis=(1, 2)) / (num_kb_1 ** 2)
#         k_m = self._kernel_func.get_gram_matrix_multi(k2_reshaped).sum(axis=(1, 2)) / (num_kb_2 ** 2)
#         chunk_size = 50
#         for i in range(0, r, chunk_size):
#             k_nm[:, i:i+chunk_size] = self._kernel_func.get_gram_matrix_multi(
#                 k1_reshaped, k2_reshaped[i:i+chunk_size]).sum(axis=(2, 3))
#         # k_nm[k_nm < 1e-12] = 0
#         k_nm /= (0.5 * num_kb_1 * num_kb_2)
#
#         return k_n[:, np.newaxis] + k_m[np.newaxis, :] - k_nm
#
#     def __call__(self, k1, k2, eval_gradient=False):
#         return self._compute_kb_distance(k1, k2)
#
#     def diag(self, X):
#         if len(X.shape) > 1:
#             return np.zeros((X.shape[0],))
#         return np.zeros(1)
#
#     def is_stationary(self):
#         return True
#
#     def set_bandwidth(self, bandwidth):
#         bandwidth = bandwidth.reshape((-1, 2)).mean(axis=0)
#         self._kernel_func.bandwidth = self.bandwidth_factor * bandwidth
#
#     def set_params(self, **params):
#         if 'bandwidth' in params:
#             self.set_bandwidth(params['bandwidth'])


class MahaKernel:
    def __init__(self, bandwidth_factor=1.0, num_processes=1):
        self.bandwidth = 1.
        if type(bandwidth_factor) in [float, int]:
            self.bandwidth_factor = bandwidth_factor
        elif type(bandwidth_factor) in [list, tuple]:
            self.bandwidth_factor = np.array(bandwidth_factor)
        else:
            self.bandwidth_factor = bandwidth_factor
        self.num_processes = num_processes

        self._preprocessor = None

    def __call__(self, X, Y=None, eval_gradient=False):
        if self._preprocessor:
            X = self._preprocessor(X)
            if Y is not None:
                Y = self._preprocessor(Y)
        return pairwise_distances(X, Y, metric='mahalanobis', n_jobs=self.num_processes, VI=self.bandwidth)

    def diag(self, X):
        return np.zeros((X.shape[0],))

    def is_stationary(self):
        return True

    def set_params(self, **params):
        if 'bandwidth' in params:
            if np.isscalar(params['bandwidth']):
                self.bandwidth = 1 / (self.bandwidth_factor * params['bandwidth'])
            else:
                self.bandwidth = np.diag(1 / (self.bandwidth_factor * params['bandwidth']))
        if 'num_processes' in params:
            self.num_processes = params['num_processes']


class MeanSwarmKernel(MahaKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from ._preprocessors import compute_mean_position
        self._preprocessor = compute_mean_position


class MeanCovSwarmKernel(MeanSwarmKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from ._preprocessors import compute_mean_and_cov_position
        self._preprocessor = compute_mean_and_cov_position


class KilobotEnvKernel:
    KBKernel = KilobotSwarmKernel

    def __init__(self, bandwidth_factor_ed=1.0, bandwidth_factor_kb=1.0, extra_dims=2, weight=.5, num_processes=1):
        self._kb_kernel = self.KBKernel(bandwidth_factor=bandwidth_factor_kb)
        self._extra_dims = extra_dims
        self._l_kernel = MahaKernel(bandwidth_factor=bandwidth_factor_ed, num_processes=num_processes)
        self._weight = weight

    @property
    def num_processes(self):
        return self._l_kernel.num_processes

    def __call__(self, X, Y=None, eval_gradient=False):
        k1 = X[:, :-self._extra_dims]
        l1 = X[:, -self._extra_dims:]

        if Y is None:
            k_l = self._l_kernel(l1)
            k_kb = self._kb_kernel(k1, k1)
        else:
            k2 = Y[:, :-self._extra_dims]
            l2 = Y[:, -self._extra_dims:]

            k_l = self._l_kernel(l1, l2)
            k_kb = self._kb_kernel(k1, k2)

        return np.exp(-self._weight * k_l - (1 - self._weight) * k_kb)

    def diag(self, X):
        kb_dims = X[:, :-self._extra_dims]
        light_dims = X[:, -self._extra_dims:]
        return np.exp(-self._weight * self._l_kernel.diag(light_dims)
                      - (1 - self._weight) * self._kb_kernel.diag(kb_dims))

    def is_stationary(self):
        return self._l_kernel.is_stationary() and self._kb_kernel.is_stationary()

    def set_params(self, **params):
        if 'bandwidth' in params:
            bandwidth_kb = params['bandwidth'][:-self._extra_dims]
            bandwidth_l = params['bandwidth'][-self._extra_dims:]

            self._kb_kernel.set_params(bandwidth=bandwidth_kb)
            self._l_kernel.set_params(bandwidth=bandwidth_l)

        if 'num_processes' in params:
            self._l_kernel.num_processes = params['num_processes']


class MeanEnvKernel(KilobotEnvKernel):
    KBKernel = MeanSwarmKernel


class MeanCovEnvKernel(KilobotEnvKernel):
    KBKernel = MeanCovSwarmKernel


class KilobotEnvKernelWithWeight(KilobotEnvKernel):
    def __init__(self, weight_dim, bandwidth_factor_weigth=1.0, num_processes=1, *args, **kwargs):
        super().__init__(*args, num_processes=num_processes, **kwargs)
        self._weight_dim = weight_dim

        self._weight_kernel = MahaKernel(bandwidth_factor=bandwidth_factor_weigth, num_processes=num_processes)

    def __call__(self, X, Y=None, *args, **kwargs):
        weights_X = X[:, self._weight_dim]
        if self._weight_dim == -1 or self._weight_dim == X.shape[-1] - 1:
            other_X = X[:, :self._weight_dim]
        else:
            other_X = np.c_[X[:, :self._weight_dim], X[:, self._weight_dim+1:]]

        if Y is not None:
            weights_Y = Y[:, self._weight_dim]
            if self._weight_dim == -1 or self._weight_dim == Y.shape[-1] - 1:
                other_Y = Y[:, :self._weight_dim]
            else:
                other_Y = np.c_[Y[:, :self._weight_dim], Y[:, self._weight_dim + 1:]]
            K_weights = np.exp(self._weight_kernel(weights_X[:, None], weights_Y[:, None]))
        else:
            K_weights = np.exp(self._weight_kernel(weights_X[:, None]))
            other_Y = None

        K_other = super().__call__(other_X, other_Y)

        return K_weights * K_other

    def set_params(self, **params):
        if 'bandwidth' in params:
            bandwidth = params['bandwidth']
            bandwidth_weight = bandwidth[self._weight_dim]
            if self._weight_dim == -1 or self._weight_dim == bandwidth.shape[-1] - 1:
                bandwidth_other = bandwidth[:self._weight_dim]
            else:
                bandwidth_other = np.r_[bandwidth[:self._weight_dim], bandwidth[self._weight_dim+1:]]

            self._weight_kernel.set_params(bandwidth=bandwidth_weight)
            params['bandwidth'] = bandwidth_other

        if 'num_processes' in params:
            self._weight_kernel.num_processes = params['num_processes']

        super(KilobotEnvKernelWithWeight, self).set_params(**params)


class MeanEnvKernelWithWeight(KilobotEnvKernelWithWeight):
    KBKernel = MeanSwarmKernel


class MeanCovEnvKernelWithWeight(KilobotEnvKernelWithWeight):
    KBKernel = MeanCovSwarmKernel
