import numpy as np

from scipy.spatial.distance import pdist, squareform

from sklearn.gaussian_process.kernels import Kernel, RBF
from sklearn.metrics.pairwise import pairwise_distances


class KilobotKernel(Kernel):
    def __init__(self, bandwidth=1., num_processes=1):
        self._kernel_func = RBF(bandwidth)
        self.num_processes = num_processes

    def _compute_kb_distance(self, k1, k2):
        """Computes the kb distance matrix between any configuration in k1 and any configuration in k2.

        :param k1: n x 2*d1 matrix of n configurations with each d1 kilobots
        :param k2: m x 2*d2 matrix of m configurations with each d2 kilobots
        :return: n x m matrix with the distances between the configurations in k1 and k2
        """
        # number of kilobots in A
        n = k1.shape[1] // 2

        # number of kilobots in B
        m = k2.shape[1] // 2

        # compute the kernel values within each configuration of k1
        k_n = np.empty((k1.shape[0], 1))
        for i in range(int(k1.shape[0])):
            k1_reshaped = np.reshape(k1[i, :], (n, 2))
            k_n[i] = self._kernel_func(k1_reshaped).sum()

        k_n /= n ** 2

        # compute the kernel values within each configuration of k2
        k_m = np.empty((1, k2.shape[0]))
        for i in range(k2.shape[0]):
            k2_reshaped = np.reshape(k2[i, :], (m, 2))
            k_m[0, i] = self._kernel_func(k2_reshaped).sum()

        k_m /= m ** 2

        k1_reshaped = np.c_[k1.flat[0::2], k1.flat[1::2]]
        k2_reshaped = np.c_[k2.flat[0::2], k2.flat[1::2]]

        # compute kernel between all kilobot positions
        k_nm = self._kernel_func(k1_reshaped, k2_reshaped)
        # add column and row of zeros
        k_nm = np.c_[np.zeros((k_nm.shape[0] + 1, 1)), np.r_[np.zeros((1, k_nm.shape[1])), k_nm]]
        # compute the cumulative sum in both directions
        k_nm = k_nm.cumsum(axis=0).cumsum(axis=1)

        # We are interested in the cumsum of the block matrices for each combination of configuration. To get this
        # cumsum for a certain block, we take the cumsum at the lower-right entry of said block and subtract the
        # cumsum at the lower-right entry of the block at its left and at its top. Since both of these two cumsums
        # include the cumsum of the whole submatrix before the block, we need to add this value again.
        k_nm = k_nm[n::n, m::m] - k_nm[0:-1:n, m::m] - k_nm[n::n, 0:-1:m] + k_nm[0:-1:n, 0:-1:m]
        k_nm / (0.5 * m * n)

        return k_n + k_m - k_nm

    def __call__(self, X, Y=None, eval_gradient=False):
        if self.num_processes > 1:
            return pairwise_distances(X, Y, self._compute_kb_distance, n_jobs=self.num_processes)
        else:
            if Y is None:
                return squareform(pdist(X, self._compute_kb_distance))
            return self._compute_kb_distance(X, Y)

    def diag(self, X):
        return np.ones((X.shape[0], 1))

    def is_stationary(self):
        return True

    def set_bandwidth(self, bandwidth):
        bandwidth = bandwidth.reshape((-1, 2)).mean(axis=0)
        self._kernel_func.set_params(length_scale=bandwidth)

    def set_params(self, **params):
        if 'bandwidth' in params:
            self.set_bandwidth(params['bandwidth'])
        if 'num_processes' in params:
            self.num_processes = params['num_processes']


class MahaKernel(Kernel):
    def __init__(self, bandwidth=1.0, num_processes=1):
        if type(bandwidth) in [float, int]:
            self.bandwidth = 1 / bandwidth
        else:
            self.bandwidth = np.diag(1 / bandwidth)
        self.num_processes = num_processes

    def __call__(self, X, Y=None, eval_gradient=False):
        return pairwise_distances(X, Y, metric='mahalanobis', n_jobs=self.num_processes, VI=self.bandwidth)

    def diag(self, X):
        return np.zeros((X.shape[0], 1))

    def is_stationary(self):
        return True

    def set_params(self, **params):
        if 'bandwidth' in params:
            if type(params['bandwidth']) in [float, int]:
                self.bandwidth = 1 / params['bandwidth']
            else:
                self.bandwidth = np.diag(1 / params['bandwidth'])
        if 'num_processes' in params:
            self.num_processes = params['num_processes']


class StateKernel(Kernel):
    _extra_dims = 2

    def __init__(self, bandwidth_light=1.0, bandwidth_kb=1.0, weight=.5, num_processes=1):
        self._kb_kernel = KilobotKernel(bandwidth_kb, num_processes=num_processes)
        self._l_kernel = MahaKernel(bandwidth_light, num_processes=num_processes)
        self._weight = weight

    @property
    def num_processes(self):
        return self._kb_kernel.num_processes

    @num_processes.setter
    def num_processes(self, num_processes):
        self._kb_kernel.num_processes = num_processes
        self._l_kernel.num_processes = num_processes

    def __call__(self, X, Y=None, eval_gradient=False):
        k1 = X[:, :-self._extra_dims]
        l1 = X[:, -self._extra_dims:]

        if Y is None:
            k_l = self._l_kernel(l1)
            k_kb = self._kb_kernel(k1)
        else:
            k2 = Y[:, :-self._extra_dims]
            l2 = Y[:, -self._extra_dims:]

            k_l = self._l_kernel(l1, l2)
            k_kb = self._kb_kernel(k1, k2)

        return np.exp(-self._weight * k_l - (1 - self._weight) * k_kb)

    def diag(self, X):
        light_dims = X[:, :self._extra_dims]
        kb_dims = X[:, self._extra_dims:]
        return np.exp(-self._weight * self._l_kernel.diag(light_dims) - (1 - self._weight) * self._kb_kernel(kb_dims))

    def is_stationary(self):
        return self._l_kernel.is_stationary() and self._kb_kernel.is_stationary()

    def set_params(self, **params):
        if 'bandwidth' in params:
            bandwidth_kb = params['bandwidth'][:-self._extra_dims]
            bandwidth_l = params['bandwidth'][-self._extra_dims:]

            self._kb_kernel.set_params(bandwidth=bandwidth_kb)
            self._l_kernel.set_params(bandwidth=bandwidth_l)


class StateActionKernel(StateKernel):
    _extra_dims = 4
