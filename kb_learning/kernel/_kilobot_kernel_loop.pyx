from cython.parallel import prange
# from cython.view cimport array as cvarray
import cython
cimport cython

import numpy as np
cimport numpy as np

from kb_learning.tools import np_chunks

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef class ExponentialQuadraticKernel:
    def __init__(self, normalized=0):
        self.normalized = normalized

    cdef np.ndarray get_gram_matrix(self, np.ndarray a, np.ndarray b=None):
        assert a.dtype == DTYPE

        cdef np.ndarray q, sqdist

        if np.isscalar(self.bandwidth):
            q = np.eye(a.shape[1]) / (self.bandwidth ** 2)
        else:
            assert (a.shape[1] == self.bandwidth.shape[0])
            q = np.diag((np.ones(self.bandwidth.shape[0]) / (self.bandwidth ** 2)))

        cdef np.ndarray aq = a.dot(q)
        cdef np.ndarray aq_a = np.sum(aq * a, axis=1)
        cdef np.ndarray bq_b
        if b is None:
            sqdist = aq_a[:, np.newaxis] + aq_a - 2 * aq.dot(a.T)
        else:
            bq_b = np.sum(b.dot(q) * b, axis=1)
            # Equivalent to MATLAB bsxfun(@plus, ..)
            sqdist = aq_a[:, np.newaxis] + bq_b - 2 * aq.dot(b.T)
        cdef np.ndarray K = np.exp(-0.5 * sqdist)

        if self.normalized:
            K = K / np.sqrt(np.prod(self.bandwidth) ** 2 * (2 * np.pi) ** a.shape[1])

        return K

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef np.ndarray get_gram_matrix_multi(self, np.ndarray[DTYPE_t, ndim=3] a, np.ndarray[DTYPE_t, ndim=3] b=None):
        """
        
        :param a: q x n x d matrix of kilobot positions
        :param b: r x m x d matrix of kilobot positions
        :return:  q x r x n x m matrix if a and b are given, q x n x n if only a is given
        """
        # assert a.dtype == DTYPE

        cdef Py_ssize_t q, n, d
        cdef Py_ssize_t r, m
        cdef Py_ssize_t i, j, k, l, s

        q = a.shape[0]
        n = a.shape[1]
        d = a.shape[2]

        cdef np.ndarray[DTYPE_t, ndim=4] sq_dist

        cdef np.ndarray bw = np.array(1 / (self.bandwidth ** 2), ndmin=3)

        cdef np.ndarray[DTYPE_t, ndim=3] aq = a * bw
        cdef np.ndarray[DTYPE_t, ndim=2] aq_a = np.sum(aq * a, axis=-1)
        cdef np.ndarray[DTYPE_t, ndim=2] bq_b
        if b is None:
            # sq_dist = aq_a[:, None, :] + aq_a[:, :, None]
            # sq_dist -= 2 * np.einsum('qid,qjd->qij', aq, a)
            sq_dist = np.zeros((q, 1, n, n))
            for i in range(q):
                for j in range(n):
                    for k in range(n):
                        sq_dist[i, 0, j, k] = aq_a[i, j] + aq_a[i, k]
                            # - 2 * np.sum(aq[i, j, :] * a[i, k, :], axis=-1)
                        for s in range(d):
                            sq_dist[i, 0, j, k] -= 2 * aq[i, j, s] * a[i, k, s]
        else:
            r = b.shape[0]
            m = b.shape[1]
            assert b.dtype == DTYPE
            bq_b = np.sum((b * bw) * b, axis=-1)
            # sq_dist = aq_a[:, None, :, None] + bq_b[None, :, None, :]
            # sq_dist -= 2 * np.einsum('qid,rjd->qrij', aq, b)
            sq_dist = np.zeros((q, r, n, m))
            for i in range(q):
                for j in range(r):
                    for k in range(n):
                        for l in range(m):
                            sq_dist[i, j, k, l] = aq_a[i, k] + bq_b[j, l]
                                # - 2 * np.sum(aq[i, k, :] * b[j, l, :], axis=-1)
                            for s in range(d):
                                sq_dist[i, j, k, l] -= 2 * aq[i, k, s] * b[j, l, s]
        cdef np.ndarray K = np.exp(np.multiply(-0.5, sq_dist.squeeze()))

        return K

    def get_gram_diag(self, np.ndarray data):
        return np.ones(data.shape[0])

    cpdef get_derivation_param(self, np.ndarray data):
        """
        
        :param data: n x d
        :return: p x n x n, where p is the number of bandwidth parameters
        """
        cdef np.ndarray gram_matrix = self.get_gram_matrix(data, data)
        cdef np.ndarray gradient_matrices = np.zeros([self.bandwidth.shape[0], gram_matrix.shape[0],
                                                      gram_matrix.shape[1]])

        for dim in range(self.bandwidth.shape[0]):
            sqdist = (data[:, dim, None] - data[:, dim]) ** 2
            gradient_matrices[dim, :, :] = gram_matrix * (sqdist / (self.bandwidth[dim] ** 3))

        return gradient_matrices, gram_matrix

    cpdef get_derivation_data(self, np.ndarray ref_data, np.ndarray cur_data):
        """

        :param ref_data: n x d
        :param cur_data: m x d
        :return: m x d x n
        """
        cdef np.ndarray gram_matrix = self.get_gram_matrix(ref_data, cur_data)
        cdef np.ndarray q = np.diag((np.ones(self.bandwidth.shape[0]) / (self.bandwidth ** 2)))

        cdef np.ndarray scaleddist = -2 * (cur_data.dot(q) - ref_data.dot(q))

        return gram_matrix.T[None, ...] * scaleddist

cdef class KilobotKernel:
    def __init__(self, bandwidth_factor=1., num_processes=1):
        self._kernel_func = ExponentialQuadraticKernel()
        self.bandwidth_factor = bandwidth_factor
        # self.num_processes = num_processes

    cdef np.ndarray _compute_kb_distance(self, np.ndarray k1, np.ndarray k2):
        """Computes the kb distance matrix between any configuration in k1 and any configuration in k2.

        :param k1: q x 2*d1 matrix of q configurations with each d1 kilobots
        :param k2: r x 2*d2 matrix of r configurations with each d2 kilobots
        :return: q x r matrix with the distances between the configurations in k1 and k2
        """
        cdef int N, m
        cdef int num_kb_1, num_kb_2
        cdef int q, r
        cdef int i

        # number of samples in k1
        q = k1.shape[0]

        # number of samples in k2
        r = k2.shape[0]

        # number of kilobots in k1
        num_kb_1 = k1.shape[1] // 2
        # number of kilobots in k2
        num_kb_2 = k2.shape[1] // 2

        # reshape matrices
        cdef np.ndarray k1_reshaped = k1.reshape(q, num_kb_1, 2)
        cdef np.ndarray k2_reshaped = k2.reshape(r, num_kb_2, 2)

        cdef np.ndarray[DTYPE_t, ndim=1] k_n, k_m
        cdef np.ndarray[DTYPE_t, ndim=2] k_nm = np.empty((q, r), dtype=DTYPE)
        k_n = self._kernel_func.get_gram_matrix_multi(k1_reshaped).sum(axis=(1, 2)) / (num_kb_1 ** 2)
        k_m = self._kernel_func.get_gram_matrix_multi(k2_reshaped).sum(axis=(1, 2)) / (num_kb_2 ** 2)
        cdef int chunk_size = 50
        for i in range(0, r, chunk_size):
            k_nm[:, i:i+chunk_size] = self._kernel_func.get_gram_matrix_multi(
                k1_reshaped, k2_reshaped[i:i+chunk_size]).sum(axis=(2, 3))
        # k_nm[k_nm < 1e-12] = 0
        k_nm = np.divide(k_nm, 0.5 * num_kb_1 * num_kb_2)

        return k_n[:, None] + k_m[None, :] - k_nm

    def __call__(self, k1, k2, eval_gradient=False):
        return self._compute_kb_distance(k1, k2)

    def diag(self, X):
        if len(X.shape) > 1:
            return np.zeros((X.shape[0],))
        return np.zeros(1)

    def is_stationary(self):
        return True

    def set_bandwidth(self, bandwidth):
        bandwidth = bandwidth.reshape((-1, 2)).mean(axis=0)
        self._kernel_func.bandwidth = self.bandwidth_factor * bandwidth

    def set_params(self, **params):
        if 'bandwidth' in params:
            self.set_bandwidth(params['bandwidth'])

