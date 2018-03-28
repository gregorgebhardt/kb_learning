from cython.parallel import parallel, prange
# from cython.view cimport array as cvarray
import cython
cimport cython

import numpy as np
cimport numpy as np

from math import exp
from libc.math cimport exp

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef class ExponentialQuadraticKernel:
    def __init__(self, normalized=0):
        self.normalized = normalized
        self.bandwidth = np.array([1.])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef double[:, :] get_gram_matrix_multi(self, double[:, :, :] a, double[:, :, :] b=None):
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

        cdef double[:, :] sq_dist
        if b is None:
            sq_dist = np.empty((q, 1))
        else:
            r = b.shape[0]
            m = b.shape[1]
            # bq_b = np.sum((b * bw) * b, axis=-1)
            sq_dist = np.empty((q, r))

        if len(self.bandwidth) > d:
            self.bandwidth = self.bandwidth.reshape((-1, d)).mean(axis=0)

        cdef double[:] bw = -1 / (2 * self.bandwidth)

        # cdef np.ndarray[DTYPE_t, ndim=3] aq = a * bw
        # cdef np.ndarray[DTYPE_t, ndim=2] aq_a = np.sum(aq * a, axis=-1)
        # cdef np.ndarray[DTYPE_t, ndim=2] bq_b
        cdef DTYPE_t kb_sum_1 = .0, kb_sum_2 = .0
        with nogil, parallel():  # num_threads=8
            if b is None:
                for i in prange(q, schedule='guided'):
                    sq_dist[i, 0] = .0
                    for j in range(n):
                        for k in range(n):
                            kb_sum_1 = .0
                            for s in range(d):
                                kb_sum_1 += (a[i, j, s]**2 + a[i, k, s]**2) * bw[s]
                                kb_sum_1 += -2 * a[i, j, s] * bw[s] * a[i, k, s]
                            sq_dist[i, 0] += exp(kb_sum_1)
                    sq_dist[i, 0] /= n**2
            else:
                for i in prange(q, schedule='guided'):
                    for j in range(r):
                        sq_dist[i, j] = .0
                        for k in range(n):
                            for l in range(m):
                                kb_sum_2 = .0
                                for s in range(d):
                                    kb_sum_2 += (a[i, k, s] ** 2 + b[j, l, s] ** 2) * bw[s]
                                    kb_sum_2 += -2 * a[i, k, s] * bw[s] * b[j, l, s]
                                sq_dist[i, j] += exp(kb_sum_2)
                        sq_dist[i, j] /= n * m
                        sq_dist[i, j] *= 2

        return sq_dist

    def get_gram_diag(self, np.ndarray data):
        return np.ones(data.shape[0])

cdef class EmbeddedSwarmDistance:
    def __init__(self):
        self._kernel_func = ExponentialQuadraticKernel()

    cdef np.ndarray _compute_kb_distance(self, np.ndarray k1, np.ndarray k2):
        """Computes the kb distance matrix between any configuration in k1 and any configuration in k2.

        :param k1: q x 2*d1 matrix of q configurations with each d1 kilobots
        :param k2: r x 2*d2 matrix of r configurations with each d2 kilobots
        :return: q x r matrix with the distances between the configurations in k1 and k2
        """
        cdef int N, m
        cdef int num_kb_1, num_kb_2
        cdef int q, r
        cdef int i, j

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

        cdef double[:, :] k_n, k_m, k_nm
        # cdef np.ndarray[DTYPE_t, ndim=2] k_nm = np.empty((q, r), dtype=DTYPE)
        k_n = self._kernel_func.get_gram_matrix_multi(k1_reshaped)  # / (num_kb_1 ** 2)
        k_m = self._kernel_func.get_gram_matrix_multi(k2_reshaped)  # / (num_kb_2 ** 2)
        # cdef int chunk_size = 50
        # for i in range(0, r, chunk_size):
        #     k_nm[:, i:i+chunk_size] = self._kernel_func.get_gram_matrix_multi(
        #         k1_reshaped, k2_reshaped[i:i+chunk_size])
        k_nm = self._kernel_func.get_gram_matrix_multi(k1_reshaped, k2_reshaped)
        # k_nm[k_nm < 1e-12] = 0
        # k_nm = np.divide(k_nm, 0.5 * num_kb_1 * num_kb_2)

        for i in range(q):
            for j in range(r):
                k_nm[i, j] = -k_nm[i, j] + k_n[i, 0] + k_m[j, 0]

        cdef np.ndarray K = np.asarray(k_nm)

        del k_n, k_m, k_nm, k1_reshaped, k2_reshaped

        return K

    def __call__(self, k1, k2, eval_gradient=False):
        return self._compute_kb_distance(k1, k2)

    def diag(self, X):
        if len(X.shape) > 1:
            return np.zeros((X.shape[0],))
        return np.zeros(1)

    def is_stationary(self):
        return True

    def set_bandwidth(self, bandwidth):
        if np.isscalar(bandwidth):
            self._kernel_func.bandwidth = np.array([bandwidth])
        else:
            self._kernel_func.bandwidth = bandwidth

    def get_bandwidth(self):
        return self._kernel_func.bandwidth
