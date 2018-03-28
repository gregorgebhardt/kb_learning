from cython.parallel import parallel, prange
# from cython.view cimport array as cvarray
import cython
cimport cython

import numpy as np
cimport numpy as np

from math import exp
from libc.math cimport exp

DTYPE = np.float64


cdef class MahaDist:
    def __init__(self):
        self.bandwidth = 1.
        self.preprocessor = None

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef np.ndarray get_distance_matrix(self, DTYPE_t[:, :] a, DTYPE_t[:, :] b=None):
        """computes the squared mahalanobis distance
        
        :param a: q x d matrix of kilobot positions
        :param b: r x d matrix of kilobot positions
        :return:  q x r matrix if a and b are given, q x q matrix if only a is given
        """
        # assert a.dtype == DTYPE

        cdef Py_ssize_t q, r, d
        cdef Py_ssize_t i, j, k

        q = a.shape[0]
        d = a.shape[1]
        cdef DTYPE_t[:, :] a_mem = a
        cdef DTYPE_t[:, :] b_mem

        cdef DTYPE_t[:, :] maha_dist
        if b is None:
            maha_dist = np.empty((q, q))
        else:
            b_mem = b
            r = b.shape[0]
            maha_dist = np.empty((q, r))

        cdef DTYPE_t[:] bw = 1 / self.bandwidth

        cdef DTYPE_t sq_dist_a = .0, sq_dist_ab = .0
        with nogil, parallel():
            if b is None:
                for i in prange(q, schedule='guided'):
                    for j in range(i, q):
                        maha_dist[i, j] = .0
                        maha_dist[j, i] = .0
                        sq_dist_a = .0
                        for k in range(d):
                            sq_dist_a += (a_mem[i, k] - a_mem[j, k]) ** 2 * bw[k]

                        maha_dist[j, i] += sq_dist_a
                        if j != i:
                            maha_dist[i, j] += sq_dist_a
            else:
                for i in prange(q, schedule='guided'):
                    for j in range(r):
                        maha_dist[i, j] = .0
                        for k in range(d):
                            maha_dist[i, j] += (a_mem[i, k] - b_mem[j, k])**2 * bw[k]

        return np.asarray(maha_dist)

    def get_gram_diag(self, np.ndarray data):
        return np.zeros(data.shape[0])

    def __call__(self, X, Y=None):
        if self.preprocessor:
            X = self.preprocessor(X)
            if Y is not None:
                Y = self.preprocessor(Y)
        return self.get_distance_matrix(X, Y)


class MeanSwarmDist(MahaDist):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from ._preprocessors import compute_mean_position
        self.preprocessor = compute_mean_position


class MeanCovSwarmDist(MeanSwarmDist):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from ._preprocessors import compute_mean_and_cov_position
        self.preprocessor = compute_mean_and_cov_position