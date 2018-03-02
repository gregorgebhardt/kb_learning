cimport numpy as np


cdef class ExponentialQuadraticKernel:
    cdef public np.ndarray bandwidth
    cdef int normalized
    cdef double[:, :] get_gram_matrix_multi(self, double[:, :, :] a, double[:, :, :] b=?)

cdef class KilobotSwarmKernel:
    cdef public double bandwidth_factor
    cdef ExponentialQuadraticKernel _kernel_func
    cdef np.ndarray _compute_kb_distance(self, np.ndarray k1, np.ndarray k2)
