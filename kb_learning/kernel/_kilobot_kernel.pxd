cimport numpy as np

cdef class Kernel:
    cdef np.ndarray get_gram_matrix(self, np.ndarray a, np.ndarray b=?)
    cdef np.ndarray get_gram_matrix_multi(self, np.ndarray a, np.ndarray b=?)

cdef class ExponentialQuadraticKernel(Kernel):
    cdef public np.ndarray bandwidth
    cdef int normalized
    cdef np.ndarray get_gram_matrix(self, np.ndarray a, np.ndarray b=?)
    cdef np.ndarray get_gram_matrix_multi(self, np.ndarray a, np.ndarray b=?)
    cpdef get_derivation_param(self, np.ndarray data)
    cpdef get_derivation_data(self, np.ndarray ref_data, np.ndarray cur_data)

cdef class KilobotKernel:
    cdef public double bandwidth_factor
    cdef Kernel _kernel_func
    cdef _compute_kb_distance(self, np.ndarray k1, np.ndarray k2)
