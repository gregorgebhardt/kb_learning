cimport numpy as np

ctypedef np.float64_t DTYPE_t

cdef class MahaDist:
    cdef public np.ndarray bandwidth
    cdef public preprocessor
    cdef np.ndarray get_distance_matrix(self, DTYPE_t[:, :] a, DTYPE_t[:, :] b=?)
