import numpy as np
cimport numpy as np

from kb_learning.tools import np_chunks

DTYPE = np.float64

ctypedef np.float_t DTYPE_t

cdef class Kernel:
    # Evaluates kernel for all elements
    # Compute gram matrix of the form
    #  -----------------------------
    #  | k(x₁,y₁) | k(x₁,y₂) | ... |
    #  -----------------------------
    #  | k(x₂,y₁) | k(x₂,y₂) | ... |
    #  -----------------------------
    #  | ...      | ...      | ... |
    #  -----------------------------
    # if y=None, K(x,x) is computed
    cdef np.ndarray get_gram_matrix(self, np.ndarray a, np.ndarray b=None):
        pass

    cdef np.ndarray get_gram_matrix_multi(self, np.ndarray a, np.ndarray b=None):
        pass

    # Returns the diagonal of the gram matrix
    # which means the kernel is evaluated between every data point and itself
    def get_gram_diag(self, data):
        diag = np.zeros(data.shape[0])

        for i in range(data.shape[0]):
            diag[i] = self.__call__(data[i, :], data[i, :])

cdef class ExponentialQuadraticKernel(Kernel):
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

    cdef np.ndarray get_gram_matrix_multi(self, np.ndarray a, np.ndarray b=None):
        """
        
        :param a: q x n x d matrix of kilobot positions
        :param b: r x m x d matrix of kilobot positions
        :return:  q x r x n x m matrix if a and b are given, q x n x n if only a is given
        """
        assert a.dtype == DTYPE

        cdef np.ndarray q, sq_dist

        if np.isscalar(self.bandwidth):
            q = np.eye(a.shape[-1]) / (self.bandwidth ** 2)
        else:
            # assert (a.shape[-1] == self.bandwidth.shape[0])
            q = np.diag((np.ones(self.bandwidth.shape[0]) / (self.bandwidth ** 2)))

        cdef np.ndarray aq = a.dot(q)
        cdef np.ndarray aq_a = np.sum(aq * a, axis=-1)
        cdef np.ndarray bq_b
        if b is None:
            sq_dist = aq_a[..., np.newaxis, :] + aq_a[..., np.newaxis]
            sq_dist -= 2 * np.sum(aq[:, :, np.newaxis, :] * a[:, np.newaxis, :, :], axis=-1)
        else:
            assert b.dtype == DTYPE
            bq_b = np.sum(b.dot(q) * b, axis=-1)
            sq_dist = aq_a[:, np.newaxis, :, np.newaxis] + bq_b[np.newaxis, :, np.newaxis, :]
            sq_dist -= 2 * np.sum(aq[:, np.newaxis, :, np.newaxis, :] * b[np.newaxis, :, np.newaxis, :, :], axis=-1)
        cdef np.ndarray K = np.exp(-0.5 * sq_dist)

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
            sqdist = (data[:, dim, np.newaxis] - data[:, dim]) ** 2
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

        return gram_matrix.T[np.newaxis, ...] * scaleddist

cdef class KilobotKernel:
    def __init__(self, bandwidth_factor=1., num_processes=1):
        self._kernel_func = ExponentialQuadraticKernel()
        # self.num_processes = num_processes

    cdef _compute_kb_distance(self, np.ndarray k1, np.ndarray k2):
        """Computes the kb distance matrix between any configuration in k1 and any configuration in k2.

        :param k1: q x 2*d1 matrix of q configurations with each d1 kilobots
        :param k2: r x 2*d2 matrix of r configurations with each d2 kilobots
        :return: q x r matrix with the distances between the configurations in k1 and k2
        """
        cdef int N, m
        cdef int num_kb_1, num_kb_2


        # number of samples in k1
        q = k1.shape[0]

        # number of samples in k2
        r = k2.shape[0]

        # number of kilobots in k1
        num_kb_1 = k1.shape[1] // 2
        # number of kilobots in k2
        num_kb_2 = k2.shape[1] // 2

        # reshape matrices
        # cdef np.ndarray k1_reshaped = np.c_[k1.flat[0::2], k1.flat[1::2]]
        # cdef np.ndarray k2_reshaped = np.c_[k2.flat[0::2], k2.flat[1::2]]

        cdef np.ndarray k1_reshaped = k1.reshape(q, num_kb_1, 2)
        cdef np.ndarray k2_reshaped = k2.reshape(r, num_kb_2, 2)

        cdef np.ndarray k_n, k_m
        cdef np.ndarray k_nm = np.empty((q, r))
        k_n = self._kernel_func.get_gram_matrix_multi(k1_reshaped).sum(axis=(1, 2)) / (num_kb_1 ** 2)
        k_m = self._kernel_func.get_gram_matrix_multi(k2_reshaped).sum(axis=(1, 2)) / (num_kb_2 ** 2)
        for i in range(0,r,10):
            k_nm[:, r:r+10] = self._kernel_func.get_gram_matrix_multi(k1_reshaped, k2_reshaped[r:r+10]).sum(axis=(2, 3))
        k_nm /= (0.5 * num_kb_1 * num_kb_2)

        # compute the kernel values within each configuration of k1
        # cdef np.ndarray k_n = np.empty((n, 1))
        # cdef np.ndarray k_m = np.empty((1, m))
        # cdef np.ndarray k_nm = np.empty((n, m))
        #
        # cdef int i, j
        # cdef np.ndarray chunk_n, chunk_m
        # for i in range(n):
        #     k_n[i] = self._kernel_func.get_gram_matrix(k1_reshaped[i*num_kb_1:(i+1)*num_kb_1]).sum()
        #
        #     for j in range(m):
        #         if i == 0:
        #             # compute the kernel values within each configuration of k2
        #             k_m.flat[j] = self._kernel_func.get_gram_matrix(k2_reshaped[j * num_kb_2:(j + 1) * num_kb_2]).sum()
        #
        #         k_nm[i, j] = self._kernel_func.get_gram_matrix(k1_reshaped[i * num_kb_1:(i + 1) * num_kb_1],
        #                                        k2_reshaped[j * num_kb_2:(j + 1) * num_kb_2]).sum()
        #
        # k_n /= num_kb_1 ** 2
        # k_m /= num_kb_2 ** 2
        # k_nm /= (0.5 * num_kb_1 * num_kb_2)

        # # compute kernel between all kilobot positions
        # k_nm = self._kernel_func(k1_reshaped, k2_reshaped)
        # # add column and row of zeros
        # k_nm = np.c_[np.zeros((k_nm.shape[0] + 1, 1)), np.r_[np.zeros((1, k_nm.shape[1])), k_nm]]
        # # compute the cumulative sum in both directions
        # k_nm = k_nm.cumsum(axis=0).cumsum(axis=1)
        #
        # # We are interested in the cumsum of the block matrices for each combination of configuration. To get this
        # # cumsum for a certain block, we take the cumsum at the lower-right entry of said block and subtract the
        # # cumsum at the lower-right entry of the block at its left and at its top. Since both of these two cumsums
        # # include the cumsum of the whole submatrix before the block, we need to add this value again.
        # k_nm = k_nm[num_kb_1::num_kb_1, num_kb_2::num_kb_2] \
        #        - k_nm[0:-1:num_kb_1, num_kb_2::num_kb_2] \
        #        - k_nm[num_kb_1::num_kb_1, 0:-1:num_kb_2] + k_nm[0:-1:num_kb_1, 0:-1:num_kb_2]
        # k_nm /= (0.5 * num_kb_1 * num_kb_2)

        return k_n[:, np.newaxis] + k_m[np.newaxis, :] - k_nm

    def __call__(self, X, Y=None, eval_gradient=False):
        # if self.num_processes > 1:
        #     return pairwise_distances(X, Y, self._compute_kb_distance, n_jobs=self.num_processes)
        # else:
        # if Y is None:
        #     return squareform(pdist(X, self._compute_kb_distance))
        return self._compute_kb_distance(X, Y)

    def diag(self, X):
        if len(X.shape) > 1:
            return np.zeros((X.shape[0], 1))
        return np.zeros(1)

    def is_stationary(self):
        return True

    def set_bandwidth(self, bandwidth):
        bandwidth = bandwidth.reshape((-1, 2)).mean(axis=0)
        self._kernel_func.bandwidth = bandwidth

    def set_params(self, **params):
        if 'bandwidth' in params:
            self.set_bandwidth(params['bandwidth'])
        if 'num_processes' in params:
            self.num_processes = params['num_processes']

