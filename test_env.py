import numpy as np
# from kb_learning.learner._learner import QuadPushingACRepsLearner
from kb_learning.kernel import KilobotSwarmKernel

from memory_profiler import profile

# zeros = np.array([[.0, .0]])
#
#
# def policy(s):
#     return zeros
#
#
# learner = QuadPushingACRepsLearner()
# learner.policy = policy
#
# learner._get_samples_parallel(100, 125, .0, 15)

kernel = KilobotSwarmKernel()
kernel.set_params(bandwidth=np.array([1., 1., 1., 1.]))


@profile
def compute_kernel_matrix(A, B):
    K = kernel(A, B)
    del K


for i in range(10):
    A = np.random.rand(10000, 32)
    B = np.random.rand(300, 32)
    compute_kernel_matrix(A, B)
