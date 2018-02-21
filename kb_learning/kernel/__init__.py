import numpy as np

import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})

from ._kilobot_kernel_loop import KilobotSwarmKernel, ExponentialQuadraticKernel
from ._kernel import KilobotEnvKernel, MeanEnvKernel, MeanCovEnvKernel, KilobotEnvKernelWithWeight, \
    MeanEnvKernelWithWeight, MeanCovEnvKernelWithWeight
from ._preprocessors import compute_median_bandwidth, select_reference_set_randomly, compute_mean_position, \
    compute_mean_and_cov_position

__all__ = [
    'KilobotSwarmKernel',
    'KilobotEnvKernel',
    'KilobotEnvKernelWithWeight',
    'MeanEnvKernel',
    'MeanEnvKernelWithWeight',
    'MeanCovEnvKernel',
    'MeanCovEnvKernelWithWeight',
    'compute_median_bandwidth',
    'select_reference_set_randomly',
    'compute_mean_position',
    'compute_mean_and_cov_position'
]
