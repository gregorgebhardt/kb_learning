import numpy as np

import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})

from ._kilobot_kernel_loop import KilobotKernel, ExponentialQuadraticKernel
# from ._kilobot_kernel import KilobotKernel, ExponentialQuadraticKernel
from ._kernel import MahaKernel, StateKernel, StateActionKernel  # , KilobotKernel
from ._preprocessors import compute_median_bandwidth, select_reference_set_randomly

__all__ = [
    'KilobotKernel',
    'MahaKernel',
    'StateKernel',
    'StateActionKernel',
    'compute_median_bandwidth',
    'select_reference_set_randomly'
]