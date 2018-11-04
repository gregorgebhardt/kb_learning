import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy.stats

import GPy

from paramz.transformations import Logexp
from GPy.core.parameterization import Param
from kb_learning.kernel import KilobotEnvKernel

# it_sars = pd.read_pickle('/home/gebhardt/Desktop/it_sars_15kbts.pkl')
#
# kernel = KilobotEnvKernel(kilobots_dim=30, light_dim=2)
#
# kernel.kilobots_bandwidth = [.04, .04]
# kernel.light_bandwidth = [.1]
kernel_function = GPy.kern.RBF(input_dim=1)

# sinus curve with Gaussian noise
sample_size = 100
samples_x = np.random.rand(sample_size) * 2 * np.pi - np.pi

noise_std = .2
sigma_sqr = noise_std ** 2
samples_y = np.sin(samples_x) + np.random.normal(0, noise_std, (sample_size))

# weights
loc = -2.
scale = .5

weights = scipy.stats.norm.pdf(samples_x, loc=loc, scale=scale)
weights /= weights.max()

# sparse set
sparse_size = 10

sparse_index = [np.random.randint(sample_size)]
sparse_x = samples_x[sparse_index]
for i in range(1, sparse_size):
    K = kernel_function.K(sparse_x[:, None], samples_x[:, None])
    sparse_index += [K.max(axis=0).argmin()]
    sparse_x = samples_x[sparse_index]

# Y_metadata = dict(output_index=np.arange(100)[:, None])
# llh = GPy.likelihoods.HeteroscedasticGaussian(variance=sigma_sqr, Y_metadata=Y_metadata)
# llh.variance *= 1 / weights[:, None]
# llh.variance.fix()
# predict_llh = GPy.likelihoods.Gaussian(variance=sigma_sqr)
# gp_sparse = GPy.core.SparseGP(X=samples_x[:, None], Y=samples_y[:, None], Z=sparse_x[:, None],
#                               kernel=kernel_function, likelihood=llh, Y_metadata=dict(output_index=np.arange(100)))
#
# gp_sparse.inducing_inputs.fix()
# gp_sparse.optimize('bfgs')
# _ = gp_sparse.plot(lower=-2, upper=2, predict_kw=dict(likelihood=predict_llh))
# _ = plt.scatter(samples_x, weights, color='grey', marker='.')
# print(gp_sparse)

Y_metadata = dict(output_index=np.arange(100))
het_llh = GPy.likelihoods.HeteroscedasticGaussian(Y_metadata=Y_metadata)
het_llh.variance = 1 / weights
het_llh.variance.fix()
noise_llh = GPy.likelihoods.Gaussian(variance=sigma_sqr)

llh2 = GPy.likelihoods.MixedNoise(likelihoods_list=[het_llh, noise_llh])

gp_sparse = GPy.core.SparseGP(X=samples_x[:, None], Y=samples_y[:, None], Z=sparse_x[:, None],
                              kernel=GPy.kern.RBF(input_dim=1), likelihood=llh2, inference_method=MyVarDTC(limit=3),
                              Y_metadata=dict(output_index=np.arange(100)))


class SemiHeteroscedasticGaussian(GPy.likelihoods.Gaussian):
    def __init__(self, het_variance, variance=1., name='semi_het_Gauss'):
        super().__init__(variance=variance, name=name)

        self.het_variance = Param('het_variance', het_variance, Logexp())
        self.link_parameter(self.het_variance)
        self.het_variance.fix()

    def gaussian_variance(self, Y_metadata=None):
        return self.het_variance[Y_metadata['output_index'].flatten()] * self.variance