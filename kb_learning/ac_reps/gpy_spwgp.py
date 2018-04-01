import numpy as np
import GPy

from paramz.transformations import Logexp
from GPy.core.parameterization import Param


class SemiHeteroscedasticGaussian(GPy.likelihoods.Gaussian):
    """
    A Gaussian likelihood that has both, a heteroscedastic variance and a homoscedastic variance.
    While both the heteroscedastic part and the homoscedastic part are returned by the function gaussian_variance
    during inference, only the homoscedastic variance is used for doing prediction
    """

    def __init__(self, het_variance, variance=1., name='semi_het_Gauss'):
        super().__init__(variance=variance, name=name)

        self.het_variance = Param('het_variance', het_variance, Logexp())
        self.link_parameter(self.het_variance)
        self.het_variance.fix()

    def gaussian_variance(self, Y_metadata=None):
        return self.het_variance[Y_metadata['output_index'].flatten()] * self.variance


class SparseWeightedGPyWrapper:
    def __init__(self, X, Y, Z, weights, kernel, noise_var=1., mean_function=None):
        semi_het_lik = SemiHeteroscedasticGaussian(variance=noise_var, het_variance=1 / weights)
        if mean_function:
            Y = Y - mean_function(X)

        self._gp = GPy.core.SparseGP(X=X, Y=Y, Z=Z, kernel=kernel, likelihood=semi_het_lik,
                                     Y_metadata=dict(output_index=np.arange(100)))
        self._mean_function = mean_function

        self.output_bounds = None
        self.output_bounds_enforce = False

    def get_mean(self, X):
        return self._gp.predict(Xnew=X)[0]

    def get_mean_sigma(self, X):
        return self._gp.predict(Xnew=X)

    def sample(self, X, num_samples_per_X=1):
        if self._mean_function:
            return self._mean_function(X) + self._gp.posterior_samples(X, size=1)
        return self._gp.posterior_samples(X, size=num_samples_per_X)