import numpy as np
import GPy

from paramz.transformations import Logexp
from GPy.core.parameterization import Param
from kb_learning.kernel import KilobotEnvKernel


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
        self.variance.fix()

    def gaussian_variance(self, Y_metadata=None):
        return self.het_variance[Y_metadata['output_index'].flatten()] * self.variance


class SparseWeightedGPyWrapper:
    def __init__(self, kernel, noise_var=1., mean_function=None, output_bounds=None):
        self.kernel = kernel
        # self.kernel.variance.fix()
        self.noise_var = noise_var

        self._gp = None
        self._mean_function = mean_function

        self.output_bounds = output_bounds

    def train(self, inputs, outputs, weights, sparse_inputs, optimize=True):
        if self._mean_function:
            outputs = outputs - self._mean_function(inputs)

        if self._gp is None:
            semi_het_lik = SemiHeteroscedasticGaussian(variance=self.noise_var, het_variance=weights.max() / weights)
            self._gp = GPy.core.SparseGP(X=inputs, Y=outputs, Z=sparse_inputs, kernel=self.kernel,
                                         likelihood=semi_het_lik,
                                         Y_metadata=dict(output_index=np.arange(inputs.shape[0])))
            self._gp.inducing_inputs.fix()

        else:
            # self._gp.set_updates(False)
            self._gp.set_XY(inputs, outputs)
            self._gp.set_Z(sparse_inputs, False)
            self._gp.inducing_inputs.fix()
            self._gp.likelihood.het_variance = weights.max() / weights
            # self._gp.set_updates()
            self._gp.update_model(True)

        if optimize:
            self._gp.optimize('bfgs', messages=True)
        print(self._gp)

    def to_dict(self):
        input_dict = dict()
        input_dict['kernel'] = self.kernel.to_dict()
        input_dict['noise_var'] = self.noise_var
        input_dict['mean_function'] = self._mean_function
        input_dict['output_bounds'] = self.output_bounds

        if self._gp:
            input_dict['gp'] = dict()
            input_dict['gp']['inputs'] = self._gp.X.values
            input_dict['gp']['outputs'] = self._gp.Y.values
            input_dict['gp']['sparse_inputs'] = self._gp.Z.values
            input_dict['gp']['weights'] = 1 / self._gp.likelihood.het_variance.values

        return input_dict

    @staticmethod
    def from_dict(input_dict):
        gp_dict = input_dict.pop('gp', None)
        kernel_dict = input_dict.pop('kernel')
        input_dict['kernel'] = KilobotEnvKernel.from_dict(kernel_dict)

        spwgp = SparseWeightedGPyWrapper(**input_dict)
        if gp_dict:
            spwgp.train(**gp_dict, optimize=False)

        return spwgp

    def get_mean(self, inputs):
        return self._gp.predict(Xnew=inputs)[0]

    def get_mean_sigma(self, inputs):
        return self._gp.predict(Xnew=inputs)

    def sample(self, inputs):
        if self._gp is None:
            variance = self.noise_var + self.kernel.variance
            # TODO remove magic number
            samples = np.random.normal(scale=np.sqrt(variance), size=(inputs.shape[0], 2))
        else:
            mean, variance = self._gp.predict(Xnew=inputs)
            samples = mean + np.random.normal(scale=np.sqrt(variance), size=(inputs.shape[0], 2))

        if self._mean_function:
            samples += self._mean_function(inputs)

        if self.output_bounds is not None:
                # check samples against bounds from action space
            samples = np.minimum(samples, self.output_bounds[1])
            samples = np.maximum(samples, self.output_bounds[0])

        return samples

    def __call__(self, inputs):
        return self.sample(inputs)
