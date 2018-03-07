import numpy as np

from . import EmbeddedSwarmDistance, MahaDist, MeanSwarmDist, MeanCovSwarmDist


class KilobotEnvKernel:
    def __init__(self, bandwidth_factor_v=1.0, bandwidth_factor_kb=1.0, v_dims=2, weight=.5,
                 kb_dist_class=EmbeddedSwarmDistance, v_dist_class=MahaDist):
        self._kb_dist = kb_dist_class(bandwidth_factor=bandwidth_factor_kb)
        self._v_dims = v_dims
        self._v_dist = v_dist_class(bandwidth_factor=bandwidth_factor_v)
        self._weight = weight

    def __call__(self, X, Y=None, eval_gradient=False):
        if self._v_dims:
            k1 = X[:, :-self._v_dims]
            v1 = X[:, -self._v_dims:]

            if Y is None:
                k_v = self._v_dist(v1)
                k_kb = self._kb_dist(k1, k1)
            else:
                k2 = Y[:, :-self._v_dims]
                v2 = Y[:, -self._v_dims:]

                k_v = self._v_dist(v1, v2)
                k_kb = self._kb_dist(k1, k2)

            return np.exp(-self._weight * k_v - (1 - self._weight) * k_kb)
        else:
            if Y is None:
                k_kb = self._kb_dist(X, X)
            else:
                k_kb = self._kb_dist(X, Y)

            return np.exp(- (1 - self._weight) * k_kb)

    def diag(self, X):
        if self._v_dims:
            kb_dims = X[:, :-self._v_dims]
            light_dims = X[:, -self._v_dims:]
            return np.exp(-self._weight * self._v_dist.diag(light_dims)
                          - (1 - self._weight) * self._kb_dist.diag(kb_dims))
        else:
            return np.exp(- (1 - self._weight) * self._kb_dist.diag(X))

    def is_stationary(self):
        return self._v_dist.is_stationary() and self._kb_dist.is_stationary()

    def set_params(self, **params):
        if 'bandwidth' in params:
            if self._v_dims:
                bandwidth_kb = params['bandwidth'][:-self._v_dims]
                bandwidth_l = params['bandwidth'][-self._v_dims:]

                self._kb_dist.set_params(bandwidth=bandwidth_kb)
                self._v_dist.set_params(bandwidth=bandwidth_l)
            else:
                self._kb_dist.set_params(bandwidth=params['bandwidth'])


class MeanEnvKernel(KilobotEnvKernel):
    def __init__(self, kb_dist_class=MeanSwarmDist, *args, **kwargs):
        super().__init__(kb_dist_class=kb_dist_class, *args, **kwargs)


class MeanCovEnvKernel(KilobotEnvKernel):
    def __init__(self, kb_dist_class=MeanCovSwarmDist, *args, **kwargs):
        super().__init__(kb_dist_class=kb_dist_class, *args, **kwargs)


class KilobotEnvKernelWithWeight(KilobotEnvKernel):
    def __init__(self, weight_dim, bandwidth_factor_weight=1.0, weight_dist_class=MahaDist, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weight_dim = weight_dim

        self._weight_kernel = weight_dist_class(bandwidth_factor=bandwidth_factor_weight)

    def __call__(self, X, Y=None, *args, **kwargs):
        weights_X = X[:, self._weight_dim]
        if self._weight_dim == -1 or self._weight_dim == X.shape[-1] - 1:
            other_X = X[:, :self._weight_dim]
        else:
            other_X = np.c_[X[:, :self._weight_dim], X[:, self._weight_dim+1:]]

        if Y is not None:
            weights_Y = Y[:, self._weight_dim]
            if self._weight_dim == -1 or self._weight_dim == Y.shape[-1] - 1:
                other_Y = Y[:, :self._weight_dim]
            else:
                other_Y = np.c_[Y[:, :self._weight_dim], Y[:, self._weight_dim + 1:]]
            K_weights = np.exp(-.5 * self._weight_kernel(weights_X[:, None], weights_Y[:, None]))
        else:
            K_weights = np.exp(-.5 * self._weight_kernel(weights_X[:, None]))
            other_Y = None

        K_other = super().__call__(other_X, other_Y)

        return K_weights * K_other

    def set_params(self, **params):
        if 'bandwidth' in params:
            bandwidth = params['bandwidth']
            bandwidth_weight = bandwidth[self._weight_dim]
            if self._weight_dim == -1 or self._weight_dim == bandwidth.shape[-1] - 1:
                bandwidth_other = bandwidth[:self._weight_dim]
            else:
                bandwidth_other = np.r_[bandwidth[:self._weight_dim], bandwidth[self._weight_dim+1:]]

            self._weight_kernel.set_params(bandwidth=bandwidth_weight)
            params['bandwidth'] = bandwidth_other

        if 'num_processes' in params:
            self._weight_kernel.num_processes = params['num_processes']

        super(KilobotEnvKernelWithWeight, self).set_params(**params)


class MeanEnvKernelWithWeight(KilobotEnvKernelWithWeight):
    def __init__(self, kb_dist_class=MeanSwarmDist, *args, **kwargs):
        super().__init__(kb_dist_class=kb_dist_class, *args, **kwargs)


class MeanCovEnvKernelWithWeight(KilobotEnvKernelWithWeight):
    def __init__(self, kb_dist_class=MeanCovSwarmDist, *args, **kwargs):
        super().__init__(kb_dist_class=kb_dist_class, *args, **kwargs)
