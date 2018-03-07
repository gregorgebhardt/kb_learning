import numpy as np

from sklearn.metrics.pairwise import pairwise_distances


class MahaDist:
    def __init__(self, bandwidth_factor=1.0):
        self.bandwidth = 1.
        if type(bandwidth_factor) in [float, int]:
            self.bandwidth_factor = bandwidth_factor
        elif type(bandwidth_factor) in [list, tuple]:
            self.bandwidth_factor = np.array(bandwidth_factor)
        else:
            self.bandwidth_factor = bandwidth_factor

        self._preprocessor = None

    def __call__(self, X, Y=None, eval_gradient=False):
        if self._preprocessor:
            X = self._preprocessor(X)
            if Y is not None:
                Y = self._preprocessor(Y)
        return pairwise_distances(X, Y, metric='mahalanobis', VI=self.bandwidth)

    @staticmethod
    def diag(X):
        return np.zeros((X.shape[0],))

    @staticmethod
    def is_stationary():
        return True

    def set_params(self, **params):
        if 'bandwidth' in params:
            if np.isscalar(params['bandwidth']):
                self.bandwidth = 1 / (self.bandwidth_factor * params['bandwidth'])
            else:
                self.bandwidth = np.diag(1 / (self.bandwidth_factor * params['bandwidth']))


class MeanSwarmDist(MahaDist):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from ._preprocessors import compute_mean_position
        self._preprocessor = compute_mean_position


class MeanCovSwarmDist(MeanSwarmDist):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from ._preprocessors import compute_mean_and_cov_position
        self._preprocessor = compute_mean_and_cov_position


class PeriodicDist:
    def __init__(self, bandwidth_factor=1.0):
        self.bandwidth = 1.
        if type(bandwidth_factor) in [float, int]:
            self.bandwidth_factor = bandwidth_factor
        elif type(bandwidth_factor) in [list, tuple]:
            self.bandwidth_factor = np.array(bandwidth_factor)
        else:
            self.bandwidth_factor = bandwidth_factor

        self._preprocessor = None

    def __call__(self, X, Y=None, eval_gradient=False):
        if self._preprocessor:
            X = self._preprocessor(X)
            if Y is not None:
                Y = self._preprocessor(Y)

        if Y is None:
            Y = X

        return np.sum((np.sin(.5 * np.einsum('nd,kd->nkd', X, -Y)) ** 2) / self.bandwidth, axis=2)

    @staticmethod
    def diag(X):
        return np.zeros((X.shape[0],))

    @staticmethod
    def is_stationary():
        return True

    def set_params(self, **params):
        if 'bandwidth' in params:
            self.bandwidth = 1 / (self.bandwidth_factor * params['bandwidth'])
