import numpy as np

from . import EmbeddedSwarmDistance, MahaDist


class KilobotEnvKernel:
    def __init__(self, bandwidth_factor_kilobots=1.0, bandwidth_factor_light=1.0, bandwidth_factor_action=1.0,
                 light_idx=None, action_idx=None, weight=.5, kb_dist_class=EmbeddedSwarmDistance,
                 light_dist_class=MahaDist, action_dist_class=MahaDist):
        self.kilobots_dist = kb_dist_class()
        self.light_dist = light_dist_class()
        self.action_dist = action_dist_class()

        self._light_idx = light_idx
        self._action_idx = action_idx
        self._split_idx = []
        if self._light_idx:
            self._split_idx.append(self._light_idx)
        if self._action_idx:
            self._split_idx.append(self._action_idx)

        self._weight = weight

    def __call__(self, X, Y=None):
        X_split = np.split(X, self._split_idx, axis=1)
        X_kilobots = X_split[0]

        if Y is not None:
            Y_split = np.split(Y, self._split_idx, axis=1)
            Y_kilobots = Y_split[0]
            k = - (1 - self._weight) * self.kilobots_dist(X_kilobots, Y_kilobots)

            if self._light_idx and self._light_idx < X.shape[1]:
                X_light = X_split[1]
                Y_light = Y_split[1]
                k -= self._weight * self.light_dist(X_light, Y_light)

            if self._action_idx and self._action_idx < X.shape[1]:
                X_action = X_split[-1]
                Y_action = Y_split[-1]
                k -= self._weight * self.action_dist(X_action, Y_action)
        else:
            k = - (1 - self._weight) * self.kilobots_dist(X_kilobots, X_kilobots)

            if self._light_idx and self._light_idx < X.shape[1]:
                X_light = X_split[1]
                k -= self._weight * self.light_dist(X_light)

            if self._action_idx and self._action_idx < X.shape[1]:
                X_action = X_split[-1]
                k -= self._weight * self.action_dist(X_action)

        return np.exp(k)

    def diag(self, X):
        X_split = np.split(X, self._split_idx, axis=1)
        X_kilobots = X_split[0]
        k = - (1 - self._weight) * self.kilobots_dist.diag(X_kilobots)

        if self._light_idx and self._light_idx < X.shape[1]:
            X_light = X_split[1]
            k -= self._weight * self.light_dist.diag(X_light)

        if self._action_idx and self._action_idx < X.shape[1]:
            X_action = X_split[-1]
            k -= self._weight * self.action_dist.diag(X_action)

        return np.exp(k)

    def is_stationary(self):
        return self.light_dist.is_stationary() and self.kilobots_dist.is_stationary()


class KilobotEnvKernelWithWeight(KilobotEnvKernel):
    def __init__(self, bandwidth_factor_weight=1.0, weight_idx=None, weight_dist_class=MahaDist, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_dist = weight_dist_class(bandwidth_factor=bandwidth_factor_weight)
        self._weight_idx = weight_idx
        if self._action_idx:
            self._split_idx[-1] -= 1

    def __call__(self, X, Y=None):
        X_weight = X[:, self._weight_idx:self._action_idx]
        X_other = np.c_[X[:, :self._weight_idx], X[:, self._action_idx:]]

        if Y is not None:
            Y_weight = Y[:, self._weight_idx:self._action_idx]
            Y_other = np.c_[Y[:, :self._weight_idx], Y[:, self._action_idx:]]
            k_weights = np.exp(-.5 * self.weight_dist(X_weight, Y_weight))
        else:
            Y_other = None
            k_weights = np.exp(-.5 * self.weight_dist(X_weight))

        k_other = super().__call__(X_other, Y_other)

        return k_weights * k_other

    def diag(self, X):
        X_weight = X[:, self._weight_idx:self._action_idx]
        X_other = np.c_[X[:, :self._weight_idx], X[:, self._action_idx:]]

        return np.exp(-.5 * self.weight_dist.diag(X_weight)) * super().diag(X_other)
