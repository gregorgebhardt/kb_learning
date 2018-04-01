import numpy as np
from GPy import Param
from GPy.kern.src.kern import Kern
from paramz.transformations import Logexp

from . import EmbeddedSwarmDistance, MahaDist


class KilobotEnvKernel(Kern):
    def __init__(self, kilobots_dim, light_dim=0, weight_dim=0, action_dim=0,
                 rho=.5, active_dims=None):
        super(KilobotEnvKernel, self).__init__(input_dim=kilobots_dim+light_dim+weight_dim+action_dim,
                                               active_dims=active_dims, name='kilobot_env')
        self.kilobots_dim = kilobots_dim
        self.light_dim = light_dim
        self.weight_dim = weight_dim
        self.action_dim = action_dim

        self.kilobots_dist = EmbeddedSwarmDistance()
        self.light_dist = MahaDist()
        self.weight_dist = MahaDist()
        self.action_dist = MahaDist()

        self.kilobots_bandwidth = Param('kilobots_bandwidth', np.array([1.] * 2), Logexp())
        self.kilobots_bandwidth.add_observer(self, self.__kilobots_bandwidth_observer)
        self.link_parameter(self.kilobots_bandwidth)

        if light_dim:
            self.light_bandwidth = Param('light_bandwidth', np.array([1.] * light_dim), Logexp())
            self.light_bandwidth.add_observer(self, self.__light_bandwidth_observer)
            self.link_parameter(self.light_bandwidth)

        if weight_dim:
            self.weight_bandwidth = Param('weight_bandwidth', np.array([1.] * weight_dim), Logexp())
            self.weight_bandwidth.add_observer(self, self.__weight_bandwidth_observer)
            self.link_parameter(self.weight_bandwidth)

        if action_dim:
            self.action_bandwidth = Param('action_bandwidth', np.array([1.] * action_dim), Logexp())
            self.action_bandwidth.add_observer(self, self.__action_bandwidth_observer)
            self.link_parameter(self.action_bandwidth)

        self.rho = Param('rho', np.array([rho]))
        self.rho.constrain_bounded(.0, 1.)
        self.variance = Param('variance', np.array([1.]), Logexp())

        self.link_parameters(self.rho, self.variance)

    def __kilobots_bandwidth_observer(self, param, which):
        self.kilobots_dist.bandwidth = self.kilobots_bandwidth.values

    def __light_bandwidth_observer(self, param, which):
        self.light_dist.bandwidth = self.light_bandwidth.values

    def __weight_bandwidth_observer(self, param, which):
        self.weight_dist.bandwidth = self.weight_bandwidth.values

    def __action_bandwidth_observer(self, param, which):
        self.action_dist.bandwidth = self.action_bandwidth.values

    def K(self, X, X2=None, return_components=False):
        X_splits = np.split(X, np.cumsum([self.kilobots_dim, self.light_dim, self.weight_dim]), axis=1)
        X_kilobots = X_splits[0]
        X_light = X_splits[1]
        X_weight = X_splits[2]
        X_action = X_splits[3]

        if X2 is not None:
            Y_splits = np.split(X2, np.cumsum([self.kilobots_dim, self.light_dim, self.weight_dim]), axis=1)
            Y_kilobots = Y_splits[0]
            Y_light = Y_splits[1]
            Y_weight = Y_splits[2]
            Y_action = Y_splits[3]
        else:
            Y_kilobots = None
            Y_light = None
            Y_weight = None
            Y_action = None

        k_kilobots = - (1 - self.rho) * self.kilobots_dist(X_kilobots, Y_kilobots)
        if self.light_dim:
            k_light = - self.rho * self.light_dist(X_light, Y_light)
        else:
            k_light = .0
        if self.weight_dim:
            k_weight = - self.rho * self.weight_dist(X_weight, Y_weight)
        else:
            k_weight = .0
        if self.action_dim:
            k_action = - self.rho * self.action_dist(X_action, Y_action)
        else:
            k_action = .0

        if return_components:
            return (self.variance * np.exp(k_kilobots + k_light + k_weight + k_action),
                    k_kilobots, k_light, k_weight, k_action)
        return self.variance * np.exp(k_kilobots + k_light + k_weight + k_action)

    def Kdiag(self, X):
        X_splits = np.split(X, np.cumsum([self.kilobots_dim, self.light_dim, self.weight_dim]), axis=1)
        X_kilobots = X_splits[0]
        X_light = X_splits[1]
        X_weight = X_splits[2]
        X_action = X_splits[3]

        k_kilobots = -(1 - self.rho) * self.kilobots_dist.diag(X_kilobots)
        if self.light_dim:
            k_light = -self.rho * self.light_dist.diag(X_light)
        else:
            k_light = .0
        if self.weight_dim:
            k_weight = -self.rho * self.weight_dist.diag(X_weight)
        else:
            k_weight = .0
        if self.action_dim:
            k_action = -self.rho * self.action_dist.diag(X_action)
        else:
            k_action = .0

        return self.variance * np.exp(k_kilobots + k_light + k_weight + k_action)

    def update_gradients_full(self, dL_dK, X, Y=None):
        X_splits = np.split(X, np.cumsum([self.kilobots_dim, self.light_dim, self.weight_dim]), axis=1)
        X_kilobots = X_splits[0]
        X_light = X_splits[1]
        X_weight = X_splits[2]
        X_action = X_splits[3]

        if Y is not None:
            Y_splits = np.split(Y, np.cumsum([self.kilobots_dim, self.light_dim, self.weight_dim]), axis=1)
            Y_kilobots = Y_splits[0]
            Y_light = Y_splits[1]
            Y_weight = Y_splits[2]
            Y_action = Y_splits[3]
        else:
            Y_kilobots = None
            Y_light = None
            Y_weight = None
            Y_action = None

        k, k_kilobots, k_light, k_weight, k_action = self.K(X, Y, return_components=True)

        # compute gradient w.r.t. kernel bandwidths
        dK_kb_d_bw = self.kilobots_dist.get_distance_matrix_gradient(X_kilobots, Y_kilobots)
        dK_kb_d_bw *= (dL_dK * k)[..., None] * (self.rho - 1)
        self.kilobots_bandwidth.gradient = np.sum(dK_kb_d_bw, axis=(0, 1))
        if self.light_dim:
            dK_l_d_bw = self.light_dist.get_distance_matrix_gradient(X_light, Y_light)
            dK_l_d_bw *= (dL_dK * k)[..., None] * (-self.rho)
            self.light_bandwidth.gradient = np.sum(dK_l_d_bw, axis=(0, 1))
        if self.weight_dim:
            dK_w_d_bw = self.weight_dist.get_distance_matrix_gradient(X_weight, Y_weight)
            dK_w_d_bw *= (dL_dK * k)[..., None] * (-self.rho)
            self.weight_bandwidth.gradient = np.sum(dK_w_d_bw, axis=(0, 1))
        if self.action_dim:
            dK_a_d_bw = self.action_dist.get_distance_matrix_gradient(X_action, Y_action)
            dK_a_d_bw *= (dL_dK * k)[..., None] * (-self.rho)
            self.action_bandwidth.gradient = np.sum(dK_a_d_bw, axis=(0, 1))

        # compute gradient w.r.t. rho
        self.rho.gradient = np.sum(dL_dK * k * (-k_kilobots - k_light - k_weight - k_action))

        # compute gradient w.r.t. variance
        self.variance.gradient = np.sum(dL_dK * k) / self.variance

    def update_gradients_diag(self, dL_dKdiag, X):
        # compute gradient w.r.t. kernel bandwidths
        # self.kilobots_bandwidth.gradient = np.zeros(2)
        # self.light_bandwidth.gradient = np.zeros(self.light_dim)
        # self.weight_bandwidth.gradient = np.zeros(self.weight_dim)
        # self.action_bandwidth.gradient = np.zeros(self.action_dim)
        # compute gradient w.r.t. rho
        # self.rho.gradient = 0
        # compute gradient w.r.t. variance
        self.variance.gradient = np.sum(dL_dKdiag)

    def __call__(self, X, Y=None):
        return self.K(X, Y)

    def diag(self, X):
        return self.Kdiag(X)
