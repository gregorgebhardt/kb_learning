import numpy as np

from matplotlib.colorbar import Colorbar
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib

matplotlib.rc('font', family='Oswald')


def reward_distribution_plot(R, R_axes: Axes):
    R_mean = np.mean(R, axis=1)
    R_std = np.std(R, axis=1)

    x = np.arange(R.shape[0])

    R_axes.plot(x, R, 'k-', alpha=.2)
    R_axes.fill_between(x, R_mean - 2 * R_std, R_mean + 2 * R_std, alpha=.5)
    R_axes.plot(x, R_mean)


def value_function_plot(V: np.ndarray, x_range, y_range, axes: Axes, cm_axes: Axes = None, **kwargs):
    im = axes.matshow(V, extent=x_range + y_range, **kwargs)  # norm=Normalize(-0.05, 0.05)

    if cm_axes is not None:
        Colorbar(cm_axes, im)
    return im


def trajectories_plot(T, x_range, y_range, axes: Axes, cm_axes=None, cmap='viridis', **kwargs):
    # light_states.index.shape
    num_episodes, num_steps, _ = T.shape

    line_collections = []

    axes.set_xlim(*x_range)
    axes.set_ylim(*y_range)

    for t in T:
        segments = np.r_['2,3,0', t[:-1], t[1:]].swapaxes(1, 2)
        # segments[:, :, 1] *= -1
        lc = LineCollection(segments, norm=Normalize(0, num_steps), cmap=cmap, **kwargs)
        color = np.arange(num_steps)
        lc.set_array(color)
        lc.set_linewidth(1.)
        line_collections.append(lc)

        axes.add_collection(lc)

    if cm_axes is not None:
        Colorbar(cm_axes, line_collections[0])

    return line_collections


def policy_plot(P, x_range, y_range, P_axes, cm_axes=None, **kwargs):
    if type(P) is tuple:
        A, sigma = P
        # color = (sigma - sigma.min()) / (sigma.max() - sigma.min()) / 2 + .5
        color = sigma
    else:
        A = P
        color = 1.

    [X, Y] = np.meshgrid(np.linspace(*x_range, A.shape[1]), np.linspace(*y_range, A.shape[0]))
    if A.shape[2] == 2:
        return P_axes.quiver(X, Y, A[..., 0], A[..., 1], color, angles='xy', **kwargs)
    else:
        A_cos = np.cos(A.squeeze())
        A_sin = np.sin(A.squeeze())
        return P_axes.quiver(X, Y, A_cos, A_sin, color, angles='xy', **kwargs)
