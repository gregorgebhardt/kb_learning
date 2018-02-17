from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib

from gym_kilobots.envs import KilobotsEnv

import mpld3

import pandas as pd
import numpy as np

import os

matplotlib.rc('font', family='Oswald')


def plot_light_trajectory(axes: Axes, light_states: pd.DataFrame):
    # light_states.index.shape
    num_episodes, num_steps = light_states.index.levshape

    line_collections = []

    for _, traj in light_states.groupby(level=0):
        segments = np.r_['2,3,0', traj[:-1], traj[1:]].swapaxes(1, 2)
        segments[:, :, 1] *= -1
        lc = LineCollection(segments, cmap='viridis', norm=Normalize(10, num_steps-10))
        color = np.arange(num_steps)
        lc.set_array(color)
        lc.set_linewidth(1.)
        line_collections.append(lc)

        axes.add_collection(lc)

    return line_collections


def plot_value_function(axes: Axes, V, x_range, y_range, **kwargs):
    im = axes.imshow(V, extent=x_range+y_range, **kwargs)
    return im


def plot_policy(axes: Axes, actions, x_range, y_range, **kwargs):
    if type(actions) is tuple:
        A, sigma = actions
        # color = (sigma - sigma.min()) / (sigma.max() - sigma.min()) / 2 + .5
        color = sigma
    else:
        A = actions
        color = 1.

    [X, Y] = np.meshgrid(np.linspace(*x_range, A.shape[1]), np.linspace(*y_range, A.shape[0]))
    return axes.quiver(X, Y, A[..., 0], A[..., 1], color, angles='xy', **kwargs)


def plot_objects(axes: Axes, env: KilobotsEnv, **kwargs):
    for o in env.get_objects():
        o.plot(axes, alpha=.5, **kwargs)


def plot_trajectory_reward_distribution(axes: Axes, reward: pd.DataFrame):
    mean_reward = reward.groupby(level=1).mean()
    std_reward = reward.groupby(level=1).std()

    x = np.arange(mean_reward.shape[0])

    axes.plot(x, reward.unstack(level=0), 'k-', alpha=.3)
    axes.fill_between(x, mean_reward-2*std_reward, mean_reward+2*std_reward, alpha=.5)
    axes.plot(x, mean_reward)


def save_plot_as_html(figure, filename=None, path=None, overwrite=True):
    if path is None:
        import tempfile
        path = tempfile.gettempdir()
    if filename is None:
        filename = 'plot.html'

    html_full_path = os.path.join(path, filename)

    if overwrite and os.path.exists(html_full_path):
        os.remove(html_full_path)
    elif os.path.exists(html_full_path):
        root, ext = os.path.splitext(html_full_path)
        root_i = root + '_{}'
        i = 1
        while os.path.exists(html_full_path):
            html_full_path = root_i.format(i) + ext
            i = i + 1

    html_data = mpld3.fig_to_html(figure)

    with open(html_full_path, mode='w') as html_file:
        # mpld3.save_html(figure, html_file)
        html_file.write(html_data)

    return html_full_path

browser_controller = None


def show_plot_in_browser(figure, filename=None, path=None, overwrite=True, browser='google-chrome',
                         save_only=False):
    html_full_path = save_plot_as_html(figure, filename, path, overwrite)

    if save_only:
        return

    global browser_controller
    if browser_controller is None:
        import webbrowser
        browser_controller = webbrowser.get(browser)

    browser_controller.open(html_full_path)
