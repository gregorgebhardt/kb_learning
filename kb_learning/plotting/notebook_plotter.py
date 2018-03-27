import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from IPython.display import display, clear_output
import ipywidgets as widgets
import traitlets

# import os
# import mpld3

import time

import plot_work

from kb_learning.learner import ACRepsLearner

from .iteration_results_plotters import *
from .plotting_helper import *

cmap_plasma = cm.get_cmap('plasma')
cmap_gray = cm.get_cmap('gray')

# def save_plot_as_html(figure, filename=None, path=None, overwrite=True):
#     if path is None:
#         import tempfile
#         path = tempfile.gettempdir()
#     if filename is None:
#         filename = 'plot.html'
#
#     html_full_path = os.path.join(path, filename)
#
#     if overwrite and os.path.exists(html_full_path):
#         os.remove(html_full_path)
#     elif os.path.exists(html_full_path):
#         root, ext = os.path.splitext(html_full_path)
#         root_i = root + '_{}'
#         i = 1
#         while os.path.exists(html_full_path):
#             html_full_path = root_i.format(i) + ext
#             i = i + 1
#
#     html_data = mpld3.fig_to_html(figure)

def reward_plot_output(R):
    f = plt.figure()
    ax_R = plt.gca()
    reward_distribution_plot(R, ax_R)

    out = widgets.Output()
    with out:
        clear_output(wait=True)
        display(f)

    return out


def value_function_plot_output(V, x_range, y_range, obj=None):
    f = plt.figure()
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    value_function_plot(V, x_range, y_range, axes=ax, cm_axes=cax, cmap=cmap_plasma)
    if obj is not None:
        obj.plot(ax, alpha=.3, fill=True)

    out = widgets.Output()
    with out:
        clear_output(wait=True)
        display(f)

    return out


def trajectory_plot_output(T, x_range, y_range, V=None, obj=None):
    f = plt.figure()
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    if V is not None:
        value_function_plot(V, x_range, y_range, axes=ax, cmap=cmap_gray)
    trajectories_plot(T, x_range, y_range, ax, cm_axes=cax)
    if obj is not None:
        obj.plot(ax, alpha=.3, fill=True)

    out = widgets.Output()
    with out:
        clear_output(wait=True)
        display(f)

    return out


@plot_work.register_iteration_plot_function('animate_trajectories')
def trajectory_animation_output(learner: ACRepsLearner, config):
    params = config['params']

    num_episodes = params['eval']['num_episodes']
    num_steps = params['eval']['num_steps_per_episode']

    kb_T = learner.eval_info[learner.kilobots_columns].values.reshape((num_episodes * num_steps, -1, 2))
    light_T = learner.eval_info.S.loc[:, 'light'].values
    reward_T = learner.eval_sars.R.values

    object_T = learner.eval_info.S.loc[:, 'object'].values

    from sklearn.gaussian_process.kernels import RBF
    kernel = RBF()
    kernel.set_params(length_scale=learner.kernel.kilobots_dist.get_bandwidth())
    kernel_l = RBF()
    kernel_l.set_params(length_scale=learner.kernel.light_dist.get_bandwidth())

    N = 40
    X, Y = np.meshgrid(np.linspace(-.4, .4, N), np.linspace(-.4, .4, N))
    XY = np.c_[X.flat, Y.flat]

    f = plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.yaxis.tick_right()

    range_R = reward_T.min(), reward_T.max()

    out = widgets.Output()
    # with out:
    #     display(f)

    def update(event):
        if isinstance(event, dict):
            i = event['new']
        else:
            i = event
        # draw kilobot density
        K = kernel(XY, kb_T[i]).sum(axis=1).reshape(N, N) * 2 / kb_T.shape[1]
        K_l = kernel_l(XY, light_T[[i]]).reshape(N, N)

        obj = get_object(object_shape=params['sampling']['object_shape'],
                         object_width=params['sampling']['object_width'],
                         object_height=params['sampling']['object_height'],
                         object_init=object_T[i])

        with out:
            ax.clear()
            ax.contourf(X, Y, K, cmap=cm.BuPu)
            ax.contour(X, Y, K_l, cmap=cm.YlGn_r, linewidths=.5)
            obj.plot(ax, alpha=.3, fill=True)
            ax.scatter(kb_T[i, :, 0], kb_T[i, :, 1], s=10)
            ax.add_patch(Circle(light_T[i], radius=.2, color=(0.4, 0.7, 0.3, 0.3), fill=False))
            ax.plot(light_T[i, 0], light_T[i, 1], 'kx', markersize=5)
            ax.set_xlim([-.4, .4])
            ax.set_ylim([-.4, .4])

            cax.clear()
            cax.bar(0, reward_T[i])
            cax.set_xlim([-.5, .5])
            cax.set_ylim([*range_R])
            cax.set_xticks([])
            # f.canvas.draw()
            # f.show()
            # plt.show()

            clear_output(wait=True)
            display(f)

    update(0)

    play = widgets.Play(value=0, min=0, step=1, max=kb_T.shape[0])
    slider = widgets.IntSlider(value=0, min=0, step=1, max=kb_T.shape[0], continuous_update=False, disabled=False)
    widgets.jslink((play, 'value'), (slider, 'value'))
    play.observe(update, names='value')

    return widgets.VBox(children=[out, widgets.HBox([play, slider])])


@plot_work.register_iteration_plot_function('fixed_weight')
def plot_fixed_weight_iteration(learner, config):
    params = config['params']

    def state_action_features(state, action):
        if state.ndim == 1:
            state = state.reshape((1, -1))
        if action.ndim == 1:
            action = action.reshape((1, -1))
        return learner.kernel(np.c_[state, action], learner.lstd_samples.values)

    # setup figure
    fig = plt.figure(figsize=(10, 20))
    gs = GridSpec(nrows=4, ncols=2, width_ratios=[20, 1], height_ratios=[1, 3, 3, 3])

    x_range = (-.4, .4)
    y_range = (-.4, .4)

    obj = get_object(object_shape=params['sampling']['object_shape'],
                     object_width=params['sampling']['object_width'],
                     object_height=params['sampling']['object_height'],
                     object_init=(.0, .0, .0))
    num_kilobots = params['sampling']['num_kilobots']
    num_episodes = params['eval']['num_episodes']
    num_steps = params['eval']['num_steps_per_episode']

    V = compute_value_function_grid(state_action_features, learner.policy, learner.theta, num_kilobots=num_kilobots,
                                    x_range=x_range, y_range=y_range)
    P = compute_policy_quivers(learner.policy, num_kilobots, x_range, y_range)
    T = learner.eval_sars['S']['light'].values.reshape((num_episodes, num_steps, 2))
    R = learner.eval_sars['R'].unstack(level=0).values

    # reward plot
    R_out = reward_plot_output(R)

    # value function plot
    V_out = value_function_plot_output(V, x_range, y_range, obj=obj)

    # trajectories plot
    T_out = trajectory_plot_output(T, x_range, y_range, V=V, obj=obj)

    # new policy plot
    # ax_aft_P = fig.add_subplot(gs[3, 0])
    # ax_aft_P_cb = fig.add_subplot(gs[3, 1])
    # value_function_plot(V, x_range, y_range, axes=ax_aft_P, cmap=cmap_gray)
    # policy_plot(P, x_range, y_range, ax_aft_P, cm_axes=ax_aft_P_cb, cmap=cmap_plasma)
    # obj.plot(ax_bef_T, alpha=.3, fill=True)

    box = widgets.VBox(children=[R_out, V_out, T_out])

    # save and show plot
    return box
