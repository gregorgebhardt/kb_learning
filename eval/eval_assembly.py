import os
import socket

import numpy as np
import pandas as pd
import yaml
import pickle

from collections import namedtuple

import matplotlib
from matplotlib.animation import FFMpegWriter

matplotlib.use('cairo')

from gym_kilobots.kb_plotting import plot_body, update_body
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import patches

from kb_learning.envs import EvalEnv
from kb_learning.ac_reps.spwgp import SparseWeightedGP
from kb_learning.planning import AssemblyPolicy, PolicyContainer
from kb_learning.controller import KilobotController


def load_policies(policy_configurations) -> PolicyContainer:
    policy_container = PolicyContainer()

    # iterator over policy configurations
    for policy_conf in policy_configurations:
        # iterate over w_factors
        for w in policy_conf['w_factors']:
            # load policy from pickle file
            with open(policy_conf['path_template'].format(w_factor=w), 'rb') as f:
                policy_dict = pickle.load(f)

            # fix policy dict (if it still contains `weight_dim` replace it with `extra_dim`
            if 'weight_dim' in policy_dict['kernel']:
                policy_dict['kernel']['extra_dim'] = policy_dict['kernel']['weight_dim']
                del policy_dict['kernel']['weight_dim']

            # construct policy and add to container
            policy = SparseWeightedGP.from_dict(policy_dict)
            policy_container.add_policy(policy, w, object_type=policy_conf['object_type'])

    return policy_container


def process_circular_path(object_trajectories: pd.DataFrame):
    # correct offset of circle center
    object_trajectories['x'] -= -.5
    object_trajectories['y'] -= .5
    # compute polar coordinates of object trajectory
    object_trajectories['r'] = np.sqrt((object_trajectories[['x', 'y']] ** 2).sum(axis=1))
    object_trajectories['a'] = np.arctan2(object_trajectories['y'], object_trajectories['x'])
    # compute error of the radius component
    object_trajectories['e'] = np.abs(object_trajectories['r'] - 1.)

    # we want to know for which angular range, we can compare the object trajectories
    # compute max of the minimum angle of the trajectories
    max_min_angle = object_trajectories.a.groupby(level=[0, 1]).min().max()
    # compute min of the maximum angle of the trajectories
    min_max_angle = object_trajectories.a.groupby(level=[0, 1]).max().min()

    print('angle range: {} - {}'.format(max_min_angle, min_max_angle))

    # compute sample points for the interpolation
    n_sample_points = 100
    sample_angles = np.round(np.linspace(max_min_angle + .01, min_max_angle - .01, n_sample_points), 2)

    # create result series
    radius_errors = pd.Series(index=pd.MultiIndex.from_product([object_trajectories.index.levels[0],
                                                                object_trajectories.index.levels[1],
                                                                sample_angles]))
    radius_errors.name = 'radius error'
    theta_errors = pd.Series(index=pd.MultiIndex.from_product([object_trajectories.index.levels[0],
                                                               object_trajectories.index.levels[1],
                                                               sample_angles]))
    theta_errors.name = 'theta error'

    for episode_idx, episode in object_trajectories.groupby(level=(0, 1)):
        # get angles and errors
        a = episode.a.values
        e = episode.e.values
        t = episode.theta.values

        # filter for unique angle values
        a, idx = np.unique(a, return_index=True)
        e = e[idx]
        t = t[idx]

        # interpolate error function
        e_fun = interp1d(a, e, kind='nearest')
        t_fun = interp1d(a, t, kind='nearest')

        series_index = pd.MultiIndex.from_product([[episode_idx[0]], [episode_idx[1]], sample_angles])
        radius_errors = radius_errors.append(pd.Series(e_fun(sample_angles), index=series_index))
        theta_errors = theta_errors.append(pd.Series(t_fun(sample_angles) - sample_angles - np.pi / 2,
                                                     index=series_index))

    return radius_errors, theta_errors


def main():
    # configuration_file = 'eval/assembly_policies/eval_circular_path.yml'
    # configuration_file = 'eval/assembly_policies/eval_straight_line.yml'
    configuration_file = 'eval/assembly_policies/eval_pose_control.yml'
    # configuration_file = 'eval/assembly_policies/eval_triangle_assembly.yml'
    # configuration_file = 'eval/assembly_policies/eval_L_assembly.yml'
    # configuration_file = 'eval/assembly_policies/eval_CT_assembly.yml'

    filename_base, _ = os.path.splitext(os.path.basename(configuration_file))
    os.makedirs(os.path.join('out', filename_base), exist_ok=True)

    movie_filename_template = os.path.join('out', filename_base, 'eval_kbc_{}_{}.mp4')
    figure_filename_template = os.path.join('out', filename_base, 'eval_kbc_{}_{}_{}.png')

    with open(configuration_file) as f:
        conf = yaml.load(f)

    Evaluation = namedtuple('Evaluation', ['env', 'controller'])

    evaluations = dict()
    # iterate over eval_configurations here (if a list)
    if 'eval_configurations' in conf:
        for k, eval_conf in conf['eval_configurations'].items():
            _env = EvalEnv(eval_conf['env_configuration'])
            if eval_conf['assembly_policy_config'].way_points[0].obj_conf is None:
                _assembly_policy = AssemblyPolicy(eval_conf['assembly_policy_config'],
                                                  eval_conf['env_configuration'].objects[0])
            else:
                _assembly_policy = AssemblyPolicy(eval_conf['assembly_policy_config'])
            _env.assembly_policy = _assembly_policy
            _policy_container = load_policies(eval_conf['pushing_policies'])
            _controller = KilobotController(_policy_container, _assembly_policy, _env)
            evaluations[k] = Evaluation(_env, _controller)
    else:
        _env = EvalEnv(conf['env_configuration'])
        _assembly_policy = AssemblyPolicy(conf['assembly_policy_config'])
        _env.assembly_policy = _assembly_policy
        _policy_container = load_policies(conf['pushing_policies'])
        _controller = KilobotController(_policy_container, _assembly_policy, _env)
        evaluations[None] = Evaluation(_env, _controller)

    # DataFrame for storing evaluation results
    num_objects = len(_env.get_objects())
    obj_trajectory_columns = pd.MultiIndex.from_product([range(num_objects), ['x', 'y', 'theta']])

    obj_trajectories_filename = 'eval_kbc_{}_object_trajectories.pkl'.format(filename_base)
    if os.path.exists(obj_trajectories_filename):
        obj_trajectories = pd.read_pickle(obj_trajectories_filename)
    else:
        obj_trajectories = pd.DataFrame(columns=obj_trajectory_columns)

    plt.ion()
    if socket.gethostname() == 'johnson':
        plt.rcParams['animation.ffmpeg_path'] = '/home/gebhardt/bin/miniconda3/envs/dme/bin/ffmpeg'
    elif socket.gethostname() == 'wallace':
        plt.rcParams['animation.ffmpeg_path'] = '/home/gebhardt16/bin/miniconda3/envs/dme/bin/ffmpeg'
    else:
        plt.rcParams['animation.ffmpeg_path'] = '/Users/gregor/Applications/miniconda3/envs/dme/bin/ffmpeg'

    # run evaluations
    for evaluation_name, evaluation in evaluations.items():
        for rep in [6]:
            print('running {} - rep {}'.format(evaluation_name, rep))
            # reset environment
            obs = evaluation.env.reset()
            for _ in range(10):
                obs, reward, done, info = evaluation.env.step(None)

            evaluation.controller.reset()
            # evaluation.env.render()

            fig = plt.figure()
            ax = fig.gca()
            ax.set_xlim(evaluation.env.world_x_range)
            ax.set_ylim(evaluation.env.world_y_range)
            ax.set_aspect('equal')
            evaluation.controller.set_axes(ax)

            movie_writer = FFMpegWriter()
            movie_writer.setup(fig, movie_filename_template.format(evaluation_name, rep))

            def plot_kb(axes, position):
                p = patches.Circle(position, 0.016, color=(.2, .2, .2, .5), fill=True, ls='-',
                                   edgecolor=(.2, .2, .2, 1.), zorder=5)
                axes.add_patch(p)
                return p

            kb_artists = [plot_kb(ax, kb[:2]) for kb in obs['kilobots']]
            obj_artists = [plot_body(ax, obj, facecolor=(0, .7, .7)) for obj in evaluation.env.get_objects()]
            obj_centers = [ax.plot(*obj.get_position(), '.r')[0] for obj in evaluation.env.get_objects()]

            # swarm_pos = obs['kilobots'][:, :2].mean(axis=0)
            # swarm_trace, = ax.plot(*swarm_pos, ':g', linewidth=1., zorder=-5)
            # light_trace, = ax.plot(*obs['light'], ':r', linewidth=1., zorder=-5)
            ax.plot([wp.position[0] for wp in evaluation.controller.assembly_policy._way_points],
                    [wp.position[1] for wp in evaluation.controller.assembly_policy._way_points],
                    ':', c='gray', linewidth=1., zorder=-6)
            # ax.plot([-.4, .4], [.0, .0], ':', c='gray', linewidth=1., zorder=-6)
            obj_trace, = ax.plot(obs['objects'][0, 0], obs['objects'][0, 1], ':k', linewidth=1., zorder=-5)

            def pos_angle2vec(x0, angle):
                return np.c_[x0, x0 + .03 * np.array([np.cos(angle), np.sin(angle)])]

            obj_orientations = [ax.plot(*pos_angle2vec(obj.get_position(), obj.get_orientation()), color='r')[0]
                                for obj in evaluation.env.get_objects()]
            # assembly_policy_plots = evaluation.controller.assembly_policy.plot(ax)
            assembly_policy_plots = evaluation.controller.assembly_policy.plot_with_object_information(ax,
                                                                                                       evaluation.env.get_objects())

            reward_sum = 0
            obj_trajectory = [obs['objects']]

            counter = 0
            while not evaluation.controller.assembly_policy.finished() and counter < conf['max_eval_steps']:
                # plot objects
                for o, a, c, v in zip(evaluation.env.get_objects(), obj_artists, obj_centers, obj_orientations):
                    update_body(o, a)
                    c.set_data(*o.get_position())
                    v.set_data(*pos_angle2vec(o.get_position(), o.get_orientation()))
                for k, p in zip(kb_artists, obs['kilobots']):
                    k.center = p[:2]
                if counter % 50 == 0:
                    # plot trace of active object
                    active_obj = evaluation.env.get_objects()[evaluation.controller.assembly_policy.get_object_idx()]
                    plot_body(ax, active_obj, fill=False, ec='0.8', zorder=-10, ls=':')
                evaluation.controller.assembly_policy.update_plot_with_object_information(assembly_policy_plots,
                                                                                          evaluation.env.get_objects())

                if counter % 5 == 0:
                    # plot swarm trace
                    # swarm_pos = obs['kilobots'][:, :2].mean(axis=0)
                    # swarm_trace.set_xdata(np.append(swarm_trace.get_xdata(), swarm_pos[0]))
                    # swarm_trace.set_ydata(np.append(swarm_trace.get_ydata(), swarm_pos[1]))
                    # plot light trace
                    # light_trace.set_xdata(np.append(light_trace.get_xdata(), obs['light'][0]))
                    # light_trace.set_ydata(np.append(light_trace.get_ydata(), obs['light'][1]))
                    # plot object trace
                    obj_trace.set_xdata(np.append(obj_trace.get_xdata(), obs['objects'][0, 0]))
                    obj_trace.set_ydata(np.append(obj_trace.get_ydata(), obs['objects'][0, 1]))

                fig.canvas.draw()
                fig.show()
                movie_writer.grab_frame()
                if not counter % 50:
                    fig.savefig(figure_filename_template.format(evaluation_name, rep, counter // 50))
                    # fig.savefig('eval_kbc_{}_{}_{}_{}.png'.format(filename_base, evaluation_name, rep, counter // 50))

                counter += 1
                action = evaluation.controller.compute_action(**obs)

                obs, reward, done, info = evaluation.env.step(action)
                reward_sum += reward
                obj_trajectory.append(obs['objects'])

            fig.savefig(figure_filename_template.format(evaluation_name, rep, 'end'))
            # fig.savefig('eval_kbc_{}_{}_{}_end.png'.format(filename_base, evaluation_name, rep))

            obj_trajectory_index = pd.MultiIndex.from_product([[evaluation_name], [rep], range(len(obj_trajectory))])
            obj_trajectory = np.asarray(obj_trajectory).reshape((-1, 3 * num_objects))
            if (evaluation_name, rep) in obj_trajectories.index:
                obj_trajectories.drop(index=(evaluation_name, rep), inplace=True)
            obj_trajectory = pd.DataFrame(obj_trajectory, index=obj_trajectory_index,
                                          columns=obj_trajectory_columns)
            obj_trajectories = pd.concat((obj_trajectories, obj_trajectory))

            movie_writer.finish()
            plt.close(fig)

    # obj_trajectories.index = pd.MultiIndex.from_tuples(obj_trajectories.index, names=['eval', 'rep', 'step'])
    obj_trajectories.to_pickle(obj_trajectories_filename.format(filename_base))

    # # postprocessing of trajectories
    # processed_results = process_circular_path(obj_trajectories)
    #
    # f = plt.figure()
    # if len(processed_results) > 3:
    #     gs_rows = len(processed_results) // 2 + len(processed_results) % 2
    #     gs_cols = 2
    # else:
    #     gs_rows = len(processed_results)
    #     gs_cols = 1
    # gs = plt.GridSpec(gs_rows, gs_cols)
    #
    # for i, result in enumerate(processed_results):
    #     # plot mean and std
    #     _mean = result.groupby(level=[0, 2]).mean()
    #     _std = result.groupby(level=[0, 2]).std()
    #     x = _mean.index.tolist()
    #
    #     ax = f.add_subplot(gs[i])
    #     ax.set_title(result.name)
    #
    #     for s_idx, s_re in _std.groupby(level=0):
    #         m = _mean.loc[s_idx].values
    #
    #         ax.fill_between(x, m - 2 * s_re.values, m + 2 * s_re.values, alpha=.3)
    #
    #     for m_idx, m in _mean.groupby(level=0):
    #         ax.plot(x, m.values, label=str(m_idx))
    #
    # plt.legend()
    # plt.show(block=True)


if __name__ == '__main__':
    main()
