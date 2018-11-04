import os
import socket
from collections import namedtuple

import matplotlib

matplotlib.use('cairo')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from gym_kilobots.kb_plotting import plot_body, update_body
from matplotlib import patches
from matplotlib.animation import FFMpegWriter

from kb_learning.controller import PoseController
from kb_learning.envs import EvalEnv
from kb_learning.planning import AssemblyPolicy


def main():
    # configuration_file = 'eval/assembly_policies/eval_straight_line.yml'
    configuration_file = 'eval/assembly_policies/eval_pose_control.yml'

    filename_base, _ = os.path.splitext(os.path.basename(configuration_file))
    os.makedirs(os.path.join('out', filename_base), exist_ok=True)

    movie_filename_template = os.path.join('out', filename_base, 'eval_pc_{}_{}.mp4')
    figure_filename_template = os.path.join('out', filename_base, 'eval_pc_{}_{}_{}.png')

    with open(configuration_file) as f:
        conf = yaml.load(f)

    Evaluation = namedtuple('Evaluation', ['env', 'controller'])

    evaluations = dict()
    # iterate over eval_configurations here (if a list)
    if 'eval_configurations' in conf:
        for k, eval_conf in conf['eval_configurations'].items():
            _env = EvalEnv(eval_conf['env_configuration'])
            assembly_policy = AssemblyPolicy(eval_conf['assembly_policy_config'])
            _env.assembly_policy = assembly_policy
            controller = PoseController(assembly_policy, _env, k_p=.1, k_t=.2, C=.1)
            evaluations[k] = Evaluation(_env, controller)
    else:
        _env = EvalEnv(conf['env_configuration'])
        assembly_policy = AssemblyPolicy(conf['assembly_policy_config'])
        _env.assembly_policy = assembly_policy
        controller = PoseController(assembly_policy, _env, k_p=.1, k_t=.2, C=.1)
        evaluations[None] = Evaluation(_env, controller)

    # DataFrame for storing evaluation results
    num_objects = len(_env.get_objects())
    obj_trajectory_columns = pd.MultiIndex.from_product([range(num_objects), ['x', 'y', 'theta']])

    obj_trajectories_filename = 'eval_pc_{}_object_trajectories.pkl'.format(filename_base)
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
        for rep in range(10):
            print('running {} - rep {}'.format(evaluation_name, rep))
            # reset environments
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
            obj_artists = [plot_body(ax, obj) for obj in evaluation.env.get_objects()]
            obj_centers = [ax.plot(*obj.get_position(), '.r')[0] for obj in evaluation.env.get_objects()]

            # swarm_pos = obs['kilobots'][:, :2].mean(axis=0)
            # swarm_trace, = ax.plot(*swarm_pos, ':g', linewidth=1., zorder=-5)
            # light_trace, = ax.plot(*obs['light'], ':r', linewidth=1., zorder=-5)
            obj_trace, = ax.plot(obs['objects'][0, 0], obs['objects'][0, 1], ':k', linewidth=1., zorder=-5)

            def pos_angle2vec(x0, angle):
                return np.c_[x0, x0 + .03 * np.array([np.cos(angle), np.sin(angle)])]

            obj_orientations = [ax.plot(*pos_angle2vec(obj.get_position(), obj.get_orientation()), color='r')[0]
                                for obj in evaluation.env.get_objects()]
            # assembly_policy_plots = evaluation.controller.assembly_policy.plot_with_object_information(ax,
            #                                                                                            evaluation.env.get_objects())

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
                    plot_body(ax, active_obj, fill=False, ec='0.8')
                # evaluation.controller.assembly_policy.update_plot_with_object_information(assembly_policy_plots,
                #                                                                           evaluation.env.get_objects())

                if counter % 5 == 0:
                    # plot swarm
                    # swarm_pos = obs['kilobots'][:, :2].mean(axis=0)
                    # swarm_trace.set_xdata(np.append(swarm_trace.get_xdata(), swarm_pos[0]))
                    # swarm_trace.set_ydata(np.append(swarm_trace.get_ydata(), swarm_pos[1]))
                    # plot light
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
                    # fig.savefig('eval_pc_{}_{}_{}.png'.format(filename_base, evaluation_name, counter // 50))

                counter += 1
                action = evaluation.controller.compute_action(**obs)

                obs, reward, done, info = evaluation.env.step(action)
                reward_sum += reward
                obj_trajectory.append(obs['objects'])

            fig.savefig(figure_filename_template.format(evaluation_name, rep, 'end'))
            # fig.savefig('eval_pc_{}_{}_end.png'.format(filename_base, k))

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

if __name__ == '__main__':
    main()