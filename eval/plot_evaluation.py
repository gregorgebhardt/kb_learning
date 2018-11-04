import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import cycle


color_cycler = mpl.rcParams['axes.prop_cycle']
style_cycler = mpl.cycler('linestyle', ['-', '--', ':', '-.'])

# load pandas file
from eval.generate_circular_path import generate_circular_path

object_trajectories_kbc = pd.read_pickle('eval_kbc_pose_control_filtered_2.pkl')
object_trajectories_pc = pd.read_pickle('eval_pc_eval_pose_control_object_trajectories.pkl')
# object_trajectories = pd.read_pickle('eval_kbc_eval_circular_path_object_trajectories.pkl')
# object_trajectories = pd.read_pickle('eval_kbc_eval_straight_line_object_trajectories.pkl')

object_trajectories_kbc.columns = object_trajectories_kbc.columns.droplevel(0)
object_trajectories_pc.columns = object_trajectories_pc.columns.droplevel(0)

f = plt.figure()
axes_array = f.subplots(1, 3, squeeze=False)
axes = dict(zip(['rect_1', 'rect_2', 'rect_3'], axes_array.flat))

for n, a in axes.items():
    # a.set_title(n)
    a.set_xlim(-.75, .75)
    a.set_ylim(-.5, .5)
    a.set_aspect('equal')

for a in axes_array[:, 1:].flat:
    a.set_yticks([])
# for a in axes_array[0].flat:
#     a.set_xticks([])

# color_cycle = cycle(color_cycler)
# style_cycle = cycle(style_cycler)

for task, kbc_group in object_trajectories_kbc.groupby(level=0):
    # if shape.endswith('_null'):
    #     shape = shape[:-5]
    #     policy_type = 'square'
    #     ls = {'linestyle': '--'}
    # else:
    #     policy_type = 'learned'
    #     ls = {'linestyle': '-'}
    # shape, radius = shape.rsplit('_', 1)
    # radius = float(radius) / 10

    # obj_mean = obj_group.groupby(level=[0, 2]).mean()
    # obj_std = obj_group.groupby(level=0).std()

    ax = axes[task]

    pc_group = object_trajectories_pc.loc[task]

    for _traj_idx, kbc_traj in kbc_group.groupby(level=1):
        l1, = ax.plot(kbc_traj.x.values, kbc_traj.y.values, c='tab:blue', ls='--')
        pc_traj = pc_group.loc[_traj_idx]
        l2, = ax.plot(pc_traj.x.values, pc_traj.y.values, c='tab:orange', ls=':')
        l3, = ax.plot(kbc_traj.x.values[-1], kbc_traj.y.values[-1], c='tab:blue', marker='x')
        l4, = ax.plot(pc_traj.x.values[-1], pc_traj.y.values[-1], c='tab:orange', marker='x')

    l1.set_label('kilobot control')
    l2.set_label('pose control')

    # plt.fill_between(obj_mean.x, obj_mean.y + 2 * obj_std.y, obj_mean.y - 2 * obj_std.y)
    # plt.plot(obj_mean.x, obj_mean.y, label=obj_group_idx)

for k, a in axes.items():
    if k is 'rect_1':
        a.legend()
    # for r in [0.2, 0.4, 0.6, 0.8]:
    #     wpts = generate_circular_path(center=(-.4, .4), radius=r, angular_range=(-np.pi/2, -np.pi/20))
    #     points = np.array([wp.position for wp in wpts])
    #     a.plot(points[:, 0], points[:, 1], '.', c=(.2, .2, .2, .8))


# plt.legend()
pass
