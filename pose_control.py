import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def get_pose_controller(goal_pose, k_p, k_t, C):
    # compute line through target pose
    a = -np.sin(goal_pose[2])
    b = np.cos(goal_pose[2])
    c = -a * goal_pose[0] - b * goal_pose[1]

    width = .05
    height = .3

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim(-.8, .8)
    ax.set_ylim(-.6, .6)
    ax.set_aspect('equal')

    def _plot_patch(pose, **kwargs):
        _a = -np.sin(pose[2])
        _b = np.cos(pose[2])
        left = -_b * width / 2 - _a * height / 2 + pose[0]
        bottom = _a * width / 2 - _b * height / 2 + pose[1]

        patch = patches.Rectangle((left, bottom), width, height, np.rad2deg(pose[2]), **kwargs)
        ax.add_patch(patch)
        ax.plot(goal_pose[0], goal_pose[1], 'xr')

        return patch

    fig.show()

    def _update_patch(patch: patches.Rectangle, pose):
        _a = -np.sin(pose[2])
        _b = np.cos(pose[2])
        left = -_b * width / 2 - _a * height / 2 + pose[0]
        bottom = _a * width / 2 - _b * height / 2 + pose[1]

        patch.set_xy((left, bottom))
        patch.angle = np.rad2deg(pose[2])

    _plot_patch(goal_pose, fill=False, clip_on=False, ls='--', ec='grey')
    x = np.linspace(-.8, .8, 100)
    plt.plot(x, -a / b * x - c / b, ':')
    o_p_line = plt.plot(x, -1 * np.ones(100), ':')[0]

    intermediate_patch = _plot_patch(goal_pose, fill=False, clip_on=False, ls=':', ec='grey')
    current_patch = _plot_patch(goal_pose, fill='blue', clip_on=False, ls='-', ec='blue')

    def _closest_point_on_line(point, _a, _b, _c):
        ab = np.array([_a, _b])
        ba = np.flipud(ab)
        abab = ab.dot(ab)
        p_x = (_b * (point * [1, -1]).dot(ba) - _a * _c) / abab
        p_y = (_a * (point * [-1, 1]).dot(ba) - _b * _c) / abab

        return np.array([p_x, p_y])

    def _which_side_of_line(point, _a, _b, _c):
        return np.sign(_a * point[0] + _b * point[1] + _c)

    def pose_controller(swarm_pos, light_pos, object_pose):
        _update_patch(current_patch, object_pose)

        # compute closest point of object on line (a, b, c)
        p = _closest_point_on_line(object_pose[:2], a, b, c)
        # d_theta = goal_pose[2] - object_pose[2]

        # distance to line
        d_o_p = object_pose[:2] - p
        d = np.linalg.norm(d_o_p)

        if d < C:
            # if distance below threshold, we do line following along (a, b, c)
            p_s = _closest_point_on_line(object_pose[:2], a, b, c)
            theta = goal_pose[2]

            _update_patch(intermediate_patch, goal_pose)
            o_p_line.set_ydata(-1 * np.ones(100))
        else:
            # if distance above threshold, compute line perpendicular to (a,b,c) through p (thus through object pos)
            theta = goal_pose[2] - np.pi / 2 * _which_side_of_line(object_pose[:2], a, b, c)
            a_t = -np.sin(theta)
            b_t = np.cos(theta)
            c_t = -a_t * p[0] - b_t * p[1]

            _update_patch(intermediate_patch, (*p, theta))
            o_p_line.set_ydata(-a_t / b_t * x - c_t / b_t)

            # compute closest point of swarm on line (a_t, b_t, c_t)
            p_s = _closest_point_on_line(object_pose[:2], a_t, b_t, c_t)

        sc_theta = np.array([-np.sin(object_pose[2]), np.cos(object_pose[2])])

        # if k_p * d_o_p[0] + k_t * np.abs(d_theta) > .1:

        d_o_p_s = object_pose[:2] - p_s
        d_theta = theta - object_pose[2]

        target_swarm = object_pose[:2] - k_p * d_o_p_s - k_t * sc_theta * d_theta

        ax.plot(target_swarm[0], target_swarm[1], '.k', markersize=.5)

        swarm_vec = target_swarm - swarm_pos
        ax.plot(swarm_pos[0], swarm_pos[1], '.g', markersize=.5)
        swarm_vec *= .05 / np.linalg.norm(swarm_vec)

        fig.canvas.draw()
        fig.show()

        return swarm_pos + swarm_vec - light_pos

    return pose_controller
