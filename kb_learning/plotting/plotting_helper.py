import numpy as np


def compute_value_function_grid(state_action_features, policy, theta, num_kilobots, x_range, y_range, resolution=40):
    if type(resolution) is not tuple:
        resolution = (resolution, resolution)

    x_space = np.linspace(*x_range, resolution[0])
    y_space = np.linspace(*y_range, resolution[1])
    [X, Y] = np.meshgrid(x_space, y_space)
    X = X.flatten()
    Y = -Y.flatten()

    # kilobots at light position
    states = np.tile(np.c_[X, Y], [1, num_kilobots + 1])

    # get mean actions
    actions = policy.get_mean(states)

    value_function = state_action_features(states, actions).dot(theta).reshape((resolution[1], resolution[0]))

    return value_function


def compute_policy_quivers(policy, num_kilobots, x_range, y_range, resolution=40):
    if type(resolution) is not tuple:
        resolution = (resolution, resolution)

    [X, Y] = np.meshgrid(np.linspace(*x_range, resolution[0]), np.linspace(*y_range, resolution[1]))
    X = X.flatten()
    Y = Y.flatten()

    # kilobots at light position
    states = np.tile(np.c_[X, Y], [1, num_kilobots + 1])

    # get mean actions
    mean_actions, sigma_actions = policy.get_mean_sigma(states)
    mean_actions = mean_actions.reshape((resolution[1], resolution[0], mean_actions.shape[1]))
    sigma_actions = sigma_actions.reshape((resolution[1], resolution[0]))

    actions = mean_actions, sigma_actions

    return actions, x_range, y_range


def get_object(object_shape, object_width, object_height, object_init):
    from gym_kilobots.lib import Quad, Triangle, Circle, LForm, TForm, CForm
    from Box2D import b2World

    fake_world = b2World()

    if object_shape in ['quad', 'rect']:
        return Quad(width=object_width, height=object_height,
                    position=object_init[:2], orientation=object_init[2],
                    world=fake_world)
    elif object_shape == 'triangle':
        return Triangle(width=object_width, height=object_height,
                        position=object_init[:2], orientation=object_init[2],
                        world=fake_world)
    elif object_shape == 'circle':
        return Circle(radius=object_width, position=object_init[:2],
                      orientation=object_init[2], world=fake_world)
    elif object_shape == 'l_shape':
        return LForm(width=object_width, height=object_height,
                     position=object_init[:2], orientation=object_init[2],
                     world=fake_world)
    elif object_shape == 't_shape':
        return TForm(width=object_width, height=object_height,
                     position=object_init[:2], orientation=object_init[2],
                     world=fake_world)
    elif object_shape == 'c_shape':
        return CForm(width=object_width, height=object_height,
                     position=object_init[:2], orientation=object_init[2],
                     world=fake_world)
