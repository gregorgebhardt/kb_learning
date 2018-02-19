from gym.envs.registration import register

from ._fixed_weight_quad import QuadEnv_w000_kb15, QuadEnv_w025_kb15, QuadEnv_w050_kb15, \
    QuadEnv_w075_kb15, QuadEnv_w100_kb15

__gym_classes = [QuadEnv_w000_kb15, QuadEnv_w025_kb15, QuadEnv_w050_kb15,
                 QuadEnv_w075_kb15, QuadEnv_w100_kb15]

kb_learning_gyms = ['QuadEnv_w000_kb15-v0',
                    'QuadEnv_w025_kb15-v0',
                    'QuadEnv_w050_kb15-v0',
                    'QuadEnv_w075_kb15-v0',
                    'QuadEnv_w100_kb15-v0']

for cl, id in zip(__gym_classes, kb_learning_gyms):
    register(id, entry_point='kb_learning.envs:' + cl.__name__)


class QuadEnvNotAvailableError(Exception):
    def __init__(self, weight, num_kilobots):
        self.message = "No quad-pushing environment for w = {} and #kb = {} availabe.".format(weight, num_kilobots)
        self.weight = weight
        self.num_kilobots = num_kilobots


def get_fixed_weight_quad_env(weight: float, num_kilobots: int):
    id = 'QuadEnv_w{:03}_kb{}-v0'.format(int(weight * 100), num_kilobots)
    if id in kb_learning_gyms:
        return id
    else:
        raise QuadEnvNotAvailableError(weight, num_kilobots)


# from typing import Iterable, Union, Iterator
#
# __registered_gyms = list()
#
#
# def register_quadpushing_environment(weight: float, num_kilobots: int):
#     """Create a subclass of the Quad environment of the Kilobot gym with the given weight and number of
#     Kilobots and register the subclass as a gym. Returns the id of the registered gym.
#
#     :param weight: the weight for the Quad environment
#     :param num_kilobots: the number of kilobots for the Quad environment
#     :return: the id of the registered environment
#     """
#     from gym.envs.registration import register
#     from .single_quad import QuadEnvWith
#
#     assert type(weight) is float, "weight has to be of type float"
#     assert .0 <= weight <= 1., "weight has to be in the interval [0.0, 1.0]"
#     assert type(num_kilobots) is int, "num_kilobots has to be of type int"
#     assert 0 < num_kilobots, "num_kilobots has to be a positive integer."
#
#     _name = 'QuadEnv_w{:03}_kb{}'.format(int(weight * 100), num_kilobots)
#     _id = 'QuadEnv_w{:03}_kb{}-v0'.format(int(weight * 100), num_kilobots)
#
#     if _id in __registered_gyms:
#         return _id
#     else:
#         __registered_gyms.append(_id)
#
#     globals()[_name] = QuadEnvWith(weight, num_kilobots)
#     register(id=_id,
#              entry_point='kb_learning.envs:' + _name)
#
#     return _id
#
#
# def register_kilobot_environments(weights: Union[Iterable[float], float] = None,
#                                   num_kilobots: Union[Iterable[int], int] = None,
#                                   iterator: Iterator=None) -> Union[dict, str]:
#     """Creates subclasses of the Quad environment of the Kilobot gym with the given weights and numbers of
#     Kilobots and register the subclass as a gym. The ids of the created and registered environments are returned as
#     string (if a single environment has been created) or as dictionary. If both, weights and num_kilobots are
#     Iterables, the returned dictionary has the weights as top-level keys and the num_kilobots as low-level keys.
#
#     :param weights: float or Iterable[float] with weights in the interval [.0, 1.]
#     :param num_kilobots: int or Iterable[int] with the requested swarm sizes.
#     :param iterator: an iterator function to combine the weights with the num_kilobots, default is itertools.product
#     :return: returns the ids of the registered environments as dictionary or string (if both, weights and
#     num_kilobots are scalars)
#     """
#
#     if weights is None:
#         weights = frozenset([.0, .25, .5, .75, 1.])
#     if num_kilobots is None:
#         num_kilobots = frozenset([10, 15, 20, 25])
#     if iterator is None:
#         from itertools import product
#         iterator = product
#
#     if type(weights) in [float]:
#         w_scalar = True
#     else:
#         w_scalar = False
#
#     if type(num_kilobots) in [int]:
#         n_scalar = True
#     else:
#         n_scalar = False
#
#     if w_scalar and n_scalar:
#         _id = register_quadpushing_environment(weights, num_kilobots)
#     elif w_scalar:
#         _id = dict()
#         for _n in num_kilobots:
#             _id[_n] = register_quadpushing_environment(weights, _n)
#     elif n_scalar:
#         _id = dict()
#         for _w in weights:
#             _id[_w] = register_quadpushing_environment(_w, num_kilobots)
#     else:
#         _id = dict()
#         for _w, _n in iterator(weights, num_kilobots):
#             if _w not in _id.keys():
#                 _id[_w] = dict()
#
#             _id[_w][_n] = register_quadpushing_environment(_w, _n)
#
#     return _id
#
#
# del Iterable, Union, Iterator
