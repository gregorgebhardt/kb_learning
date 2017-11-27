from itertools import product

from gym.envs.registration import register

from .single_quad import QuadPushingEnvWith

weights = frozenset([.0, .25, .5, .75, 1.])
num_kilobots = frozenset([10, 15, 20, 25])

for _w, _n in product(weights, num_kilobots):
    _name = 'QuadPushingEnv_w{:03}_kb{}'.format(int(_w*100), _n)
    globals()[_name] = QuadPushingEnvWith(_w, _n)
    register(id='Kilobots-QuadPushingEnv_w{:03}_kb{}-v0'.format(int(_w * 100), _n),
             entry_point='kb_learning.envs:'+_name)

    del _w, _n, _name
del QuadPushingEnvWith
del register
del product
