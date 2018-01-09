import itertools
import numpy as np


def chunks(iterable, n):
    it = iter(iterable)

    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def np_chunks(array: np.ndarray, n):
    for chunk in chunks(array, n):
        yield np.array(chunk)
