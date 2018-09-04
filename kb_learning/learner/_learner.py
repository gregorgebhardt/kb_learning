import abc

from cluster_work import ClusterWork


class KilobotLearner(ClusterWork, abc.ABC):
    @abc.abstractmethod
    def iterate(self, config: dict, rep: int, n: int) -> dict:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, config: dict, rep: int) -> None:
        raise NotImplementedError
