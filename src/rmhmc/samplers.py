from abc import ABC, abstractmethod


@abstractmethod
class Sampler(ABC):
    @abstractmethod
    def sample(self):
        pass
