from abc import ABC, abstractmethod

from qgen.model import Model
from qgen.result import Result


class Backend(ABC):
    def __init__(self, model: Model) -> None:
        self.model = model

    @abstractmethod
    def simulate(self, *args, **kwargs) -> Result: ...
