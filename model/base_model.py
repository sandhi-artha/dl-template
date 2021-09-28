'''Abstract base model, functions as a strict template
to create a model class. If a method is not implemented
will raise error
'''

from abc import ABC, abstractmethod
from utils.config import Config


class BaseModel(ABC):
    '''Abstract Model class that is inherited to all models'''
    def __init__(self, cfg):
        self.config = Config.from_json(cfg)
        # self.config = cfg

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass


