from abc import ABC, abstractmethod


class AbstractFlow(ABC):
    def __init__(self, type: str):
        self.type = type

    @abstractmethod
    def Init(self):
        print(f'Initializing {self.type} resources...')

    @abstractmethod
    def run_flow(self, args=None):
        pass

    def __repr__(self):
        return f'Flow: {self.type}'
