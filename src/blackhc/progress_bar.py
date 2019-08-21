import abc


class ProgressBar(abc.ABC):
    def __init__(self, length):
        self.length = length

    @abc.abstractmethod
    def start(self):
        pass

    @abc.abstractmethod
    def update(self, delta_processed=1):
        pass

    @abc.abstractmethod
    def finish(self):
        pass
