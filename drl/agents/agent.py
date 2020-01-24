from abc import abstractmethod


class Agent:
    @abstractmethod
    def predict(self, observation):
        pass

    @abstractmethod
    def update(self, observation, reward, done, info):
        pass
