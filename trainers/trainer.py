from abc import ABC, abstractmethod

# Interface
class TrainerInterface(ABC):
    @abstractmethod
    def store_episode(self, episode):
        pass

    @abstractmethod
    def attach_models(self, dict_of_models):
        pass

    @abstractmethod
    def attach_losses(self, dict_of_losses):
        pass

    @abstractmethod
    def attach_agent(self, agent):
        pass

    @abstractmethod
    def _episode_to_batch(self, episode):
        pass

    @abstractmethod
    def build_batch(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def reset_episodes(self):
        pass

    @abstractmethod
    def save_checkpoint(self, path):
        pass

    @abstractmethod
    def load_checkpoint(self, path):
        pass



class Trainer:
    def __init__(self, models=None, optimizers=None, reward_shaper=None, losses=None, episodes=None, batch_size=None, agent=None):
        self.models = models
        self.optimizers = optimizers if optimizers is not None else {}
        self.reward_shaper = reward_shaper
        self.losses = losses if losses is not None else {}
        self.episodes = episodes if episodes is not None else []
        self.batch_size = batch_size
        self.agent = agent
