from environment import Environment


class Agent(object):
    def __init__(self, env):
        self.env = env

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent from observation
        """
        raise NotImplementedError("Subclasses should implement this!")

    def init_game_setting(self):
        """
        Init things for test time
        """
        raise NotImplementedError("Subclasses should implement this!")
