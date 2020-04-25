import numpy as np

from pommerman import constants
from pommerman.envs import v0


class Pomme(v0.Pomme):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_board(self):
        num_agents = len(self._agents)
        size = self._board_size
        self._board = np.ones((size, size)).astype(np.uint8) * constants.Item.Passage.value

        if num_agents == 2:
            self._board[1, 1] = constants.Item.Agent0.value
            self._board[size - 2, size - 2] = constants.Item.Agent1.value
        else:
            self._board[1, 1] = constants.Item.Agent0.value
            self._board[size - 2, 1] = constants.Item.Agent1.value
            self._board[size - 2, size - 2] = constants.Item.Agent2.value
            self._board[1, size - 2] = constants.Item.Agent3.value

