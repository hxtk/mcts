from typing import Optional
from typing import Protocol
from typing import Sequence

import numpy as np
import numpy.typing as npt

Move = npt.NDArray[np.float64]
State = npt.NDArray[np.float64]
Evaluation = Sequence[float]


class Game(Protocol):

    @staticmethod
    def move_mask(state: State) -> Move:
        """Compute an 0-1 array with 1s representing legal moves.

        The returned array MUST be of the same shape as the input accepted by
        `play_move`.
        """
        raise Exception('Not Implemented')

    @staticmethod
    def new() -> State:
        """Return an initial game state."""
        raise Exception('Not Implemented')

    @staticmethod
    def play_move(state: State, move: Move) -> State:
        """Modify the state of the game to record `move` being played."""
        raise Exception('Not Implemented')

    @staticmethod
    def evaluate(state: State) -> Optional[Evaluation]:
        """Evaluate the value of the position if it is a terminal state.

        Returns:
            None if the game is not a terminal state; otherwise returns the
            evaluation of the position for the first player.
        """
        raise Exception('Not Implemented')

    @staticmethod
    def player(state: State) -> int:
        """Return the ordinal number of the player to move in the given state."""
        raise Exception('Not Implemented')

    @staticmethod
    def state_shape() -> Sequence[int]:
        raise Exception('Not Implemented')

    @staticmethod
    def move_shape() -> Sequence[int]:
        raise Exception('Not Implemented')

    @staticmethod
    def eval_shape() -> Sequence[int]:
        raise Exception('Not Implemented')


class Player(Protocol):
    """A Player returns a move to be played.

    Players have access to the current state of the game and
    a list of legal moves.
    """

    def __call__(self, state: State, mask: Move) -> Move:
        return np.empty(0)


def play_classical(
    game: Game,
    players: Sequence[Player],
) -> Evaluation:
    """Play a classical game.

    Classical games are characterized by players sharing a game state
    with perfect information and taking turns.

    Args:
        game: the rules of the game being played.
        players: a list of players to cycle through in order.

    Returns:
        A list of scores corresponding to each player, respectively,
        at the end of the game, after a terminal state is reached.
    """
    if len(players) == 0:
        raise ValueError('At least one player is required for a game.')

    g = game.new()

    idx = 0
    value = game.evaluate(g)
    while value is None:
        mask = game.move_mask(g)
        g = game.play_move(g, players[idx](g, mask))

        idx = (idx + 1) % len(players)
        value = game.evaluate(g)

    return value
