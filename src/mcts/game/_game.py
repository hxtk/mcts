from typing import Protocol
from typing import Sequence
from typing import Tuple
from typing import TypeVar

import tensorflow as tf

Move = TypeVar("Move")
State = TypeVar("State")
Evaluation = TypeVar("Evaluation")


class Game(Protocol[State, Move, Evaluation]):
    @staticmethod
    def move_mask(state: State) -> Move:
        """Compute an 0-1 array with 1s representing legal moves.

        The returned array MUST be of the same shape as the input accepted by
        `play_move`.
        """
        raise NotImplementedError()

    @staticmethod
    def new() -> State:
        """Return an initial game state."""
        raise NotImplementedError()

    @staticmethod
    def play_move(state: State, move: Move) -> State:
        """Modify the state of the game to record `move` being played."""
        raise NotImplementedError()

    @staticmethod
    def evaluate(state: State) -> Evaluation:
        """Evaluate the value of the position if it is a terminal state.

        Returns:
            None if the game is not a terminal state; otherwise returns the
            evaluation of the position for the first player.
        """
        raise NotImplementedError()

    @staticmethod
    def player(state: State) -> int:
        """Return the index of the player to move in the given state."""
        raise NotImplementedError()

    @staticmethod
    def state_shape() -> Tuple[int, ...]:
        raise NotImplementedError()

    @staticmethod
    def policy_shape() -> Tuple[int, ...]:
        raise NotImplementedError()

    @staticmethod
    def eval_size() -> int:
        raise NotImplementedError()


class Player(Protocol[State, Move]):
    """A Player returns a move to be played.

    Players have access to the current state of the game and
    a list of legal moves.
    """

    def __call__(self, state: State, mask: Move) -> Move:
        raise NotImplementedError()


class InsufficientPlayersError(ValueError):
    """Exception indicating not enough players were provided for the game."""

    def __init__(
        self,
        got: int,
        minimum: int = 1,
    ) -> None:
        super().__init__(f"need at least {minimum} players; got {got}")


class GameAlreadyOverError(Exception):
    """Exception indicating that the game has ended and cannot be played."""

    def __init__(self):
        super().__init__("cannot play play in a game that had already ended")


def play_classical(
    game: Game[State, Move, Evaluation],
    players: Sequence[Player[State, Move]],
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
        raise InsufficientPlayersError(len(players))

    g = game.new()

    idx = 0
    value = game.evaluate(g)[0]
    while tf.math.reduce_any(tf.math.is_nan(value)).numpy():
        mask = game.move_mask(g)
        g = game.play_move(g, players[idx](g, mask))

        idx = (idx + 1) % len(players)
        value = game.evaluate(g)[0]

    return value
