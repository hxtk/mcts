"""Module implemeting game.Game for Tic-Tac-Toe."""
import enum
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import numpy.typing as npt

DIM = 3  # The dimension of the board


class EndState(enum.Enum):
    X = 1
    O = -1  # noqa: E741
    DRAW = 0


Move = npt.NDArray[np.float32]
State = npt.NDArray[np.float32]


def new() -> State:
    """Return a game state for a new Tic-Tac-Toe game."""
    return np.zeros((3, 3, 3))


def move_mask(state: State) -> Move:
    """Evaluate the set of legal moves for a state.

    Returns:
        A 3x3 0-1 array where 1s represent legal moves.
    """
    return np.ones(state.shape[1:]) - (state[0] + state[1])


def play_move(state: State, move: Move) -> State:
    """Apply move to the game represented by state.

    If the player layer is all 0s then an "X" is placed
    in the coordinate having the highest value in move.

    An "O" is placed in the coordinate having the
    highest value in move otherwise.
    """
    move = move.reshape((3, 3))
    board = np.copy(state[0:2])
    p = player(state)

    # pylint: disable=unbalanced-tuple-unpacking
    row, col = np.unravel_index(move.argmax(), (3, 3))
    board[p][row][col] = 1.0

    if p == 0:
        return np.concatenate((board, [np.ones((3, 3))]))
    return np.concatenate((board, [np.zeros((3, 3))]))


def player(state: State) -> int:
    """Get the ordinal number of the player whose turn it is.

    Returns:
        0 if it is X to play.
        1 if it is O to play.
    """
    return int(state[2][0][0])


def evaluate(state: State) -> Optional[List[float]]:
    """Evaluate a GameState board.

    Args:
        state: A 3x3x3 numpy array where the first two layers represent the
            Xs and Os on the board, respectively, and the third layer is all
            1s if it is Xs turn; otherwise it is all 0s.

    Returns:
        None if the game is not in a terminal state; otherwise returns a list
        representing the evaluation for X and O, respectively. A victory for X
        is [1, -1], a victory for O is [-1, 1], and a draw is [0, 0].
    """
    evaluation = _evaluate(state[0:2])
    if evaluation is None:
        return None
    if evaluation is EndState.X:
        return [1, -1]
    if evaluation is EndState.O:
        return [-1, 1]
    return [0, 0]


def state_shape() -> Tuple[int]:
    return new().shape


def policy_shape() -> Tuple[int, ...]:
    return move_mask(new()).shape


def eval_size() -> int:
    return 2


def _evaluate(board: State) -> Optional[EndState]:
    for dim in range(1, board.ndim):
        extreme = board.shape[dim]
        val = np.apply_along_axis(np.sum, dim, board)
        if np.max(val[0]) == extreme:
            return EndState.X
        if np.max(val[1]) == extreme:
            return EndState.O

    if board[0][1][1] + board[1][1][1] == 0:
        return None

    diagonals = np.trace(board, axis1=1, axis2=2)
    if diagonals[0] == DIM:
        return EndState.X
    if diagonals[1] == DIM:
        return EndState.O

    diagonals = np.trace(np.flip(board, 2), axis1=1, axis2=2)
    if diagonals[0] == DIM:
        return EndState.X
    if diagonals[1] == DIM:
        return EndState.O

    if np.count_nonzero(board) == 9:
        return EndState.DRAW

    return None
