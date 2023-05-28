"""Module implementing game.Game for Tic-Tac-Toe."""
from typing import List, Tuple
from typing import Optional

import tensorflow as tf

Move = tf.Tensor
State = tf.Tensor


@tf.function
def new() -> State:
    """Return a game state for a new Tic-Tac-Toe game."""
    return tf.zeros(state_shape())


@tf.function
def move_mask(state: State) -> Move:
    """Evaluate the set of legal moves for a state.

    Returns:
        A 3x3 0-1 array where 1s represent legal moves.
    """
    return tf.ones(state.shape[:-1]) - tf.math.reduce_max(state[:, :, :2], axis=-1)


@tf.function
def play_move(state: State, move: Move) -> State:
    """Apply move to the game represented by state.

    If the player layer is all 0s then an "X" is placed
    in the coordinate having the highest value in move.

    An "O" is placed in the coordinate having the
    highest value in move otherwise.
    """
    p = player(state)
    changed = tf.reshape(state[:, :, p] + move, shape=(3, 3, 1))
    if p == 0:
        return tf.concat(
            [changed, state[:, :, 1:2], tf.ones_like(changed)],
            axis=-1,
        )

    return tf.concat(
        [state[:, :, :1], changed, tf.zeros_like(changed)],
        axis=-1,
    )


@tf.function
def player(state: State) -> int:
    """Get the ordinal number of the player whose turn it is.

    Returns:
        0 if it is X to play.
        1 if it is O to play.
    """
    return tf.cast(state[0][0][2], dtype=tf.int32)


_ROW_KERNELS = tf.constant(
    [
        [[[1, 0, 0]], [[1, 0, 0]], [[1, 0, 0]]],
        [[[0, 1, 0]], [[0, 1, 0]], [[0, 1, 0]]],
        [[[0, 0, 1]], [[0, 0, 1]], [[0, 0, 1]]],
    ],
    dtype=tf.float32,
)

_COL_KERNELS = tf.constant(
    [
        [[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]]],
        [[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]]],
        [[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]]],
    ],
    dtype=tf.float32,
)

_DIAGONAL_KERNELS = tf.constant(
    [
        [[[1, 0]], [[0, 0]], [[0, 1]]],
        [[[0, 0]], [[1, 1]], [[0, 0]]],
        [[[0, 1]], [[0, 0]], [[1, 0]]],
    ],
    dtype=tf.float32,
)

_ALL_KERNELS = tf.concat(
    [_ROW_KERNELS, _COL_KERNELS, _DIAGONAL_KERNELS],
    axis=-1,
)


@tf.function
def _assign_eval(x: tf.Tensor) -> tf.Tensor:
    if x[0] >= 3:
        return tf.constant([1.0, -1.0])
    if x[1] <= -3:
        return tf.constant([-1.0, 1.0])
    if x[2] >= 9:
        return tf.constant([0.0, 0.0])
    return tf.constant([float("nan"), float("nan")])


@tf.function
def evaluate(state: State) -> tf.Tensor:
    """Evaluate a GameState board.

    Args:
        state: A 3x3x3 numpy array where the first two layers represent the
            Xs and Os on the board, respectively, and the third layer is all
            1s if it is Xs turn; otherwise it is all 0s.

    Returns:
        None if the game is not in a terminal state.
        Otherwise, returns a list representing the evaluation for X and O,
        respectively. A victory for X is [1, -1], a victory for O is [-1, 1],
        and a draw is [0, 0].
    """
    state = tf.reshape(state, (-1,) + state_shape())
    out = tf.nn.conv2d(
        state[:, :, :, :1] - state[:, :, :, 1:2],
        _ALL_KERNELS,
        strides=1,
        padding="VALID",
    )
    out = tf.reshape(
        tf.concat(
            [
                tf.math.reduce_max(out, axis=[-3, -2, -1], keepdims=True),
                tf.math.reduce_min(out, axis=[-3, -2, -1], keepdims=True),
                tf.cast(
                    tf.math.count_nonzero(
                        state[:, :, :, 0:2],
                        axis=[-3, -2, -1],
                        keepdims=True,
                    ),
                    dtype=tf.float32,
                ),
            ],
            axis=-1,
        ),
        shape=(-1, 3),
    )
    return tf.map_fn(
        fn=_assign_eval,
        elems=out,
    )


def state_shape() -> Tuple[int, ...]:
    return 3, 3, 3


def policy_shape() -> Tuple[int, ...]:
    return 3, 3


def eval_size() -> int:
    return 2
