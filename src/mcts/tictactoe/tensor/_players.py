"""Miscellaneous game.Player implementations for Tic-Tac-Toe."""

import random
import sys
from typing import IO
from typing import MutableMapping
from typing import Optional
from typing import Tuple

import numpy as np
import tensorflow as tf

from mcts import game
from mcts.tictactoe.tensor import _game


def _hash_state(state: game.State) -> int:
    h = 0
    for x in state.flatten():
        if int(x) == 1:
            h |= 1
        h <<= 1
    return h


def _render_board(state: game.State):
    """Print the tic-tac-toe board as ASCII text."""
    line = "+-+-+-+\n"
    out = line
    for row in (state[:, :, 0] - state[:, :, 1]).numpy():
        out += "|"
        for cell in row:
            out += ["O", " ", "X"][int(cell + 1)] + "|"
        out += "\n" + line
    return out


def _read_line(
    source: IO,
    out: Optional[IO],
    mask: game.Move,
) -> Tuple[int, int]:
    """Read a set of coordinates corresponding to a move.

    Lines that cannot be parsed or do not correspond to legal moves are
    ignored and another line is read from source.

    Args:
        source: input stream containing lines matching the regex
            /[012][ ,][012]/. Lines not matching that regex will be ignored.
        out: output stream to which prompts shall be printed. These prompts MAY
            change and SHOULD NOT be parsed.
        mask: a 0-1 tensor with 1s representing legal moves.

    Returns:
        A tuple corresponding to the row and column in which to play.
    """
    row, sep, col = source.readline().partition(" ")
    if sep == "":
        row, sep, col = row.partition(",")

    col = col[:-1]  # Remove trailing newline
    if sep == "" or not row.isnumeric() or not col.isnumeric():
        print(f"You entered: {row} {col}", file=out)
        print("Format: [row] SPACE [col]\nExample: 0 2", file=out)
        return _read_line(source, out, mask)

    shape = _game.policy_shape()
    row_n = int(row)
    col_n = int(col)
    if (
        not 0 <= row_n < shape[0]
        or not 0 <= col_n < shape[1]
        or mask[row_n][col_n] == 0
    ):
        print(f"{row_n} {col_n} is not a legal move. Try another:", file=out)
        return _read_line(source, out, mask)

    return row_n, col_n


class TextIOPlayer:
    """Tic-Tac-Toe Player that uses text IO to obtain moves."""

    def __init__(self, in_: IO = sys.stdin, out: Optional[IO] = sys.stdout):
        self._in = in_
        self._out = out

    def __call__(self, state: game.State, mask: game.Move) -> game.Move:
        """Implements the game.Player interface."""
        print(_render_board(state), file=self._out)

        player = "X" if state[2][0][0] == 0 else "O"

        print(f"{player} to play ([row] [col]): ", file=self._out)
        row, col = _read_line(self._in, self._out, mask)

        return tf.scatter_nd(
            indices=[[row, col]],
            updates=[1.0],
            shape=_game.policy_shape(),
        )


class RandomPlayer:
    """Tic-Tac-Toe Player that makes uniform random moves."""

    def __init__(self, rand: random.Random = random.SystemRandom()):
        self._rand = rand

    def __call__(self, _: game.State, mask: game.Move) -> game.Move:
        """Implements the game.Player interface."""
        ps = (tf.reshape(mask, (9,)) / tf.math.count_nonzero(mask)).numpy()

        # pylint: disable=unbalanced-tuple-unpacking
        index = np.unravel_index(
            np.random.choice(mask.size, p=ps),
            mask.shape,
        )

        return tf.scatter_nd(
            indices=[index],
            updates=[1.0],
            shape=_game.policy_shape(),
        )


class MinMaxPlayer:
    def __init__(self) -> None:
        self.value_cache: MutableMapping[
            int,
            Tuple[float, Optional[int]],
        ] = dict()

    def value(
        self,
        state: game.State,
        mask: game.Move,
        depth: float = 1.0,
    ) -> Tuple[float, Optional[int]]:
        sh = _hash_state(state)
        if sh in self.value_cache:
            value, move = self.value_cache[sh]
            return (value / depth), move

        evaluation = _game.evaluate(state)
        if evaluation is not None:
            value = evaluation[state[2][0][0]]
            self.value_cache[sh] = (value, None)
            return (value / depth), None

        values = tf.zeros_like(mask)
        for i, x in enumerate(mask.flatten()):
            if x == 0:
                # Illegal moves are worse than losing.
                values[i] = float("-inf")
                continue

            # pylint: disable=unbalanced-tuple-unpacking
            index = np.unravel_index(
                i,
                mask.shape,
            )
            move = tf.scatter_nd(
                indices=[index],
                updates=[1.0],
                shape=_game.policy_shape(),
            )

            state_next = _game.play_move(
                state,
                move,
            )
            values[i] = -self.value(
                state_next,
                _game.move_mask(state_next),
                depth + 1.0,
            )[0]

        value = float(values.max())
        move = values.argmax()
        self.value_cache[sh] = (value, int(move))
        return (value / depth), int(move)

    def __call__(self, state: game.State, mask: game.Move) -> game.Move:
        val, move = self.value(state, mask)
        print(f"Evaluation: {val}")
        if move is None:
            raise game.GameAlreadyOverError()

        # pylint: disable=unbalanced-tuple-unpacking
        index = np.unravel_index(
            move,
            mask.shape,
        )
        return tf.scatter_nd(
            indices=[index],
            updates=[1.0],
            shape=_game.policy_shape(),
        )
