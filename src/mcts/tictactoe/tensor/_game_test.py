"""Unit tests for _game module."""
import unittest

import tensorflow as tf

from mcts.tictactoe.tensor import _game


class TestPlayMove(unittest.TestCase):
    """Test that PlayMove has the desired effect."""

    def test_first_move(self):
        g = _game.play_move(
            _game.new(),
            tf.scatter_nd(
                indices=[[0, 0]],
                updates=[1.0],
                shape=_game.policy_shape(),
            ),
        )

        self.assertEqual(g[0][0][0], 1.0)
        self.assertEqual(g[0][0][1], 0.0)
        self.assertEqual(g[0][0][2], 1.0)


class TestEvaluate(unittest.TestCase):
    """Test that we can evaluate end states."""

    def test_empty_board(self):
        board = _game.new()
        self.assertTrue(
            tf.math.reduce_all(tf.math.is_nan(_game.evaluate(board))),
        )

    def test_x_row(self):
        for x in range(3):
            board = tf.scatter_nd(
                indices=[[x, 0, 0], [x, 1, 0], [x, 2, 0]],
                updates=[1.0, 1.0, 1.0],
                shape=_game.state_shape(),
            )
            self.assertTrue(
                tf.math.reduce_all(
                    tf.math.equal(_game.evaluate(board), [[1.0, -1.0]]),
                ),
            )

    def test_x_row_batch(self):
        boards = tf.scatter_nd(
            indices=[[y, x, y, 0] for x in range(3) for y in range(3)],
            updates=[1.0 for _ in range(9)],
            shape=(3, *_game.state_shape()),
        )
        self.assertTrue(
            tf.math.reduce_all(
                tf.math.equal(
                    _game.evaluate(boards),
                    [[1.0, -1.0] for _ in range(3)],
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
