"""Unit tests for _game module."""
import unittest

import numpy as np

from tictactoe import _game


class TestPlayMove(unittest.TestCase):
    """Test that PlayMove has the desired effect."""

    def test_first_move(self):
        g = _game.play_move(
            _game.new(),
            np.array([
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]),
        )

        self.assertEqual(g[0][0][0], 1)
        self.assertEqual(g[2][0][0], 1)


class TestEvaluate(unittest.TestCase):
    """Test that we can evaluate end states."""

    def test_empty_board(self):
        board = np.zeros((2, 3, 3))
        self.assertEqual(_game.evaluate(board), None)

    def test_x_row(self):
        board = np.zeros((2, 3, 3))
        for x in range(3):
            board[0][0][x] = 1
        self.assertEqual(_game.evaluate(board), [1, -1])

    def test_x_col(self):
        board = np.zeros((2, 3, 3))
        for x in range(3):
            board[0][x][0] = 1
        self.assertEqual(_game.evaluate(board), [1, -1])

    def test_x_eye(self):
        board = np.zeros((2, 3, 3))
        for x in range(3):
            board[0][x][x] = 1
        self.assertEqual(_game.evaluate(board), [1, -1])

    def test_x_diagonal(self):
        board = np.zeros((2, 3, 3))
        for x in range(3):
            board[0][x][2 - x] = 1
        self.assertEqual(_game.evaluate(board), [1, -1])

    def test_o_row(self):
        board = np.zeros((2, 3, 3))
        for x in range(3):
            board[1][0][x] = 1
        self.assertEqual(_game.evaluate(board), [-1, 1])

    def test_o_col(self):
        board = np.zeros((2, 3, 3))
        for x in range(3):
            board[1][x][0] = 1
        self.assertEqual(_game.evaluate(board), [-1, 1])

    def test_o_eye(self):
        board = np.zeros((2, 3, 3))
        for x in range(3):
            board[1][x][x] = 1
        self.assertEqual(_game.evaluate(board), [-1, 1])

    def test_o_diagonal(self):
        board = np.zeros((2, 3, 3))
        for x in range(3):
            board[1][x][2 - x] = 1
        self.assertEqual(_game.evaluate(board), [-1, 1])


if __name__ == '__main__':
    unittest.main()
