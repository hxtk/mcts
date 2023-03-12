import unittest

import numpy as np

from tictactoe import _game
from tictactoe import _players


class TestMinMaxPlayer(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.engine = _players.MinMaxPlayer()

    def test_recognizes_loss(self):
        board = np.array(
            [
                [
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                ], [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 0, 0],
                ], [
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ]
            ]
        )
        value, move = self.engine.value(
            board,
            _game.move_mask(board),
        )

        # We lost.
        self.assertLess(value, 0)
        self.assertEqual(move, None)

    def test_takes_win(self):
        board = np.array(
            [
                [
                    [1, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ], [
                    [0, 0, 0],
                    [1, 1, 0],
                    [0, 0, 0],
                ], [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            ]
        )
        value, move = self.engine.value(
            board,
            _game.move_mask(board),
        )

        # We can win.
        self.assertGreater(value, 0)

        # We have to play in 0, 2 to win.
        self.assertEqual(move, 2)

    def test_blocks_enemy_win(self):
        board = np.array(
            [
                [
                    [1, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ], [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 0, 0],
                ], [
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ]
            ]
        )
        value, move = self.engine.value(
            board,
            _game.move_mask(board),
        )

        # We have to play in 0, 2 to stop X winning.
        self.assertEqual(move, 2)


if __name__ == '__main__':
    unittest.main()
