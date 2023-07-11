import pathlib
import tempfile
import unittest

import tensorflow as tf

import mcts.tictactoe.tensor as ttt
from mcts.gym import _storage


class TestStorage(unittest.TestCase):
    def test_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            s = _storage.Storage(
                ttt,
                pathlib.Path(tmpdir),
            )
            with s.writer() as w:
                state = ttt.new()
                move = tf.zeros(ttt.policy_shape())
                outcome = tf.zeros((ttt.eval_size(),))
                w.add(state, move, outcome)
                w.add(state, move, outcome)
            data = s.get_dataset(2)
            self.assertEqual(
                len(list(data.take(3))),
                2,
                "wanted 2 records written",
            )
