"""Shared storage module for distributed replay buffers."""
import os
import pathlib
import types

import tensorflow as tf
import uuid6

from mcts import agent
from mcts import game


class PathModelStore:
    def __init__(self, path: str, g: game.Game) -> None:
        self.g = g
        self.path = path

    def load_model(self) -> tf.keras.Model:
        try:
            return tf.keras.models.load_model(self.path)
        except IOError:
            return agent.residual_model(
                self.g,
                residual_layers=2,
                residual_conv_filters=32,
            )

    def save_model(self, model: tf.keras.Model) -> None:
        model.save(self.path)


def _tensor_to_feature(t: tf.Tensor) -> tf.train.Feature:
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(t).numpy(),
            ],
        ),
    )


class ReplayWriter:
    def __init__(self, writer: tf.io.TFRecordWriter):
        self.writer = writer

    def __enter__(self) -> "ReplayWriter":
        return self

    def __exit__(
        self,
        exc_type: type,
        exc_val: Exception,
        exc_tb: types.TracebackType,
    ) -> None:
        del exc_type, exc_val, exc_tb
        self.close()

    def close(self) -> None:
        self.writer.close()

    def add(
        self,
        state: tf.Tensor,
        move: tf.Tensor,
        outcome: tf.Tensor,
    ) -> None:
        record_bytes = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "state": _tensor_to_feature(state),
                    "move": _tensor_to_feature(move),
                    "outcome": _tensor_to_feature(outcome),
                },
            ),
        ).SerializeToString()
        self.writer.write(record_bytes)


class Storage:
    def __init__(
        self,
        g: game.Game,
        base_path: pathlib.Path,
    ):
        self.g = g
        self.base_path = base_path

    def writer(self) -> ReplayWriter:
        self.base_path.mkdir(exist_ok=True)
        return ReplayWriter(
            tf.io.TFRecordWriter(
                os.fspath(
                    self.base_path.joinpath(str(uuid6.uuid7()) + ".record"),
                ),
            ),
        )

    def _decode_fn(self, record_bytes: bytes) -> tf.train.Example:
        tf.io.parse_example(
            record_bytes,
            features={
                "state": tf.io.FixedLenFeature(
                    self.g.state_shape(),
                    tf.float32,
                ),
                "move": tf.io.FixedLenFeature(
                    self.g.policy_shape(),
                    tf.float32,
                ),
                "outcome": tf.io.FixedLenFeature(
                    [self.g.eval_size()],
                    tf.float32,
                ),
            },
        )

    def get_dataset(self, size: int) -> tf.data.TFRecordDataset:
        return tf.data.TFRecordDataset(
            sorted(
                os.path.join(self.base_path, x)
                for x in os.listdir(self.base_path)
            )[:size],
        )  # .map(self._decode_fn)
