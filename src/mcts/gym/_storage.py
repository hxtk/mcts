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
                    "state": tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(state).numpy()],
                        ),
                    ),
                    "move": tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(move).numpy()],
                        ),
                    ),
                    "outcome": tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(outcome).numpy()],
                        ),
                    ),
                },
            ),
        ).SerializeToString()
        self.writer.write(record_bytes)


def _decode_fn(record_bytes: bytes) -> tf.train.Example:
    example = tf.io.parse_single_example(
        record_bytes,
        features={
            "state": tf.io.RaggedFeature(
                tf.string,
            ),
            "move": tf.io.RaggedFeature(
                tf.string,
            ),
            "outcome": tf.io.RaggedFeature(
                tf.string,
            ),
        },
    )

    return (
        tf.io.parse_tensor(example["state"][0], tf.float32),
        (
            tf.io.parse_tensor(example["move"][0], tf.float32),
            tf.io.parse_tensor(example["outcome"][0], tf.float32),
        ),
    )


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
        # We use a UUID7 for time-ordered file names that can be generated
        # without locks or inter-process communication.
        return ReplayWriter(
            tf.io.TFRecordWriter(
                os.fspath(
                    self.base_path.joinpath(str(uuid6.uuid7()) + ".tfrecord"),
                ),
            ),
        )

    def get_dataset(self, size: int) -> tf.data.TFRecordDataset:
        return tf.data.TFRecordDataset(
            sorted(
                os.path.join(self.base_path, x)
                for x in os.listdir(self.base_path)
            )[:size],
        ).map(_decode_fn)
