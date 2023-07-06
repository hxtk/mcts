"""Model construction to evaluate game states."""
from typing import List
from typing import Sequence

import numpy as np
import tensorflow as tf

from mcts import game

State = tf.Tensor
Move = tf.Tensor
Evaluation = tf.Tensor


def residual_model(
    g: game.Game[State, Move, Evaluation],
    *,
    residual_layers: int = 1,
    residual_conv_filters: int = 8,
    residual_kernel_size: int = 3,
) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=g.state_shape())
    x = tf.keras.layers.Conv2D(
        residual_conv_filters,
        residual_kernel_size,
        activation="relu",
        padding="same",
    )(inputs)

    residual_in = x
    for _ in range(residual_layers):
        x = tf.keras.layers.Conv2D(
            residual_conv_filters,
            residual_kernel_size,
            padding="same",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            residual_conv_filters,
            residual_kernel_size,
            padding="same",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, residual_in])
        x = tf.keras.layers.ReLU()(x)
        residual_in = x

    policy = value = x
    for layer in _policy_head(g.policy_shape()):
        policy = layer(policy)

    for layer in _value_head([g.eval_size()]):
        value = layer(value)

    return tf.keras.Model(inputs, [policy, value])


def _policy_head(shape: Sequence[int]) -> List[tf.keras.layers.Layer]:
    return [
        tf.keras.layers.Conv2D(2, 1, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(np.prod(shape)),
        tf.keras.layers.Softmax(),
        tf.keras.layers.Reshape(shape, name="policy"),
    ]


def _value_head(shape: Sequence[int]) -> List[tf.keras.layers.Layer]:
    return [
        tf.keras.layers.Conv2D(1, 1, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(
            np.prod(shape),
            activation=tf.keras.activations.tanh,
        ),
        tf.keras.layers.Reshape(shape, name="value"),
    ]
