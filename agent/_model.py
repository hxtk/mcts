import logging
import random
from typing import List
from typing import Protocol
from typing import Sequence

import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tqdm

import game
from agent import _agent


class ModelStore(Protocol):

    def load_model(self) -> tf.keras.Model:
        """Load the last model saved."""

    def save_model(self, model: tf.keras.Model) -> None:
        """Save a new model."""


def compete_models(
    g: game.Game,
    model1: tf.keras.Model,
    model2: tf.keras.Model,
    n_games: int = 100,
    limit: int = 10,
) -> float:
    logging.info('Competing models.')
    players = [
        _agent.CompetitivePlayer(g, model1, limit=limit),
        _agent.CompetitivePlayer(g, model2, limit=limit),
    ]
    outcomes = np.empty((0, 2))
    for x in tqdm.trange(n_games):
        outcomes = np.concatenate(
            (outcomes,
             np.array([
                 game.play_classical(
                     g,
                     players if x % 2 == 0 else list(reversed(players)),
                 )
             ])))
    result = np.apply_along_axis(np.sum, 0, outcomes)
    return result[0] / n_games


def train(
    model: tf.keras.Model,
    g: game.Game,
    store: ModelStore,
    threshold: float = 0.55,
    *,
    games_per_batch: int = 100,
    samples_per_batch: int = 300,
    learning_rate: float = 0.001,
    test_games: int = 100,
) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=[
            tf.keras.losses.CategoricalCrossentropy(),
            tf.keras.losses.MeanSquaredError(),
        ],
    )
    logging.info('Running training batch.')
    store.save_model(model)
    training_batch(
        model,
        g,
        games_per_batch,
        samples_per_batch,
    )
    rate = compete_models(
        g,
        model,
        store.load_model(),
        n_games=test_games,
    )
    logging.info(f'New model beats old model in {rate*100}% of games.')
    if rate < threshold:
        return

    train(
        model=model,
        g=g,
        store=store,
        threshold=threshold,
        games_per_batch=games_per_batch,
        samples_per_batch=samples_per_batch,
        learning_rate=learning_rate,
        test_games=test_games,
    )


def training_batch(
        model: tf.keras.Model,
        g: game.Game,
        num_games: int,
        num_samples: int,
        r: random.Random = random.Random(),
):
    states: List[game.State] = []
    moves: List[game.Move] = []
    outcomes: List[float] = []

    logging.info('Generating training set...')
    n = -1
    for _ in tqdm.trange(num_games):
        x_player = _agent.TrainingPlayer(g, model, limit=10)
        o_player = _agent.TrainingPlayer(g, model, limit=10)
        outcome = game.play_classical(g, [x_player, o_player])

        batch_states = x_player.states + o_player.states
        batch_moves = x_player.moves + o_player.moves

        # reservoir sampling labeled data for training.
        for state, move in zip(batch_states, batch_moves):
            n += 1
            if len(states) < num_samples:
                states.append(state)
                moves.append(move)
                outcomes.append(outcome[0])
                continue

            j = r.randrange(n)
            if j < len(states):
                states[j] = state
                moves[j] = move
                outcomes[j] = outcome[0]

    logging.info('Fitting model...')
    ss = np.array(states)
    ms = np.array(moves)
    oc = np.array(outcomes)
    model.fit(ss, [ms, oc])


def build_model(
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    *,
    residual_layers=1,
    residual_conv_filters=8,
    residual_kernel_size=3,
) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = residual_in = inputs
    for _ in range(residual_layers):
        x = tf.keras.layers.Conv2D(
            residual_conv_filters,
            residual_kernel_size,
            padding='same',
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            residual_conv_filters,
            residual_kernel_size,
            padding='same',
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Concatenate()([x, residual_in])
        x = tf.keras.layers.ReLU()(x)
        residual_in = x

    policy = value = x
    for layer in _policy_head(output_shape):
        policy = layer(policy)

    for layer in _value_head():
        value = layer(value)

    model = tf.keras.Model(inputs, [policy, value])
    return model


def _policy_head(shape: npt.ArrayLike) -> List[tf.keras.layers.Layer]:
    return [
        tf.keras.layers.Conv2D(2, 1, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(np.prod(shape)),
        tf.keras.layers.Softmax(name='policy'),
    ]


def _value_head() -> List[tf.keras.layers.Layer]:
    return [
        tf.keras.layers.Conv2D(16, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.tanh),
        tf.keras.layers.Reshape((1,), name='value')
    ]
