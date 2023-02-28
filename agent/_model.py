import logging
import random
from typing import Generator
from typing import List
from typing import Protocol
from typing import Sequence
from typing import Tuple

import numpy as np
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
        outcome = game.play_classical(
            g,
            players if x % 2 == 0 else list(reversed(players)),
        )
        outcome = outcome if x % 2 == 0 else list(reversed(outcome))
        outcomes = np.concatenate((outcomes, np.array([outcome])),)
    result = np.apply_along_axis(np.sum, 0, 0.5 * (outcomes + 1))
    logging.debug(result)
    return result[0] / n_games


def train(
    model: tf.keras.Model,
    g: game.Game,
    store: ModelStore,
    threshold: float = 0.55,
    *,
    optimizer: tf.keras.optimizers.Optimizer,
    games_per_batch: int = 100,
    samples_per_batch: int = 300,
    learning_rate: float = 0.001,
    test_games: int = 100,
    node_count: int = 10,
    max_retries: int = 3,
    _retry_count: int = 0,
) -> tf.keras.Model:
    model.compile(
        optimizer=optimizer,
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
        node_count=node_count,
    )
    rate = compete_models(
        g,
        model,
        store.load_model(),
        n_games=test_games,
        limit=node_count,
    )
    logging.info(f'New model beats old model in {rate*100}% of games.')
    if rate < threshold:
        if _retry_count >= max_retries:
            return store.load_model()
        return train(
            model=store.load_model(),
            g=g,
            store=store,
            optimizer=optimizer,
            threshold=threshold,
            games_per_batch=games_per_batch,
            samples_per_batch=samples_per_batch,
            learning_rate=learning_rate,
            test_games=test_games,
            max_retries=max_retries,
            _retry_count=_retry_count + 1,
        )

    return train(
        model=model,
        g=g,
        store=store,
        optimizer=optimizer,
        threshold=threshold,
        games_per_batch=games_per_batch,
        samples_per_batch=samples_per_batch,
        learning_rate=learning_rate,
        test_games=test_games,
        max_retries=max_retries,
    )


def training_data(
    model: tf.keras.Model,
    g: game.Game,
    num_games: int,
    node_count: int,
) -> Generator[Tuple[game.State, game.Move, game.Evaluation], None, None]:
    for _ in tqdm.trange(num_games):
        player = _agent.TrainingPlayer(g, model, limit=node_count)
        players = [player for _ in range(g.eval_shape()[0])]
        outcome = game.play_classical(g, players)
        for state, move in zip(player.states, player.moves):
            yield state, move, outcome


def training_batch(
        model: tf.keras.Model,
        g: game.Game,
        num_games: int,
        num_samples: int,
        node_count: int,
        r: random.Random = random.Random(),
):
    states: List[game.State] = []
    moves: List[game.Move] = []
    outcomes: List[game.Evaluation] = []

    logging.info('Generating training set...')
    n = -1
    for _ in tqdm.trange(num_games):
        player = _agent.TrainingPlayer(
            g,
            model,
            limit=node_count,
            temperature=1.0,
            alpha=0.4,
        )
        players = [player for _ in range(g.eval_shape()[0])]
        outcome = game.play_classical(g, players)

        # reservoir sampling labeled data for training.
        for state, move in zip(player.states, player.moves):
            n += 1
            if len(states) < num_samples:
                states.append(state)
                moves.append(move)
                outcomes.append(outcome)
                continue

            j = r.randrange(n)
            if j < len(states):
                states[j] = state
                moves[j] = move
                outcomes[j] = outcome

    logging.info('Fitting model...')
    ss = np.array(states)
    ms = np.array(moves)
    oc = np.array(outcomes)
    model.fit(ss, [ms, oc])


def build_model(
    g: game.Game,
    *,
    residual_layers=1,
    residual_conv_filters=8,
    residual_kernel_size=3,
) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=g.state_shape())

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
    for layer in _policy_head(g.policy_shape()):
        policy = layer(policy)

    for layer in _value_head(g.eval_shape()):
        value = layer(value)

    model = tf.keras.Model(inputs, [policy, value])
    return model


def _policy_head(shape: Sequence[int]) -> List[tf.keras.layers.Layer]:
    return [
        tf.keras.layers.Conv2D(2, 1, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(np.prod(shape)),
        tf.keras.layers.Softmax(name='policy'),
    ]


def _value_head(shape: Sequence[int]) -> List[tf.keras.layers.Layer]:
    return [
        tf.keras.layers.Conv2D(16, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(np.prod(shape),
                              activation=tf.keras.activations.tanh),
        tf.keras.layers.Reshape(shape, name='value')
    ]
