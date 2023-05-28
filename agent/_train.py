import logging
import random
from typing import Generator
from typing import List
from typing import Protocol
from typing import Tuple

import numpy as np
import tensorflow as tf
import tqdm

from agent import _agent
import game

State = tf.Tensor
Move = tf.Tensor
Evaluation = tf.Tensor


class ModelStore(Protocol):
    def load_model(self) -> tf.keras.Model:
        """Load the last model saved."""

    def save_model(self, model: tf.keras.Model) -> None:
        """Save a new model."""


def compete_models(
    g: game.Game[State, Move, Evaluation],
    model1: tf.keras.Model,
    model2: tf.keras.Model,
    n_games: int = 100,
    limit: int = 10,
) -> float:
    logging.info("Competing models.")
    players = [
        _agent.TreeNodePlayer(g, model1, limit=limit),
        _agent.TreeNodePlayer(g, model2, limit=limit),
    ]
    outcomes = np.empty((0, 2))
    for x in tqdm.trange(n_games):
        outcome = game.play_classical(
            g,
            players if x % 2 == 0 else list(reversed(players)),
        )
        outcome = outcome if x % 2 == 0 else tf.reverse(outcome, axis=(0,))
        outcomes = np.concatenate(
            (outcomes, np.array([outcome])),
        )
        for p in players:
            p.root = None
            p.states.clear()
            p.moves.clear()

    result = np.apply_along_axis(np.sum, 0, 0.5 * (outcomes + 1))
    logging.debug(result)
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
    node_count: int = 10,
    max_retries: int = 3,
    _retry_count: int = 0,
) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.experimental.SGD(learning_rate=learning_rate),
        loss=[
            tf.keras.losses.CategoricalCrossentropy(),
            tf.keras.losses.MeanSquaredError(),
        ],
    )
    print("Running training batch.")
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
    print(f"New model beats old model in {rate*100}% of games.")
    if rate < threshold:
        if _retry_count >= max_retries:
            return store.load_model()
        return train(
            model=store.load_model(),
            g=g,
            store=store,
            threshold=threshold,
            games_per_batch=games_per_batch,
            samples_per_batch=samples_per_batch,
            learning_rate=learning_rate,
            test_games=test_games,
            max_retries=max_retries,
            _retry_count=_retry_count + 1,
        )

    store.save_model(model)
    return train(
        model=model,
        g=g,
        store=store,
        threshold=threshold,
        games_per_batch=games_per_batch,
        samples_per_batch=samples_per_batch,
        learning_rate=learning_rate,
        test_games=test_games,
        max_retries=max_retries,
    )


def training_game(
    store: ModelStore,
    g: game.Game[tf.Tensor, tf.Tensor, tf.Tensor],
    alpha: float,
    temperature: float,
    node_count: int,
    reload_interval: int = 100,
) -> Generator[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], None, None]:
    while True:
        model = store.load_model()
        for _ in range(reload_interval):
            player = _agent.TreeNodePlayer(
                g,
                model,
                limit=node_count,
                temperature=temperature,
                alpha=alpha,
            )
            outcome = game.play_classical(g, [player])
            outcome = tf.repeat(outcome, len(player.states))


def training_batch(
    model: tf.keras.Model,
    g: game.Game[tf.Tensor, tf.Tensor, tf.Tensor],
    num_games: int,
    num_samples: int,
    node_count: int,
    r: random.Random = random.Random(),
):
    states: List[game.State] = []
    moves: List[game.Move] = []
    outcomes: List[game.Evaluation] = []

    logging.info("Generating training set...")
    n = -1
    players = [
        _agent.TreeNodePlayer(
            g,
            model,
            limit=node_count,
            temperature=1.0,
            alpha=0.3,
        )
        for _ in range(g.eval_size())
    ]

    for _ in tqdm.trange(num_games):
        outcome = game.play_classical(g, players)

        # reservoir sampling labeled data for training.
        for player in players:
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
            player.states.clear()
            player.moves.clear()
            player.root = None

    logging.info("Fitting model...")
    ss = np.array(states)
    ms = np.array(moves)
    oc = np.array(outcomes)
    model.fit(ss, [ms, oc])
