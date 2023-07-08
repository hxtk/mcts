import argparse
import multiprocessing
import pathlib
from typing import NoReturn

import tensorflow as tf
import tqdm

import mcts.tictactoe.tensor as ttt
from mcts import agent
from mcts import game
from mcts.gym import _storage


def build_batch(
    g: game.Game[tf.Tensor, tf.Tensor, tf.Tensor],
    model: tf.keras.Model,
    storage: _storage.Storage,
    num_items: int,
    alpha: float,
    temperature: float,
    node_count: int,
) -> None:
    for _ in tqdm.trange(num_items):
        with storage.writer() as w:
            agent.training_game(
                model,
                g,
                alpha=alpha,
                temperature=temperature,
                node_count=node_count,
                w=w,
            )


def run(
    g: game.Game[tf.Tensor, tf.Tensor, tf.Tensor],
    model_store: _storage.PathModelStore,
    storage: _storage.Storage,
    refresh_interval: int = 5000,
    alpha: float = 0.3,
    temperature: float = 1.0,
    node_count: int = 30,
) -> NoReturn:
    while True:
        model = model_store.load_model()
        build_batch(
            g,
            model,
            storage,
            refresh_interval,
            alpha,
            temperature,
            node_count,
        )


def main() -> NoReturn:
    parser = argparse.ArgumentParser(
        "data-generate",
    )
    parser.add_argument(
        "data_path",
        type=pathlib.Path,
        help="path to write generated training data",
    )
    parser.add_argument(
        "model_path",
        type=pathlib.Path,
        help="path from which to load models",
    )
    parser.add_argument(
        "-R",
        "--refresh",
        type=int,
        default=100,
        help="the number of games to play between each refresh of the model",
    )
    parser.add_argument(
        "-T",
        "--temperature",
        type=float,
        default=1.0,
        help=(
            "(0, 1] value controlling how much the model favors exploration "
            "vs exploitation, with larger values being more greedy"
        ),
    )
    parser.add_argument(
        "-n",
        "--nodes",
        type=int,
        default=50,
        help=(
            "the number of game tree nodes to explore in MCTS "
            "before choosing a move"
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help=(
            "parameter controlling the Dirichlet noise applied to the root "
            "node of the search tree for non-deterministic play, usually "
            "chosen to be proportional to the inverse of the mean number of "
            "legal moves"
        ),
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=1,
        help="number of processes to use for parallel execution",
    )
    args = parser.parse_args()
    game_module = ttt
    kwargs = {
        "g": game_module,
        "storage": _storage.Storage(game_module, args.data_path),
        "model_store": _storage.PathModelStore(args.model_path, game_module),
        "refresh_interval": args.refresh,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "node_count": args.nodes,
    }
    processes = [
        multiprocessing.Process(
            target=run,
            kwargs=kwargs,
        )
        for _ in range(args.threads - 1)
    ]
    for p in processes:
        p.start()

    try:
        run(**kwargs)
    finally:
        for p in processes:
            p.terminate()


if __name__ == "__main__":
    main()
