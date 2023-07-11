import argparse
import importlib
import multiprocessing
import pathlib
import sys
import typing
from typing import NoReturn

import tensorflow as tf
import tqdm

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
    simulator: str,
    data_path: pathlib.Path,
    model_path: pathlib.Path,
    refresh_interval: int = 5000,
    alpha: float = 0.3,
    temperature: float = 1.0,
    node_count: int = 30,
) -> NoReturn:
    game_module = _load_simulator(simulator)
    storage = _storage.Storage(game_module, data_path)
    model_store = _storage.PathModelStore(model_path, game_module)
    while True:
        model = model_store.load_model()
        build_batch(
            game_module,
            model,
            storage,
            refresh_interval,
            alpha,
            temperature,
            node_count,
        )


def _load_simulator(path: str) -> game.Game[tf.Tensor, tf.Tensor, tf.Tensor]:
    module, sep, name = typing.cast(str, path).partition(":")
    if sep != "" and name == "":
        sys.exit(1)

    print(repr(module), repr(sep), repr(name))
    mod = importlib.import_module(module)
    game_module: game.Game = mod if sep == "" else getattr(mod, name)
    if not isinstance(game_module, game.Game):
        raise TypeError()
    return game_module


def main() -> NoReturn:
    parser = argparse.ArgumentParser(
        "generate",
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
        "simulator",
        type=str,
        help=(
            "module path of the simulator to be learned, e.g., foo.bar to "
            "specify a module, or foo.bar:Baz to specify a class within that "
            "module"
        ),
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
    print(args.simulator)
    kwargs = {
        "simulator": args.simulator,
        "data_path": args.data_path,
        "model_path": args.model_path,
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
