import argparse
import importlib
import pathlib
import sys
import time
import typing
from typing import NoReturn

import tensorflow as tf

from mcts import game
from mcts.gym import _storage


def main() -> NoReturn:
    parser = argparse.ArgumentParser(
        "train",
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
        "-I",
        "--interval",
        type=int,
        default=60,
        help="time in seconds to wait between model trainings",
    )
    parser.add_argument(
        "-L",
        "--learn_rate",
        default=0.02,
        type=float,
        help="SGD learning rate to use for training",
    )
    parser.add_argument(
        "-S",
        "--size",
        default=5000,
        type=int,
        help="number of data to include in each training",
    )
    args = parser.parse_args()

    module, sep, name = typing.cast(str, args.simulator).partition(":")
    if sep != "" and name == "":
        sys.exit(1)

    mod = importlib.import_module(module)
    game_module: game.Game = mod if sep == "" else getattr(mod, name)

    storage = _storage.Storage(game_module, args.data_path)
    model_store = _storage.PathModelStore(args.model_path, game_module)
    while True:
        model = model_store.load_model()
        model.compile(
            optimizer=tf.keras.optimizers.experimental.SGD(
                learning_rate=args.learn_rate,
            ),
            loss=[
                tf.keras.losses.CategoricalCrossentropy(),
                tf.keras.losses.MeanSquaredError(),
            ],
        )

        dataset = storage.get_dataset(args.size).shuffle(args.size).batch(100)
        model.fit(dataset)
        model.save(args.model_path)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
