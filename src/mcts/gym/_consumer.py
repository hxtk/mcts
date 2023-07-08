import argparse
import pathlib
from typing import NoReturn

import tensorflow as tf

import mcts.tictactoe.tensor as ttt
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

    game_module = ttt
    storage = _storage.Storage(game_module, args.data_path)
    model_store = _storage.PathModelStore(args.model_path, game_module)
    for x in range(1):
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

        dataset = storage.get_dataset(1)
        for x in dataset.as_numpy_iterator():
            print(x)


if __name__ == "__main__":
    main()
