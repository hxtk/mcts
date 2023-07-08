import logging

import tensorflow as tf

from mcts import agent
from mcts.tictactoe import tensor


class PathModelStore:  # noqa: D101
    def __init__(self, path: str) -> None:
        self.path = path

    def load_model(self) -> tf.keras.Model:
        try:
            return tf.keras.models.load_model(self.path)
        except IOError:
            return agent.residual_model(
                tensor,
                residual_layers=2,
                residual_conv_filters=32,
            )

    def save_model(self, model: tf.keras.Model) -> None:
        model.save(self.path, overwrite=True)


def main() -> None:
    ms = PathModelStore("model/")
    model = ms.load_model()

    print(model.summary())
    for x in [0.02, 0.002, 0.0002, 0.00002]:
        logging.info("learning_rate=%f", x)
        agent.train(
            model,
            tensor,
            ms,
            learning_rate=x,
            node_count=10,
            games_per_batch=100,
            samples_per_batch=2000,
            test_games=10,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
