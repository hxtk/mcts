import logging

import tensorflow as tf

import agent
from tictactoe import tensor


class PathModelStore(object):

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


def main():
    ms = PathModelStore('model/')
    model = ms.load_model()

    print(model.summary())
    for x in [0.02, 0.002, 0.0002, 0.00002]:
        logging.info(f'learning_rate={x}')
        agent.train(
            model,
            tensor,
            ms,
            learning_rate=x,
            node_count=30,
            games_per_batch=100,
            samples_per_batch=2000,
            test_games=10,
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
