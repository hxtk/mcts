import logging

import tensorflow as tf

import agent
from tictactoe import tensor


class PathModelStore(object):

    def __init__(self, path: str) -> None:
        self.path = path

    def load_model(self) -> tf.keras.Model:
        return tf.keras.models.load_model(self.path)

    def save_model(self, model: tf.keras.Model) -> None:
        model.save(self.path, overwrite=True)


def main():
    ms = PathModelStore('data/')
    try:
        model = ms.load_model()
        ms.save_model(model)
        logging.info('Loaded model successfully.')
    except IOError:
        model = agent.build_model(tensor)
        logging.info('Constructed new model.')

    for x in [0.02, 0.002, 0.0002, 0.00002]:
        logging.info(f'learning_rate={x}')
        agent.train(
            model,
            tensor,
            ms,
            learning_rate=x,
            node_count=30,
            games_per_batch=10,
            samples_per_batch=20,
            test_games=10,
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
