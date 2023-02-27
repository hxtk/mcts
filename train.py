import logging

import tensorflow as tf

import agent
import tictactoe as ttt


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
        logging.info('Loaded model successfully.')
    except IOError:
        g = ttt.new()
        m = ttt.move_mask(g)
        model = agent.build_model(g.shape, m.shape)
        logging.info('Constructed new model.')

    for x in [0.002, 0.0002, 0.00002]:
        logging.info(f'learning_rate={x}')
        agent.train(
            model,
            ttt,
            ms,
            learning_rate=x,
            games_per_batch=100,
            test_games=100,
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
