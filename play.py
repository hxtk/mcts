import datetime
import logging

import tensorflow as tf

import agent
import game
import tictactoe.tensor as ttt


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
        model = agent.residual_model(ttt)
        logging.info('Constructed new model.')

    human = ttt.players.TextIOPlayer()
    cpu = agent.CompetitivePlayer(
        ttt, model, datetime.timedelta(seconds=1), show_eval=True
    )
    print(game.play_classical(ttt, [human, cpu]))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
