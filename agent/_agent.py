import datetime
from typing import List
from typing import Union

import tensorflow as tf

import game
from agent import _mcts


class TrainingPlayer(object):
    """Player that makes weighted random moves and saves a replay buffer."""

    def __init__(
        self,
        g: game.Game,
        model: tf.keras.Model,
        limit: Union[int, datetime.timedelta],
    ) -> None:
        self.game = g
        self.model = model
        self.limit = limit
        self.states: List[game.State] = []
        self.moves: List[game.Move] = []

    def __call__(self, state: game.State, mask: game.Move) -> game.Move:
        move = _mcts.training_choose_move(
            g=self.game,
            model=self.model,
            state=state,
            limit=self.limit,
            temperature=1.0,
            mask=mask,
        )
        self.states.append(state)
        self.moves.append(move)
        return move


class CompetitivePlayer(object):

    def __init__(
        self,
        g: game.Game,
        model: tf.keras.Model,
        limit: Union[int, datetime.timedelta],
        show_eval: bool = False,
    ) -> None:
        self.game = g
        self.model = model
        self.limit = limit
        self.show_eval = show_eval

    def __call__(self, state: game.State, mask: game.Move) -> game.Move:
        move = _mcts.competitive_choose_move(
            g=self.game,
            model=self.model,
            state=state,
            limit=self.limit,
            mask=mask,
        )

        if self.show_eval:
            inputs = tf.convert_to_tensor(state.reshape((1,) + state.shape))
            _, value = self.model(inputs)
            print(f'Evaluation: {value}')

        return move