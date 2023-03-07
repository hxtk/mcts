import datetime
from typing import List
from typing import MutableMapping
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
        temperature: float,
        alpha: float,
    ) -> None:
        self.game = g
        self.model = model
        self.limit = limit
        self.temperature = temperature
        self.alpha = alpha
        self.tree_builder = _mcts.TreeBuilder({})

        self.states: List[game.State] = []
        self.moves: List[game.Move] = []

    def __call__(self, state: game.State, mask: game.Move) -> game.Move:
        move = _mcts.training_choose_move(
            g=self.game,
            model=self.model,
            state=state,
            limit=self.limit,
            temperature=self.temperature,
            alpha=self.alpha,
            builder=self.tree_builder,
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
        self.tree_builder = _mcts.TreeBuilder({})

    def __call__(self, state: game.State, mask: game.Move) -> game.Move:
        move = _mcts.competitive_choose_move(
            g=self.game,
            model=self.model,
            state=state,
            limit=self.limit,
            builder=self.tree_builder,
        )

        if self.show_eval:
            inputs = tf.convert_to_tensor(state.reshape((1,) + state.shape))
            _, value = self.model(inputs)
            print(f'Evaluation: {value}')

        return move


def _hash_state(state: game.State, mask: game.Move) -> int:
    h = 0
    for x in state.flatten():
        if int(x) == 1:
            h |= 1
        h <<= 1
    for x in mask.flatten():
        if int(x) == 1:
            h |= 1
        h <<= 1
    return h


class CachingPlayer(object):

    def __init__(self, player: game.Player):
        self.player = player
        self.cache: MutableMapping[int, game.Move] = {}

    def __call__(
        self,
        state: game.State,
        mask: game.Move,
    ) -> game.Move:
        h = _hash_state(state, mask)
        if h in self.cache:
            return self.cache[h]
        return self.player(state, mask)
