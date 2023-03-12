"""game.Player implementation using MCTS."""
import datetime
import logging
from typing import List
from typing import Optional
from typing import Protocol
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf

import game
from agent import _mcts

State = tf.Tensor
Move = tf.Tensor
Evaluation = tf.Tensor


class LimitCondition(Protocol):

    def increment(self) -> bool:
        """Report an operation taking place.

        Returns:
            True if the limiting condition has not been reached.
            False otherwise.
        """
        raise Exception('Not implemented')


class TimeLimit(object):
    """TimeLimit is a LimitCondition that expires at a certain time."""

    def __init__(self, end: datetime.datetime):
        self.end = end

    def increment(self) -> bool:
        """Report an operation taking place.

        Returns:
            True if the current time is before the end time.
            False otherwise.
        """
        return datetime.datetime.utcnow() < self.end


class CountLimit(object):
    """CountLimit is a LimitCondition that limits the number of operations."""

    def __init__(self, limit: int, count: int = 0):
        self.limit = limit
        self.count = count

    def increment(self) -> bool:
        """Report an operation taking place.

        Returns:
            True until the number of calls plus the initial `count` (default: 0)
            totals more than the configured limit.
            False for all subsequent calls.
        """
        self.count += 1
        return self.count <= self.limit


class TreeNodePlayer(object):
    """Player that performs MCTS and evaluates leaf nodes with an ANN.

    Args:
        g: tf.Tensor-based game to play.
        model: a callable accepting batches of game states as inputs and
            returning batches of policy probability distributions and
            estimated scores for each player.
        limit: a limit condition telling the agent when to terminate MCTS.
        alpha: the alpha parameter used for adding Dirichlet noise to the
            root node for MCTS (or None if no noise).
        deterministic: if True, select the "best" move evaluated by MCTS.
            Otherwise, select a weighted random move based on MCTS predicted
            move probabilities.
        show_eval: if True, print the estimated win-probability of the
            current player.
    """

    def __init__(
        self,
        g: game.Game[State, Move, Evaluation],
        model: tf.keras.Model,
        limit: Union[LimitCondition, int, datetime.timedelta],
        temperature: float = 1.0,
        alpha: Optional[float] = None,
        deterministic: bool = False,
        show_eval: bool = False,
    ):
        self.game = g
        self.model = model
        self.limit = limit
        self.temperature = temperature
        self.alpha = alpha
        self.deterministic = deterministic
        self.show_eval = show_eval

        self.root: Optional[_mcts.TreeNode] = None
        self.states: List[State] = []
        self.moves: List[Move] = []

    def _noise(self, alpha: float):
        shape = self.game.policy_shape()
        size = np.prod(shape)
        return np.random.dirichlet([alpha] * size).reshape(shape)

    def _choose_move_nd(
        self,
        moves: List[int],
        ns: List[float],
    ) -> Tuple[int, Move]:
        ps = np.array(ns)**(1 / self.temperature)
        ps = np.exp(ps) / sum(np.exp(ps))
        move = np.random.choice(moves, p=ps)

        shape = self.game.policy_shape()
        index = np.unravel_index(move, shape)
        return move, tf.scatter_nd(
            indices=[index],
            updates=[1.],
            shape=shape,
        )

    def _choose_move(
        self,
        moves: List[int],
        ns: List[float],
    ) -> Tuple[int, Move]:
        move = moves[np.argmax(ns)]

        shape = self.game.policy_shape()
        index = np.unravel_index(move, shape)
        return move, tf.scatter_nd(
            indices=[index],
            updates=[1.],
            shape=shape,
        )

    def __call__(self, state: State, mask: Move) -> Move:
        if isinstance(self.limit, int):
            limit = CountLimit(self.limit)
        elif isinstance(self.limit, datetime.timedelta):
            limit = TimeLimit(datetime.datetime.utcnow() + self.limit)
        else:
            limit = self.limit

        if self.root is None:
            self.root = _mcts.TreeNode.build(
                g=self.game,
                model=self.model,
                parent=None,
                state=state,
                p=1.0,
            )
        else:
            self.root.build_children()
            for v in self.root.children.values():
                if tf.math.reduce_all(v.state == state):
                    self.root = v
                    break
            else:
                logging.warning(
                    (
                        'state does not follow from previous state. '
                        'Rebuilding MCTS tree.'
                    ),
                )
                print(state)
                print(
                    list(child.state for child in self.root.children.values())
                )
                self.root = _mcts.TreeNode.build(
                    g=self.game,
                    model=self.model,
                    parent=None,
                    state=state,
                    p=1.0,
                )

        if self.alpha is not None:
            self.root.policy += self._noise(self.alpha)

        while limit.increment():
            self.root.grow_tree()

        moves = []
        ns = []
        for k, v in self.root.children.items():
            moves.append(k)
            ns.append(v.n)

        if self.deterministic:
            move, policy = self._choose_move(moves, ns)
        else:
            move, policy = self._choose_move_nd(moves, ns)

        self.root = self.root.children[move]

        if self.show_eval:
            print(f'Evaluation: {self.root.v}')

        self.states.append(state)
        self.moves.append(policy)

        return policy
