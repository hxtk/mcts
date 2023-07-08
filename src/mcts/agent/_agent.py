"""game.Player implementation using MCTS."""
import datetime
from typing import List
from typing import Optional
from typing import Protocol
from typing import Union

import numpy as np
import tensorflow as tf

from mcts import game
from mcts.agent import _mcts

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
        raise NotImplementedError()


class TimeLimit:
    """TimeLimit is a LimitCondition that expires at a certain time."""

    def __init__(self, end: datetime.datetime):
        self.end = end

    def increment(self) -> bool:
        """Report an operation taking place.

        Returns:
            True if the current time is before the end time.
            False otherwise.
        """
        return datetime.datetime.now(tz=datetime.UTC) < self.end


class CountLimit:
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


class TreeNodePlayer:
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

    def __call__(self, state: State, mask: Move) -> Move:
        if isinstance(self.limit, int):
            limit = CountLimit(self.limit)
        elif isinstance(self.limit, datetime.timedelta):
            limit = TimeLimit(
                datetime.datetime.now(tz=datetime.UTC) + self.limit,
            )
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
            for v in self.root.children.values():
                if tf.math.reduce_all(v.state == state):
                    self.root = v
                    break
            else:
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
            move_ref = moves[np.argmax(ns)]
        else:
            ps = np.array(ns) ** (1 / self.temperature)
            ps = np.exp(ps) / sum(np.exp(ps))
            move_ref = np.random.choice(moves, p=ps)

        self.root = self.root.children[move_ref]
        self.root.parent = None

        move = move_ref.deref()

        if self.show_eval:
            print(f"Evaluation: {self.root.v}")

        self.states.append(state)
        self.moves.append(move)

        return move
