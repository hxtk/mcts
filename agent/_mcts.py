import dataclasses
import datetime
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Protocol
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf

import game


@dataclasses.dataclass(slots=True)
class TreeNode(object):
    g: game.Game
    model: tf.keras.Model
    parent: Optional['TreeNode']
    children: MutableMapping[int, 'TreeNode']
    state: game.State
    policy: game.Move
    mask: game.Move

    # Number of times this node has been visited.
    n: float

    # Total weight of all child nodes.
    w: float

    # Average weight of all child nodes.
    q: float

    # Prior probability of visiting this node.
    p: float

    # Estimated value of state for that state's player.
    v: float

    # Exploration function.
    @property
    def u(self):
        return self.p / (1 + self.n)

    @property
    def qu(self):
        return self.q + self.u

    @classmethod
    def build(
        cls,
        g: game.Game,
        model: tf.keras.Model,
        state: game.State,
        parent: Optional['TreeNode'] = None,
        p: float = 0,
    ) -> 'TreeNode':
        inputs = tf.convert_to_tensor(state.reshape((1,) + state.shape))
        mask = g.move_mask(state)

        policy, value = model(inputs)
        policy = policy.numpy().reshape(mask.shape)
        return cls(
            g=g,
            model=model,
            parent=parent,
            children=dict(),
            state=state,
            policy=policy,
            mask=mask,
            n=0,
            w=0,
            q=0,
            p=p,
            v=value.numpy()[0][g.player(state)],
        )

    def get_child(self, move: int) -> Optional['TreeNode']:
        if move in self.children:
            return self.children[move]
        if self.mask.flatten()[move] == 0:
            return None

        policy = np.zeros(self.policy.size)
        policy[move] = 1
        policy.reshape(self.policy.shape)

        child = self.build(
            g=self.g,
            model=self.model,
            state=self.g.play_move(self.state, policy),
            parent=self,
            p=self.policy.flatten()[move],
        )
        self.children[move] = child
        return child

    def max_qu_child(self) -> Optional['TreeNode']:
        max_child: Optional['TreeNode'] = None
        for i, v in enumerate(self.mask.flatten()):
            if v == 0:
                continue
            if max_child is None:
                max_child = self.get_child(i)
                continue

            child = self.get_child(i)
            if child is None:
                continue

            # If this move wins, play it unconditionally.
            evaluation = self.g.evaluate(child.state)
            if evaluation is not None:
                child.v = evaluation[self.g.player(child.state)]
                return child

            if child.qu > max_child.qu:
                max_child = child
        return max_child

    def grow_tree(self):
        leaf = self
        while leaf.n != 0:
            child = leaf.max_qu_child()
            if child is None:
                break
            leaf = child

        node = leaf
        while node.parent is not None:
            node.n += 1
            node.w += leaf.v
            node.q = node.w / node.n
            node = node.parent


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


def _build_tree(
    g: game.Game,
    model: tf.keras.Model,
    state: game.State,
    limit: Union[LimitCondition, int, datetime.timedelta],
) -> Tuple[List[int], List[float]]:
    root = TreeNode.build(
        g=g,
        model=model,
        parent=None,
        state=state,
        p=1.0,
    )
    root.n = 1

    if isinstance(limit, int):
        limit = CountLimit(limit)
    elif isinstance(limit, datetime.timedelta):
        limit = TimeLimit(datetime.datetime.utcnow() + limit)

    while limit.increment():
        root.grow_tree()
    moves = []
    ns = []
    for k, v in root.children.items():
        moves.append(k)
        ns.append(v.n)
    return moves, ns


def training_choose_move(
    g: game.Game,
    model: tf.keras.Model,
    state: game.State,
    limit: Union[int, datetime.timedelta],
    temperature: float,
    mask: game.Move,
) -> game.Move:
    moves, ns = _build_tree(g, model, state, limit)

    ps = np.array(ns)**(1 / temperature)
    ps = np.exp(ps) / sum(np.exp(ps))
    move = np.random.choice(moves, p=ps)

    policy = np.zeros(mask.size)
    policy[move] = 1
    policy.reshape(mask.shape)

    return policy


def competitive_choose_move(
    g: game.Game,
    model: tf.keras.Model,
    state: game.State,
    limit: Union[int, datetime.timedelta],
    mask: game.Move,
) -> game.Move:
    moves, ns = _build_tree(g, model, state, limit)

    policy = np.zeros(mask.size)
    policy[moves] = ns
    policy.reshape(mask.shape)

    return policy
