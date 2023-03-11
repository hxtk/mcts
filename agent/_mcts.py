import dataclasses
import datetime
from typing import Callable
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Protocol
from typing import Tuple
from typing import Union

import numpy as np
import numpy.typing as npt
import tensorflow as tf

import game

State = tf.Tensor
Move = tf.Tensor
Evaluation = tf.Tensor


@dataclasses.dataclass
class TreeNode(object):
    g: game.Game[State, Move, Evaluation]
    model: tf.keras.Model
    parent: Optional['TreeNode']
    children: MutableMapping[int, 'TreeNode']
    state: State
    policy: Move
    mask: Move
    end: bool

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
        g: game.Game[State, Move, Evaluation],
        model: tf.keras.Model,
        state: State,
        parent: Optional['TreeNode'] = None,
        cache: Optional[MutableMapping] = None,
        p: float = 0,
    ) -> 'TreeNode':
        mask = g.move_mask(state)
        if cache is not None:
            h = _hash_state(state, mask).numpy()
            if h in cache:
                policy, value, evaluation = cache[h]
            else:
                inputs = tf.reshape(state, (1,) + g.state_shape())
                policy, value = model(inputs)
                policy = tf.reshape(policy, mask.shape)
                evaluation = g.evaluate(state)[0]
                cache[h] = policy, value, evaluation
        else:
            inputs = tf.reshape(state, (1,) + g.state_shape())
            policy, value = model(inputs)
            policy = tf.reshape(policy, mask.shape)
            evaluation = g.evaluate(state)[0]

        game_continues = tf.math.reduce_any(tf.math.is_nan(evaluation))
        if game_continues:
            value = value.numpy()[0][g.player(state)]
        else:
            value = evaluation[g.player(state)]

        return cls(
            g=g,
            model=model,
            parent=parent,
            children=dict(),
            state=state,
            policy=policy,
            mask=mask,
            end=not game_continues,
            n=0,
            w=0,
            q=0,
            p=p,
            v=value,
        )

    def get_child(
        self,
        move: int,
        cache: Optional[MutableMapping[int, Tuple[tf.Tensor,
                                                  tf.Tensor]]] = None,
    ) -> Optional['TreeNode']:
        if move in self.children:
            return self.children[move]
        size = np.prod(self.g.policy_shape())
        if tf.reshape(self.mask, (size,))[move] == 0:
            return None

        index = np.unravel_index(move, self.g.policy_shape())
        policy = tf.scatter_nd(
            indices=[index],
            updates=[1.],
            shape=self.g.policy_shape(),
        )

        child = self.build(
            g=self.g,
            model=self.model,
            state=self.g.play_move(self.state, policy),
            parent=self,
            p=self.policy[index],
            cache=cache,
        )
        self.children[move] = child
        return child

    def max_qu_child(
        self,
        cache: Optional[MutableMapping[int, Tuple[tf.Tensor,
                                                  tf.Tensor]]] = None,
    ) -> Optional['TreeNode']:
        max_child: Optional['TreeNode'] = None
        for i in range(np.prod(self.g.policy_shape())):
            if max_child is None:
                max_child = self.get_child(i, cache)
                continue

            child = self.get_child(i, cache)
            if child is None:
                continue

            # If this move wins, play it unconditionally.
            if child.end:
                return child

            if child.qu > max_child.qu:
                max_child = child
        return max_child

    def grow_tree(
        self,
        cache: Optional[MutableMapping[int, Tuple[tf.Tensor,
                                                  tf.Tensor]]] = None,
    ):
        leaf = self
        while leaf.n != 0:
            child = leaf.max_qu_child(cache=cache)
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


class TreeBuilder(object):

    def __init__(
        self,
        cache: Optional[MutableMapping[int, Tuple[tf.Tensor,
                                                  tf.Tensor]]] = None,
    ):
        self.cache = cache

    def build_tree(
        self,
        g: game.Game,
        model: tf.keras.Model,
        state: game.State,
        limit: Union[LimitCondition, int, datetime.timedelta],
        noise: Optional[Callable[[], np.ndarray]] = None
    ) -> Tuple[List[int], List[float]]:
        root = TreeNode.build(
            g=g,
            model=model,
            parent=None,
            state=state,
            p=1.0,
            cache=self.cache,
        )
        if noise is not None:
            root.policy = root.policy + noise().reshape(root.policy.shape)

        root.n = 1

        if isinstance(limit, int):
            limit = CountLimit(limit)
        elif isinstance(limit, datetime.timedelta):
            limit = TimeLimit(datetime.datetime.utcnow() + limit)

        while limit.increment():
            root.grow_tree(cache=self.cache)
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
        alpha: float,
        builder: TreeBuilder = TreeBuilder(),
) -> game.Move:

    def noise():
        size = np.prod(g.policy_shape())
        return np.random.dirichlet([alpha] * size)

    moves, ns = builder.build_tree(g, model, state, limit, noise=noise)

    ps = np.array(ns)**(1 / temperature)
    ps = np.exp(ps) / sum(np.exp(ps))
    move = np.random.choice(moves, p=ps)

    shape = g.policy_shape()
    index = np.unravel_index(move, shape)
    policy = tf.scatter_nd(
        indices=[index],
        updates=[1.],
        shape=shape,
    )

    return policy


def competitive_choose_move(
    g: game.Game,
    model: tf.keras.Model,
    state: game.State,
    limit: Union[int, datetime.timedelta],
    builder: TreeBuilder = TreeBuilder()
) -> game.Move:
    moves, ns = builder.build_tree(g, model, state, limit)

    move = moves[np.argmax(ns)]

    shape = g.policy_shape()
    index = np.unravel_index(move, shape)
    policy = tf.scatter_nd(
        indices=[index],
        updates=[1.],
        shape=shape,
    )

    return policy


@tf.function
def _hash_state(
    state: tf.Tensor,
    mask: tf.Tensor,
) -> tf.Tensor:
    entries = tf.concat([
        tf.reshape(state, shape=(-1,)),
        tf.reshape(mask, shape=(-1,)),
    ],
                        axis=0)
    return tf.foldl(
        fn=lambda a, x: tf.bitwise.bitwise_or(
            tf.bitwise.left_shift(a, tf.constant(1)),
            tf.constant(1) if x > 0.5 else tf.constant(0),
        ),
        elems=entries,
        initializer=tf.constant(0),
    )
