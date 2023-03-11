import dataclasses
import datetime
from typing import Callable
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import Protocol
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf

import game

State = tf.Tensor
Move = tf.Tensor
Evaluation = tf.Tensor


@tf.function(reduce_retracing=True)
def _infer_batch(
    g: game.Game[State, Move, Evaluation],
    model: tf.keras.Model,
    states: State,
) -> Tuple[Move, Evaluation, Evaluation]:
    policies, values = model(states)
    evals = g.evaluate(states)
    return policies, values, evals


@dataclasses.dataclass
class TreeNode(object):
    g: game.Game[State, Move, Evaluation]
    model: tf.keras.Model
    parent: Optional['TreeNode']
    children: Optional[Mapping[int, 'TreeNode']]
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

    def _build_children(self) -> Mapping[int, 'TreeNode']:
        outputs = {}
        states = []
        for i in range(np.prod(self.policy.shape)):
            index = np.unravel_index(i, self.g.policy_shape())
            if self.mask[index] == 0:
                continue
            policy = tf.scatter_nd(
                indices=[index],
                updates=[1.],
                shape=self.g.policy_shape(),
            )
            state = tf.reshape(self.g.play_move(self.state, policy),
                               shape=(1,) + self.g.state_shape())
            mask = self.g.move_mask(state[0])
            states.append((i, state, mask))

        if len(states) == 0:
            return outputs

        policies, values, evals = _infer_batch(
            self.g, self.model, tf.concat([x[1] for x in states], axis=0))

        data = zip(states, policies, values, evals)
        for x, policy, value, evaluation in data:
            i, state, mask = x
            game_continues = tf.math.reduce_any(tf.math.is_nan(evaluation))
            if game_continues:
                value = value[self.g.player(state[0])].numpy()
            else:
                value = evaluation[self.g.player(state[0])].numpy()

            outputs[i] = TreeNode(
                g=self.g,
                model=self.model,
                parent=self,
                children=None,
                state=state[0],
                policy=policy,
                mask=mask,
                end=not game_continues,
                n=0,
                w=0,
                q=0,
                p=self.policy[np.unravel_index(x[0], self.g.policy_shape())],
                v=value,
            )

        return outputs

    @classmethod
    def build(
        cls,
        g: game.Game[State, Move, Evaluation],
        model: tf.keras.Model,
        state: State,
        parent: Optional['TreeNode'] = None,
        p: float = 0,
    ) -> 'TreeNode':
        mask = g.move_mask(state)
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
            children=None,
            state=state,
            policy=policy,
            mask=mask,
            end=not game_continues,
            n=0.,
            w=0.,
            q=0.,
            p=p,
            v=value,
        )

    def get_child(
        self,
        move: int,
    ) -> Optional['TreeNode']:
        if self.children is None:
            self.children = self._build_children()
        if move in self.children:
            return self.children[move]
        return None

    def max_qu_child(self) -> Optional['TreeNode']:
        max_child: Optional['TreeNode'] = None
        for i in range(np.prod(self.g.policy_shape())):
            if max_child is None:
                max_child = self.get_child(i)
                continue

            child = self.get_child(i)
            if child is None:
                continue

            # If this move wins, play it unconditionally.
            if child.end:
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


class TreeBuilder(object):

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
        )
        if noise is not None:
            root.policy = root.policy + noise().reshape(root.policy.shape)

        root.n = 1

        if isinstance(limit, int):
            limit = CountLimit(limit)
        elif isinstance(limit, datetime.timedelta):
            limit = TimeLimit(datetime.datetime.utcnow() + limit)

        while limit.increment():
            root.grow_tree()
        moves = []
        ns = []
        assert root.children is not None
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
