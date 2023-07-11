"""Monte-Carlo Tree Search with ANN to evaluate leaf nodes."""
import dataclasses
from typing import Any
from typing import Dict
from typing import Optional
from typing import Protocol
from typing import Tuple

import numpy as np
import tensorflow as tf

from mcts import game

State = tf.Tensor
Move = tf.Tensor
Evaluation = tf.Tensor


class Reference(Protocol):
    def __hash__(self) -> Any:  # noqa: ANN401, inheriting builtin type.
        pass

    def deref(self) -> tf.Tensor:
        pass


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
class TreeNode:
    g: game.Game[State, Move, Evaluation]
    model: tf.keras.Model
    parent: Optional["TreeNode"]
    children: Dict[Reference, "TreeNode"]
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

    def build_children(self) -> None:
        if self.children:
            return
        states = []
        policy_shape = self.g.policy_shape()
        legal_policies = tf.reshape(
            tf.one_hot(
                indices=tf.where(
                    tf.reshape(self.mask, shape=(-1,)),
                ),
                depth=np.prod(policy_shape),
            ),
            shape=(-1, *policy_shape),
        )
        for move in legal_policies:
            state = tf.reshape(
                self.g.play_move(self.state, move),
                shape=(1, *self.g.state_shape()),
            )
            mask = self.g.move_mask(state[0])
            states.append((move, state, mask))

        if len(states) == 0:
            return

        policies, values, evals = _infer_batch(
            self.g,
            self.model,
            tf.concat(
                values=[x[1] for x in states],
                axis=0,
            ),
        )

        data = zip(states, policies, values, evals)
        for x, policy, value, evaluation in data:
            move, state, mask = x
            game_continues = tf.math.reduce_any(tf.math.is_nan(evaluation))
            if game_continues:
                value = value[self.g.player(state[0])].numpy()
            else:
                value = evaluation[self.g.player(state[0])].numpy()

            self.children[move.ref()] = TreeNode(
                g=self.g,
                model=self.model,
                parent=self,
                children={},
                state=state[0],
                policy=policy,
                mask=mask,
                end=not game_continues,
                n=0,
                w=0,
                q=0,
                p=tf.math.reduce_sum(self.policy * move),
                v=value,
            )

    @classmethod
    def build(
        cls,
        g: game.Game[State, Move, Evaluation],
        model: tf.keras.Model,
        state: State,
        parent: Optional["TreeNode"] = None,
        p: float = 0,
    ) -> "TreeNode":
        mask = g.move_mask(state)
        inputs = tf.reshape(state, (1, *g.state_shape()))
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
            children={},
            state=state,
            policy=policy,
            mask=mask,
            end=not game_continues,
            n=0.0,
            w=0.0,
            q=0.0,
            p=p,
            v=value,
        )

    def max_qu_child(self) -> Optional["TreeNode"]:
        self.build_children()

        max_child: Optional["TreeNode"] = None
        for child in self.children.values():
            if max_child is None:
                max_child = child
                continue

            # If this move wins, play it unconditionally.
            if child.end:
                return child

            if child.qu > max_child.qu:
                max_child = child
        return max_child

    def grow_tree(self):
        leaf = self.max_qu_child()
        while leaf.n != 0:
            child = leaf.max_qu_child()
            if child is None:
                break
            leaf = child

        node = leaf
        while node is not None:
            node.n += 1
            node.w += leaf.v
            node.q = node.w / node.n
            node = node.parent
