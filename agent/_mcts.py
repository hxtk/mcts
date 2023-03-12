"""Monte-Carlo Tree Search with ANN to evaluate leaf nodes."""
import dataclasses
from typing import Optional
from typing import Tuple

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
    children: dict[int, 'TreeNode']
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
        for i in range(np.prod(self.policy.shape)):
            index = np.unravel_index(i, self.g.policy_shape())
            if self.mask[index] == 0:
                continue
            policy = tf.scatter_nd(
                indices=[index],
                updates=[1.],
                shape=self.g.policy_shape(),
            )
            state = tf.reshape(
                self.g.play_move(self.state, policy),
                shape=(1,) + self.g.state_shape()
            )
            mask = self.g.move_mask(state[0])
            states.append((i, state, mask))

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
            i, state, mask = x
            game_continues = tf.math.reduce_any(tf.math.is_nan(evaluation))
            if game_continues:
                value = value[self.g.player(state[0])].numpy()
            else:
                value = evaluation[self.g.player(state[0])].numpy()

            self.children[i] = TreeNode(
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
                p=self.policy[np.unravel_index(x[0], self.g.policy_shape())],
                v=value,
            )

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
            children={},
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
        self.build_children()
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
        while node is not None:
            node.n += 1
            node.w += leaf.v
            node.q = node.w / node.n
            node = node.parent
