import tensorflow as tf
import tqdm

from mcts import agent
from mcts import game
from mcts.gym import _storage


def build_batch(
    g: game.Game[tf.Tensor, tf.Tensor, tf.Tensor],
    ms: _storage.PathModelStore,
    storage: _storage.Storage,
    num_items: int,
    alpha: float,
    temperature: float,
    node_count: int,
) -> None:
    model = ms.load_model()
    for _ in tqdm.trange(num_items):
        with storage.writer() as w:
            agent.training_game(
                model,
                g,
                alpha=alpha,
                temperature=temperature,
                node_count=node_count,
                w=w,
            )
