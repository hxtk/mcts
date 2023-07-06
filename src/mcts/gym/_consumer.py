import os
from typing import Union

import p_tqdm

from mcts.gym import _producer


def training_loop(
    batches: int,
    step_per_batch: int,
    num_cpus: Union[int, float, None] = None,
) -> None:
    if num_cpus is None:
        num_cpus = os.cpu_count()
    if isinstance(num_cpus, float):
        num_cpus = int(os.cpu_count() * num_cpus)

    p_tqdm.p_umap(_producer)
