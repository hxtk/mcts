"""Monte-Carlo Tree Search Agent for Games."""
from mcts.agent import _agent
from mcts.agent import _model
from mcts.agent import _train

residual_model = _model.residual_model
train = _train.train
training_batch = _train.training_batch
training_game = _train.training_game

MCTSPlayer = _agent.TreeNodePlayer
ModelStore = _train.ModelStore
