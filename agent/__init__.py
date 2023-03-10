"""Monte-Carlo Tree Search Agent for Games."""
from agent import _agent
from agent import _model
from agent import _train

residual_model = _model.residual_model
train = _train.train
training_batch = _train.training_batch

MCTSPlayer = _agent.TreeNodePlayer
