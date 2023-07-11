from mcts.gym import _consumer
from mcts.gym import _producer
from mcts.gym import _storage

ReplayWriter = _storage.ReplayWriter

producer = _producer.main
consumer = _consumer.main
