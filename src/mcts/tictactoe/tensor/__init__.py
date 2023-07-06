from mcts.tictactoe.tensor import _game
from mcts.tictactoe.tensor import _players

players = _players

new = _game.new
play_move = _game.play_move
move_mask = _game.move_mask
evaluate = _game.evaluate
player = _game.player
state_shape = _game.state_shape
policy_shape = _game.policy_shape
eval_size = _game.eval_size
