import agent
import tictactoe as ttt

g = ttt.new()
m = ttt.move_mask(g)
model = agent.build_model(g.shape, m.shape)
agent.training_batch(model, ttt, 10, 10)
