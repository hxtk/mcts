import agent
import tictactoe as ttt

model = agent.build_model(ttt)
agent.training_batch(model, ttt, 10, 10, node_count=30)
