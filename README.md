# Monte-Carlo Tree Search

This repository is a collection of reinforcement learning algorithms and
training environments using Monte-Carlo Tree Search with neural network evaluation
of leaf nodes.

## Purpose

This is a hobby project to replicate the AlphaZero paper's algorithm and address
training issues by using the Data Generation appendix of the MuZero paper. I hope
to implement the MuZero algorithm in the future.

At present, we provide two executables that use a shared storage to share a replay
buffer of game data that is generated asynchronously by arbitrarily many consumers
and used by a consumer to update the model.

This approach does not rely on any locks and can scale to a very large number of
data generation processes. Arguments are provided to scale vertically by using
arbitrarily many processes on a single computer, but with a shared network file
system, this data generation process can safely be scaled to quite a large number
of machines.

This horizontal scalability is to get around the problem that, even on an optimized
tensorflow implementation of TicTacToe, playing a training game takes on the order
of seconds. For less optimized or more complicated simulators, such as chess, this
will only make things harder as the time per node, nodes per move, and moves per
game all increase. Relying on optimized single-thread performance for these problems
proved to be untenable.

As required arguments, the producer and consumer take a path for saving/loading the
model, a path for saving/loading replay buffers, and the module path of the
simulation to train.

At present, the only supported simulations are classical games, due to limitations
of both the agent's ability to learn and the gym's ability to encode arbitrary
reinforcement learning simulations.

## Usage

Implement your environment as a `mcts.game.Game`. See `mcts.tictactoe.tensor` for an
example implementation of the game TicTacToe, which will be used as an example.

Open two terminals. In one of them, run the following command, substituting
"4" with the number appropriate for CPU availability on your system.

```commandline
mcts_generate --threads 4 data model mcts.tictactoe.tensor
```

In the other terminal, run `mcts_train data momdel mcts.tictactoe.tensor`.

Together, these two commands shall asynchronously generate data and use that data to
train a model for the simulated environment. It rapidly becomes unbeatable
in tic-tac-toe. See [play.py](./play.py) for an example usage of the resulting
trained model.
