import tensorflow as tf

import agent
import tictactoe.tensor as ttt


def main():
    model = agent.residual_model(ttt)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        loss=[
            tf.keras.losses.CategoricalCrossentropy(),
            tf.keras.losses.MeanSquaredError(),
        ],
    )
    agent.training_batch(model, ttt, 10, 10, node_count=10)


if __name__ == "__main__":
    main()
