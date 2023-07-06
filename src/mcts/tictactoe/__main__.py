import mcts.tictactoe._game as ttt
import mcts.tictactoe._players as players
from mcts import game

if __name__ == "__main__":
    print(
        game.play_classical(
            ttt,
            [players.TextIOPlayer(), players.MinMaxPlayer()],
        ),
    )
