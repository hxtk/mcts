import game
import tictactoe._game as ttt
import tictactoe._players as players

if __name__ == '__main__':
    print(
        game.play_classical(ttt,
                            [players.TextIOPlayer(),
                             players.MinMaxPlayer()]))
