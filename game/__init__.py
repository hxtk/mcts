"""Generalized framework for Games."""
from game import _game

Evaluation = _game.Evaluation
Game = _game.Game
Move = _game.Move
Player = _game.Player
State = _game.State

play_classical = _game.play_classical

__all__ = [
    'Game',
    'Move',
    'Player',
    'State',
    'play_classical',
]
