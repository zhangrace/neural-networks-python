#! /usr/bin/env python3
import json
import numpy as np
from os import listdir as ls

RED   = u"\033[1;31m"
BLUE  = u"\033[1;34m"
RESET = u"\033[0;0m"
CIRCLE = u"\u25CF"

RED_DISK = RED + CIRCLE + RESET
BLUE_DISK = BLUE + CIRCLE + RESET
RED_BORDER = RED + "-" + RESET
BLUE_BORDER = BLUE + "\\" + RESET

def print_char(i):
    if i > 0:
        return BLUE_DISK
    if i < 0:
        return RED_DISK
    return u'\u00B7' # empty cell

def brighntess_scale(minimum, maximum, x):
    bucket_size = (maximum - minimum) / 23
    bucket = 233 + int((x - minimum) / bucket_size)
    return u"\033[38;5;" + str(bucket) + "m" + CIRCLE + RESET

def show_values(board, values):
    board = board.reshape(board.shape[0], board.shape[1])
    minimum = values[board == 0].min()
    maximum = values[board == 0].max()
    size = board.shape[0]
    s = u"\n" + (" " + RED_BORDER)*size +"\n"
    for i in range(size):
        s += " " * i + BLUE_BORDER + " "
        for j in range(size):
            if board[i,j] == 0:
                s += brighntess_scale(minimum, maximum, values[i,j]) + " "
            else:
                s += print_char(board[i,j]) + " "
        s += BLUE_BORDER + "\n"
    s += " "*(size) + " " + (" " + RED_BORDER) * size
    return s

def show_move(board, move, turn=None):
    board = board.reshape(board.shape[0], board.shape[1])
    size = board.shape[0]
    s = u"\n" + (" " + RED_BORDER)*size +"\n"
    for i in range(size):
        s += " " * i + BLUE_BORDER + " "
        for j in range(size):
            if i == move[0] and j == move[1]:
                if turn == 1:
                    s += "B "
                elif turn == -1:
                    s += "R "
                else:
                    s += "* "
            else:
                s += print_char(board[i,j]) + " "
        s += BLUE_BORDER + "\n"
    s += " "*(size) + " " + (" " + RED_BORDER) * size
    return s

def show_board(board):
    board = board.reshape(board.shape[0], board.shape[1])
    size = board.shape[0]
    s = u"\n" + (" " + RED_BORDER)*size +"\n"
    for i in range(size):
        s += " " * i + BLUE_BORDER + " "
        for j in range(size):
            s += print_char(board[i,j]) + " "
        s += BLUE_BORDER + "\n"
    s += " "*(size) + " " + (" " + RED_BORDER) * size
    return s
