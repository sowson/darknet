#!/usr/bin/env python3
import chess
import chess.pgn
import time
import math
import subprocess
import matplotlib.pyplot as plt
import sys
import os

from chess.engine import Limit

if len(sys.argv) < 2:
    print("Usage: python3 estimator.py <engine_binary>")
    sys.exit(1)

MY_ENGINE_PATH = "./" + sys.argv[1]
if not os.path.isfile(MY_ENGINE_PATH):
    print(f"Error: Engine binary '{MY_ENGINE_PATH}' not found.")
    sys.exit(1)

STOCKFISH_PATH = "/usr/local/bin/stockfish"
INITIAL_ELO = 1400
STOCKFISH_ELO = min(max(1350, 1600), 2850)
K_FACTOR = 20
GAMES_TO_PLAY = 50
MOVE_TIME_MS = 2000
FIRST_MOVE_TIME_MS = 5000

def expected_score(elo1, elo2):
    return 1 / (1 + 10 ** ((elo2 - elo1) / 400))

def update_elo(elo, expected, score, k=K_FACTOR):
    value = elo + k * (score - expected)
    return min(max(value, 1350), 2850)

def start_engine(path):
    p = subprocess.Popen(path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, bufsize=1)
    send_cmd(p, "uci")
    wait_for(p, "uciok")
    send_cmd(p, "isready")
    wait_for(p, "readyok")
    return p

def send_cmd(engine, command):
    engine.stdin.write(command + "\n")
    engine.stdin.flush()

def wait_for(engine, expected):
    while True:
        line = engine.stdout.readline().strip()
        if line == expected:
            break

def get_bestmove(engine, board, time_ms):
    moves = " ".join(move.uci() for move in board.move_stack)
    pos_cmd = f"position startpos moves {moves}" if moves else "position startpos"
    send_cmd(engine, pos_cmd)
    send_cmd(engine, f"go movetime {time_ms}")
    while True:
        line = engine.stdout.readline().strip()
        if line.startswith("bestmove"):
            tokens = line.split()
            if len(tokens) >= 2:
                return tokens[1].lower()
            return None

elo_history = [INITIAL_ELO]

my_engine = start_engine(MY_ENGINE_PATH)
stockfish = start_engine(STOCKFISH_PATH)
send_cmd(stockfish, f"setoption name UCI_LimitStrength value true")
send_cmd(stockfish, f"setoption name UCI_Elo value {STOCKFISH_ELO}")

current_elo = INITIAL_ELO

my_engine_name = sys.argv[1]

for game_num in range(1, GAMES_TO_PLAY + 1):
    board = chess.Board()
    if game_num % 2 == 1:
        engines = [my_engine, stockfish]
        names = [my_engine_name, "Stockfish"]
    else:
        engines = [stockfish, my_engine]
        names = ["Stockfish", my_engine_name]

    print(f"Game {game_num}: {names[0]} (White) vs {names[1]} (Black) — Current Elo: {round(current_elo)}")

    send_cmd(engines[0], "ucinewgame")
    send_cmd(engines[1], "ucinewgame")

    ply = 0
    while not board.is_game_over():
        turn = 0 if board.turn == chess.WHITE else 1
        is_my_turn = (engines[turn] == my_engine)
        time_ms = FIRST_MOVE_TIME_MS if (ply == 0 and is_my_turn) else MOVE_TIME_MS

        move_uci = get_bestmove(engines[turn], board, time_ms)
        try:
            move = board.parse_uci(move_uci)
            board.push(move)
        except:
            print(f"Invalid move: {move_uci}")
            break
        ply += 1

    result_str = board.result()

    if (game_num % 2 == 1 and result_str == "1-0") or (game_num % 2 == 0 and result_str == "0-1"):
        score = 1
    elif result_str == "1/2-1/2":
        score = 0.5
    else:
        score = 0

    if (score == 1):   print("+Result:", result_str)
    if (score == 0.5): print("=Result:", result_str)
    if (score == 0):   print("-Result:", result_str)

    # Create PGN game
    game_pgn = chess.pgn.Game()
    game_pgn.headers["White"] = names[0]
    game_pgn.headers["Black"] = names[1]
    game_pgn.headers["Result"] = result_str

    node = game_pgn
    for move in board.move_stack:
        node = node.add_main_variation(move)

    pgn_filename = f"game_{game_num:03}.pgn"
    with open(pgn_filename, "w") as pgn_file:
        print(game_pgn, file=pgn_file)

    exp = expected_score(current_elo, STOCKFISH_ELO)
    current_elo = update_elo(current_elo, exp, score)
    elo_history.append(current_elo)

send_cmd(my_engine, "quit")
send_cmd(stockfish, "quit")

print("Final Elo:", round(current_elo))
