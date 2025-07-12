import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import random

# --- Representación del tablero como vector fractal ---
# 0 = vacío, 1 = X, 2 = O

def draw_board(ax, board, highlight=None, title=None):
    # board: vector de 9 elementos
    for i in range(3):
        for j in range(3):
            idx = i*3 + j
            x, y = j, 2-i
            val = board[idx]
            color = 'white'
            if val == 1:
                color = 'orange'
            elif val == 2:
                color = 'deepskyblue'
            if highlight and idx == highlight:
                edge = 'red'
                lw = 4
            else:
                edge = 'gray'
                lw = 2
            circle = plt.Circle((x, y), 0.38, color=color, ec=edge, lw=lw, alpha=0.8)
            ax.add_patch(circle)
            if val == 1:
                ax.text(x, y, 'X', ha='center', va='center', fontsize=28, color='black')
            elif val == 2:
                ax.text(x, y, 'O', ha='center', va='center', fontsize=28, color='black')
    ax.set_xlim(-0.7, 2.7)
    ax.set_ylim(-0.7, 2.7)
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=16)

# --- Lógica de juego y bots ---
def check_winner(board):
    wins = [
        [0,1,2],[3,4,5],[6,7,8], # filas
        [0,3,6],[1,4,7],[2,5,8], # columnas
        [0,4,8],[2,4,6]          # diagonales
    ]
    for w in wins:
        vals = [board[i] for i in w]
        if vals[0] != 0 and all(v == vals[0] for v in vals):
            return vals[0]
    if all(x != 0 for x in board):
        return 0 # empate
    return None


def random_bot(board, player):
    moves = [i for i, v in enumerate(board) if v == 0]
    if not moves:
        return None
    return random.choice(moves)

# --- Bot inteligente: gana o bloquea si puede, si no aleatorio ---
def smart_bot(board, player):
    opponent = 3 - player
    moves = [i for i, v in enumerate(board) if v == 0]
    # 1. ¿Puedo ganar?
    for m in moves:
        b = board[:]
        b[m] = player
        if check_winner(b) == player:
            return m
    # 2. ¿Debo bloquear?
    for m in moves:
        b = board[:]
        b[m] = opponent
        if check_winner(b) == opponent:
            return m
    # 3. Si no, aleatorio
    if not moves:
        return None
    return random.choice(moves)

# --- Aurora simple: aprende de partidas y luego juega ---
class AuroraTicTacToe:
    def __init__(self):
        self.memory = [] # (tablero, jugada, resultado)
    def observe(self, board, move, result):
        self.memory.append((tuple(board), move, result))
    def play(self, board, player):
        # Busca en memoria jugadas ganadoras para este tablero
        for b, m, r in self.memory:
            if b == tuple(board) and r == player:
                return m
        # Si no sabe, juega aleatorio
        return random_bot(board, player)

# --- Simulación de aprendizaje ---
def simulate_learning_games(n_games=20):
    aurora = AuroraTicTacToe()
    history = []
    for _ in range(n_games):
        board = [0]*9
        player = 1
        moves = []
        while True:
            # Uno de los bots es inteligente, el otro aleatorio
            if player == 1:
                move = smart_bot(board, player)
            else:
                move = random_bot(board, player)
            if move is None:
                break
            board[move] = player
            moves.append((list(board), move, player))
            winner = check_winner(board)
            if winner is not None:
                for b, m, p in moves:
                    aurora.observe(b, m, winner)
                break
            player = 3 - player
        history.append([b for b,_,_ in moves])
    return aurora, history

# --- Animación ---
def animate_tictactoe():
    aurora, learn_history = simulate_learning_games(20)
    # Ahora Aurora juega contra smart_bot
    test_games = []
    for _ in range(3):
        board = [0]*9
        player = 1
        game = []
        while True:
            if player == 1:
                move = aurora.play(board, player)
            else:
                move = smart_bot(board, player)
            if move is None:
                break
            board[move] = player
            game.append((list(board), move, player))
            winner = check_winner(board)
            if winner is not None:
                break
            player = 3 - player
        test_games.append(game)

    # Prepara frames para animación
    frames = []
    # Fase de aprendizaje
    for game in learn_history[-3:]:
        for b in game:
            frames.append({'board': b, 'highlight': None, 'title': 'Aprendizaje observando bots'})
    # Fase de juego
    for game in test_games:
        for b, m, p in game:
            frames.append({'board': b, 'highlight': m, 'title': f'Aurora juega (jugador {p})'})

    fig, ax = plt.subplots(figsize=(5,5))
    def update(frame):
        ax.clear()
        draw_board(ax, frame['board'], highlight=frame['highlight'], title=frame['title'])
    ani = FuncAnimation(fig, update, frames=frames, interval=800, repeat=False)
    plt.show()

if __name__ == "__main__":
    animate_tictactoe()
