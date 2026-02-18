import streamlit as st
import random

def initialize_game():
    st.session_state.board = [' '] * 9
    st.session_state.current_player = 'X'
    st.session_state.game_over = False
    st.session_state.history = []
    if 'q_table' not in st.session_state:
        st.session_state.q_table = {}
    if 'stats' not in st.session_state:
        st.session_state.stats = {'wins': 0, 'losses': 0, 'draws': 0}

    ai_move()

def check_winner(board):
    # Check rows
    for i in range(0, 9, 3):
        if board[i] == board[i+1] == board[i+2] != ' ':
            return board[i]
    # Check columns
    for i in range(3):
        if board[i] == board[i+3] == board[i+6] != ' ':
            return board[i]
    # Check diagonals
    if board[0] == board[4] == board[8] != ' ':
        return board[0]
    if board[2] == board[4] == board[6] != ' ':
        return board[2]
    # Check draw
    if ' ' not in board:
        return 'Draw'
    return None

def get_state(board):
    return ''.join(board)

def epsilon_greedy_action(board, epsilon=0.1):
    state = get_state(board)
    possible_actions = [i for i, cell in enumerate(board) if cell == ' ']
    
    if state not in st.session_state.q_table:
        st.session_state.q_table[state] = {a: 0 for a in possible_actions}
    
    if random.random() < epsilon:
        return random.choice(possible_actions)
    else:
        q_values = st.session_state.q_table[state]
        max_q = max(q_values.values())
        best_actions = [a for a in q_values if q_values[a] == max_q]
        return random.choice(best_actions)

def update_q_table(reward):
    alpha = 0.1
    gamma = 0.9
    history = st.session_state.history
    
    for idx, (state, action) in enumerate(history):
        discount = gamma ** (len(history) - idx - 1)
        future_reward = discount * reward
        
        if state not in st.session_state.q_table:
            st.session_state.q_table[state] = {}
        
        old_q = st.session_state.q_table[state].get(action, 0)
        st.session_state.q_table[state][action] = old_q + alpha * (future_reward - old_q)

def ai_move():
    if st.session_state.game_over:
        return
    
    board = st.session_state.board.copy()
    state_before = get_state(board)
    action = epsilon_greedy_action(board)
    
    st.session_state.history.append((state_before, action))
    board[action] = 'X'
    st.session_state.board = board
    
    winner = check_winner(board)
    if winner == 'X':
        st.session_state.game_over = True
        st.session_state.stats['wins'] += 1
        update_q_table(1)
    elif winner == 'Draw':
        st.session_state.game_over = True
        st.session_state.stats['draws'] += 1
        update_q_table(0)
    else:
        st.session_state.current_player = 'O'

def handle_click(index):
    if st.session_state.current_player == 'O' and not st.session_state.game_over:
        if st.session_state.board[index] == ' ':
            st.session_state.board[index] = 'O'
            st.session_state.current_player = 'X'
            
            winner = check_winner(st.session_state.board)
            if winner:
                st.session_state.game_over = True                                                                                                                                                                                                              
                if winner == 'O':
                    st.session_state.stats['losses'] += 1
                    update_q_table(-1)
                elif winner == 'Draw':
                    st.session_state.stats['draws'] += 1
                    update_q_table(0)
            else:
                ai_move()

# Streamlit UI
st.title("Tic-Tac-Toe RL Agent")

if 'board' not in st.session_state:
    initialize_game()

# Game board
st.write("### Game Board")
cols = st.columns(3)
for i in range(3):
    for j in range(3):  
        index = i * 3 + j

        with cols[j]:
            if st.button(
                st.session_state.board[index] if st.session_state.board[index] != ' ' else ' ',
                key=f"cell{index}",
                on_click=lambda idx=index: handle_click(idx)
            ):
                pass

# Controls
if st.button("New Game"):
    initialize_game()

# Statistics
st.write("### Statistics")
st.write(f"Wins: {st.session_state.stats['wins']}")
st.write(f"Losses: {st.session_state.stats['losses']}")
st.write(f"Draws: {st.session_state.stats['draws']}")

# Q-table display
st.write("### Q-Table Snapshot")
current_state = get_state(st.session_state.board)
if current_state in st.session_state.q_table:
    st.write(st.session_state.q_table[current_state])
else:
    st.write("No Q-values available for current state")