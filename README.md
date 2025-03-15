# Tic-Tac-Toe with Vanilla Monte Carlo Tree Search (MCTS)

This repository contains an implementation of the classic Tic-Tac-Toe game that lets you play against an AI powered by a Vanilla Monte Carlo Tree Search (MCTS) algorithm. The AI uses the MCTS approach to simulate and evaluate moves, ultimately selecting the best move based on statistical outcomes from many random game simulations.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
  - [Game Environment](#game-environment)
  - [Monte Carlo Tree Search (MCTS)](#monte-carlo-tree-search-mcts)
- [Code Structure](#code-structure)
  - [TicTacToe Class](#tictactoe-class)
  - [Node Class](#node-class)
  - [MCTS Class](#mcts-class)
  - [play_game Function](#play_game-function)
- [Usage](#usage)
  - [Main Code](#main-code)
  - [Unit Test](#unit-test)

---

## Overview

This project implements a playable Tic-Tac-Toe game where one player is controlled by a human and the other by an AI that uses the Monte Carlo Tree Search algorithm. Unlike many AI techniques that rely on predefined heuristics or exhaustive search, MCTS uses randomized simulations to evaluate moves and build a decision tree over many iterations.

---

## How It Works

### Game Environment

- **TicTacToe Class:**  
  The game environment is encapsulated in the `TicTacToe` class. It maintains the board as a 3x3 NumPy array and tracks the current player (Player 1 represented by `1` and Player 2 represented by `2`). Key functionalities include:
  - **Move Validation:** Checks whether a selected move is valid.
  - **Winner Detection:** Examines rows, columns, and diagonals to determine if a player has won or if the game ends in a draw.
  - **Random Simulation:** Implements a method to simulate a random game (useful during MCTS simulations).

### Monte Carlo Tree Search (MCTS)

MCTS is an algorithm that balances exploration (trying out less-visited moves) and exploitation (favoring moves that have historically led to wins). The algorithm consists of four main phases:

1. **Selection:**  
   Starting from the root node (current game state), the algorithm traverses the tree by selecting the most promising child node using the Upper Confidence Bound (UCB1) formula.

2. **Expansion:**  
   When a node with untried moves is found, one of these moves is expanded, adding a new child node representing the new state after the move.

3. **Simulation:**  
   From the new node, the algorithm simulates a random game until a terminal state (win, loss, or draw) is reached. This is achieved using the `simulate_random_game` method from the `TicTacToe` class.

4. **Backpropagation:**  
   The result of the simulation is then propagated back up the tree, updating win counts and visit counts for each node along the path.

The MCTS implementation in this project is parameterized by:
- **Iterations:** The number of times the MCTS algorithm simulates games (default is 1000 iterations).
- **Exploration Constant:** A parameter (default 1.41) that controls the trade-off between exploration and exploitation when selecting nodes.

---

## Code Structure

### TicTacToe Class

- **Attributes:**
  - `board`: A 3x3 NumPy array representing the game board.
  - `current_player`: Tracks whose turn it is (1 or 2).

- **Key Methods:**
  - `reset()`: Resets the game to its initial state.
  - `is_valid_move(row, col)`: Checks if a move at the specified row and column is allowed.
  - `check_winner()`: Determines if there is a winner, a draw, or if the game should continue.
  - `make_move(row, col)`: Places a player's mark on the board and switches the turn.
  - `get_available_moves()`: Returns a list of empty positions on the board.
  - `simulate_random_game()`: Plays out a random game from the current state until completion, returning the result.

### Node Class

- **Purpose:**  
  Represents a node in the MCTS tree. Each node stores a game state along with statistical information for MCTS.

- **Attributes:**
  - `state`: The TicTacToe game state at this node.
  - `parent`: The parent node in the tree.
  - `move`: The move that led to this state.
  - `children`: A list of child nodes.
  - `wins`: The number of wins resulting from simulations that passed through this node.
  - `visits`: The number of times this node has been visited.
  - `untried_moves`: A list of possible moves that have not yet been explored.
  - `player_just_moved`: The player who made the move leading to this node.

- **Key Methods:**
  - `select_child()`: Uses the UCB1 formula to choose the most promising child node.
  - `expand()`: Expands the current node by taking an untried move and creating a corresponding child node.
  - `is_fully_expanded()`: Checks if all possible moves have been explored.
  - `is_terminal_node()`: Determines if the node represents a terminal game state.
  - `update(result)`: Updates the node’s statistics based on the outcome of a simulation.

### MCTS Class

- **Purpose:**  
  Implements the Monte Carlo Tree Search algorithm.

- **Parameters:**
  - `iterations`: Number of iterations to run the search (affects decision quality and speed).
  - `exploration_constant`: Balances exploration versus exploitation in the selection phase.

- **Key Method:**
  - `search(root_state)`: Executes the MCTS algorithm starting from the given game state and returns the best move as determined by the highest number of visits among the child nodes.

### play_game Function

- **Purpose:**  
  Provides an interactive interface to play Tic-Tac-Toe.
  - **Player Roles:**  
    - AI (MCTS) plays as Player 1 (X).
    - Human plays as Player 2 (O).
  - The function continuously alternates turns between the AI and the human player, displays the board after each move, and checks for game termination (win or draw).

---

## Usage
  ### Main Code

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Ibrahimkhan4real/MCTS-Tic-Tac-Toe.git
   cd tic-tac-toe-mcts

2. **Install Requirements:**
   ```bash
   pip install numpy
3. **Run the Game:**
   ```bash
   python main.py

  ### Unit Test

1. **Navigate to the Repository**
   ```bash
   cd tic-tac-toe-mcts
2. **Run the Unit Test**
   ```bash
       python -m unittest -v test_main
  Or
   ```bash
    python test_main.py
