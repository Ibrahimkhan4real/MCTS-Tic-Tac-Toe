"""
Tic-Tac-Toe with Vanilla Monte Carlo Tree Search Algorithm (MCTS).
This module lets you play Tic-Tac-Toe against AI.

Author
------
M.I.Khan

Date
----
13th March 2025
"""
ITERATIONS = 1000
EXPLORATION_CONSTANT = 1.41


import numpy as np
import random
from copy import deepcopy
import math

# Game class responsible for all game actions, acting as an environment
class TicTacToe:
    """
    Tic-Tac-Toe game environment that manages the board state and game rules.
    
    Attributes:
        board (numpy.ndarray): 3x3 grid representing the game board
        current_player (int): Current player (1 or 2)
    """
    
    def __init__(self):
        """Initialize an empty Tic-Tac-Toe board with player 1 to start."""
        self.board = np.zeros((3, 3), dtype=int)  # 3x3 board represented by "X" and "0"
        self.current_player = 1  # player 1 is represented by "1", player 2 by "2" on the board

    def reset(self):
        """Reset the game to initial state."""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
    
    def is_valid_move(self, row, col):
        """
        Check if a move is valid.
        
        Args:
            row (int): Row index (0-2)
            col (int): Column index (0-2)
            
        Returns:
            bool: True if the move is valid, False otherwise
        """
        return 0 <= row < 3 and 0 <= col < 3 and self.board[row][col] == 0
    
    def check_winner(self):
        """
        Check if there's a winner or a draw.
        
        Returns:
            int or None: 
                - Player number (1 or 2) if there's a winner
                - 0 if the game is a draw
                - None if the game is still ongoing
        """
        # Check rows and columns
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != 0:
                return self.board[i][0] # Row win
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != 0:
                return self.board[0][i] # Column win
        
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            return self.board[0][2]
        
        # Check for draw (if no empty cells remain)
        if not any(self.board[i][j] == 0 for i in range(3) for j in range(3)):
            return 0
        
        return None # Game is still going

    def make_move(self, row, col):
        """
        Place the current player's mark at the specified position.
        
        Args:
            row (int): Row index (0-2)
            col (int): Column index (0-2)
            
        Returns:
            bool: True if the move was valid and made, False otherwise
        """
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
            self.current_player = 3 - self.current_player  # Switch player (1->2, 2->1)
            return True
        return False
    
    def get_available_moves(self):
        """
        Get all available (empty) positions on the board.
        
        Returns:
            list: List of (row, col) tuples representing available moves
        """
        return [(row, col) for row in range(3) for col in range(3) if self.board[row][col] == 0]

    def simulate_random_game(self):
        """
        Simulate a random game from the current position until completion.
        
        Returns:
            int: The result of the game (0 for draw, 1 for player 1 win, 2 for player 2 win)
        """
        temp_game = deepcopy(self) # Create a deep copy to avoid modifying the real game
        
        while True:
            winner = temp_game.check_winner()
            if winner is not None:
                return winner # Return winner if game is over
                
            moves = temp_game.get_available_moves()
            if not moves:
                return 0  # Draw if no moves are available
                
            move = random.choice(moves) # Select a random move
            temp_game.make_move(*move) # Apply the move

class Node:
    """
    Node class for the MCTS tree.
    
    Attributes:
        state (TicTacToe): Game state at this node
        parent (Node): Parent node
        move (tuple): Move that led to this state
        children (list): Child nodes
        wins (int): Number of wins from this node
        visits (int): Number of visits to this node
        untried_moves (list): List of moves not yet expanded
        player_just_moved (int): The player who just moved to reach this state
    """
    
    def __init__(self, state, parent=None, move=None):
        """
        Initialize a new node in the MCTS tree.
        
        Args:
            state (TicTacToe): Current game state
            parent (Node, optional): Parent node. Defaults to None.
            move (tuple, optional): Move that led to this state. Defaults to None.
        """
        self.state = deepcopy(state)
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = state.get_available_moves()
        # Track which player made the move to this state
        self.player_just_moved = 3 - state.current_player  # The player who just moved to reach this state
    
    def select_child(self, exploration_constant=EXPLORATION_CONSTANT):
        """
        Select the most promising child node using UCB1 formula.
        
        Args:
            exploration_constant (float, optional): Controls exploration vs exploitation.
                                                   Defaults to 1.41 (sqrt(2)).
        
        Returns:
            Node: The selected child node
        """
        # Upper Confidence Bound (UCB1) formula
        return max(
            self.children,
            key=lambda c: c.wins / c.visits + 
                 exploration_constant * math.sqrt(2 * math.log(self.visits) / c.visits)
        )

    def expand(self):
        """
        Expand the tree by adding a child node with a random untried move.
        
        Returns:
            Node: The newly created child node
        """
        # Select a random untried move
        move = self.untried_moves.pop(random.randrange(len(self.untried_moves)))
        # Create new game state
        new_state = deepcopy(self.state)
        new_state.make_move(*move)
        # Create child node
        child = Node(new_state, self, move)
        self.children.append(child)
        return child
    
    def is_fully_expanded(self):
        """
        Check if all possible moves from this node have been expanded.
        
        Returns:
            bool: True if fully expanded, False otherwise
        """
        return len(self.untried_moves) == 0
    
    def is_terminal_node(self):
        """
        Check if this node represents a terminal game state.
        
        Returns:
            bool: True if this is a terminal state, False otherwise
        """
        return self.state.check_winner() is not None
    
    def update(self, result):
        """
        Update node statistics after a simulation.
        
        Args:
            result (int): The simulation result (0 for draw, 1 or 2 for player win)
        """
        self.visits += 1
        # Update win count based on the player perspective
        if result == self.player_just_moved:  # If this node's player won
            self.wins += 1
        elif result == 0:  # Draw (counting as half a win)
            self.wins += 0.5
        # Loss: no update needed as we only track wins

class MCTS:
    """
    Monte Carlo Tree Search implementation.
    
    Attributes:
        iterations (int): Number of iterations to run
        exploration_constant (float): Parameter controlling exploration vs exploitation
    """
    def __init__(self, iterations=ITERATIONS, exploration_constant=EXPLORATION_CONSTANT):
        """
        Initialize the MCTS algorithm.
        
        Args:
            iterations (int, optional): Number of search iterations. Defaults to 1000.
            exploration_constant (float, optional): UCB1 exploration parameter. Defaults to 1.41.
        """
        self.iterations = iterations
        self.exploration_constant = exploration_constant

    def search(self, root_state):
        """
        Run the MCTS algorithm to find the best move.
        
        The algorithm consists of four phases:
        1. Selection: Traverse the tree to find the most urgent leaf node
        2. Expansion: Expand the tree by adding a new node
        3. Simulation: Simulate a random game from the new node
        4. Backpropagation: Update node statistics based on the simulation result
        
        Args:
            root_state (TicTacToe): The current game state
            
        Returns:
            tuple: The best move as (row, col)
        """
        root = Node(root_state)
        
        for _ in range(self.iterations):
            # 1. Selection - find the most promising leaf node
            node = root
            
            # Select child until reaching a leaf node
            while node.is_fully_expanded() and not node.is_terminal_node():
                node = node.select_child(self.exploration_constant)
            
            # 2. Expansion - unless we've reached a terminal state
            if not node.is_terminal_node():
                if node.untried_moves:  # If there are untried moves
                    node = node.expand()
            
            # 3. Simulation - play a random game from this node
            state_to_simulate = deepcopy(node.state)
            result = state_to_simulate.simulate_random_game()
            
            # 4. Backpropagation - update statistics up the tree
            while node is not None:
                node.update(result)
                node = node.parent
        
        # Return the best move from the root
        return max(root.children, key=lambda c: c.visits).move

def play_game():
    """
    Run an interactive Tic-Tac-Toe game between a human player and the MCTS AI.
    
    Human plays as player 2 (O), and the AI plays as player 1 (X).
    """
    game = TicTacToe()
    mcts = MCTS(iterations=ITERATIONS)
    
    print("Initial board:")
    print(game.board)
    
    while True:
        if game.current_player == 1:  # AI's turn
            print("\nMCTS is thinking...")
            move = mcts.search(game)
            game.make_move(*move)
            print(f"AI played at position: {move}")
        else:  # Human's turn
            while True:
                try:
                    row, col = map(int, input("\nEnter row and column (0-2): ").split())
                    if game.make_move(row, col):
                        break
                    print("Invalid move, try again.")
                except ValueError:
                    print("Invalid input. Please enter two numbers separated by space.")
        
        # Display the current board state
        print("\nCurrent board:")
        for i in range(3):
            row = ['X' if cell == 1 else 'O' if cell == 2 else '.' for cell in game.board[i]]
            print(' '.join(row))
        
        # Check for game end conditions
        winner = game.check_winner()
        if winner is not None:
            if winner == 0:
                print("\nIt's a draw!")
            else:
                print(f"\nPlayer {winner} wins!")
            break

if __name__ == "__main__":
    play_game()
