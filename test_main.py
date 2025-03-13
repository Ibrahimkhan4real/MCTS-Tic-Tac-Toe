"""
Unit tests for the Tic-Tac-Toe MCTS implementation.
"""
import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import random

from Main import TicTacToe, Node, MCTS, EXPLORATION_CONSTANT

class TestTicTacToe(unittest.TestCase):
    """Test cases for the TicTacToe class."""
    
    def setUp(self):
        """Set up a fresh game instance before each test."""
        self.game = TicTacToe()
    
    def test_init(self):
        """Test that the game initializes with an empty board and player 1."""
        self.assertTrue(np.array_equal(self.game.board, np.zeros((3, 3))))
        self.assertEqual(self.game.current_player, 1)
    
    def test_reset(self):
        """Test that reset returns the game to initial state."""
        # Make some moves
        self.game.make_move(0, 0)
        self.game.make_move(1, 1)
        
        # Reset the game
        self.game.reset()
        
        # Check if board is reset and player is 1
        self.assertTrue(np.array_equal(self.game.board, np.zeros((3, 3))))
        self.assertEqual(self.game.current_player, 1)
    
    def test_is_valid_move(self):
        """Test valid and invalid moves."""
        # Initially all moves are valid
        self.assertTrue(self.game.is_valid_move(0, 0))
        self.assertTrue(self.game.is_valid_move(2, 2))
        
        # Make a move and check
        self.game.make_move(1, 1)
        self.assertFalse(self.game.is_valid_move(1, 1))  # Occupied
        self.assertTrue(self.game.is_valid_move(0, 0))   # Still free
        
        # Out of bounds moves should be invalid
        self.assertFalse(self.game.is_valid_move(-1, 0))
        self.assertFalse(self.game.is_valid_move(0, 3))
    
    def test_check_winner_rows(self):
        """Test row win conditions."""
        # No winner initially
        self.assertIsNone(self.game.check_winner())
        
        # Create a row win for player 1
        self.game.board[0] = [1, 1, 1]
        self.assertEqual(self.game.check_winner(), 1)
        
        # Reset and create a row win for player 2
        self.game.reset()
        self.game.board[1] = [2, 2, 2]
        self.assertEqual(self.game.check_winner(), 2)
    
    def test_check_winner_columns(self):
        """Test column win conditions."""
        # Create a column win for player 1
        self.game.board[:, 0] = [1, 1, 1]
        self.assertEqual(self.game.check_winner(), 1)
        
        # Reset and create a column win for player 2
        self.game.reset()
        self.game.board[:, 2] = [2, 2, 2]
        self.assertEqual(self.game.check_winner(), 2)
    
    def test_check_winner_diagonals(self):
        """Test diagonal win conditions."""
        # Create a diagonal win for player 1
        self.game.board = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(self.game.check_winner(), 1)
        
        # Reset and create another diagonal win for player 2
        self.game.reset()
        self.game.board = np.array([[0, 0, 2], [0, 2, 0], [2, 0, 0]])
        self.assertEqual(self.game.check_winner(), 2)
    
    def test_check_winner_draw(self):
        """Test draw condition."""
        # Create a draw scenario (full board, no winner)
        self.game.board = np.array([[1, 2, 1], [1, 2, 2], [2, 1, 1]])
        self.assertEqual(self.game.check_winner(), 0)
    
    def test_make_move(self):
        """Test making moves and player switching."""
        # Player 1 starts
        self.assertEqual(self.game.current_player, 1)
        
        # Make a move as player 1
        result = self.game.make_move(0, 0)
        self.assertTrue(result)
        self.assertEqual(self.game.board[0, 0], 1)
        self.assertEqual(self.game.current_player, 2)  # Now player 2's turn
        
        # Make a move as player 2
        result = self.game.make_move(1, 1)
        self.assertTrue(result)
        self.assertEqual(self.game.board[1, 1], 2)
        self.assertEqual(self.game.current_player, 1)  # Back to player 1
        
        # Try an invalid move
        result = self.game.make_move(0, 0)  # Already occupied
        self.assertFalse(result)
        self.assertEqual(self.game.current_player, 1)  # Player doesn't change
    
    def test_get_available_moves(self):
        """Test getting available moves."""
        # All positions available initially
        moves = self.game.get_available_moves()
        self.assertEqual(len(moves), 9)
        
        # Make some moves
        self.game.make_move(0, 0)
        self.game.make_move(1, 1)
        
        # Check remaining moves
        moves = self.game.get_available_moves()
        self.assertEqual(len(moves), 7)
        self.assertNotIn((0, 0), moves)
        self.assertNotIn((1, 1), moves)
    
    def test_simulate_random_game(self):
        """Test game simulation runs to completion."""
        # Set a seed for reproducibility
        random.seed(42)
        
        result = self.game.simulate_random_game()
        # Result should be 0, 1, or 2
        self.assertIn(result, [0, 1, 2])
        
        # Original board should remain unchanged
        self.assertTrue(np.array_equal(self.game.board, np.zeros((3, 3))))


class TestNode(unittest.TestCase):
    """Test cases for the Node class."""
    
    def setUp(self):
        """Set up a fresh game and node instance before each test."""
        self.game = TicTacToe()
        self.node = Node(self.game)
    
    def test_init(self):
        """Test node initialization."""
        self.assertEqual(self.node.visits, 0)
        self.assertEqual(self.node.wins, 0)
        self.assertEqual(len(self.node.untried_moves), 9)  # All moves untried initially
        self.assertEqual(self.node.player_just_moved, 2)  # Since player 1 is current
    
    def test_is_fully_expanded(self):
        """Test detection of fully expanded nodes."""
        # Initially not fully expanded
        self.assertFalse(self.node.is_fully_expanded())
        
        # Remove all untried moves
        self.node.untried_moves = []
        self.assertTrue(self.node.is_fully_expanded())
    
    def test_is_terminal_node(self):
        """Test detection of terminal nodes."""
        # Initial state is not terminal
        self.assertFalse(self.node.is_terminal_node())
        
        # Create a winning state for player 1
        self.node.state.board[0] = [1, 1, 1]
        self.assertTrue(self.node.is_terminal_node())
    
    def test_expand(self):
        """Test node expansion."""
        # Remember initial number of untried moves
        initial_untried = len(self.node.untried_moves)
        
        # Expand the node
        child = self.node.expand()
        
        # Check that a move was used
        self.assertEqual(len(self.node.untried_moves), initial_untried - 1)
        
        # Check child attributes
        self.assertIsInstance(child, Node)
        self.assertEqual(child.parent, self.node)
        self.assertIn(child, self.node.children)
        self.assertEqual(child.visits, 0)
    
    def test_update(self):
        """Test node statistics update."""
        # Update for a win for the player who just moved
        self.node.update(2)  # Player 2 win
        self.assertEqual(self.node.visits, 1)
        self.assertEqual(self.node.wins, 1)  # Full win
        
        # Update for a draw
        self.node.update(0)  # Draw
        self.assertEqual(self.node.visits, 2)
        self.assertEqual(self.node.wins, 1.5)  # Win + half win
        
        # Update for a loss
        self.node.update(1)  # Player 1 win (loss for player 2)
        self.assertEqual(self.node.visits, 3)
        self.assertEqual(self.node.wins, 1.5)  # No change in wins
    
    def test_select_child(self):
        """Test child selection using UCB1."""
        # Create three children with different statistics
        game = TicTacToe()
        
        # Create child 1 (high win rate but few visits)
        child1 = Node(game, self.node, (0, 0))
        child1.visits = 5
        child1.wins = 4
        
        # Create child 2 (low win rate but many visits)
        child2 = Node(game, self.node, (0, 1))
        child2.visits = 20
        child2.wins = 8
        
        # Create child 3 (medium stats)
        child3 = Node(game, self.node, (0, 2))
        child3.visits = 10
        child3.wins = 6
        
        # Add children to parent
        self.node.children = [child1, child2, child3]
        self.node.visits = 35  # Total parent visits
        
        # With default exploration constant
        selected = self.node.select_child()
        self.assertIn(selected, [child1, child2, child3])
        
        # With custom exploration constant
        # With low exploration constant, should prefer child1 (highest win rate)
        selected = self.node.select_child(exploration_constant=0.1)
        self.assertEqual(selected, child1)


class TestMCTS(unittest.TestCase):
    """Test cases for the MCTS class."""
    
    def setUp(self):
        """Set up MCTS and game instances before each test."""
        self.game = TicTacToe()
        self.mcts = MCTS(iterations=100)  # Fewer iterations for testing
    
    def test_init(self):
        """Test MCTS initialization."""
        self.assertEqual(self.mcts.iterations, 100)
        self.assertEqual(self.mcts.exploration_constant, EXPLORATION_CONSTANT)
        
        # Test with custom parameters
        custom_mcts = MCTS(iterations=200, exploration_constant=1.0)
        self.assertEqual(custom_mcts.iterations, 200)
        self.assertEqual(custom_mcts.exploration_constant, 1.0)
    
    @patch('random.choice')
    def test_search_returns_valid_move(self, mock_random):
        """Test that search returns a valid move."""
        # Mock random.choice to return predictable moves
        mock_random.side_effect = lambda moves: moves[0]
        
        # Run search
        move = self.mcts.search(self.game)
        
        # Verify the move is valid
        self.assertIsInstance(move, tuple)
        self.assertEqual(len(move), 2)
        row, col = move
        self.assertTrue(0 <= row < 3)
        self.assertTrue(0 <= col < 3)
        self.assertTrue(self.game.is_valid_move(*move))
    
    def test_search_winning_move(self):
        """Test that MCTS finds an obvious winning move."""
        # Set up a board where player 1 can win in one move
        self.game.board = np.array([
            [1, 1, 0],
            [2, 2, 0],
            [0, 0, 0]
        ])
        # Set player 1's turn
        self.game.current_player = 1
        
        # MCTS should find the winning move (0,2)
        move = self.mcts.search(self.game)
        self.assertEqual(move, (0, 2))
    
    def test_search_blocking_move(self):
        """Test that MCTS blocks an opponent's winning move."""
        # Set up a board where player 2 could win in one move
        self.game.board = np.array([
            [1, 0, 0],
            [2, 2, 0],
            [1, 0, 0]
        ])
        # Set player 1's turn
        self.game.current_player = 1
        
        # MCTS should block at (1,2)
        move = self.mcts.search(self.game)
        self.assertEqual(move, (1, 2))


if __name__ == '__main__':
    unittest.main()
