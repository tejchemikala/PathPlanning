import unittest
import random
from QLearning import QLearning  # Assuming your class is in a file named q_learning.py

class TestQLearning(unittest.TestCase):
    def setUp(self):
        # Create a simple 3x3 board for testing
        self.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    
    def test_qtable_initialization(self):
        qlearning = QLearning(self.board)
        
        # Check if Q-table is initialized properly
        for cellx in range(len(self.board)):
            for celly in range(len(self.board[0])):
                for action in qlearning.qTable[(cellx, celly)]:
                    self.assertEqual(qlearning.qTable[(cellx, celly)][action], 0)
    
    
    def test_qfunction_evaluation(self):
        qlearning = QLearning(self.board)
        coord = (0, 0)
        action = (1, 0)
        qlearning.qTable[coord][action] = 5
        
        qlearning.evalQFunction(coord, action)
        
        expected_qvalue = 5 + qlearning.alpha * (-1 + qlearning.discount * 0 - 5)
        self.assertAlmostEqual(qlearning.qTable[coord][action], expected_qvalue)
    
    def test_learn(self):
        qlearning = QLearning(self.board)
        qlearning.learn()
        
        # You can add more assertions here based on your expectations
        
if __name__ == '__main__':
    unittest.main()
