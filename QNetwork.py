import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]

# Define the Q-network using PyTorch
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class QLearning_NET(object):
    discount = 0.9
    alpha = 0.9
    epsilon = 3

    def __init__(self, b) -> None:
        self.board = b
        self.TargetState = (len(self.board) - 1, len(self.board[0]) - 1)
        self.q_network = QNetwork(input_size=2, output_size=len(directions))
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)

    def learn(self, num_episodes):
        for episode in range(num_episodes):
            current_state = (0, 0)
            done = False

            while not done:
                action = self.epsilon_greedy(current_state)
                next_state, reward, done = self.take_action(current_state, action)
                self.update_q_function(current_state, action, reward, next_state)
                current_state = next_state

    def epsilon_greedy(self, state):
        if random.randint(1,11) <= self.epsilon:
            return random.choice(directions)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values).item()
            return directions[action]

    def take_action(self, current_state, action):
        next_state = (current_state[0] + action[0], current_state[1] + action[1])
        if self.is_valid_cell(next_state):
            reward = -1 + 2*np.sqrt(action[0] ** 2 + action[1] ** 2)
        else:
            reward = -1  # Invalid move
            next_state = current_state
        done = (next_state == self.TargetState)
        return next_state, reward, done

    def update_q_function(self, current_state, action, reward, next_state):
        state_tensor = torch.tensor(current_state, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        action_index = directions.index(action)

        q_values = self.q_network(state_tensor)
        next_q_values = self.q_network(next_state_tensor)

        # Calculate the target Q-value
        target_q_value = reward + self.discount * torch.max(next_q_values) 

        # Calculate the loss
        loss = nn.MSELoss()(q_values[action_index], target_q_value)

        # Perform a gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def is_valid_cell(self, state):
        return (state[0] >= 0 and state[1] >= 0 and state[1] < len(self.board[0]) and state[0] < len(self.board))
    
    def get_q_value(self, state, action):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        q_values = self.q_network(state_tensor)
        action_index = directions.index(action)
        return q_values[action_index].item()

def hueristic_distance_Qnetwork(start_node,direction,qTable):
    return  qTable.get_q_value(start_node, direction)
# Sample usage:
if __name__ == "__main__":
    board = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    q_learning = QLearning_NET(board)
    q_learning.learn(num_episodes=300)
    state = (0, 0)
    action = (0, 1)  # Example action
    q_value = q_learning.get_q_value(state, action)
    print(f"Q-value for state {state} and action {action}: {q_value}")
