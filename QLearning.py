import random 
import numpy as np
directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]
NUM_EPISODES = 5000
EPSILON = 3

class QLearning(object):
    qTable = {}
    discount = 0.9 
    alpha = 0.9
    CurrentState = (0,0)

    def __init__(self,b) -> None:
        self.board = b
        self.TargetState = (len(self.board)-1,len(self.board[0])-1)
        self.initializeQTable()
    
    def initializeQTable(self):
        for cellx in range(len(self.board)):
            for celly in range(len(self.board[0])):
                self.qTable[(cellx,celly)] = {}
                for action in directions:
                    self.qTable[(cellx,celly)][action] = 0
    
    def learn(self):
        count = 0
        for episode in range(NUM_EPISODES):
            print(episode)
            self.currState = (0, 0)
            count+=1
            while self.currState != self.TargetState:
                action = self.epsilonGreedy(self.currState)
                if(self.isValidCell((self.currState[0] + action[0],self.currState[1] + action[1]))):
                    self.evalQFunction(self.currState, action)
                    self.currState = (self.currState[0] + action[0],self.currState[1] + action[1])
                else:
                    continue
        

    
    def isValidCell(self,state):
        return (state[0]>=0 and state[1]>=0 and state[1]<len(self.board[0]) and state[0]<len(self.board))
    
    def epsilonGreedy(self, state):
        randInt = random.randint(1,11)
        if randInt <= EPSILON:
            validActions = list(filter(lambda action: self.isValidCell((state[0]+action[0],state[1]+action[1])), directions))
            return random.choice(validActions)
        else:
            # Gets all qValues for specified state for all q values
            arr = {key: val for key, val in self.qTable.items() if key == state}
            # returns action that yields highest q value
            return max(arr[state], key=arr[state].get)
        
    def evalQFunction(self,coord, action):
        nextCell = (coord[0] + action[0], coord[1]+action[1])
        # reward = -1 for lowest number of steps
        reward = -1 + np.sqrt(action[0]**2 + action[1]**2)
        if(self.isValidCell(nextCell)):
            maxQSPrime = max([self.qTable[nextCell][action2] for action2 in directions if self.isValidCell((nextCell[0]+action2[0],nextCell[1]+action2[1]))])
            self.qTable[coord][action] += (self.alpha * (reward + self.discount * maxQSPrime - self.qTable[coord][action]))
        else:
            return


