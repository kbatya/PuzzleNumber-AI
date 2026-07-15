from Action import Action
from itertools import permutations
import numpy as np
import torch
from State import State
from Puzzle import Puzzle
import pygame

PATH = 'Data/V_4.pth'
gamma = 0.95

class AI_Agent:
    def __init__(self, env: Puzzle) -> None:
        self.V = dict()
        self.env : Puzzle = env

    def get_Action (self, events):
        state =  self.env.state
        best_value = -1
        best_action = None
        for action in self.env.get_actions(state):
            next_state, reward = self.env(state, action)
            next_value = reward + gamma * self.get_V(next_state)
            if next_value > best_value:
                best_value = next_value
                best_action = action
        pygame.time.wait(300)
        return best_action
    
    def Init_Value_Table (self):
        # get all states as tuple
        states = permutations([0,1,2,3,4,5,6,7,8])
        for s in states:
            self.V[s] = 0
        # self.save_v()

    def Value_Iteration(self):
        if len(self.V) == 0:
            self.Init_Value_Table()

        goal_key = tuple(self.env.goal.board.flatten())
        self.V[goal_key] = 0

        accuracy = 1e-4
        while True:
            delta = 0
            for key in list(self.V.keys()):
                if key == goal_key:
                    continue

                state = self.key_to_state(key)
                best_value = float('-inf')
                for action in self.env.get_actions(state):
                    next_state, reward = self.env(state, action)
                    next_key = tuple(next_state.board.flatten())
                    value = reward + gamma * self.V[next_key]
                    if value > best_value:
                        best_value = value

                if best_value == float('-inf'):
                    best_value = 0

                delta = max(delta, abs(self.V[key] - best_value))
                self.V[key] = best_value

            if delta < accuracy:
                break

        return self.V
    
    def save_v (self):
        torch.save(self.V, PATH)

    def load_v (self):
        self.V = torch.load(PATH, weights_only=False)
    
    def get_V(self, state: State):
        key = tuple(state.board.flatten())
        return self.V.get(key, 0)
    
    def set_V(self,state: State, value):
        arr = state.board.flatten()
        key = tuple(arr)
        self.V[key] = value

    def key_to_state(self, key):
        board = np.asarray(key).reshape([3,3])
        return State(board)

