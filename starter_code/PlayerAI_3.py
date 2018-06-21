from random import randint
from BaseAI_3 import BaseAI
import time
import math
import numpy

class node(object):
    def __init__(self, value, parent=None, action = None, depth = 0):
        self.parent = parent
        self.action = action
        self.value = value
        self.children = []
        self.depth = depth
      
class PlayerAI(BaseAI):
    def getMove(self, grid):
        self.start = time.clock()
        self.max = float("-inf")
        initial_state = node(float("-inf"), parent = None, action = None, depth = 1)
        alpha = float("-inf")
        beta = float("inf")
        grid_test = grid.clone()
        self.depth = 4
        max_value = float("-inf")
        action = None
        while (time.clock() - self.start) < 0.19:
            self.cap = []
            state = self.maximize(initial_state, alpha, beta, grid_test)
            while state.depth != 2:
                state = state.parent
            if max_value < state.value:
                max_value = state.value
                action = state.action 
            self.depth = self.depth + 2
        return action
    
    def maximize(self, state, alpha, beta, grid):
        moves = grid.getAvailableMoves()
        maxchild = node(float("-inf"), parent = state, action = None, depth = state.depth + 1)

        if len(moves) == 0:
            grid_test = grid.clone()
            value = self.heuristic(grid_test)    
            maxchild.value = value
        
        if len(state.children) == 0:
            for i in moves:
                state.children.append(node(0, parent = state, action = i, depth = state.depth + 1))
        
        else:
            state.children = []
            for i in moves:
                state.children.append(node(0, parent = state, action = i, depth = state.depth + 1))
        for child in state.children:
            grid_test = grid.clone()
            grid_test.move(child.action)

            if child.depth == self.depth:
                #heuristic
                self.cap.append(1)
                child.value = self.heuristic(grid_test)
                minchild = child
            else:
                minchild = self.minimize(child, alpha, beta, grid_test)
            end = time.clock()
            if minchild.value > maxchild.value:
                maxchild = minchild
            if maxchild.value >= beta:
                break
            if maxchild.value > alpha:
                alpha = maxchild.value
            if end - self.start > 0.19:
                break
        return maxchild

    def minimize(self, state, alpha, beta, grid):
        minchild = node(float("inf"), parent = None, action = None, depth = 0)
        cells = grid.getAvailableCells()

        for cell in cells:
            child = node(0, parent = state, action = state.action, depth = state.depth+1)
            grid_test = grid.clone()
            grid_test.insertTile(cell, 2)
            maxchild_1 = self.maximize(child, alpha, beta, grid_test)
            grid_test = grid.clone()
            grid_test.insertTile(cell, 4)
            maxchild_2 = self.maximize(child, alpha, beta, grid_test)
            value = maxchild_1.value*0.9 + maxchild_2.value*0.1
            maxchild = node(value, parent = state, action = state.action, depth = state.depth + 2)   
        
            if minchild.value > maxchild.value:
                minchild = maxchild
            if minchild.value <= alpha:
                break
            if minchild.value < beta:
                beta = minchild.value
            end = time.clock()
            if end - self.start > 0.19:
                break
        return minchild

    def heuristic(self, grid):
        m1 = numpy.array(grid.map)
        h1 = len(grid.getAvailableCells())        
        weight = numpy.array([[6,5,4,3],[5,4,3,2],[4,3,2,1],[3,2,1,0]])
        h2 = numpy.sum(numpy.exp(numpy.log2(m1 * weight)))
        h3 = grid.getMaxTile()
        h4 = numpy.sum(abs(numpy.diff(m1))) + numpy.sum(abs(numpy.diff(m1, axis = 0)))
        h = h1*150 + h2 + 4 * h3 + 0*h4
        return h