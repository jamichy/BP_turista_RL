import numpy as np
import pandas as pd
from math import sqrt, ceil
from PIL import Image, ImageDraw 


class GridWorld(object):
    def __init__(self, m, n, filepathTerrain, altitudeFactor):
        self.m = m
        self.n = n
        self.terrain = pd.read_csv(filepathTerrain)
        self.altitudeFactor = altitudeFactor
        #print(self.terrain.values)
        self.coloursRGBB = [(173,255,47), (144,238,144), (240,230,140), (255,127,80), (184,134,11), (128,128,128), (105,105,105), (248,248,255)]
        self.sizeOfStep = 600/max(m, n)
        self.stateSpacePlus = []
        for i in range(self.m):
            for j in range(self.n):
                self.stateSpacePlus.append((i, j))
        self.stateSpace = self.stateSpacePlus.copy()
        self.stateSpace.remove((self.m-1, self.n-1))
        self.actionSpace = {0: (1, 0), 1: (1, 1),
                            2: (0, 1), 3: (-1, 1),
                            4: (-1, 0), 5: (-1, -1), 
                            6:(0, -1), 7: (1, -1)}
        self.possibleActions = [0, 1, 2, 3, 4, 5, 6, 7]
        #self.agentPosition = (0, 0)
        self.playgroundImage = Image.new("RGB", (ceil(self.sizeOfStep*m), ceil(self.sizeOfStep*n)), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.playgroundImage)
        self.drawPlaygroundImg()
        self.playgroundImage.save('terrain.png')
        self.workingImage = Image.open('terrain.png')

    def isTerminalState(self, state):
        return state in self.stateSpacePlus and state not in self.stateSpace
    
    def getAgentRowAndColumn(self):
        x, y = self.agentPosition
        return x, y

    def setState(self, state):
        self.agentPosition = state
        
    def offGridMove(self, newState):
        # pokud se nový stav nenachází v naší mapě
        return newState not in self.stateSpacePlus
        """
        if newState not in self.stateSpacePlus:
            return True
        else:
            return False
        """

    def step(self, action):
        agentX, agentY = self.agentPosition
        actionX, actionY = self.actionSpace[action]
        
        
        step_size_2D = sqrt(2) if (action % 2 == 1) else 1
        
        resultingState = (agentX + actionX,agentY + actionY)
        
        if not self.offGridMove(resultingState):
            rew = -sqrt(((self.terrain.iloc[self.n-agentY-1,agentX] -\
            self.terrain.iloc[self.n-1-agentY-actionY,agentX + actionX])**2)\
            *self.altitudeFactor**2 + step_size_2D**2) 
            
            reward = rew if not self.isTerminalState(resultingState) else 5
            (X, Y) = resultingState
            
            self.draw.line((self.sizeOfStep*(agentX+0.5),self.sizeOfStep*(self.n-agentY-0.5),
                            self.sizeOfStep*(X+0.5),self.sizeOfStep*(self.n-Y-0.5)),
                           fill = (0,0,0), width=2)
            self.setState(resultingState)
            
            return resultingState, reward, \
                   self.isTerminalState(resultingState)
        else:
            return self.agentPosition, -10, \
                   self.isTerminalState(self.agentPosition)
    def reset(self):
        self.setState((0, 0))
        self.workingImage = Image.open('terrain.png')
        self.draw = ImageDraw.Draw(self.workingImage)
        return self.agentPosition
    
    def drawSquareToImg(self, x, y):
        self.draw.rectangle((x*self.sizeOfStep, y*self.sizeOfStep,\
                        (x+1)*self.sizeOfStep, (y+1)*self.sizeOfStep),
                       fill= self.coloursRGBB[self.terrain.iloc[y,x]],
                       outline=(0,0,0))
    
    def drawPlaygroundImg(self):
        for i in range(self.m):
            for j in range(self.n):
                self.drawSquareToImg(i, j)