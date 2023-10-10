# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
       
        allFood = newFood.asList()
        # calculate distance from ghost
        minGhostDistance = float('inf')
        listOfDistanceToValidFood = []

        for ghost in successorGameState.getGhostPositions():
            if (manhattanDistance(newPos, ghost) < 2):
                return -float('inf')
      
        minDistanceToFood = float('inf')
        for foodPos in allFood:
            for ghost in newGhostStates:
                distanceToGhost = manhattanDistance(foodPos, ghost.getPosition())
                if distanceToGhost < minGhostDistance:
                    minGhostDistance = distanceToGhost
            distanceToPacman = manhattanDistance(foodPos, newPos)
            if distanceToPacman < minDistanceToFood:
                minDistanceToFood = distanceToPacman
            if (distanceToGhost - distanceToPacman) > 0:
                listOfDistanceToValidFood.append(distanceToPacman)

        minDistanceGhostToPacman = float('inf')      
        for ghost in newGhostStates:
            ghostToPacman = manhattanDistance(newPos, ghost.getPosition())
            if ghostToPacman < minDistanceGhostToPacman:
                minDistanceGhostToPacman = ghostToPacman
                
        if len(listOfDistanceToValidFood)==0:
            return successorGameState.getScore() +  1/minDistanceToFood 
        
        minimumDistance = min(listOfDistanceToValidFood)
        return successorGameState.getScore() + 1/minimumDistance


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        numOfAgents = gameState.getNumAgents()        
        output, action = self.max(gameState, 0, -float('inf'), numOfAgents)
        return action
    
    def max(self, gamestate, depth, currentMax, numOfAgents):
        if gamestate.isWin() or gamestate.isLose() or depth == self.depth:
            # print("in max", self.evaluationFunction(gamestate), depth)
            return self.evaluationFunction(gamestate), ""
        
        actions = gamestate.getLegalActions(0)
        currentAction = ''
        for action in actions:
            successorGameState = gamestate.generateSuccessor(0, action)
            output = self.min(successorGameState, depth, float('inf'), numOfAgents, 1)
            # print("output is", output,"curmax is", currentMax)
            if output > currentMax:
                currentMax = output
                currentAction = action
        # print("max", currentMax, depth)

        return currentMax, currentAction
    
    def min(self, gamestate, depth, currentMin, numOfAgents, agentIndex):
        if gamestate.isWin() or gamestate.isLose() or depth == self.depth:
            # print("in min", self.evaluationFunction(gamestate))
            return self.evaluationFunction(gamestate)
        

        if agentIndex == numOfAgents-1:            
            ghostAction = gamestate.getLegalActions(agentIndex)

            for action in ghostAction:
                successorGameState = gamestate.generateSuccessor(agentIndex, action)
                output, action = self.max(successorGameState, depth+1, -float('inf'), numOfAgents)                
                if output < currentMin:
                    currentMin = output
            # print("min", currentMin, depth)
            return currentMin

        ghostAction = gamestate.getLegalActions(agentIndex)

        for action in ghostAction:
            successorGameState = gamestate.generateSuccessor(agentIndex, action)
            output = self.min(successorGameState, depth, float('inf'), numOfAgents, agentIndex+1)
            if output < currentMin:
                currentMin = output
        # print("min", currentMin, depth)
        return currentMin
            
            


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numOfAgents = gameState.getNumAgents()        
        output, action = self.max(gameState, 0, -float('inf'), numOfAgents, -float('inf'), float('inf'))
        return action
    
    def max(self, gamestate, depth, currentMax, numOfAgents,alpha, beta):
        if gamestate.isWin() or gamestate.isLose() or depth == self.depth:
            # print("in max", self.evaluationFunction(gamestate), depth)
            return self.evaluationFunction(gamestate), ""
        
        actions = gamestate.getLegalActions(0)
        currentAction = ''
        for action in actions:
            successorGameState = gamestate.generateSuccessor(0, action)
            output = self.min(successorGameState, depth, float('inf'), numOfAgents, 1, alpha, beta)
            # print("output is", output,"curmax is", currentMax)
            if output > currentMax:
                currentMax = output
                currentAction = action
            if currentMax > beta:
                return currentMax, currentAction
            alpha = max(alpha, currentMax)
        # print("max", currentMax, depth)

        return currentMax, currentAction
    
    def min(self, gamestate, depth, currentMin, numOfAgents, agentIndex, alpha, beta):
        if gamestate.isWin() or gamestate.isLose() or depth == self.depth:
            # print("in min", self.evaluationFunction(gamestate))
            return self.evaluationFunction(gamestate)
        

        if agentIndex == numOfAgents-1:            
            ghostAction = gamestate.getLegalActions(agentIndex)

            for action in ghostAction:
                successorGameState = gamestate.generateSuccessor(agentIndex, action)
                output, action = self.max(successorGameState, depth+1, -float('inf'), numOfAgents, alpha, beta)                
                if output < currentMin:
                    currentMin = output
                if currentMin < alpha:
                    return currentMin
                beta = min(beta, currentMin)
            # print("min", currentMin, depth)
            return currentMin

        ghostAction = gamestate.getLegalActions(agentIndex)

        for action in ghostAction:
            successorGameState = gamestate.generateSuccessor(agentIndex, action)
            output = self.min(successorGameState, depth, float('inf'), numOfAgents, agentIndex+1, alpha, beta)
            if output < currentMin:
                currentMin = output
            if currentMin < alpha:
                return currentMin
            beta = min(beta, currentMin)
        # print("min", currentMin, depth)
        return currentMin

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        numOfAgents = gameState.getNumAgents()        
        output, action = self.max(gameState, 0, -float('inf'), numOfAgents)
        return action
    
    def max(self, gamestate, depth, currentMax, numOfAgents):
        if gamestate.isWin() or gamestate.isLose() or depth == self.depth:
            # print("in max", self.evaluationFunction(gamestate), depth)
            return self.evaluationFunction(gamestate), ""
        
        actions = gamestate.getLegalActions(0)
        currentAction = ''
        for action in actions:
            successorGameState = gamestate.generateSuccessor(0, action)
            output = self.exp(successorGameState, depth, float('inf'), numOfAgents, 1)
            # print("output is", output,"curmax is", currentMax)
            if output > currentMax:
                currentMax = output
                currentAction = action
        return currentMax, currentAction

    def exp(self, gamestate, depth, currentMin, numOfAgents, agentIndex):
        v = 0
        if gamestate.isWin() or gamestate.isLose() or depth == self.depth:
            # print("in min", self.evaluationFunction(gamestate))
            return self.evaluationFunction(gamestate)
        
        if agentIndex == numOfAgents-1:            
            ghostAction = gamestate.getLegalActions(agentIndex)
            for action in ghostAction:
                successorGameState = gamestate.generateSuccessor(agentIndex, action)
                output, action = self.max(successorGameState, depth+1, -float('inf'), numOfAgents)                
                v += output

            # print("min", currentMin, depth)
            return v / len(ghostAction)

        ghostAction = gamestate.getLegalActions(agentIndex)

        for action in ghostAction:
            successorGameState = gamestate.generateSuccessor(agentIndex, action)
            output = self.exp(successorGameState, depth, float('inf'), numOfAgents, agentIndex+1)
            v += output
            
        # print("min", currentMin, depth)
        return v / len(ghostAction)
    
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    
    Pacman prefers going to the closest valid food where valid foods 
    are foods that are closer to the pacman than the ghost. In other words, pacman 
    wants to avoid the ghost.

    If there are no valid foods, pacman goes to the closest food.

    Pacman will not go to a state where the manhattan distance between pacman and 
    the ghost is less than 2.
    
    Pacman also picks the nearest biggest food in order to 
    make the ghost enters its scared state.

    we evaluate our pacman based on the sum of the current score, reciprocal of distance to closest 
    valid food or closest any food if there is no valid food, and the total ghost scared time.

    we take the distance to closest valid food or closest any food 
    if there is no valid food in reciprocal form since it results in higher evaluation function the shorter the distance

    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    allFood = newFood.asList()
    # calculate distance from ghost
    minGhostDistance = float('inf')
    listOfDistanceToValidFood = []

    for ghost in currentGameState.getGhostPositions():
        if (manhattanDistance(newPos, ghost) < 2):
            return -float('inf')
    
    minDistanceToFood = float('inf')
    for foodPos in allFood:
        for ghost in newGhostStates:
            distanceToGhost = manhattanDistance(foodPos, ghost.getPosition())
            if distanceToGhost < minGhostDistance:
                minGhostDistance = distanceToGhost
        distanceToPacman = manhattanDistance(foodPos, newPos)
        if distanceToPacman < minDistanceToFood:
            minDistanceToFood = distanceToPacman
        if (distanceToGhost - distanceToPacman) > 0:
            listOfDistanceToValidFood.append(distanceToPacman)

    minDistanceGhostToPacman = float('inf')      
    for ghost in newGhostStates:
        ghostToPacman = manhattanDistance(newPos, ghost.getPosition())
        if ghostToPacman < minDistanceGhostToPacman:
            minDistanceGhostToPacman = ghostToPacman
            
    if len(listOfDistanceToValidFood)==0:
        return currentGameState.getScore() +  1/minDistanceToFood + sum(newScaredTimes)
    
    minimumDistance = min(listOfDistanceToValidFood)
    return currentGameState.getScore() + 1/minimumDistance + sum(newScaredTimes)


# Abbreviation
better = betterEvaluationFunction
