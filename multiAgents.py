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
from multiprocessing.reduction import steal_handle

from autograder import runTest
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        curPos = currentGameState.getPacmanPosition()
        curFood = currentGameState.getFood()
        curGhostStates = currentGameState.getGhostStates()

        "*** YOUR CODE HERE ***"

        # Get the closest food
        closestFood = float('inf')
        for food in newFood.asList():
            closestFood = min(closestFood, manhattanDistance(newPos, food))

        # Get the closest ghost
        closestGhost = float('inf')
        for ghost in newGhostStates:
            closestGhost = min(closestGhost, manhattanDistance(newPos, ghost.getPosition()))

        # Get the closest food from the current state
        closestFoodCur = float('inf')
        for food in curFood.asList():
            closestFoodCur = min(closestFoodCur, manhattanDistance(curPos, food))

        # Get the closest ghost from the current state
        closestGhostCur = float('inf')
        for ghost in curGhostStates:
            closestGhostCur = min(closestGhostCur, manhattanDistance(curPos, ghost.getPosition()))

        # If the new state is a win, return the maximum value
        # if successorGameState.isWin():
        #     return float('inf')

        # If the new state is a lose, return the minimum value
        # if successorGameState.isLose():
        #     return float('-inf')

        # if closestFood == 1 and closestGhost == 4:
        #     return float('-inf')

        curFoodNr = len(curFood.asList())
        newFoodNr = len(newFood.asList())

        foodDiff = curFoodNr - newFoodNr
        # foodDiff = closestFoodCur - closestFood

        print(foodDiff)

        if foodDiff > 0 and closestGhostCur > 3:
            return float('inf')

        if closestGhost < 4 :
            return float('-inf')

        return (successorGameState.getScore() - closestFood)

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        # util.raiseNotDefined()
        ghostIds = [i for i in range(1, gameState.getNumAgents())]

        def over(state: GameState, depth: int):
            return state.isWin() or state.isLose() or depth == self.depth


        def vMin(state, ghostId, depth):
            if over(state, depth):
                # return self.evaluationFunction(state)
                return scoreEvaluationFunction(state)

            minValue = float('+inf')
            for action in state.getLegalActions(ghostId):
                if ghostId != ghostIds[-1]:
                    minValue = min(
                        minValue,
                        vMin(state.generateSuccessor(ghostId, action), ghostId + 1, depth)
                    )
                else:
                    minValue = min(
                        minValue,
                        vMax(state.generateSuccessor(ghostId, action), depth + 1)
                    )
            return minValue

        def vMax(state, depth):
            if over(state, depth):
                # return self.evaluationFunction(state)
                return scoreEvaluationFunction(state)

            maxValue = float('-inf')
            for action in state.getLegalActions(0):
                maxValue = max(
                    maxValue,
                    vMin(state.generateSuccessor(0, action), ghostIds[0], depth)
                )
            return maxValue

        res = [(action, vMin(gameState.generateSuccessor(0, action), ghostIds[0], 0)) for action in gameState.getLegalActions(0)]
        # res = [item for item in res if item[1] is not None]
        res.sort(key=lambda k: k[1])

        return res[-1][0]


        # moves = list(map(vMin, gameState.get))

        # return

        # def term(state, depth):
        #     return state.isWin() or state.isLose() or depth == self.depth
        #
        # def vMin(state, depth, ghost):
        #     if term(state, depth):
        #         return self.evaluationFunction(state)
        #
        #     value = float('-inf')
        #     for action in state.getLegalActions(ghost):
        #         if ghost == ghostIds[-1]:
        #             value = min(value, vMax(state.generateSuccessor(ghost, action), depth + 1))
        #         else:
        #             value = min(value, vMin(state.generateSuccessor(ghost, action), depth, ghost + 1))
        #
        #     return value
        #
        # def vMax(state, depth):
        #     if term(state, depth):
        #         return self.evaluationFunction(state)
        #
        #     value = float('-inf')
        #     for action in state.getLegalActions(0):
        #         value = max(value, vMin(state.generateSuccessor(0, action), depth, ghostIds[0]))
        #
        #     return value
        #
        # res = [(action, vMin(gameState.generateSuccessor(0, action), 0, ghostIds[0])) \
        #        for action in gameState.getLegalActions(0)]
        # res.sort(key=lambda k: k[1])
        #
        # return res[-1][0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
