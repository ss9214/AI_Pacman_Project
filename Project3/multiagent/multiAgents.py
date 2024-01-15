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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        incentive = 0

        for i in range(0, len(newGhostStates)):
            ghostPos = newGhostStates[i].getPosition()
            scareTime = newScaredTimes[i]

            dist = manhattanDistance(newPos, ghostPos)

            if (dist < 5 and scareTime == 0):
                incentive += dist
            elif (scareTime > 0):
                incentive -= dist

        for radius in range(-3, 3):
            try:
                if (newFood[newPos[0] + radius][newPos[1]] or newFood[newPos[0]][newPos[1] + radius]):
                    incentive += 4 - abs(radius)
                    break
            except:
                continue

        return successorGameState.getScore() + incentive

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

        # Evaluates the score if we reach max depth or a win/lose state
        # Otherwise it calls maxAgent again to go to the next depth
        def getScore(state: GameState, depth: int):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
        
            return maxAgent(state, depth)[1]

        # Returns a tuple since we care about both the action and the score
        def maxAgent(state: GameState, depth=0, agent=0):
            actionScores = list()

            for action in state.getLegalActions(agent):
                newState = state.generateSuccessor(agent, action)
                actionScores.append((action, minAgent(newState, depth, agent + 1)))

            # Sort actions from highest to lowest score
            actionScores.sort(reverse=True, key=lambda x: x[1])

            return actionScores[0]
        
        # Returns only the score since we don't care about the ghosts' actions
        def minAgent(state: GameState, depth=0, agent=0):
            actionScores = list()
            
            if (agent == state.getNumAgents()):
                actionScores.append(getScore(state, depth + 1))
            else:
                for action in state.getLegalActions(agent):
                    newState = state.generateSuccessor(agent, action)
                    actionScores.append(minAgent(newState, depth, agent + 1))

            # Sort scores in ascending order
            actionScores.sort()

            if (len(actionScores) == 0):
                return minAgent(state, depth, agent + 1)

            return actionScores[0]

        bestAction = maxAgent(gameState)

        return bestAction[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # Evaluates the score if we reach max depth or a win/lose state
        # Otherwise it calls maxAgent again to go to the next depth
        def getScore(state: GameState, depth: int, a, b):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
        
            return maxAgent(state, depth, 0, a, b)[1]

        # Returns a tuple since we care about both the action and the score
        def maxAgent(state: GameState, depth, agent, a, b):
            actionScores = list()

            for action in state.getLegalActions(agent):
                newState = state.generateSuccessor(agent, action)
                score = minAgent(newState, depth, agent + 1, a, b)

                actionScores.append((action, score))
                a = max(a, score)

                if b < a:
                    break

            # Sort actions from highest to lowest score
            actionScores.sort(reverse=True, key=lambda x: x[1])

            return actionScores[0]
        
        # Returns only the score since we don't care about the ghosts' actions
        def minAgent(state: GameState, depth, agent, a, b):
            actionScores = list()
            
            if (agent == state.getNumAgents()):
                actionScores.append(getScore(state, depth + 1, a, b))
            else:
                for action in state.getLegalActions(agent):
                    newState = state.generateSuccessor(agent, action)
                    score = minAgent(newState, depth, agent + 1, a, b)

                    actionScores.append(score)
                    b = min(b, score)

                    if b < a:
                        break

            # Sort scores in ascending order
            actionScores.sort()

            if (len(actionScores) == 0):
                return minAgent(state, depth, agent + 1, a, b)

            return actionScores[0]

        bestAction = maxAgent(gameState, 0, 0, float('-inf'), float('inf'))

        return bestAction[0]

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
        
        # Evaluates the score if we reach max depth or a win/lose state
        # Otherwise it calls maxAgent again to go to the next depth
        def getScore(state: GameState, depth: int):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
        
            return maxAgent(state, depth)[1]

        # Returns a tuple since we care about both the action and the score
        def maxAgent(state: GameState, depth=0, agent=0):
            actionScores = list()

            for action in state.getLegalActions(agent):
                newState = state.generateSuccessor(agent, action)
                actionScores.append((action, expectAgent(newState, depth, agent + 1)))

            # Sort actions from highest to lowest score
            actionScores.sort(reverse=True, key=lambda x: x[1])

            return actionScores[0]
        
        # Returns only the score since we don't care about the ghosts' actions
        def expectAgent(state: GameState, depth=0, agent=0):
            actionScores = list()
            
            if (agent == state.getNumAgents()):
                actionScores.append(getScore(state, depth + 1))
            else:
                for action in state.getLegalActions(agent):
                    newState = state.generateSuccessor(agent, action)
                    actionScores.append(expectAgent(newState, depth, agent + 1))

            # Sort scores in ascending order
            actionScores.sort()

            if (len(actionScores) == 0):
                return expectAgent(state, depth, agent + 1)

            # Instead of minimizing, calculate the average

            total = 0

            for score in actionScores:
                total += score

            return total / len(actionScores)

        bestAction = maxAgent(gameState)

        return bestAction[0]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    score = currentGameState.getScore()

    pos = currentGameState.getPacmanPosition()

    food = currentGameState.getFood()
    capsules = currentGameState.getCapsules()

    ghostStates = currentGameState.getGhostStates()
    ghostScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    
    # Pacman should go towards nearest visible food
    for radius in range(-3, 3):
        try:
            if (food[pos[0] + radius][pos[1]] or food[pos[0]][pos[1] + radius]):
                score += 4 - abs(radius)
                break
        except:
            continue

    # Pacman should prioritize eating capsules if nearby
    for capsule in capsules:
        capsuleDist = manhattanDistance(pos, capsule)

        if (capsuleDist < 3):
            if (capsule[0] == pos[0]):
                score += 10
            
            if (capsule[1] == pos[1]):
                score += 10

            score -= 3 * capsuleDist

    # Pacman should avoid ghosts unless they are scared (then he should eat them)
    for i in range(0, len(ghostStates)):
        ghostPos = ghostStates[i].getPosition()
        scareTime = ghostScaredTimes[i]

        dist = manhattanDistance(pos, ghostPos)

        if (dist < 3 and scareTime == 0):
            score += dist

        elif (scareTime > 0):
            score -= dist

    # Tiebreaker
    score -= 0.1 * pos[1]

    return score

# Abbreviation
better = betterEvaluationFunction
