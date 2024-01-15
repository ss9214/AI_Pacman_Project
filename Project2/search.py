# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # DFS uses a Stack
    fringe = util.Stack()

    fringe.push({'path': [], 'cur_state': problem.getStartState()})
 
    visited_nodes = set()

    while fringe:
        step = fringe.pop()

        while (step['cur_state'] in visited_nodes):
            step = fringe.pop()

        if problem.isGoalState(step['cur_state']): # if cur state is the goal return it
            return step['path']
        
        visited_nodes.add(step['cur_state'])  # state is not goal, add to visited
        successors = problem.getSuccessors(step['cur_state'])

        for successor, action, stepCost in successors:
            if successor not in visited_nodes:
                step_path = step['path'].copy()
                step_path.append(action)

                fringe.push({'path': step_path, 'cur_state': successor}) # push current path and next successor state
    
    return [] # no goal found

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    # BFS uses a Queue
    fringe = util.Queue()

    fringe.push({'path': [], 'cur_state': problem.getStartState()})
 
    visited_nodes = set()

    while fringe:
        step = fringe.pop()

        while (step['cur_state'] in visited_nodes):
            step = fringe.pop()

        if problem.isGoalState(step['cur_state']): # if cur state is the goal return it
            return step['path']
        
        visited_nodes.add(step['cur_state'])  # state is not goal, add to visited
        successors = problem.getSuccessors(step['cur_state'])

        for successor, action, stepCost in successors:
            if successor not in visited_nodes:
                step_path = step['path'].copy()
                step_path.append(action)

                fringe.push({'path': step_path, 'cur_state': successor}) # push current path and next successor state
    
    return [] # no goal found

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    # UCS uses a Priority Queue
    fringe = util.PriorityQueue()

    fringe.push({
        'path': [], 
        'cur_state': problem.getStartState(), 
        'cost': 0
        }, 0)
 
    visited_nodes = set()

    while fringe:
        step = fringe.pop()

        while (step['cur_state'] in visited_nodes):
            step = fringe.pop()

        if problem.isGoalState(step['cur_state']): # if cur state is the goal return it
            return step['path']
        
        visited_nodes.add(step['cur_state'])  # state is not goal, add to visited
        successors = problem.getSuccessors(step['cur_state'])

        for successor, action, stepCost in successors:
            if successor not in visited_nodes:
                step_path = step['path'].copy()
                step_path.append(action)

                total_cost = step['cost'] + stepCost

                # push current path, next successor state, and total cost
                fringe.push({
                    'path': step_path, 
                    'cur_state': successor, 
                    'cost': total_cost
                    }, total_cost)

    return [] # no goal found

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    fringe = util.PriorityQueue() # A* uses a Priority Queue

    fringe.push({
        'path': [], 
        'cur_state': problem.getStartState(), 
        'cost': 0
        }, 0)
    
    visited_nodes = set()

    while fringe:
        step = fringe.pop()

        while (step['cur_state'] in visited_nodes):
            step = fringe.pop()

        if problem.isGoalState(step['cur_state']): # if cur state is the goal return it
            return step['path']
        
        visited_nodes.add(step['cur_state'])  # state is not goal, add to visited
        successors = problem.getSuccessors(step['cur_state'])

        for successor, action, stepCost in successors:
            if successor not in visited_nodes:
                step_path = step['path'].copy()
                step_path.append(action)

                total_cost = step['cost'] + stepCost
                
                # push current path, next successor state, and total cost
                fringe.push({
                    'path': step_path, 
                    'cur_state': successor, 
                    'cost': total_cost
                    }, total_cost + heuristic(successor, problem))

    return [] # no goal found

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
