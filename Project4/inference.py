# inference.py
# ------------
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


import random
import itertools
from typing import List, Dict, Tuple
import busters
import game
import bayesNet as bn
from bayesNet import normalize
import hunters
from util import manhattanDistance, raiseNotDefined
from factorOperations import joinFactorsByVariableWithCallTracking, joinFactors
from factorOperations import eliminateWithCallTracking

########### ########### ###########
########### QUESTION 1  ###########
########### ########### ###########

def constructBayesNet(gameState: hunters.GameState):
    """
    Construct an empty Bayes net according to the structure given in Figure 1
    of the project description.

    You *must* name all variables using the constants in this function.

    In this method, you should:
    - populate `variables` with the Bayes Net nodes
    - populate `edges` with every edge in the Bayes Net. we will represent each
      edge as a tuple `(from, to)`.
    - set each `variableDomainsDict[var] = values`, where `values` is a list
      of the possible assignments to `var`.
        - each agent position is a tuple (x, y) where x and y are 0-indexed
        - each observed distance is a noisy Manhattan distance:
          it's non-negative and |obs - true| <= MAX_NOISE
    - this uses slightly simplified mechanics vs the ones used later for simplicity
    """
    # constants to use
    PAC = "Pacman"
    GHOST0 = "Ghost0"
    GHOST1 = "Ghost1"
    OBS0 = "Observation0"
    OBS1 = "Observation1"
    X_RANGE = gameState.getWalls().width
    Y_RANGE = gameState.getWalls().height
    MAX_NOISE = 7

    variables = []
    edges = []
    variableDomainsDict = {}

    # Fill up variables and edges with given values
    variables = variables + [PAC, GHOST0, GHOST1, OBS0, OBS1]
    edges = edges + [(GHOST0, OBS0), (PAC, OBS0), (PAC, OBS1), (GHOST1, OBS1)]

    # Fill up domains for each variable
    for var in variables:
        possibleValues = []

        # OBS0 and OBS1 have integer values while the other variables have tuple values
        if var == OBS0 or var == OBS1:
            maxNoisyManhattanDistance = (X_RANGE - 1) + (Y_RANGE - 1) + MAX_NOISE

            for x in range(0, maxNoisyManhattanDistance + 1):
                possibleValues.append(x)
        else:
            for x in range(0, X_RANGE):
                for y in range(0, Y_RANGE):
                    possibleValues.append((x, y))        
        
        variableDomainsDict[var] = possibleValues 

    net = bn.constructEmptyBayesNet(variables, edges, variableDomainsDict)
    return net


def inferenceByEnumeration(bayesNet: bn, queryVariables: List[str], evidenceDict: Dict):
    """
    An inference by enumeration implementation provided as reference.
    This function performs a probabilistic inference query that
    returns the factor:

    P(queryVariables | evidenceDict)

    bayesNet:       The Bayes Net on which we are making a query.
    queryVariables: A list of the variables which are unconditioned in
                    the inference query.
    evidenceDict:   An assignment dict {variable : value} for the
                    variables which are presented as evidence
                    (conditioned) in the inference query. 
    """
    callTrackingList = []
    joinFactorsByVariable = joinFactorsByVariableWithCallTracking(callTrackingList)
    eliminate = eliminateWithCallTracking(callTrackingList)

    # initialize return variables and the variables to eliminate
    evidenceVariablesSet = set(evidenceDict.keys())
    queryVariablesSet = set(queryVariables)
    eliminationVariables = (bayesNet.variablesSet() - evidenceVariablesSet) - queryVariablesSet

    # grab all factors where we know the evidence variables (to reduce the size of the tables)
    currentFactorsList = bayesNet.getAllCPTsWithEvidence(evidenceDict)

    # join all factors by variable
    for joinVariable in bayesNet.variablesSet():
        currentFactorsList, joinedFactor = joinFactorsByVariable(currentFactorsList, joinVariable)
        currentFactorsList.append(joinedFactor)

    # currentFactorsList should contain the connected components of the graph now as factors, must join the connected components
    fullJoint = joinFactors(currentFactorsList)

    # marginalize all variables that aren't query or evidence
    incrementallyMarginalizedJoint = fullJoint
    for eliminationVariable in eliminationVariables:
        incrementallyMarginalizedJoint = eliminate(incrementallyMarginalizedJoint, eliminationVariable)

    fullJointOverQueryAndEvidence = incrementallyMarginalizedJoint

    # normalize so that the probability sums to one
    # the input factor contains only the query variables and the evidence variables, 
    # both as unconditioned variables
    queryConditionedOnEvidence = normalize(fullJointOverQueryAndEvidence)
    # now the factor is conditioned on the evidence variables

    # the order is join on all variables, then eliminate on all elimination variables
    return queryConditionedOnEvidence

########### ########### ###########
########### QUESTION 4  ###########
########### ########### ###########

def inferenceByVariableEliminationWithCallTracking(callTrackingList=None):

    def inferenceByVariableElimination(bayesNet: bn, queryVariables: List[str], evidenceDict: Dict, eliminationOrder: List[str]):
        """
        This function should perform a probabilistic inference query that
        returns the factor:

        P(queryVariables | evidenceDict)

        It should perform inference by interleaving joining on a variable
        and eliminating that variable, in the order of variables according
        to eliminationOrder.  See inferenceByEnumeration for an example on
        how to use these functions.

        You need to use joinFactorsByVariable to join all of the factors 
        that contain a variable in order for the autograder to 
        recognize that you performed the correct interleaving of 
        joins and eliminates.

        If a factor that you are about to eliminate a variable from has 
        only one unconditioned variable, you should not eliminate it 
        and instead just discard the factor.  This is since the 
        result of the eliminate would be 1 (you marginalize 
        all of the unconditioned variables), but it is not a 
        valid factor.  So this simplifies using the result of eliminate.

        The sum of the probabilities should sum to one (so that it is a true 
        conditional probability, conditioned on the evidence).

        bayesNet:         The Bayes Net on which we are making a query.
        queryVariables:   A list of the variables which are unconditioned
                          in the inference query.
        evidenceDict:     An assignment dict {variable : value} for the
                          variables which are presented as evidence
                          (conditioned) in the inference query. 
        eliminationOrder: The order to eliminate the variables in.

        Hint: BayesNet.getAllCPTsWithEvidence will return all the Conditional 
        Probability Tables even if an empty dict (or None) is passed in for 
        evidenceDict. In this case it will not specialize any variable domains 
        in the CPTs.

        Useful functions:
        BayesNet.getAllCPTsWithEvidence
        normalize
        eliminate
        joinFactorsByVariable
        joinFactors
        """

        # this is for autograding -- don't modify
        joinFactorsByVariable = joinFactorsByVariableWithCallTracking(callTrackingList)
        eliminate             = eliminateWithCallTracking(callTrackingList)
        if eliminationOrder is None: # set an arbitrary elimination order if None given
            eliminationVariables = bayesNet.variablesSet() - set(queryVariables) -\
                                   set(evidenceDict.keys())
            eliminationOrder = sorted(list(eliminationVariables))

        # Get factors with the given domain from evidenceDict
        factors = bayesNet.getAllCPTsWithEvidence(evidenceDict)

        # Join and eliminate for each variable in eliminationOrder
        for var in eliminationOrder:
            factors, joinedFactor = joinFactorsByVariable(factors, var)

            # Only eliminate if there is more than 1 unconditioned variable
            if len(joinedFactor.unconditionedVariables()) > 1:
                joinedFactor = eliminate(joinedFactor, var)
                factors.append(joinedFactor)

        # Do the final join and normalize
        return normalize(joinFactors(factors))

    return inferenceByVariableElimination

inferenceByVariableElimination = inferenceByVariableEliminationWithCallTracking()

def sampleFromFactorRandomSource(randomSource=None):
    if randomSource is None:
        randomSource = random.Random()

    def sampleFromFactor(factor, conditionedAssignments=None):
        """
        Sample an assignment for unconditioned variables in factor with
        probability equal to the probability in the row of factor
        corresponding to that assignment.

        factor:                 The factor to sample from.
        conditionedAssignments: A dict of assignments for all conditioned
                                variables in the factor.  Can only be None
                                if there are no conditioned variables in
                                factor, otherwise must be nonzero.

        Useful for inferenceByLikelihoodWeightingSampling

        Returns an assignmentDict that contains the conditionedAssignments but 
        also a random assignment of the unconditioned variables given their 
        probability.
        """
        if conditionedAssignments is None and len(factor.conditionedVariables()) > 0:
            raise ValueError("Conditioned assignments must be provided since \n" +
                            "this factor has conditionedVariables: " + "\n" +
                            str(factor.conditionedVariables()))

        elif conditionedAssignments is not None:
            conditionedVariables = set([var for var in conditionedAssignments.keys()])

            if not conditionedVariables.issuperset(set(factor.conditionedVariables())):
                raise ValueError("Factor's conditioned variables need to be a subset of the \n"
                                    + "conditioned assignments passed in. \n" + \
                                "conditionedVariables: " + str(conditionedVariables) + "\n" +
                                "factor.conditionedVariables: " + str(set(factor.conditionedVariables())))

            # Reduce the domains of the variables that have been
            # conditioned upon for this factor 
            newVariableDomainsDict = factor.variableDomainsDict()
            for (var, assignment) in conditionedAssignments.items():
                newVariableDomainsDict[var] = [assignment]

            # Get the (hopefully) smaller conditional probability table
            # for this variable 
            CPT = factor.specializeVariableDomains(newVariableDomainsDict)
        else:
            CPT = factor
        
        # Get the probability of each row of the table (along with the
        # assignmentDict that it corresponds to)
        assignmentDicts = sorted([assignmentDict for assignmentDict in CPT.getAllPossibleAssignmentDicts()])
        assignmentDictProbabilities = [CPT.getProbability(assignmentDict) for assignmentDict in assignmentDicts]

        # calculate total probability in the factor and index each row by the 
        # cumulative sum of probability up to and including that row
        currentProbability = 0.0
        probabilityRange = []
        for i in range(len(assignmentDicts)):
            currentProbability += assignmentDictProbabilities[i]
            probabilityRange.append(currentProbability)

        totalProbability = probabilityRange[-1]

        # sample an assignment with probability equal to the probability in the row 
        # for that assignment in the factor
        pick = randomSource.uniform(0.0, totalProbability)
        for i in range(len(assignmentDicts)):
            if pick <= probabilityRange[i]:
                return assignmentDicts[i]

    return sampleFromFactor

sampleFromFactor = sampleFromFactorRandomSource()

class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))
    
    ########### ########### ###########
    ########### QUESTION 5a ###########
    ########### ########### ###########

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """

        # Divide each value by the total of the values in order to normalize
        weightSum = self.total()

        # Check if weightSum is not zero since you cannot divide by zero
        # If weightSum is zero, do nothing
        if weightSum != 0:
            for key, value in self.items():
                self[key] = value / weightSum
            
    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """
        # Gets keys with weights (values)
        keys = list(self.keys())
        values = list(self.values())

        # Built-in python function to get a random value with weights (returns a list of length 1 so we return just the list item)
        return random.choices(keys, weights=values, k=1)[0] 


class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)
    
    ########### ########### ###########
    ########### QUESTION 5b ###########
    ########### ########### ###########

    def getObservationProb(self, noisyDistance: int, pacmanPosition: Tuple, ghostPosition: Tuple, jailPosition: Tuple):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """

        # Take care of edge cases where the noisyDistance is None and when the ghost is in jail
        if noisyDistance == None or ghostPosition == jailPosition:
            return noisyDistance == None and ghostPosition == jailPosition

        # Use getObservationProbability function to calculate probability
        return busters.getObservationProbability(noisyDistance, manhattanDistance(pacmanPosition, ghostPosition))

    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()
    
    ########### ########### ###########
    ########### QUESTION 6  ###########
    ########### ########### ###########

    def observeUpdate(self, observation: int, gameState: busters.GameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        # Updates beliefs for each position using the getObservationProb function
        for position in self.allPositions:    
            self.beliefs[position] *= self.getObservationProb(observation, gameState.getPacmanPosition(), position, self.getJailPosition())

        self.beliefs.normalize()
    
    ########### ########### ###########
    ########### QUESTION 7  ###########
    ########### ########### ###########

    def elapseTime(self, gameState: busters.GameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
        # Distribution of updated beliefs with their probabilities
        newBeliefs = DiscreteDistribution()
        # Stores calculated distributions to avoid excessive calls to getPositionDistribution
        posDists = dict()

        # Update belief for all positions
        for curPosition in self.allPositions:
            prob = 0.0
            calculatedPositions = list(posDists.keys())

            # Calculate the probability using all previous positions
            for prevPosition in self.allPositions:
                # Avoid calling getPositionDistribution unless it has not already been calculated for the given position
                if not prevPosition in calculatedPositions:
                    posDists[prevPosition] = self.getPositionDistribution(gameState, prevPosition)

                # Get the position distribution and use it to update the belief with the current position
                newPosDist = posDists[prevPosition]
                updatedBelief = self.beliefs[prevPosition] * newPosDist[curPosition]

                prob += updatedBelief

            newBeliefs[curPosition] = prob

        # Update beliefs
        self.beliefs = newBeliefs

    def getBeliefDistribution(self):
        return self.beliefs

class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles
    
    ########### ########### ###########
    ########### QUESTION 9  ###########
    ########### ########### ###########

    def initializeUniformly(self, gameState: busters.GameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        # Initializes self.particles as an empty list
        self.particles = []

        # Creates an amount of particles equal to self.numParticles
        for num in range(0, self.numParticles):
            # Uses mod to evenly distribute particles
            distributionIndex = num % len(self.legalPositions)

            self.particles.append(self.legalPositions[distributionIndex])

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.

        This function should return a normalized distribution.
        """
        beliefDistribution = DiscreteDistribution()

        for particle in self.particles:
            beliefDistribution[particle] = beliefDistribution[particle] + 1 if particle in beliefDistribution else 0

        beliefDistribution.normalize()

        return beliefDistribution
    
    ########### ########### ###########
    ########### QUESTION 10 ###########
    ########### ########### ###########

    def observeUpdate(self, observation: int, gameState: busters.GameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        beliefs = self.getBeliefDistribution()
        newBeliefs = DiscreteDistribution()

        # Updates beliefs for each position using the getObservationProb function
        for position in beliefs.keys():    
            newBeliefs[position] = beliefs[position] * self.getObservationProb(observation, gameState.getPacmanPosition(), position, self.getJailPosition())

        # Check for special case
        if newBeliefs.total() == 0:
            self.initializeUniformly(gameState)
        else:
            # Re-sample particles
            self.particles = [newBeliefs.sample() for _ in range(self.numParticles)]
    
    ########### ########### ###########
    ########### QUESTION 11 ###########
    ########### ########### ###########

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        # Stores calculated distributions to avoid excessive calls to getPositionDistribution
        posDists = dict()

        # Fills posDists with distributions for each position in particles while avoiding calling
        # getPositionDistribution too often
        for position in self.allPositions:
            if position not in posDists and position in self.particles:
                posDists[position] = self.getPositionDistribution(gameState, position)

        # Re-sample particles
        self.particles = [posDists[pos].sample() for pos in self.particles]


class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    ########### ########### ###########
    ########### QUESTION 12 ###########
    ########### ########### ###########

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        # Initializes self.particles as an empty list
        self.particles = []

        # Get permutations for legalPositions for the given number of ghosts
        permutations = list(itertools.product(self.legalPositions, repeat=self.numGhosts))

        # itertools returns permutations in a non-random order so it needs to be shuffled
        random.shuffle(permutations)

        # Creates an amount of particles equal to self.numParticles
        for num in range(0, self.numParticles):
            # Uses mod to evenly distribute particles
            distributionIndex = num % len(permutations)

            self.particles.append(permutations[distributionIndex])

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.
        """
        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)

    ########### ########### ###########
    ########### QUESTION 13 ###########
    ########### ########### ###########

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.
        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.
        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        beliefs = self.getBeliefDistribution()
        newBeliefs = DiscreteDistribution()

        # Updates beliefs for each position using the getObservationProb function
        for ghostPositions in beliefs.keys():
            prob = 1.0

            # Calls getObservationProb for each ghost and calculates the probability of each ghost being in that position
            for ghostIndex, position in enumerate(ghostPositions):
                prob *= self.getObservationProb(observation[ghostIndex], gameState.getPacmanPosition(), position, self.getJailPosition(ghostIndex))

            # Updates newBeliefs
            newBeliefs[ghostPositions] = beliefs[ghostPositions] * prob

        # Check for special case
        if newBeliefs.total() == 0:
            self.initializeUniformly(gameState)
        else:
            # Re-sample particles
            self.particles = [newBeliefs.sample() for _ in range(self.numParticles)]

    ########### ########### ###########
    ########### QUESTION 14 ###########
    ########### ########### ###########

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions

            # now loop through and update each entry in newParticle...

            # Stores calculated distributions to avoid excessive calls to getPositionDistribution
            posDists = dict()

            # Fills posDists with distributions for each position in particles while avoiding calling
            # getPositionDistribution too often
            for ghostIndex, position in enumerate(list(oldParticle)):
                if position not in posDists:
                    posDists[position] = self.getPositionDistribution(gameState, list(oldParticle), ghostIndex, self.ghostAgents[ghostIndex])
            
            # Re-sample particles
            newParticle = [posDists[pos].sample() for pos in newParticle]

            newParticles.append(tuple(newParticle))
        self.particles = newParticles

# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist
