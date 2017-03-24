from random import random as rnd
import random
import numpy as np


"""Kuhn poker definitions"""
PASS = 0
BET = 1
NUM_ACTIONS = 2
nodeMap = {}

def cdl(expr, value1, value2):
    assert isinstance(expr, bool)
    if expr:
        return value1
    else:
        return value2

class Node():
    def __init__(self):
        """Kuhn node definitions"""
        self.regretSum = np.zeros((NUM_ACTIONS,))
        self.strategy = np.zeros((NUM_ACTIONS,))
        self.strategySum = np.zeros((NUM_ACTIONS,))
        self.infoSet = ''

    """set mixed strategy"""
    def getStrategy(self, realizationWeight):
        self.strategy = np.maximum(self.regretSum, 0).copy()
        normalizingSum = np.sum(self.strategy)
        if normalizingSum > 0:
            self.strategy /= normalizingSum
        else:
            self.strategy = np.ones((NUM_ACTIONS,))/float(NUM_ACTIONS)
        self.strategySum += realizationWeight*self.strategy
        return self.strategy

    def getAverageStrategy(self):
        avgStrategy = np.zeros((NUM_ACTIONS,))
        normalizingSum = np.sum(self.strategySum)
        #Adhere to for-loop as in paper, but realize that more efficient code is possible
        if normalizingSum > 0:
            avgStrategy = self.strategySum/ normalizingSum
        else:
            avgStrategy = np.ones((NUM_ACTIONS,))/NUM_ACTIONS
        return avgStrategy

    """Set string representation"""
    def __str__(self):
        return "%4s: %s"%(infoSet, str(getAverageStrategy()))

def train(iterations = 100):
    cards = [1, 2, 3]
    util = 0.0
    for i in range(iterations):
        random.shuffle(cards)
        util += cfr(cards,"",1,1)
    print("Average game value: %.3f"%(util/iterations))


def cfr(cards, history, p0, p1):
    plays = len(history)
    player = plays%2
    opponent = 1-player
    """Return payoff"""
    if plays > 1:
        terminalPass = history[plays-1] == 'p'
        doubleBet = history[plays-2:plays] == 'bb'
        isPlayerCardHigher = cards[player] > cards[opponent]
        if terminalPass:
            if history == 'pp':
                return int(isPlayerCardHigher)*2-1
            else:
                return 1
        elif doubleBet:
            return int(isPlayerCardHigher)*4-2

    infoSet = str(cards[player]) + history
    """get information set or create if nonexistant"""
    node = nodeMap.get(infoSet, None)
    if node is None:
        node = Node()
        node.infoSet = infoSet
        nodeMap[infoSet] = node
    """Recursively call cfr"""
    strategy = node.getStrategy(cdl(player == 0, p0, p1))
    util = np.zeros((NUM_ACTIONS,))
    nodeUtil = 0.0
    for a in range(NUM_ACTIONS):
        nextHistory = history + str(cdl(a==0, "p", "b"))
        if player == 0:
            util[a] = -1*cfr(cards, nextHistory, p0*strategy[a], p1)
        else:
            util[a] = -1*cfr(cards, nextHistory, p0, p1*strategy[a])
        nodeUtil += strategy[a] * util[a]

    """For each action, compute and accumulate counterfactual regret"""
    for a in range(NUM_ACTIONS):
        regret = util[a] - nodeUtil
        node.regretSum[a] += cdl(player == 0, p1, p0)*regret
    history = nextHistory
    return nodeUtil

if __name__ == '__main__':
    iterations = 10
    train(iterations)









