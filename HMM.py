import random
import argparse
import numpy as np
import sys

# Vinay Bojja
def forward(fileName, model):
    file = open(fileName)

    count = 0
    while True:
        line = file.readline()
        if not line:
            break
        if count % 2 != 0:
            lineSplitArray = line.split()
            observations = Observation([], lineSplitArray)
            model.forwardAlgorithm(observations)
            # Create a new file and write the obs object to the file
            outputFile = open('forward.output.obs', 'a')
            outputFile.write(str(observations))
            outputFile.close()
        count += 1
    file.close()

def viterbi(file_name, model):
    file = open(file_name)

    count = 0
    while True:
        line = file.readline()
        if not line:
            break
        if count % 2 != 0:
            lineSplitArray = line.split()
            obs = Observation ([], lineSplitArray)
            model.viterbiAlgorithm(obs)
            # Create a new file and write the obs object to the file
            output_file = open('viterbi.output.obs', 'a')
            output_file.write(str(obs))
            output_file.close()
        count += 1
    # print('No of words, word_count)
    file.close()

# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# hmm model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        ## Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        # trans
        with open(basename+".trans", 'r') as file:
            for line in file:
                # Process each line here
                words = line.split()
                self.insertInTransitionDictionary(words)

        # emit
        with open(basename+".emit", 'r') as file:
            for line in file:
                # Process each line here
                words = line.split()
                self.insertInEmissionDictionary(words)

    def insertInEmissionDictionary(self,words):
        keys = self.emissions.keys()
        if words[0] in keys:
            innerDictionary = self.emissions[words[0]]
            innerDictionary[words[1]] = float(words[2])
            self.emissions[words[0]] = innerDictionary
        else:
            self.emissions[words[0]] = { words[1]:float(words[2]) }

    def insertInTransitionDictionary(self,words):
        keys = self.transitions.keys()
        if words[0] in keys:
            innerDictionary = self.transitions[words[0]]
            innerDictionary[words[1]] = float(words[2])
            self.transitions[words[0]] = innerDictionary
        else:
            self.transitions[words[0]] = { words[1]:float(words[2]) }

    ## you do this.
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""
        observations = Observation([], [])
        startState = '#'
        count = 0

        while count != n:
            if startState in self.transitions:
                innerDictionary = self.transitions[startState]
                values = list(innerDictionary.keys())
                probabilites = list(innerDictionary.values())

                randomChoice = random.choices(values, probabilites, k = 1)
                observations.stateseq.append(randomChoice[0])
                startState = randomChoice[0]
            count += 1

        for item in observations.stateseq:
            if item in self.emissions:
                innerDictionary = self.emissions[item]
                values = list(innerDictionary.keys())
                probabilites = list(innerDictionary.values())
                randomChoice = random.choices(values, weights=probabilites, k = 1)
                observations.outputseq.append(randomChoice[0])
        return observations

    ##forward algorithm
    def forwardAlgorithm(self, observation):
        columnList = observation.outputseq
        rowList = list(self.transitions.keys())
        for element in rowList:
            indexOne = rowList.index(rowList[0])
            indexTwo = rowList.index('#')
            if (element == '#'):
                rowList[indexOne], rowList[indexTwo] = rowList[indexTwo], rowList[indexOne]

        column = len(columnList)
        row = len(rowList)
        matrix =[[0 for _ in range(column)] for _ in range(row)]

        for i in range(1, row):
            key = rowList[i]
            if key in self.emissions:
                emissionInnerDict = self.emissions[key]
                firstColumn = columnList[0]
                if (firstColumn in emissionInnerDict):
                    transInnerDict = self.transitions[rowList[0]]
                    transProbability = transInnerDict[rowList[i]]
                    emitProbability = emissionInnerDict[firstColumn]
                    probability = transProbability * emitProbability
                    matrix[i][0] = probability
                else:
                    matrix[i][0] = 0

        for j in range(1, column):
            for i in range(1, row):
                probability = 0
                key = rowList[i]
                if key in self.emissions:
                    emissionInnerDict = self.emissions[key]
                    if columnList[j] in emissionInnerDict:
                        p1 = emissionInnerDict[columnList[j]]
                        transInnerDict = self.transitions[rowList[i]]
                        for k in range(1, row):
                            p2 = transInnerDict[rowList[k]]
                            probability += p1 * p2 * matrix[k][j - 1]
                        matrix[i][j] = probability
        npMatrix = np.array(matrix)
        maxRow = npMatrix.argmax(axis=0)
        maxElement = [rowList[element] for element in maxRow]
        observation.stateseq = maxElement

    def viterbiAlgorithm(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """
        columnList = observation.outputseq
        rowList = list(self.transitions.keys())
        for element in rowList:
            indexOne = rowList.index(rowList[0])
            indexTwo = rowList.index('#')
            if (element == '#'):
                rowList[indexOne], rowList[indexTwo] = rowList[indexTwo], rowList[indexOne]

        column = len(columnList)
        row = len(rowList)
        matrix = [[0 for _ in range(column)] for _ in range(row)]
        backPointers = [[None for _ in range(column)] for _ in range(row)]

        backPointers[0][0] = 0
        for i in range(1, row):
            key = rowList[i]
            if key in self.emissions:
                emissionInnerDict = self.emissions[key]
                firstColumn = columnList[0]
                if (firstColumn in emissionInnerDict):
                    transInnerDict = self.transitions[rowList[0]]
                    transProbability = transInnerDict[rowList[i]]
                    emitProbability = emissionInnerDict[firstColumn]
                    probability = transProbability * emitProbability
                    matrix[i][0] = probability
                else:
                    matrix[i][0] = 0
            backPointers[i][0] = 0

        maxValue = -1.0
        maxState = -sys.maxsize - 1
        for j in range(1, column):
            for i in range(1, row):
                probability = 0
                key = rowList[i]
                if key in self.emissions:
                    emissionInnerDict = self.emissions[key]
                    if columnList[j] in emissionInnerDict:
                        p1 = emissionInnerDict[columnList[j]]
                        transInnerDict = self.transitions[rowList[i]]
                        for k in range(1, row):
                            p2 = transInnerDict[rowList[k]]
                            probability += p1 * p2 * matrix[k][j - 1]
                            if(probability>maxValue):
                                maxValue = probability
                                maxState = k
                        matrix[i][j] = probability
                        maxValue = -1.0
                        backPointers[i][j] = maxState
        npMatrix = np.array(matrix)
        maxRow = npMatrix.argmax(axis=0)
        maxElement = [rowList[element] for element in maxRow]
        observation.stateseq = maxElement
