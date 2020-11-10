import os
import math
import time
import argparse
import itertools
import warnings
import random
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('dark_background')

class TSP():
    '''
    Traveling sales person class.
    '''
    def __init__(self, iterations, crossoverPercent, mutationChance, verbose=False):
        '''
        Initialize class & properties.
        '''
        self.DIMENSIONIndex = 4
        self.DATAIndex = 7
        self.dataCount = 0
        self.locs = []
        self.dists = []
        self.cwd = os.path.dirname(os.path.abspath(__file__))
        self.files = [f for f in os.listdir(self.cwd) if f.endswith('.tsp')]
        self.locationsFile = 'Random10.tsp'#"Random100.tsp"

        ## plotting
        self.fig, self.ax = plt.subplots(figsize=(12,8))
        self.perms = []
        self.verbose = verbose
        self.colorIndex = 0

        ## project3
        self.cycle = []
        # self.cycleDistance = float("inf")

        ## project4
        self.crossoverpercent, self.mutationchance = crossoverPercent, mutationChance

        self.iterations = int(iterations)
        self.mutationRate = 0.1
        self.permuationIndicies = []
        self.bestPermutation = []
        self.currentGlobalFitness = float("inf")
        self.previousGlobalFitness = float("inf")
        self.fitnesses = [0 for i in range(100)]
        self.globalFitnesses = []
        self.first, self.last = [], []
        self.selectedPopulation = []

    def readData(self):
        '''
        Read data from provided source files.
        Store data in locs/locations and skip the useless stuff at the beginning.
        '''
        with open(self.cwd + "\\" + self.locationsFile) as f:
            for i, row in enumerate(f):
                if i == self.DIMENSIONIndex:
                    self.dataCount = int(row.strip().split(" ")[1])
                if i >= self.DATAIndex:
                    values = row.strip().split(" ")
                    self.locs.append([float(values[1]), float(values[2])])

        ## for plotting
        x, y = np.array(self.locs)[:,0], np.array(self.locs)[:,1]
        (x_min, x_max), (y_min, y_max) = (int(min(x)-5), int(max(x)+5)), (int(min(y)-5), int(max(y)+5))
        self.ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    def plotter(self, i):
        '''
        Simple plotter to visualize the travler traveling.
        '''
        x,y = np.array(self.perms[i])[:,0], np.array(self.perms[i])[:,1]
        self.plot.set_data(x,y)
        
        if self.colorIndex == -1:
            time.sleep(.500)
            self.colorIndex +=1
        
        self.colorIndex +=1
        if self.colorIndex == 1:
            self.plot.set_color('mediumspringgreen')

        elif self.colorIndex == 2:
            self.plot.set_color('yellow')
        else:
            self.plot.set_color('orange')
            self.colorIndex = -1
        # self.plt.set_cmap = 'RdYlGn'
        return self.plot,
            
    def elucidianDistance(self, arr1, arr2):
        '''
        Distance equation provided in class for euclidian distance.
        '''
        x2, x1 = arr2[0], arr1[0]
        y2, y1 = arr2[1], arr1[1]
        
        return math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2))
    
    def calculateDistances(self, permutations):
        '''
        Calculate the distance measure for each permuatation created.
        '''
        dist = 0.0
        
        ## go to all destinations
        for i in range(0, len(permutations)-1):
            dist += self.elucidianDistance(permutations[i], permutations[i+1])
        ## go back to start
        dist += self.elucidianDistance(permutations[len(permutations)-1], permutations[0])
        return dist ## could have some multiplier or change to total distance


    '''#############################################################
    ################################################################
    '''

    def getFirstAndLast(self):
        self.first, self.last = self.locs[0], self.locs[len(self.locs)-1]

    def globalFitness(self, permutations):
        '''
        Global fitness is defined as the total distance of a permuation of the
        dataset. So we want to minize this global value. The smallest global fit
        is the optimal path.
        '''
        newGlobalFitness = self.calculateDistances(permutations)
        self.globalFitnesses.append(newGlobalFitness)
        if self.currentGlobalFitness > newGlobalFitness:
            self.currentGlobalFitness = newGlobalFitness
            self.bestPermutation = permutations


    def fitness(self, locs, fitnesses):
        '''
        Determine the current sample's fitness  
        '''
        ## do some operation

        # for i in range(5):
        for i in range(99):
            fitnesses[i] = self.elucidianDistance(locs[i], locs[i+1])
        
        fitnesses[len(fitnesses)-1] = self.elucidianDistance(locs[len(locs)-1], locs[0])
        return fitnesses



    def selection(self, fitnesses, selectionPercent=90):
        '''
        Selection will occur to a random portion of the population. We want to
        minimize the path size so we will randomly select <= 50% of the population
        that is the "worst" for the optimal path.
        '''
        ## percent of population for selection
        # selectionPopulation = random.randint(0, cullingPercent)

        # locations = np.array(self.locs.copy())
        # self.selectedPopulation = list(locations[np.argsort(locations)[-selectionPopulation:]])
        locations = np.array(fitnesses.copy())
        selectedPopulation = list(locations[np.argsort(locations)[-selectionPercent:]])
        return selectedPopulation


    # def crossover(self, perm1, perm2, cullingPercent):
    def crossover(self, locs, selectedPopulation, fitnesses, crossOverChance=0.8, crossOverPercent=10):
    
        '''
        A crossover is defined in this program as a location somewhere 
        in the selection population. The crossover takes the city pairs that
        will "mate" or find a new connection if randomness allows them. The maiting
        is done by swapping an index with the new one. This means more than one 
        cross over can occur.
        '''
        
        # locs = np.array(self.locs)
        locsIdxs = np.array(fitnesses)
        ## smallest ones are better so they go first since fitness is higher
        selectedIndicies = list(reversed(locsIdxs.argsort()[-crossOverPercent:][::-1]))
        # boi1 = selectedPopulation.pop(0)
        # boi2 = selectedPopulation.pop(0)
        # boiIdx1 = selectedIndicies.pop(0)
        # boiIdx2 = selectedIndicies.pop(0)

        ## swap if chance allows (high chance of swapping) --> mating should almost 
        ## always occur but in some cases you could say there is complications
        ## could increase swap chance as the fuction continues as the 
        ## selected group progressively is worse so we want more of these to swap
        while selectedIndicies:
            boi1 = selectedPopulation.pop(0)
            boi2 = selectedPopulation.pop(0)
            boiIdx1 = selectedIndicies.pop(0)
            boiIdx2 = selectedIndicies.pop(0)
            if random.random() > crossOverChance:
                a = locs[boiIdx2]
                b = locs[boiIdx1]
                # a = locs[self.fitnesses.index(boi2)]
                # b = locs[self.fitnesses.index(boi1)]
                if a != b:
                    locs[boiIdx1] = a
                    locs[boiIdx2] = b
                else:
                    print()
                

        ## reset the first and last locations (shouldnt have swapped)
        # lastIdx = locs.index(self.last)
        # startIdx = locs.index(self.first)
        # _ = locs.pop(lastIdx)
        # _ = locs.pop(startIdx)
        # locs.insert(0,self.first)
        # locs.append(self.last)

        locs.remove(self.last)
        locs.remove(self.first)


        locs.insert(0,self.first.copy())
        locs.append(self.last.copy())

        # self.fitnesses = [0 for i in range(100)]
        self.fitnesses = [0 for i in range(6)]
        return locs


    # def mutation(self, parent1, parent2, children):
    def mutation(self, locs, mutationChance=0.1, mutationPercent=50):
        '''
        If a crossover is to occur then the mutation will be the new location
        that was given by the a second location randomly selected with a random independent chance.
        '''
        idxs = random.sample(range(mutationPercent), mutationPercent)
        # idxs = random.sample(range(100), 100)
        # idxs = random.sample(range(6), 6)
        for i in range(mutationPercent):
        # for i in range(len(locs)):
            if random.random() > mutationChance:
                # idx = random.sample(range(100), 1)[0]
                b = locs[idxs[i]]
                c = locs[i]
                if b != c:
                    locs[i] = b
                    locs[idxs[i]] = c


        ## reset the first and last locations (shouldnt have swapped)
        # lastIdx = locs.index(self.last)
        # startIdx = locs.index(self.first)
        # last = locs.pop(lastIdx)
        # start = locs.pop(startIdx)
        locs.remove(self.last)
        locs.remove(self.first)


        locs.insert(0,self.first.copy())
        locs.append(self.last.copy())
        return locs

    def evolve(self, selectionPercent=90, crossOverPercent=50, crossOverChance=0.8, mutationPercent=20, mutationChance=0.5):
        locs = self.locs.copy()

        ### 11
        selectionPercents = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
        crossoverPercents = [8, 10, 12, 14, 16, 18, 20]
        mutationchances =   [0.06, 0.08, 0.1, 0.12, 0.14, 0.16]


        ### 22
        selectionPercents = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
        crossoverPercents = [8, 10, 12, 14, 16, 18, 20]
        mutationchances =   [0.06, 0.08, 0.1, 0.12, 0.14, 0.16]


        ### 44
        selectionPercents = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
        crossoverPercents = [8, 10, 12, 14, 16, 18, 20]
        mutationchances =   [0.06, 0.08, 0.1, 0.12, 0.14, 0.16]


        ### 77
        selectionPercents = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
        crossoverPercents = [8, 10, 12, 14, 16, 18, 20]
        mutationchances =   [0.06, 0.08, 0.1, 0.12, 0.14, 0.16]


        ### 97
        selectionPercents = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
        crossoverPercents = [8, 10, 12, 14, 16, 18, 20]
        mutationchances =   [0.06, 0.08, 0.1, 0.12, 0.14, 0.16]


        ### 222
        selectionPercents = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
        crossoverPercents = [8, 10, 12, 14, 16, 18, 20]
        mutationchances =   [0.06, 0.08, 0.1, 0.12, 0.14, 0.16]
        

        selectionIdxs = random.sample(range(len(selectionPercents)-1), 3)
        crossoverIdxs = random.sample(range(len(selectionPercents)-1), 3)
        mutationIdxs = random.sample(range(len(mutationchances)-1), 3)

        curentLocs = self.locs.copy()
        finalLocs = []

        for i in range(0, 9):

            inputCrossOverPercent, inputMutationchance = self.crossoverpercent, self.mutationchance

            ## find the fitnesses per location
            fitnesses = [0 for i in range(100)]
            # fitnesses = [0 for i in range(6)]
            fitnesses = self.fitness(curentLocs, fitnesses)
            
            ## calculate current global fitness
            self.globalFitness(curentLocs)

            ## do selection or dating process
            selectedPopulation = self.selection(fitnesses, selectionPercent)

            ## perform crossover
            locs = self.crossover(curentLocs, selectedPopulation, fitnesses, crossOverChance,  self.crossoverpercent)

            ## mutate
            locs = self.mutation(locs, self.mutationchance, mutationPercent)

            ## evaluation if worse revert to old state?

            ## add for plotting
            self.perms.append(locs)
            finalLocs.append(locs)
        
        bestLocFitness = self.calculateDistances(curentLocs)
        # bestLocFitnessOriginal = bestLocFitness
        bestIdx = -1
        for i, fl in enumerate(finalLocs):
            flFitness = self.calculateDistances(fl)
            if flFitness < bestLocFitness:
                bestLocFitness = flFitness
                bestIdx = i
        if i != -1:
            self.locs = finalLocs[i]
        else:
            self.locs = curentLocs



    def startRandom(self):
        '''
        After the data is loaded, initialize a random layout
        always from the same random order for one scenario.
        Updates the actual array of input data.
        '''

        locs = self.locs.copy()
        random.seed(42)
        indicies = random.sample(range(100), 100)
        # indicies = random.sample(range(6), 6)
        self.locs = [locs[i] for i in indicies]



    def firstLocation(self):
        '''
        This creates a random number for the first location that will be used in the cycle.
        It adds this to the list twice because this is the first and last location in a cycle
        Then it removes it from the list of locations. An integer could be specified instead of random.
        '''
        firstIdx = random.randint(1, len(self.locs)) - 1
        firstLocation = self.locs.pop(firstIdx)
        
        print("⚶"*90)
        print("Inital Location Index: {}\nInital Location: {}".format(firstIdx, firstLocation))
        print("⚶"*90)

        self.cycle.append(firstLocation)
        self.cycle.append(firstLocation)
        self.perms.append(self.cycle)

    def findNextClosest(self):
        '''
        For the first part of this method it is handling the case when there is only the inital location.
        It finds the second location that will be closest to the first one and inserts in between the head and tail.
        The latter half performs the same thing for every remaining location.
        '''
        currentDistances = []
        if len(self.cycle) == 2:
            cycle = self.cycle.copy()
            for location in self.locs:
                currentDistances.append(self.elucidianDistance(cycle[0], location))
            currentMinDistanceIdx = sorted(range(len(currentDistances)), key=currentDistances.__getitem__)[0]
            self.cycle = [cycle[0], self.locs[currentMinDistanceIdx], cycle[1]]
            self.locs.pop(currentMinDistanceIdx)
            self.perms.append(self.cycle)
        else:
            currentLeast = float('inf')
            test = self.cycle.copy()
            currentBestDist = float('inf')
            bestCycle = []
            bestLocation = [-1,-1]
            cycle = self.cycle.copy()

            actualBestCycle, actualBestDist, actualBestLocation = [], float('inf'), [-1,-1]
            ## Every remaining location
            for location in self.locs:
                ## Get the best cycle using this new location
                bestCycle, currentBestDist, bestLocation = self.insertMinDistance(location, bestCycle, currentBestDist, bestLocation)
                ## if its better than what is present make it the best
                if currentBestDist < actualBestDist:
                    actualBestCycle, actualBestDist, actualBestLocation = bestCycle, currentBestDist, bestLocation
            ## for plotting
            self.perms.append(bestCycle)
            ## update the new cycle with the new cycle just created
            self.cycle = bestCycle
            ## remove the location just used
            if bestLocation != [-1,-1]:
                self.locs.remove(bestLocation)

    def insertMinDistance(self, nextShortestLocation, bestCycle, currentBestDist, bestLocation):
        '''
        Calculates the actual cycle distance and deals with inserting the new location into each 
        position/index of the cycle execept the first and last. 
        Returns the best cycle, best distance, best current location or node for connecting.
        '''
        ## for every position in the cycle
        for i in range(1, len(self.cycle) - 1):
            test = self.cycle.copy()
            ## insert the new location into the ith index
            test.insert(i, nextShortestLocation)
            ## calculate the new distance
            currentCycleTestDistance = self.calculateDistances(test)
            ## if this distance at ith location is best: store all the information
            if currentCycleTestDistance < currentBestDist:
                currentBestDist = currentCycleTestDistance
                bestCycle = test
                bestLocation = nextShortestLocation
        return bestCycle, currentBestDist, bestLocation


    def travelPerson(self):
        '''
        "Main" function for combining: reads data, creates permutations,
        visits each location, and find the best path/distance. Also can plot the traversal.
        '''
        self.readData()
        initialize = True
        self.startRandom()
        self.getFirstAndLast()

        ## for plotting
        # self.perms.append(self.locs)

        bestFitnesses = []
        ## combines all genetic methods for the desired number of iterations
        for i in range(self.iterations):
            self.evolve()
            
            print("Current Best Fitness: ", self.currentGlobalFitness)
            # print("Current Fitness: ", self.globalFitnesses[len(self.globalFitnesses)-1])
            
            bestFitnesses.append(self.currentGlobalFitness)
            bestFitnesses.append(self.currentGlobalFitness)
            bestFitnesses.append(self.currentGlobalFitness)
            bestFitnesses.append(self.currentGlobalFitness)
            bestFitnesses.append(self.currentGlobalFitness)
            bestFitnesses.append(self.currentGlobalFitness)
            bestFitnesses.append(self.currentGlobalFitness)
            bestFitnesses.append(self.currentGlobalFitness)
            bestFitnesses.append(self.currentGlobalFitness)



        if self.verbose:
            self.plot, = self.ax.plot(range(self.dataCount),np.zeros(self.dataCount)*np.NaN, 'mediumspringgreen')
            anim = FuncAnimation(self.fig, self.plotter, frames=len(self.perms), repeat=False, interval=100)#, blit=True)
            plt.show()
        

        # plt.show()    ## for debugging

        ## plots the iterations of search
        plt.plot(np.array([i for i in range(len(bestFitnesses))]), np.array(bestFitnesses))
        plt.plot(np.array([i for i in range(len(bestFitnesses))]), np.array(self.globalFitnesses))
        plt.show()
        
        
        ## plots the trails path of travel
        x,y = np.array(self.bestPermutation)[:,0], np.array(self.bestPermutation)[:,1]
        plt.plot(x,y)
        plt.show()

        print("⚶"*90)
        print("Shortest Distance: {:.2f}\nPath of Travel: {}".format(self.currentGlobalFitness, str(self.bestPermutation)))
        print("⚶"*90)

def inputParser():
    '''
    Simple command line input parser.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("iterations", type=str)
    parser.add_argument("crossoverPercent", type=str)
    parser.add_argument("mutationChance", type=str)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


###############
if __name__ == "__main__":
    # parser = inputParser()
    # tsp = TSP(int(parser.iterations), int(parser.crossoverPercent), float(parser.mutationChance), parser.verbose)


    # t0 = time.time()
    # tsp = TSP(50000, True)# parser.verbose)
    tsp = TSP(500, crossoverPercent=16, mutationChance=0.14, verbose=True)
    tsp.travelPerson()
    # t1 = time.time()
    # print(t1-t0)

