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
    def __init__(self, locationFile, verbose=False):
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

        ## plotting
        self.fig, self.ax = plt.subplots(figsize=(12,8))
        self.perms = []
        self.locationsFile = "Random" + locationFile + ".tsp"
        self.verbose = verbose

        ##project4
        self.cycle = []
        # self.cycleDistance = float("inf")

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
        return dist


    def firstLocation(self):
        firstIdx = random.randint(0, len(self.locs))
        firstLocation = self.locs.pop(firstIdx)
        self.cycle.append(firstLocation)
        self.cycle.append(firstLocation)
        self.perms.append(self.cycle)

    def findNextClosest(self):
        currentDistances = []
        if len(self.cycle) == 2:
            cycle = self.cycle.copy()
            for location in self.locs:
                currentDistances.append(self.elucidianDistance(cycle[0], location))
            currentMinDistanceIdx = sorted(range(len(currentDistances)), key=currentDistances.__getitem__)[0]
            self.cycle = [cycle[0], self.locs[currentMinDistanceIdx], cycle[1]]
            self.perms.append(self.cycle)
        else:
            currentLeast = float('inf')
            test = self.cycle.copy()
            currentBestDist = float('inf')
            bestCycle = []
            bestLocation = [-1,-1]


            cycle = self.cycle.copy()

            actualBestCycle, actualBestDist, actualBestLocation = [], float('inf'), [-1,-1]
            for location in self.locs:
                bestCycle, currentBestDist, bestLocation = self.insertMinDistance(location, bestCycle, currentBestDist, bestLocation)
                if currentBestDist < actualBestDist:
                    actualBestCycle, actualBestDist, actualBestLocation = bestCycle, currentBestDist, bestLocation
            self.perms.append(bestCycle)
            self.cycle = bestCycle
            if bestLocation != [-1,-1]:
                self.locs.remove(bestLocation)

            #     for cyc in cycle:
            #         currentDistances.append(self.elucidianDistance(location, cyc))
            # shortestLocationIdx = sorted(range(len(currentDistances)), key=currentDistances.__getitem__)[0]
            # nextShortestLocation = self.locs[shortestLocationIdx]
            # self.insertMinDistance(nextShortestLocation, bestCycle, currentBestDist)

    def insertMinDistance(self, nextShortestLocation, bestCycle, currentBestDist, bestLocation):

        # print("Remaining Locations: {}".format(len(self.locs)))
        
        
        for i in range(1, len(self.cycle) - 1):
            test = self.cycle.copy()
            test.insert(i, nextShortestLocation)
            currentCycleTestDistance = self.calculateDistances(test)
            if currentCycleTestDistance < currentBestDist:
                currentBestDist = currentCycleTestDistance
                bestCycle = test
                bestLocation = nextShortestLocation
        return bestCycle, currentBestDist, bestLocation

        # self.perms.append(bestCycle)
        # self.perms.append(self.cycle)


    def cycleDistance(self, cycle):
        dist = 0.0
        for i in range(0, len(cycle)-2):
            dist+= self.elucidianDistance(cycle[i], cycle[i+1])
        return dist

    # def findNextClosest(self):
    #     currentDistances = []
    #     if len(self.cycle) == 2:
    #         cycle = self.cycle.copy()
    #         for location in self.locs:
    #             currentDistances.append(self.elucidianDistance(cycle[0], location))
    #         currentMinDistanceIdx = sorted(range(len(currentDistances)), key=currentDistances.__getitem__)[0]
    #         self.cycle = [cycle[0], self.locs[currentMinDistanceIdx], cycle[1]]
    #         self.perms.append(self.cycle)
    #     else:
    #         cycle = self.cycle.copy()
    #         for location in self.locs:
    #             for cyc in cycle:
    #                 currentDistances.append(self.elucidianDistance(location, cyc))
    #         shortestLocationIdx = sorted(range(len(currentDistances)), key=currentDistances.__getitem__)[0]
    #         nextShortestLocation = self.locs[shortestLocationIdx]


    # def insertMinDistance(self, nextShortestLocation):
    #     currentLeast = float('inf')
    #     test = self.cycle.copy()
    #     currentBestCycle = float('inf')
    #     bestCycle = []

    #     print("Remaining Locations: {}".format(len(self.locs)))
        
        
    #     for i in range(1, len(self.cycle) - 1):
    #         test = self.cycle.copy()
    #         test.insert(i, nextShortestLocation)
    #         currentCycleTestDistance = self.cycleDistance(test)
    #         if currentCycleTestDistance < currentBestCycle:
    #             currentBestCycle = currentCycleTestDistance
    #             bestCycle = test

    #     self.perms.append(bestCycle)
    #     # self.perms.append(self.cycle)
    


    def travelPerson(self):
        '''
        "Main" function for combining: reads data, creates permutations,
        visits each location, and find the best path/distance. Also can plot the traversal.
        '''
        self.readData()
        initialize = True
        self.firstLocation()
        while self.locs:
            self.findNextClosest()


        permutations = list(itertools.permutations(self.locs, self.dataCount))
        

        ## find the distance of each permutation location to location, aggregate, minimum
        ## store index of permulations list as value and aggregate as key?
        for i in range(0, len(permutations)):
            self.dists.append(self.calculateDistances(permutations[i]))

        if self.verbose:
            # self.perms = permutations
            self.plot, = self.ax.plot(range(self.dataCount),np.zeros(self.dataCount)*np.NaN, 'mediumspringgreen')
            anim = FuncAnimation(self.fig, self.plotter, frames=len(self.perms), repeat=False, interval=10)#, blit=True)
            plt.show()

        ## this

        print("⚶"*90)
        print("Shortest Distance: {:.2f}\nPath of Travel: {}".format(self.calculateDistances(self.cycle), str(self.cycle)))
        print("⚶"*90)

def inputParser():
    '''
    Simple command line input parser.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("locationCount", type=str)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


###############
if __name__ == "__main__":
    # parser = inputParser()
    # tsp = TSP(parser.locationCount, parser.verbose)


    t0 = time.time()
    tsp = TSP("30",True)# parser.verbose)
    tsp.travelPerson()
    t1 = time.time()
    print(t1-t0)

