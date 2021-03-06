import os
import re
import math
import time
import argparse
import itertools
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('dark_background')

class TSP():
    '''
    Traveling sales person class.
    '''
    def __init__(self, mode, verbose=False):
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
        self.locationsFile = "11PointDFSBFS.tsp"#locationFile + "PointDFSBFS.tsp"
        # self.locationsFile = "Random" + locationFile + ".tsp"
        self.verbose = verbose

        ## project 2
        self.mode = mode
        self.visitedDFS = []#set()
        self.visitedBFS = []
        self.queBFS = []
        self.map = {
            "1" : [2, 3, 4],
            "2" : [3],
            "3" : [4, 5],
            "4" : [5, 6, 7],
            "5" : [7, 8],
            "6" : [8],
            "7" : [9, 10],
            "8" : [9, 10, 11],
            "9" : [11],
            "10" : [11],
            "11" : []
        }
        self.weights = {
        }
        self.fullPaths = []
        self.completePaths = set()

        self.pathDFS = [1]
        self.fullPathDFS = []
        self.currentPathDFS = []
        self.initalDFS = True
        self.shouldExitDFS = False

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
    
    def calculateWeights(self):
        '''
        Calculate the input weights for the adjacent neighbors of each node.
        '''
        for key, adjacents in self.map.items():
            for adjacent in adjacents:
                distance = self.elucidianDistance(self.locs[int(key)-1], self.locs[int(adjacent)-1])
                self.weights[str(key)+"-"+str(adjacent)] = distance

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

    
    def dfsSearch(self, test, countBack, location, currentTest):
        '''
        Another attempt at trying to complete the backtracking method.
        This function was supposed to be entirely based off of depth and 
        be much more effiencet and straightforward but it did not compute correctly.
        '''
        depth = 0
        newWeight = 0
        newDepth = True
        for l_idx in reversed(range(test[len(test)-1])):
            back = test[len(test)-2]
            while l_idx != back:
                l_idx -= 1

            if newDepth:
                depth += 1
                newDepth = True

            for w_idx in range(len(self.map[str(l_idx)])):
                weightAtLocation = self.map[str(w_idx)]
                if weightAtLocation != 11:
                    newWeight = weightAtLocation
                    
                    newDepth = False
                    depth = 0




    def goBack(self, test, countBack, location, currentTest):
        '''
        This is an attempt at trying to complete the backtracking method.
        If there was an update to the location remove the most
        recent subpaths that are completed then add the connection
        between the current location and the most recent location.
        '''

        wentBack = False
        isFirst = True
        isBreak = False
        currentSubSet = []
        for l_idx in reversed(range(self.pathDFS[len(self.pathDFS)-1])):
            if isBreak:
                break
            if l_idx+1 in self.pathDFS:
                ## remove a location if all adj are visited
                if l_idx+1 != test[-1] and not isFirst:
                    # del test[-1]
                    del currentTest[-1]
                currentLocations = self.map[str(l_idx+1)]
                countBack+=1
                # if wentBack:

                # for w_idx in range(len(self.map[str(location)])):
                for w_idx in range(len(self.map[str(l_idx+1)])):
                    currentWeight = currentLocations[w_idx]
                    currentTest.append(currentWeight)
                    if test != currentTest:
                        # neighbor = currentWeight
                        location = currentWeight
                        isBreak = True
                        break
                    else:
                        del currentTest[-1]
                        if isFirst:
                            del test[-1]
                            isFirst = False
                        wentBack = True


        if countBack > 0:
            skip = True
            for i in range(countBack+1):
                if skip:
                    del self.pathDFS[-1]
                    skip = False
                del self.currentPathDFS[-1]

            last = self.pathDFS[len(self.pathDFS)-1]
            self.pathDFS.append(location)
            self.currentPathDFS.append(str(last)+"-"+str(location))
        
        return location

    def DFS(self, visited, location):
        '''
        This is the core of the DFS search that contains the skelton of the 
        algorithm. It should have a way for finding the sub path based on a 
        backtracking method. The implemention of the back tracking is needed to
        consider all the possible visited sub paths on a given "parent" complete
        path.
        '''
        
        while not self.shouldExitDFS:
            print(location)
            visited.append(str(location))

            for neighbor in self.map[str(location)]:
                self.pathDFS.append(neighbor)
                self.currentPathDFS.append(str(location)+'-'+ str(neighbor))
                self.fullPaths.append(self.currentPathDFS.copy())

                if str(neighbor) == "11":
                    self.completePaths.append(self.currentPathDFS.copy())
                    test = self.pathDFS.copy()
                    del self.pathDFS[-1]  

                    countBack = -1
                    currentTest = self.pathDFS.copy()

                    neighbor = self.goBack(test, countBack, location, currentTest)

                    if test == currentTest:
                        self.shouldExitDFS = True
                if neighbor == 11:
                    self.shouldExitDFS = True
                self.DFS(visited, neighbor)


    def getFullPath(self, currentPaths):
        '''
        This is feels very bad. 
        Gathers adjecent paths to solve for the "effiecent route"
        part of this question. Also used to create the pairing for the weights.
        Combines or extends all paths to represent the location to be traveled 
        The idea is to creat this weights array: [1-3, 3-5, 5-8, 8-11] to test for shortest distance.
        '''
        fullPaths = self.fullPaths.copy()
        for path in currentPaths:
            for partialPaths in self.fullPaths:
                for partialPath in partialPaths:
                    splitPaths = partialPath.split("-")
                    if splitPaths[len(splitPaths)-1] == path.split("-")[0]:
                        self.fullPaths.append([partialPath+"-"+path])
                        if path.split("-")[len(path.split("-"))-1] == "11":
                            self.completePaths.add(partialPath+"-"+path)

    def calculateDistancesBFS(self):
        '''
        Mostly due to the format of the generated paths this function converts the
        string representation of the complete paths to be hashed by the graph or map
        dictionary containing the weights and distance measures from each location to
        its adjacent neighbors.
        '''
        best = 999999
        resultPath = []

        ## we want the bad version of self.fullpaths for plotting
        ## instead of lines 134 and 135 could use line below
        ## paths= [list(x) for x in set(tuple(x) for x in self.fullPaths)]
        for completePath in self.completePaths:
            adjList = re.findall("[^-]+-[^-]+", completePath)
            dist = 0.0
            for adj in adjList:
                dist += self.weights[adj]
            if dist < best:
                best = dist
                resultPath = adjList
        return best, resultPath



    def BFS(self, visited, location):
        '''
        Simple shell for the BFS solution. It contains the fundamental aspect of the 
        BFS search algorithm. It uses the function getfullpath to compute the set of
        subsets for based on the current node and its adjacent values.
        '''
        visited.append(location)
        self.queBFS.append(location)
        paths = []
        count = 0
        while self.queBFS:
            if count == 1:
                self.fullPaths.append(paths)
            elif count > 1:
                self.getFullPath(paths)
            else:
                pass
            paths = []

            current = self.queBFS.pop(0)
            # path.append(current)
            # print(current, end= " ")
            
            for neighbor in self.map[str(current)]:
                paths.append(str(current)+"-"+str(neighbor))
                if neighbor not in visited:
                    visited.append(str(neighbor))
                    self.queBFS.append(neighbor)

            count+=1

    def getPlotData(self):
        initial = True
        result = []
        for fullPath in self.fullPaths:
            if initial:
                for i in range(len(fullPath)):
                    path = self.fullPaths[0]
                    indexes = list(set([int(s) for s in path[i].split("-") if s.isdigit()]))
                    result.append([self.locs[i-1] for i in indexes])
                initial = False
            else:
                path = fullPath[0]
                # sorted(list(map(int, set(path.replace("-", ",").split(",")))))

                indexes = list(set([int(s) for s in path.split("-") if s.isdigit()]))
                result.append([self.locs[i-1] for i in indexes])
        return result


    def travelPerson(self):
        '''
        "Main" function for combining: reads data, creates permutations,
        visits each location, and find the best path/distance. Also can plot the traversal.
        '''
        self.readData()

        self.calculateWeights()
        
        if self.mode == "bfs":
            self.BFS(self.visitedBFS, "1")
            bestDistanceBFS, bestPathBFS =  self.calculateDistancesBFS()
        else:
            self.DFS(self.visitedDFS, "1")
            bestDistanceDFS, bestPathDFS =  self.calculateDistancesBFS()
        
        # permutations = list(itertools.permutations(self.locs, self.dataCount))
        

        ## find the distance of each permutation location to location, aggregate, minimum
        ## store index of permulations list as value and aggregate as key?
        # for i in range(0, len(permutations)):
        #     self.dists.append(self.calculateDistances(permutations[i]))

        if self.verbose:
            self.perms = self.getPlotData()
            self.plot, = self.ax.plot(range(self.dataCount),np.zeros(self.dataCount)*np.NaN, 'mediumspringgreen')
            anim = FuncAnimation(self.fig, self.plotter, frames=len(self.perms), repeat=False, interval=10)#, blit=True)
            plt.show()

        ## this
        # shortestPathIdx1 = sorted(range(len(self.dists)), key=self.dists.__getitem__)[0]
        # subset1 = permutations[shortestPathIdx1]
        print("⚶"*90)
        a = list(set([int(s) for s in '-'.join(bestPathBFS).split("-") if s.isdigit()]))
        actualPath = [self.locs[i-1] for i in a]
        plt.plot([x[0] for x in actualPath],[x[1] for x in actualPath])
        # plt.show()

        print("Shortest Distance: {:.2f}\nPath of Travel: {}\nActual Path: {}".format(bestDistanceBFS, str(bestPathBFS), str(actualPath)))
        print("⚶"*90)


def inputParser():
    '''
    Simple command line input parser.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("modeInput", type=str)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


###############
if __name__ == "__main__":
    # parser = inputParser()
    
    # tsp = TSP("11", parser.verbose)
    
    # tsp = TSP(parser.modeInput, parser.verbose)
    tsp = TSP("bfs",False)

    t0 = time.time()
    
    tsp.travelPerson()
    t1 = time.time()
    print(t1-t0)

