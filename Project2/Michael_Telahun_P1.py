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
        self.locationsFile = locationFile + "PointDFSBFS.tsp"
        # self.locationsFile = "Random" + locationFile + ".tsp"
        self.verbose = verbose

        ## project 2
        self.visitedDFS = set()
        self.visitedBFS = []
        self.queBFS = []
        self.map = {
            "1" : [2,3,4],
            "2" : [3],
            "3" : [4,5],
            "4" : [5,6,7],
            "5" : [7,8],
            "6" : [8],
            "7" : [9, 10],
            "8" : [9, 10,11],
            "9" : [11],
            "10" : [11],
            "11" : []
        }
        self.weights = {
        }
        self.fullPaths = []
        self.completePaths = set()

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

    def DFS(self, visited, location):
        if location not in visited:
            print(location)
            visited.add(location)
            for neighbor in self.map[str(location)]:
                self.DFS(visited, neighbor)


    def getFullPath(self, currentPaths):
        '''
        This is very bad. Gathers adjecent paths to solve for the "effiecent route"
        part of this question. Also plotting needs this.
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
            print(current, end= " ")
            
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

        self.BFS(self.visitedBFS, "1")
        bestDistanceBFS, bestPathBFS =  self.calculateDistancesBFS()

        # self.DFS(self.visitedDFS, "1")
        
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
        plt.show()

        print("Shortest Distance: {:.2f}\nPath of Travel: {}".format(bestDistanceBFS, str(bestPathBFS)))
        print("⚶"*90)

def inputParser():
    '''
    Simple command line input parser.
    '''
    parser = argparse.ArgumentParser()
    # parser.add_argument("locationCount", type=str)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


###############
if __name__ == "__main__":
    # parser = inputParser()
    # tsp = TSP("11", parser.verbose)
    
    tsp = TSP("11", True)

    # t0 = time.time()
    
    tsp.travelPerson()
    # t1 = time.time()
    print(t1-t0)

