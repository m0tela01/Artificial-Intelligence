import os
import math
import itertools

import numpy as np
import matplotlib.pyplot as plt


class TSP():
    def __init__(self):
        self.DIMENSIONIndex = 4
        self.DATAIndex = 7
        self.dataCount = 0
        self.locations = {}
        self.locs = []
        self.distances = {}
        self.dists = []
        self.orderedPaths = []
        self.cwd = os.path.dirname(os.path.abspath(__file__))
        self.files = [f for f in os.listdir(self.cwd) if f.endswith('.tsp')]
        
        # 4th file is Random5.tsp for tests
        self.locations5File = self.files[4]

    def readData(self):
        with open(self.cwd + "\\" + self.locations5File) as f:
            for i, row in enumerate(f):
                if i == self.DIMENSIONIndex:
                    self.dataCount = int(row.strip().split(" ")[1])
                if i >= self.DATAIndex:
                    values = row.strip().split(" ")
                    self.locs.append([float(values[1]), float(values[2])])
                    self.locations[str(values[1]+","+values[2])] = values[0]#str(values[1]+","+values[2])
        self.locs = np.array(self.locs)

    def elucidianDistance(self, arr):
        arr1 = arr[0]
        arr2 = arr[1]
        x2, x1 = arr2[0], arr1[0]
        y2, y1 = arr2[1], arr1[0]

        return math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2))
    
    def getShortestPathSet(self, permuations):
        shortestPathIdx = np.argsort(np.array(self.dists))[0]
        subset = permuations[shortestPathIdx]
        self.orderedPaths.append(subset)

        ## should find the next best distance containing either of the values in permutation
        ## then delete the array that is not in both 
        ## right now it just drops the first one

        np.delete(self.locs, np.where(self.locs==subset[0]), 0)

    def getValuesFromKeys(self, array):
        '''
        Used to return values from dict. Should be faster than searching an array?
        '''
        key = str(self.locs[0]).strip('[]').replace(" ", ",")
        return self.locations[key]

    def deleteFromDict(self, array):
        '''
        Used to return values from dict. Should be faster than searching an array?
        '''
        key = str(self.locs[0]).strip('[]').replace(" ", ",")
        del self.locations[key]
        

    def getShortestDistance(self, permutations):


    def orderPaths(self):
        locations = self.locs
        
        while len(self.locs) > 1:
            perumations = np.array(list(itertools.permutations(self.locs, 2)))
            for i in range(0, len(perumations)):
                self.dists.append(self.elucidianDistance(perumations[i]))
            shortestPathSet = self.getShortestPathSet(perumations)

            

        self.orderPaths.append(self.locs)
        print(self.orderPaths)





    # def splitData()
tsp = TSP()
tsp.readData()
tsp.orderPaths()


plt.scatter(tsp.locs[:,0], tsp.locs[:,1])
plt.show()

plt.plot(tsp.locs[:,0], tsp.locs[:,1], 'o-')
plt.show()


print(tsp.locations)


