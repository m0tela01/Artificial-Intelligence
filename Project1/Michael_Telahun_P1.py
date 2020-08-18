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
        self.pathOfTravel = []
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
                    self.locations[str(values[1]+", "+values[2])] = values[0]#str(values[1]+","+values[2])
        # self.locs = np.array(self.locs)

    def elucidianDistance(self, arr):
        arr1 = arr[0]
        arr2 = arr[1]
        x2, x1 = arr2[0], arr1[0]
        y2, y1 = arr2[1], arr1[0]

        return math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2))
    
    # def getShortestPathSet(self, permutations):
    #     shortestPathIdx = np.argsort(np.array(self.dists))[0]
    #     subset = permutations[shortestPathIdx]
    #     self.orderedPaths.append(subset)

    #     ## should find the next best distance containing either of the values in permutation
    #     ## then delete the array that is not in both 
    #     ## right now it just drops the first one
    #     np.delete(self.locs, np.where(self.locs==subset[0]), 0)

    # def getValuesFromKeys(self, array):
    #     '''
    #     Used to return values from dict. Should be faster than searching an array?
    #     '''
    #     key = str(self.locs[0]).strip('[]').replace(" ", ",")
        # return self.locations[key]

    def getValuesFromKey(self, key):
        '''
        Used to return values from dict. Should be faster than searching an array?
        '''
        array = []
        array.append(float(key.strip('[]').split(", ")[0]))
        array.append(float(key.strip('[]').split(", ")[1]))
        return array

    def getValuesFromKey2(self, key):
        '''
        Used to return values from dict. Should be faster than searching an array?
        '''
        array = []
        key = key.strip('[]').replace("][", "|").split("|")[0].split(", ")
        array.append(float(key[0]))
        array.append(float(key[1]))
        return array   

    def deleteFromDict(self, key):
        '''
        Delete the visited location from the dictionary.
        '''
        key = str(key).strip('[]')
        del self.locations[key]
        

    # def getShortestDistance(self, permutations):


    def travelPerson(self):
        initialize = True
        # locations = self.locs
        
        while len(self.locations) > 1:
            permutationsList = list(itertools.permutations(self.locs, 2))
            a = list(itertools.permutations(self.locations, 2))
            permutations = {repr(a)+repr(b):b for a,b in permutationsList}

            if initialize:
                ## arbitrarily pick first cooridinate as start location
                start = list(permutations.keys())[0]
                arr1, arr2 = self.getValuesFromKey2(start), list(permutations.values())[0]
                
                ## calculate distances
                self.dists.append(self.elucidianDistance([arr1, arr2]))


                ## store the arrays in order of travel
                self.pathOfTravel.append(arr1)
                self.pathOfTravel.append(arr2)
                ## remove the start location from the dictonary
                self.deleteFromDict(arr1)
                ## clear and recreate locs for new permutations
                self.locs = []
                [self.locs.append(self.getValuesFromKey(x)) for x in list(self.locations.keys())]
                initialize = False
            
            else:

                for i in range(0, len(permutations)):
                    self.dists.append(self.elucidianDistance(permutations[i]))
                shortestPathSet = self.getShortestPathSet(permutations)

                

        self.pathOfTravel.append(self.locs)
        print(self.pathOfTravel)





    # def splitData()
tsp = TSP()
tsp.readData()
tsp.travelPerson()


plt.scatter(tsp.locs[:,0], tsp.locs[:,1])
plt.show()

plt.plot(tsp.locs[:,0], tsp.locs[:,1], 'o-')
plt.show()


print(tsp.locations)


