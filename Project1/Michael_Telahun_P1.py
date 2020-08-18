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
        # self.locations5File = self.files[4]

    def readData(self, locationsFile):
        with open(self.cwd + "\\" + locationsFile) as f:
            for i, row in enumerate(f):
                if i == self.DIMENSIONIndex:
                    self.dataCount = int(row.strip().split(" ")[1])
                if i >= self.DATAIndex:
                    values = row.strip().split(" ")
                    self.locs.append([float(values[1]), float(values[2])])
                    self.locations[str(values[1]+", "+values[2])] = values[0]#str(values[1]+","+values[2])
        # self.locs = np.array(self.locs)

    def distance(self, arr1, arr2):
        x2, x1 = arr2[0], arr1[0]
        y2, y1 = arr2[1], arr1[0]
        
        return math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2))
    
    def elucidianDistance(self, permutation, verbose=False):

        dist = 0.0
        ## go to all destinations
        for i in range(0, len(permutation)-1):
            ## plot the traversal here?
            # if verbose:
            #     print("show plot")
            dist += self.distance(permutation[i], permutation[i+1])
        ## go back to start
        dist += self.distance(permutation[len(permutation)-1], permutation[0])
        return dist
        

    

    # region: not used
    ## not used
    def elucidianDistance2(self, arr, verbose=False):
        arr1 = arr[0]
        arr2 = arr[1]
        x2, x1 = arr2[0], arr1[0]
        y2, y1 = arr2[1], arr1[0]

        if verbose:
            print("show plot")

        return math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2))
    ## not used
    def getShortestPathSet(self, permutations):
        shortestPathIdx = np.argsort(np.array(self.dists))[0]
        subset = permutations[shortestPathIdx]
        self.orderedPaths.append(subset)

        ## should find the next best distance containing either of the values in permutation
        ## then delete the array that is not in both 
        ## right now it just drops the first one
        np.delete(self.locs, np.where(self.locs==subset[0]), 0)
    ## not used
    def getValuesFromKeys(self, array):
        '''
        Used to return values from dict. Should be faster than searching an array?
        '''
        key = str(self.locs[0]).strip('[]').replace(" ", ",")
        return self.locations[key]
    ## not used
    def getValuesFromKey(self, key):
        '''
        Used to return values from dict. Should be faster than searching an array?
        '''
        array = []
        array.append(float(key.strip('[]').split(", ")[0]))
        array.append(float(key.strip('[]').split(", ")[1]))
        return array
    ## not used
    def getValuesFromKey2(self, key):
        '''
        Used to return values from dict. Should be faster than searching an array?
        '''
        array = []
        key = key.strip('[]').replace("][", "|").split("|")[0].split(", ")
        array.append(float(key[0]))
        array.append(float(key[1]))
        return array   
    ## not used
    def deleteFromDict(self, key):
        '''
        Delete the visited location from the dictionary.
        '''
        key = str(key).strip('[]')
        del self.locations[key]
    ## not used
    def travelPerson2(self):
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
    # endregion 

    def travelPerson(self):
        initialize = True
        # locations = self.locs
        
        # while len(self.locations) > 1:
        permutations = list(itertools.permutations(self.locs, self.dataCount))
        

        ## find the distance of each permutation location to location, aggregate, minimum
        ## store index of permulations list as value and aggregate as key?
        for i in range(0, len(permutations)):
            self.dists.append(self.elucidianDistance(permutations[i]))
            ## plot the traversal here?


        ## this
        shortestPathIdx1 = sorted(range(len(self.dists)), key=self.dists.__getitem__)[0]
        subset1 = permutations[shortestPathIdx1]
        print(subset1)


        ## or this
        # shortestPathIdx2 = np.argsort(np.array(self.dists))[0]
        # subset2 = permutations[shortestPathIdx2]




        ####### wasted?
        ## shortest overall distance (key in dict)
        # shortestDistanceKey = min(self.distances, key=self.distances.get)
        # ## index of shortest overall distance (value in dict)
        # shortestDistanceIdx = self.distances[shortestDistanceKey]
        # ## permutation w/ shortest overal distance (indexed at value in dict)
        # shortestPath = permutationsList[shortestDistanceIdx]



                    

tsp = TSP()
tsp.readData(tsp.files[4])
tsp.travelPerson()


plt.scatter(tsp.locs[:,0], tsp.locs[:,1])
plt.show()

plt.plot(tsp.locs[:,0], tsp.locs[:,1], 'o-')
plt.show()


print(tsp.locations)


