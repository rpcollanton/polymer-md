import freud
import numpy as np

def getAllPairs(maxIdx, minIdx=0):

    pairs = []
    for i in range(minIdx,maxIdx+1):
        for j in range(minIdx,i):
            pairs.append([i,j])
    return pairs


def meanSqInternalDist(coord, molecules, box):
    '''
    Args:
        coord (np.ndarray):             Nx3 array for the coordinates of N particles
        molecules (List[List[int]]):    list of indices of particles in each molecule
        box (freud.box.Box):            
    Returns:
        n (np.ndarray):         1 x max(molecule lengths)-1 containing the corresponding segment lengths
        avgRsq (np.ndarray):    1 x max(molecule lengths)-1 array containing average internal distances 
                                along the chains. Entry i corresponds with segments of length i+2 
    '''

    # find max molecule length and initialize
    molSize = [len(mol) for mol in molecules]
    maxLength = max(molSize)
    avgRsq = np.zeros((maxLength))
    count = np.zeros((maxLength))

    # loop over molecules and identify indices of distances to compute
    # this way we only make one call to compute distances.. much faster!
    points1 = []
    points2 = []
    for mol in molecules:
        minidx = min(mol)
        maxidx = max(mol)
        idxrange = list(range(minidx,maxidx+1))
        for i in idxrange:
            for j in range(minidx,i):
                points1.append(i)
                points2.append(j)

    # use box object to compute distances
    distances = box.compute_distances(coord[points1], coord[points2])

    # average squared segment distances
    distancesSquared = np.square(distances)
    for dsq,(i,j) in zip(distancesSquared,zip(points1,points2)):
        avgRsq[i-j] += dsq
        count[i-j] += 1
    
    # compute average, and remove the 0 element
    avgRsq[1:] = avgRsq[1:]/count[1:]
    avgRsq = avgRsq[1:]
    n = np.arange(2,len(avgRsq)+2)

    return n, avgRsq