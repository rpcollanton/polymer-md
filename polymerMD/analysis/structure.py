import freud
import numpy as np
from polymerMD.analysis.utility import wrap_coords

def getAllPairs(maxIdx, minIdx=0):

    pairs = []
    for i in range(minIdx,maxIdx+1):
        for j in range(minIdx,i):
            pairs.append([i,j])
    return pairs

def meanEndToEnd(coord, molecules, box, power=2):
    '''
    Args:
        coord (np.ndarray):             Nx3 array for the coordinates of N particles
        molecules (List[List[int]]):    list of indices of particles in each molecule
        box (freud.box.Box):            box used to compute distances
        power (float or int):           power to raise distances to inside average         
    Returns:
        avgDistToPower (float): average of end to end distances to inputted power
    '''

    # loop over molecules and identify indices of distances to compute
    # this way we only make one call to compute distances.. much faster!
    points1 = []
    points2 = []
    for mol in molecules:
        points1.append(min(mol))
        points2.append(max(mol))

    # use box object to compute distances
    distances = box.compute_distances(coord[points1], coord[points2])

    # average the distances
    distToPower = np.power(distances,power)
    avgDistToPower = np.mean(distToPower,axis=0)

    return avgDistToPower, distances

def meanRadiusGyration(coord, molecules, box, power=2):
    '''
    Args:
        coord (np.ndarray):             Nx3 array for the coordinates of N particles
        molecules (List[List[int]]):    list of indices of particles in each molecule
        box (freud.box.Box):            box used to compute distances
        power (float or int):           power to raise distances to inside average         
    Returns:
        avgRgToPower (float): average of end to end distances to inputted power
    '''

    # loop over molecules and identify indices of distances to compute
    # this way we only make one call to compute distances.. much faster!
    points1 = []
    points2 = []
    pos = coord
    for mol in molecules:
        # unwrap coordinates in molecule to be continuous based on first particle
        r0 = pos[mol[0],:]
        pos[mol,:] = r0 + wrap_coords(pos[mol,:] - r0, box.L)
        for i in mol:
            points1.append(i)
            points2.append(pos.shape[0]) # the eventual location of the com
        com = np.mean(pos[mol,:],axis=0).reshape(1,-1)
        pos = np.append(pos,com,axis=0)
        
    # use box object to compute distances from com
    distances = box.compute_distances(pos[points1], pos[points2])

    # square and average the distances
    distancesSquared = np.square(distances)
    RgSquared = np.array([np.mean(
        distancesSquared[[points1.index(i) for i in mol]], axis=0
    ) for mol in molecules])
    avgRgToPower = np.mean(np.power(RgSquared,power/2),axis=0) # already raised to 2nd power

    return avgRgToPower, RgSquared


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

def meanSqDistanceFromJunction(coord, blocks, junctions, box):
    '''
    Args:
        coord (np.ndarray):             Nx3 array for the coordinates of N particles
        blocks (List[List[int]]):       list of indices of particles in each block, ordered from junction to "end" (might be middle for a midblock)
        junctions (np.ndarray):         Nbx3 array with the coordinate of the terminal junction for each of the Nb blocks or block segments
        box (freud.box.Box):            
    Returns:
        n (np.ndarray):         1 x max(molecule lengths)-1 containing the corresponding distance from the block junction. 
                                starts at "0" which is actually a half segment away from the juntion
        avgRsq (np.ndarray):    1 x max(molecule lengths)-1 array containing average internal distances 
                                along the chains. Entry i corresponds with segments of length i+0.5
    
    NOTE that all inputted blocks should be exactly the same. 
    This is because results will be averaged together and returned in a single array

    '''
    # find block length and initialize arrays
    blockSize = len(blocks[0])
    for block in blocks:
        if len(block) != blockSize:
            raise ValueError("Inputted blocks are not the same length and thus probably shouldn't be treated as identical.")
    coordWithJunction = coord
    points1 = []
    points2 = []
    segmentlength = []
    for idx_block,block in enumerate(blocks):
        coordWithJunction = np.append(coordWithJunction, junctions[[idx_block],:],axis=0) # can do this cleaner for sure
        idx_junction = coordWithJunction.shape[0]-1 #index of last coordinate, which is the junction that was just appended
        
        # loop over coordinates in the block
        idx_next_to_junction = block[0]
        for idx_segment in block:
            points1.append(idx_junction)
            points2.append(idx_segment)
            segmentlength.append(abs(idx_segment-idx_next_to_junction) + 0.5) # corrected with +0.5 because junction is 0.5 away from last block

    # use box object to compute distances
    distances = box.compute_distances(coordWithJunction[points1], coordWithJunction[points2])        

    # sum up the squared segment distances
    distancesSquared = np.square(distances)
    avgRsq = np.zeros(blockSize)
    count = np.zeros(blockSize)
    for dsq, length in zip(distancesSquared,segmentlength):
        idx = int(length-0.5)
        avgRsq[idx] += dsq
        count[idx] += 1

    # compute average
    avgRsq = avgRsq/count
    n = np.arange(0,len(avgRsq))+0.5

    return avgRsq, n