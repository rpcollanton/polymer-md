import freud
import numpy as np

def getBondedClusters(snapshot):
    cluster = freud.cluster.Cluster()
    # get bond indices
    idx_querypts = []
    idx_pts = []
    for bond in snapshot.bonds.group:
        idx_querypts.append(bond[0])
        idx_pts.append(bond[1])

    box = freud.box.Box.from_box(snapshot.configuration.box)
    dist = box.compute_distances(snapshot.particles.position[idx_querypts], 
                                    snapshot.particles.position[idx_pts])
    N = snapshot.particles.N
    bondedneighbors = freud.locality.NeighborList.from_arrays(N, N, idx_querypts, idx_pts, dist)
    cluster.compute(snapshot,neighbors=bondedneighbors)

    return cluster

def getAllPairs(maxIdx, minIdx=0):

    pairs = []
    for i in range(minIdx,maxIdx+1):
        for j in range(minIdx,i):
            pairs.append([i,j])
    return pairs


def meanSqInternalDist(snapshot):

    '''
    Args:
        snapshot (gsd.snapshot/hoomd.gsd.snapshot): system state containing bonds and positions

    Returns:
        avgRsq (np.ndarray):    1 x max(polymer lengths) array containing average internal distances 
                                along the chains. Entry i corresponds with segments of length i 
    '''
    
    # get box information
    box = freud.box.Box.from_box(snapshot.configuration.box)
    
    # get cluster
    cluster = getBondedClusters(snapshot)
    nCluster = len(cluster.cluster_keys)
    
    # find max length, initialize
    avgRsq = np.zeros((nCluster,1))
    count = np.zeros((nCluster,1))

    # loop over clusters
    for cl in cluster.cluster_keys:
        minkey = min(cl)
        maxkey = max(cl)
        idxrange = list(range(minkey,maxkey+1))
        distances = box.compute_all_distances(snapshot.particles.position[idxrange], snapshot.particles.position[idxrange])
        for i in idxrange:
            for j in range(minkey,i):
                avgRsq[i-j] += distances[i,j]**2
                count[i-j] += 1
        
    avgRsq = avgRsq/count

    return avgRsq
        

                

    

    # assume indices correspond to relative position on chain.. could change with bridging technique! (or for bridging technique just switch all positions)
    
    # store separation of pair in avgRsq[i-j] for i > j
    # increment count[i-j] by 1

    # divide avgRsq[i] by count[i] to average
    
    # return avgRsq

    return