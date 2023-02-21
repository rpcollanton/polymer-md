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
    clSize = [len(cl) for cl in cluster.cluster_keys]
    nCluster = len(cluster.cluster_keys)

    # find max length, initialize
    maxLength = max(clSize)
    avgRsq = np.zeros((maxLength))
    count = np.zeros((maxLength))

    # loop over clusters
    for numitr,cl in enumerate(cluster.cluster_keys):
        if not numitr % 100:
            print("{:d}/{:d}".format(numitr,nCluster))
        minkey = min(cl)
        maxkey = max(cl)
        idxrange = list(range(minkey,maxkey+1))
        distances = box.compute_all_distances(snapshot.particles.position[idxrange], snapshot.particles.position[idxrange])
        for i in idxrange:
            for j in range(minkey,i):
                avgRsq[i-j] += distances[i-minkey,j-minkey]**2
                count[i-j] += 1
    avgRsq[1:] = avgRsq[1:]/count[1:]
    return avgRsq
