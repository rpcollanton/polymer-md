def meanSqInternalDist(snapshot):

    '''
    Args:
        snapshot (gsd.snapshot/hoomd.gsd.snapshot): system state containing bonds and positions

    Returns:
        avgRsq (np.ndarray):    1 x max(polymer lengths) array containing average internal distances 
                                along the chains. Entry i corresponds with segments of length i 
    '''
    
    # get cluster

    # find max length, initialize
    #avgRsq = np.zeros
    #count = np.zeros

    # loop over clusters

    # loop over pairs

    # assume indices correspond to relative position on chain.. could change with bridging technique! (or for bridging technique just switch all positions)
    
    # store separation of pair in avgRsq[i-j] for i > j
    # increment count[i-j] by 1

    # divide avgRsq[i] by count[i] to average
    
    # return avgRsq

    return