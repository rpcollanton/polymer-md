import numpy as np
import systemspec
import gsd.hoomd

def wrap_coords(coords,boxsize):

    # wrap coordinates into a rectangular box with side lengths given by boxsize

    dims = len(boxsize)
    if dims > 3:
        dims = 3
        boxsize = boxsize[0:3]
    
    wrapped = np.zeros_like(coords)
    for i in range(dims):
        wrapped[:,i] = coords[:,i] - boxsize[i] * np.rint(coords[:,i]/boxsize[i])

    return wrapped 

def mc_chain_walk(N, l):

    PolymerCoords = np.zeros((N,3))
    twostepcutoff = 1.02/0.97 * l
    
    # "Monte carlo" loop where a random step is taken and only accepted if the distance isn't too far
    idx_mnr = 1
    while idx_mnr < N:
        randstep = np.random.rand(1,3) - 0.5
        newcoord = PolymerCoords[idx_mnr-1,:] + randstep * l/np.linalg.norm(randstep)

        if idx_mnr >= 2:
            twostepdist = np.linalg.norm(newcoord - PolymerCoords[idx_mnr-2])
            if twostepdist < twostepcutoff:
                continue
        # This "update"/"acceptance" code is only reached if twostepdist is greater 
        # than the cut off or if this is the first step being taken along the chain
        PolymerCoords[idx_mnr,:] = newcoord
        idx_mnr += 1

    return PolymerCoords

def connect_chains(chains, l):
    
    # monte carlo cutoff 
    twostepcutoff = 1.02/0.97 * l

    # chains should be a list of numpy arrays of coordinates of polymers
    chain = chains[0]
    
    for i in range(1,len(chains)):
        chainend = chain[-1,:]
        # get a valid starting point for the next block
        while True:
            randstep = np.random.rand(1,3) - 0.5
            chainstart = chainend + l * randstep/np.linalg.norm(randstep)
            twostepdist = np.linalg.norm(chainstart - chain[-2,:])
            if twostepdist > twostepcutoff:
                break
        addedchaincoords = chains[i] - chains[i][0,:] + chainstart
        chain.append(chain, addedchaincoords, axis = 0)

    return chain

def walk_linearPolymer(polymer):

    # walk each block of the polymer
    blockcoords = []
    for block in polymer:
        blockcoords.append(mc_chain_walk(block.length, block.monomer.l))

    # connect them
    l = polymer.block[0].monomer.l
    chain = connect_chains(blockcoords, l)

    return chain

def walkComponent(component):
    # generate a set of random walk coordinates for this component that are the right length
    num = component.N
    coordlist = []
    for i in range(num):
        if isinstance(component.species, systemspec.LinearPolymerSpec):
            coordlist.append(walk_linearPolymer(component.species))  
    
    # coordlist is a list of numpy arrays
    return coordlist

def placeComponent(coordlist, region, COM=True):
    # take a list of random walk coordinates and randomly place COM of each one within the specified region
    # for now, region is just a rectangular prism centered on the origin of a certain size. In future, region should be able to be
    # any geometric region with some common descriptor/interface (i.e.: a sphere or cylinder!)
    # if COM is true, place center of mass of chain at random point. Otherwise, place first point
    comlist = [np.average(coord,axis=0) for coord in coordlist]
    randcomlist = np.multiply(region, np.random.rand(len(comlist),3)-0.5)
    newcoordlist = []
    for i,coord in enumerate(coordlist):
        newcoord = coord - comlist[i] + randcomlist[i,:]
        newcoord = wrap_coords(newcoord, region)
        newcoordlist.append(newcoord)

    return newcoordlist

def systemCoordsRandom(system):

    box = system.box[0:3]
    
    syscoords = np.zeros(system.numparticles,3)

    totaladded = 0        
    for component in system.components:
        # a list of coordinates for each instance of this component
        coordlist = walkComponent(component)
        # place center of mass of each chain at a random point
        coordlist = placeComponent(coordlist, box, COM=True)
        # sequentially add each 
        for coord in coordlist:
            nchain = coord.shape[0]
            syscoords[totaladded, totaladded+nchain,:] = coord
            totaladded += nchain

    return



def getParticleTypes(system):

    types = system.monomerlabels
    N = system.numparticles # number of particles in system
    typeid = [system.particleType(i) for i in range(N)]

    return types, typeid

def build_snapshot(system):

    # get system coords 
    pos = systemCoordsRandom(system)

    # get particle indices, types, and type ids
    types, typeid = getParticleTypes(system)

    # get bond indices

    # get bond types and type ids

    # 

    return