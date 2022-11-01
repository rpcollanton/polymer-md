import numpy as np
from . import systemspec
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
        combinedchaincoords = chains[i] - chains[i][0,:] + chainstart
        chain = np.append(chain, combinedchaincoords, axis = 0)

    return chain

def walk_linearPolymer(polymer):

    # walk each block of the polymer
    blockcoords = []
    for block in polymer.blocks:
        blockcoords.append(mc_chain_walk(block.length, block.monomer.l))

    # connect them
    # NOTE: this is an escapist solution. 
    # I need to find a way to account for different bond lengths l cleanly
    l = polymer.blocks[0].monomer.l
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

def placeComponent(coordlist, region, regioncenter=[0,0,0], COM=True):
    # take a list of random walk coordinates and randomly place COM of each one within the specified region
    # for now, region is just a rectangular prism centered on the origin of a certain size. In future, region should be able to be
    # any geometric region with some common descriptor/interface (i.e.: a sphere or cylinder!)
    # if COM is true, place center of mass of chain at random point. Otherwise, place first point
    comlist = [np.average(coord,axis=0) for coord in coordlist]
    randcomlist = np.multiply(region, np.random.rand(len(comlist),3)-0.5) + np.array(regioncenter)
    newcoordlist = []
    for i,coord in enumerate(coordlist):
        newcoord = coord - comlist[i] + randcomlist[i,:]
        newcoordlist.append(newcoord)

    return newcoordlist

def systemCoordsRandom(system):

    box = system.box[0:3]
    
    syscoords = np.zeros((system.numparticles,3))

    totaladded = 0        
    for component in system.components:
        # a list of coordinates for each instance of this component
        coordlist = walkComponent(component)
        # place center of mass of each chain at a random point
        coordlist = placeComponent(coordlist, box, COM=True)
        # sequentially add each 
        for coord in coordlist:
            nchain = coord.shape[0]
            syscoords[totaladded:totaladded+nchain,:] = coord
            totaladded += nchain

    syscoords = wrap_coords(syscoords, box)
    return syscoords

def systemCoordsBoxRegions(system, regions, regioncenters):

    # regions is a list of regions with length equal to the number of components
    # the intention is that these should be used to seed components as phase-separated in advance
     
    box = system.box[0:3]
    
    syscoords = np.zeros((system.numparticles,3))

    totaladded = 0        
    for i,component in enumerate(system.components):
        # a list of coordinates for each instance of this component
        coordlist = walkComponent(component)
        # place center of mass of each chain at a random point
        coordlist = placeComponent(coordlist, regions[i], regioncenters[i], COM=True)
        # sequentially add each 
        for coord in coordlist:
            nchain = coord.shape[0]
            syscoords[totaladded:totaladded+nchain,:] = coord
            totaladded += nchain

    syscoords = wrap_coords(syscoords, box)
    return syscoords

def getParticleTypes(system):

    types = system.monomerlabels
    N = system.numparticles # number of particles in system
    alltypes = system.particleTypes()
    typeid = [types.index(label) for label in alltypes]

    return types, typeid

def getBondTypes(system):
    bonds, allbondtypes = system.bonds()
    bondtypes = list(set(allbondtypes))
    bondtypeid = [bondtypes.index(bondlabel) for bondlabel in allbondtypes]

    return bonds, bondtypes, bondtypeid

def build_snapshot(system, type='random', regions=[], regioncenters=[]):

    # get system box size, total number of particles, 
    box = system.box
    N = system.numparticles

    # get system coords 
    if type == 'random':
        pos = systemCoordsRandom(system)
    elif type == 'boxregions':
        if len(regions)==0 or len(regioncenters)==0:
            raise ValueError("No regions specified.")
        pos = systemCoordsBoxRegions(system, regions, regioncenters)

    # get particle indices, types, and type ids
    types, typeid = getParticleTypes(system)

    # get bond indices, types, and type ids
    bondgroup, bondtypes, bondtypeid = getBondTypes(system)
    nBonds = len(bondgroup)

    # generate snapshot!!
    snapshot = gsd.hoomd.Snapshot()
    snapshot.configuration.box = box
    snapshot.particles.N = N
    snapshot.particles.position = pos
    snapshot.particles.types = types
    snapshot.particles.typeid = typeid
    snapshot.bonds.N = nBonds
    snapshot.bonds.types = bondtypes
    snapshot.bonds.typeid = bondtypeid
    snapshot.bonds.group = bondgroup

    debugging = True
    if debugging:
        print("Number of particles:             N = {:d}".format(N))
        print("Number of positions:             pos.shape[0] = {:d}".format(pos.shape[0]))
        print("Number of particle types:        len(types) = {:d}".format(len(types)))
        print("Number of particle type ids:     len(typeid) = {:d}".format(len(typeid)))
        print("Number of bonds:                 nBonds = {:d}".format(nBonds))
        print("Number of bond types:            len(bondtypes) = {:d}".format(len(bondtypes)))
        print("Number of bond type ids:         len(bondtypeid) = {:d}".format(len(bondtypeid)))
        print("Number of bond groups:           len(bondgroup) = {:d}".format(len(bondgroup)))
    return snapshot