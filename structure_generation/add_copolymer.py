import gsd.hoomd
import gsd.pygsd
import numpy as np
from polygen import chain_walk, wrap_coords

def read_snapshot_from_gsd(fname):

    f = gsd.pygsd.GSDFile(open(fname, 'rb'))
    t = gsd.hoomd.HOOMDTrajectory(f)

    return t[-1] # return the last snapshot/frame!

def generate_linearcoords(N, M, l, boxsize):

    # assume box is rectangular for now!
    if len(boxsize) > 3:
        boxsize = boxsize[0:3]
    
    PolymerList = []
    
    for idx_chain in range(M):
        # choose random starting position in box centered on origin
        pos0 = np.multiply(boxsize, (np.random.rand(1,3) - 0.5))
        chain_coords = pos0 + chain_walk(N,l)
        chain_coords = wrap_coords(chain_coords, boxsize)
        PolymerList.append(chain_coords)

    return PolymerList

def linear_bonds(blocktypes, blocklengths):
    
    Ntot = np.sum(blocklengths)
    
    # generate particle and bond types, which will be same for each polymer (with indices shifted
    types = []
    typeid = []
    
    for type,length in zip(blocktypes,blocklengths):
        if type not in types:
            types.append(type)
        
        typeid += [types.index(type)] * length

    bonds = [] 
    bondtypes = []
    bondtypeid = []
    for idx_mnr in range(Ntot-1):
        type1 = types[typeid[idx_mnr]]
        type2 = types[typeid[idx_mnr+1]]
        bondtype = "{:s}-{:s}".format(type1,type2)
        bonds.append([idx_mnr, idx_mnr + 1])

        if bondtype not in bondtypes:
            bondtypes.append(bondtype)

        bondtypeid.append(bondtypes.index(bondtype))

    return types, typeid, bondtypes, bondtypeid, bonds

def add_linearcopolymer(snapshot, blocktypes, blocklengths, M, l):

    box = snapshot.configuration.box[0:3]
    
    nBlocks = len(blocktypes)
    if len(blocklengths) != nBlocks:
        raise ValueError("Number of blocks inconsistent between block types and lengths.")

    Ntot = np.sum(blocklengths)

    copolymerCoords = generate_linearcoords(Ntot, M, l, box)
    cp_types, cp_typeid, cp_bondtypes, cp_bondtypeid, cp_bonds = linear_bonds(blocktypes, blocklengths)

    # Record number of particles
    Nprev = snapshot.particles.N
    Nadded = M*Ntot

    # Add new coordinates
    newpos = np.array(snapshot.particles.position)
    for i in range(M):
        newpos = np.append(newpos, copolymerCoords[i], axis=0)
    
    
    ### Add on type ids for particles
    newtypes = []
    newtypeid = []
    # first, the types already in the snapshot
    newtypes += snapshot.particles.types
    newtypeid += snapshot.particles.typeid.tolist()
    # next, the types from the copolymer
    for type in cp_types:
        if type not in newtypes:
            newtypes.append(type)
    # reassign type ids for the copolymer
    cp_typeid = [newtypes.index(cp_types[i]) for i in cp_typeid]
    newtypeid += (M * cp_typeid)

    ### Add on type ids for bonds
    newbondtypes = []
    newbondtypeid = []
    # first, the bond types already in the snapshot
    newbondtypes += snapshot.bonds.types
    newbondtypeid += snapshot.bonds.typeid.tolist()
    # next, the bond types from the copolymer
    for bondtype in cp_bondtypes:
        if bondtype not in newbondtypes:
            newbondtypes.append(bondtype)
    # reassign type ids for the copolymer
    cp_bondtypeid = [newbondtypes.index(cp_bondtypes[i]) for i in cp_bondtypeid]
    newbondtypeid += (M*cp_bondtypeid)

    newgroup = []
    # first, the bonds already recorded in snapshot
    newgroup += snapshot.bonds.group.tolist()
    # next, the ones from the copolymer
    for i in range(M):
        shift = Nprev + i*Ntot
        bonds_shifted = (np.array(cp_bonds) + shift).tolist()
        newgroup += bonds_shifted

    # store in new snapshot
    newsnapshot = gsd.hoomd.Snapshot()
    newsnapshot.configuration.box = snapshot.configuration.box
    newsnapshot.particles.N = Nprev + Nadded
    newsnapshot.particles.position = newpos
    newsnapshot.particles.types = newtypes
    newsnapshot.particles.typeid = newtypeid
    newsnapshot.bonds.N = len(newbondtypeid)
    newsnapshot.bonds.types = newbondtypes
    newsnapshot.bonds.typeid = newbondtypeid
    newsnapshot.bonds.group = newgroup

    return newsnapshot


fname = "/Users/ryancollanton/Desktop/N_0064_A_0512_B_0512.3_nvt.eps_1.50.gsd" 
blocktypes = ["A", "B", "A"]
blocklengths = [64, 64, 64]
M = 20
l = 1
snapshot = read_snapshot_from_gsd(fname)
newsnapshot = add_linearcopolymer(snapshot, blocktypes, blocklengths, M, l)

newf = "/Users/ryancollanton/Desktop/N_0064_A_0512_B_0512.ABA_20.init.gsd" 
with gsd.hoomd.open(name=newf, mode='wb') as f:
        f.append(newsnapshot)


