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
        pos0 = np.multiply(boxsize * (np.random.rand(1,3) - 0.5))
        chain_coords = pos0 + chain_walk(N,l)
        chain_coords = wrap_coords(chain_coords, boxsize)
        PolymerList.append(chain_coords)

    return PolymerList

def bond_particle_types(blocktypes, blocklengths):

    nBlocks = len(blocktypes)
    if len(blocklengths) != nBlocks:
        raise ValueError("Number of blocks inconsistent between block types and lengths.")
    
    Ntot = np.sum(blocklengths)
    
    # generate particle and bond types, which will be same for each polymer (with indices shifted
    types = []
    typeidxs = []
    
    for type,length in zip(blocktypes,blocklengths):
        if type not in types:
            types.append(type)
        
        typeidxs += [types.index(type)] * length

    bonds = [] 
    bondtypes = []
    bondtypeidxs = []
    for idx_mnr in range(Ntot-1):
        type1 = types[typeidxs[idx_mnr]]
        type2 = types[typeidxs[idx_mnr+1]]
        bondtype = "{%s}-{%s}".format(type1,type2)
        bonds.append([idx_mnr, idx_mnr + 1])

        if bondtype not in bondtypes:
            bondtypes.append(bondtype)

        bondtypeidxs.append(bondtypes.index(bondtype))

    return types, typeidxs, bondtypes, bondtypeidxs, bonds



