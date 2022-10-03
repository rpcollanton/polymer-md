import numpy as np
import gsd.hoomd

def wrap_coords(coords,boxsize):

    # wrap coordinates into a rectangular box with side lengths given by boxsize

    dims = len(boxsize)
    wrapped = np.zeros_like(coords)
    for i in range(dims):
        wrapped[:,i] = coords[:,i] - boxsize[i] * np.rint(coords[:,i]/boxsize[i])

    return wrapped 

def chain_walk(N, l):

    PolymerCoords = np.zeros((N,3))
    stepcutoff = 1.02/0.97 * l
    
    # "Monte carlo" loop where a random step is taken and only accepted if the distance isn't too far
    idx_mnr = 1
    while idx_mnr < N:
        randstep = np.random.rand(1,3) - 0.5
        newcoord = PolymerCoords[idx_mnr-1,:] + randstep * l/np.linalg.norm(randstep)

        if idx_mnr >= 2:
            twostepdist = np.linalg.norm(newcoord - PolymerCoords[idx_mnr-2])
            if twostepdist < stepcutoff:
                continue
        # This "update"/"acceptance" code is only reached if twostepdist is greater 
        # than the cut off or if this is the first step being taken along the chain
        PolymerCoords[idx_mnr,:] = newcoord
        idx_mnr += 1

    return PolymerCoords

def check_stretching(PolymerList, lp):

    # How to do this??? 

    M = len(PolymerList)
    N = np.size(PolymerList[0], axis=0)
    l = np.linalg.norm(PolymerList[0][1,:] - PolymerList[0][0,:])
    
    avgh2 = 0
    projections = np.zeros(M)
    for i in range(M):
        l0 = PolymerList[i][1,:] - PolymerList[i][0,:]
        R = PolymerList[i][N-1,:] - PolymerList[i][0,:]
        projections[i] = 1/l * np.dot(l0,R)
        avgh2 += np.dot(R,R)
    
    avgh2 /= M
    calc_lp = np.mean(projections)

    calc_lp = 1/(N*l) * avgh2 + l/2
    return calc_lp

def kremer_grest_mc(N, M, rho, l):

    V = N*M/rho
    L = V**(1/3)
    boxsize = [L,L,L]
    
    # for each chain
    PolymerList = []
    for idx_chain in range(M):
        # choose random starting position in box centered on origin
        pos0 = L * (np.random.rand(1,3) - 0.5)
        chain_coords = pos0 + chain_walk(N,l)
        chain_coords = wrap_coords(chain_coords, boxsize)
        PolymerList.append(chain_coords)
        #print("Chain walk {} complete.".format(idx_chain))
    

    return PolymerList, L

def mc_phase_separated_AB(N_A, M_A, N_B, M_B, rho, l):

    V = (N_A*M_A + N_B*M_B)/rho
    Lratio = (N_A*M_A)**(1/3)/(N_A*M_B)**(1/3)
    L = (V/(Lratio+1))**(1/3)
    L_A = Lratio*L
    L_B = 1*L
    
    boxsize = [L*(1+Lratio), L, L]

    # generate A chains
    Alist = []
    Acenter = np.array([-L_A/2 - L_B/2, 0, 0])
    Aboxsize = [L_A,L,L]
    for idx_chain in range(M_A):
        # choose random starting position in box centered on left edge of A domain
        pos0 = np.multiply(Aboxsize, np.random.rand(1,3)-0.5)+Acenter
        chain_coords = chain_walk(N_A,l)
        com = np.average(chain_coords,axis=0)
        chain_coords = chain_coords - com + pos0
        chain_coords = wrap_coords(chain_coords, boxsize)
        Alist.append(chain_coords)
    
    # generate B chains
    Blist = []
    Bcenter = np.array([0, 0, 0])
    Bboxsize = [L,L,L]
    for idx_chain in range(M_B):
        # choose random starting position in box centered on origin
        pos0 = np.multiply(Bboxsize, np.random.rand(1,3)-0.5+Bcenter)
        chain_coords = chain_walk(N_B,l)
        com = np.average(chain_coords,axis=0)
        chain_coords = chain_coords - com + pos0
        chain_coords = wrap_coords(chain_coords, boxsize)
        Blist.append(chain_coords)

    return Alist, Blist, boxsize

def get_bonds(polys, idx_shift):

    bonds = []    
    total_poly = 0
    for poly in polys:
        Npoly = np.size(poly,0)
        for idx_mnr in range(0,Npoly-1):
            idx_atom = idx_shift + total_poly + idx_mnr
            bond_idxs = [idx_atom, idx_atom + 1]
            bonds.append(bond_idxs)
        total_poly += Npoly
    
    return bonds

def gen_snapshot(polys, boxsize):

    # only for linear homopolymers!
    M = len(polys)
    N = np.size(polys[0], 0)
    
    # Initialize snapshot using hoomd file type
    snapshot = gsd.hoomd.Snapshot()

    ## Box size
    snapshot.configuration.box = boxsize # last ones are 

    ## Particles defined
    snapshot.particles.N = N*M

    # positions for each particle extracted from polymer list
    positions = []
    for i in range(M):
        for j in range(N):
            positions.append(polys[i][j,:])
    snapshot.particles.position = positions

    # particle type (all A monomers)
    snapshot.particles.types = ['A']
    snapshot.particles.typeid = [0] * N*M

    # generate bonds list
    nBonds = 0
    for poly in polys:
        nBonds += np.size(poly,0) - 1
    snapshot.bonds.N = nBonds
    snapshot.bonds.types = ['A-A']
    snapshot.bonds.typeid = [0] * nBonds
    
    snapshot.bonds.group = get_bonds(polys, idx_shift=0)

    return snapshot

def gen_snapshot_AB(Apolys, Bpolys, boxsize):

    # linear homopolymers
    M_A = len(Apolys)
    M_B = len(Bpolys)
    N_A = np.size(Apolys[0],0)
    N_B = np.size(Bpolys[0],0)

    # Initialize snapshot using hoomd file type
    snapshot = gsd.hoomd.Snapshot()

    ## Box size
    if len(boxsize)==3:
        boxsize += [0,0,0]
    snapshot.configuration.box = boxsize # last ones are 

    ## Particles defined
    snapshot.particles.N = N_A*M_A + N_B*M_B

    # positions for each particle extracted from polymer list
    positions = []
    for i in range(M_A):
        for j in range(N_A):
            positions.append(Apolys[i][j,:])
    for i in range(M_B):
        for j in range(N_B):
            positions.append(Bpolys[i][j,:])
    snapshot.particles.position = positions

    # particle type (A and B monomers)
    snapshot.particles.types = ['A','B']
    snapshot.particles.typeid = [0] * N_A*M_A + [1] * N_B*M_B

    # generate bonds list
    nABonds = 0
    nBBonds = 0
    for poly in Apolys:
        nABonds += np.size(poly,0) - 1
    for poly in Bpolys:
        nBBonds += np.size(poly,0) - 1
    
    snapshot.bonds.N = nABonds+nBBonds
    snapshot.bonds.types = ['A-A','B-B']
    snapshot.bonds.typeid = [0] * nABonds + [1] * nBBonds
    
    snapshot.bonds.group = get_bonds(Apolys,idx_shift=0) + get_bonds(Bpolys,idx_shift=N_A*M_A)

    return snapshot