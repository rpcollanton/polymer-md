import numpy as np
import sys
import gsd.hoomd
import gsd.pygsd

def read_snapshot_from_gsd(fname):

    f = gsd.pygsd.GSDFile(open(fname, 'rb'))
    t = gsd.hoomd.HOOMDTrajectory(f)

    return t[-1] # return the last snapshot/frame!

def convert_A_to_B(snapshot, fraction):

    # ASSUME!!! That linear homopolymers were constructed in order.
    # So, polymer 0 of length N has particles with index 0 to N-1

    n_particles = snapshot.particles.N
    
    # determine length and number of linear polymers
    n_bonds = snapshot.bonds.N

    print(n_particles)
    print(n_bonds)
    M = int(n_particles-n_bonds)
    N = int(n_particles/M)

    # pick at random some fraction of M molecules to convert to B
    rng = np.random.default_rng()
    convert_poly_idx = rng.choice(M, size=int(fraction*M), replace=False)

    # determine what particle IDs to convert to 'B'
    convert_part_idx = []
    for i in convert_poly_idx:
        idx_start = i*N
        idx_end = (i+1)*N 
        for j in range(idx_start,idx_end):
            convert_part_idx.append(j)
    
    newtypes = ['A','B']
    newtypeid = np.zeros(n_particles)
    for i in range(n_particles):
        if i in convert_part_idx:
            newtypeid[i] = 1
        else:
            newtypeid[i] = 0

    # determine what bond IDs to convert to 'A' 
    bondgroups = snapshot.bonds.group
    newbondtypes = ['A-A','B-B']
    newbondtypeid = np.zeros(n_bonds)
    for i in range(n_bonds):
        if bondgroups[i,0] in convert_part_idx: # if one is, both are
            newbondtypeid[i] = 1
        else:
            newbondtypeid[i] = 0

    # convert 
    snapshot.particles.types = newtypes
    snapshot.particles.typeid = newtypeid
    snapshot.bonds.types = newbondtypes
    snapshot.bonds.typeid = newbondtypeid

    return snapshot, N, M

def main(argv):

    fname = argv[0]
    fraction = float(argv[1])

    snap_initial = read_snapshot_from_gsd(fname)
    snap_converted, N, M = convert_A_to_B(snap_initial, fraction)

    M_A = int((1-fraction)*M)
    M_B = int(fraction*M)

    fname = "struct/init.N_{:04d}_A_{:04d}_B_{:04d}.gsd".format(N, M_A, M_B)
    with gsd.hoomd.open(name=fname, mode='wb') as f:
        f.append(snap_converted)

    return

if __name__ == "__main__":
   main(sys.argv[1:])