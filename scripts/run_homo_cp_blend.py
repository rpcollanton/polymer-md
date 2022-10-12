import numpy as np
import sim_routines
import gsd.hoomd
import gsd.pygsd
import hoomd

def read_snapshot_from_gsd(fname):

    f = gsd.pygsd.GSDFile(open(fname, 'rb'))
    t = gsd.hoomd.HOOMDTrajectory(f)

    return t[-1] # return the last snapshot/frame!

# read blend initial state
f_init = "/Users/ryancollanton/Desktop/N_0064_A_0512_B_0512.A64_B64_A64_20.init.gsd"
idstr = "N_0064_A_0512_B_0512.A64_B64_A64_20"
snap_init = read_snapshot_from_gsd(f_init)

# system parameters
kT = 1.0
epsAB = 1.5

# simulation parameters
period = 50000
cpu = hoomd.device.CPU()
gpu = hoomd.device.GPU()

# remove expected overlaps
iterations = 100000
prefactor_min = 1
prefactor_max = 120
fname = "struct/{:s}.init.gsd".format(idstr)
state_overlap = sim_routines.remove_overlaps(snap_init, gpu, kT, 
                                            [prefactor_min, prefactor_max], iterations, fname)

# relax remaining overlaps with displacement capped NVE dynamics
iterations = 10000
fname = "struct/{:s}.relax.gsd".format(idstr)
state_relax = sim_routines.relax_overlaps_AB(state_overlap, cpu, epsAB, iterations, fname)

# equilibrate A/B homopolymer blend
fname = "struct/{:s}.equil.gsd".format(idstr)
ftraj = "traj/{:s}.equil.traj.gsd".format(idstr)
iterations = 20000000
state_equil = sim_routines.equilibrate_AB(state_relax, gpu, epsAB, kT, iterations, fname, ftraj)


