import gsd.hoomd
import hoomd
from polymerMD.simtools import sim_routines

def read_snapshot_from_gsd(fname):

    f = gsd.pygsd.GSDFile(open(fname, 'rb'))
    t = gsd.hoomd.HOOMDTrajectory(f)

    return t[-1] # return the last snapshot/frame!

# read converted blend
f_init = "struct/init.N_0064_A_512_B_512.gsd"
snap_init = read_snapshot_from_gsd(f_init)
N = 64
M_A = 512
M_B = 512

# equilibrate A/B homopolymer blend
fname = "struct/equil.N_{:04d}_A_{:04d}_B_{:04d}.gsd".format(N, M_A, M_B)
ftraj = "traj/equil.N_{:04d}_A_{:04d}_B_{:04d}.gsd".format(N, M_A, M_B)
kT = 1.0
gpu = hoomd.device.GPU()
iterations = 5000000
state_equil = sim_routines.equilibrate(snap_init, gpu, kT, iterations, fname, ftraj)