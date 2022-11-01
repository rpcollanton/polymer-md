import gsd.hoomd
import hoomd
from polymerMD.simtools import sim_routines

def read_snapshot_from_gsd(fname):

    f = gsd.pygsd.GSDFile(open(fname, 'rb'))
    t = gsd.hoomd.HOOMDTrajectory(f)

    return t[-1] # return the last snapshot/frame!

# read blend initial state
f_init = "init/A064_1063_B064_1063.A064_B064_A064_0034.init.gsd"
idstr = "A064_1063_B064_1063.A064_B064_A064_0034"
snap_init = read_snapshot_from_gsd(f_init)

# system parameters
kT = 1.0
epsAB = 1.5

# simulation parameters
cpu = hoomd.device.CPU()
gpu = hoomd.device.GPU()

# remove expected overlaps
iterations = 100000
prefactor_min = 1
prefactor_max = 120
fname = "struct/{:s}.0_init.gsd".format(idstr)

print("Starting soft overlap relaxation on GPU, for iterations = {:d}".format(iterations))
state_overlap = sim_routines.remove_overlaps(snap_init, gpu, kT, 
                                            [prefactor_min, prefactor_max], iterations, fname)

# relax remaining overlaps with displacement capped NVE dynamics
iterations = 10000
fname = "struct/{:s}.1_relax.gsd".format(idstr)

print("\nFurther relaxing overlaps on CPU with displacement capped NVE, for iterations = {:d}".format(iterations))
state_relax = sim_routines.relax_overlaps_AB(state_overlap, cpu, epsAB, iterations, fname)

# equilibrate A/B homopolymer blend
fname = "struct/{:s}.equil.gsd".format(idstr)
ftraj = "traj/{:s}.2_equil.traj.gsd".format(idstr)
iterations = 40000000

print("\nStarting equilibration with FENE/LJ potential and Langevin thermostat on GPU, for iterations = {:d}".format(iterations))
state_equil = sim_routines.equilibrate_AB(state_relax, gpu, epsAB, kT, iterations, fname, ftraj)


