import numpy as np
from polymerMD.simtools import sim_routines
from polymerMD.analysis.utility import read_snapshot_from_gsd
import gsd.hoomd
import gsd.pygsd
import hoomd
import sys 
from pathlib import Path

# read blend initial state
if __name__=="__main__":
    f_init = sys.argv[1]
    parentdir = Path(f_init).parent
    idstr = ".".join(str(Path(f_init).stem).split(".")[0:-1])
snap_init = read_snapshot_from_gsd(f_init)

# system parameters
kT = 1.0
epsAB = 5.0

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
fname = "struct/{:s}.2_equil.gsd".format(idstr)
#ftraj = "traj/{:s}.2_equil.traj.gsd".format(idstr)
ftraj=None
iterations = 40000000

print("\nStarting equilibration with FENE/LJ potential and Langevin thermostat on GPU, for iterations = {:d}".format(iterations))
state_equil = sim_routines.equilibrate_AB(state_relax, gpu, epsAB, kT, iterations, fname, ftraj)


