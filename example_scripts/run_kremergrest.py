import gsd.hoomd
import hoomd
from polymerMD.simtools import sim_routines
from polymerMD.structure import polygen


# polymer attributes
N = 16
M = 16
l = 0.97
lp = 1.34
rho = 0.85

# generate and save initial state
polys, L = polygen.kremer_grest_mc(N,M,rho,l)
boxsize = [L,L,L,0,0,0]
snap_rand = polygen.gen_snapshot(polys, boxsize)
fname = "struct/rand.N_{:04d}_M_{:04d}.gsd".format(N,M)
with gsd.hoomd.open(name=fname, mode='wb') as f:
    f.append(snap_rand)

# remove expected overlaps
iterations = 100000
prefactor_min = 1
prefactor_max = 120
kT = 1.0
cpu = hoomd.device.CPU()
gpu = hoomd.device.GPU()

fname = "struct/init.N_{:04d}_M_{:04d}.gsd".format(N,M)
state_init = sim_routines.remove_overlaps(snap_rand, gpu, kT, 
                                            [prefactor_min, prefactor_max], iterations, fname)

# relax remaining overlaps with displacement capped NVE dynamics
iterations = 15000
fname = "struct/relax.N_{:04d}_M_{:04d}.gsd".format(N,M)
state_relax = sim_routines.relax_overlaps(state_init, cpu, iterations, fname)


# equilibrate
fname = "struct/equil.N_{:04d}_M_{:04d}.gsd".format(N,M)
ftraj = "traj/equil.N_{:04d}_M_{:04d}.gsd".format(N,M)
iterations = 5000000
state_equil = sim_routines.equilibrate(state_relax, gpu, kT, iterations, fname, ftraj)
