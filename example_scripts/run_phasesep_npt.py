import sim_routines
import gsd.hoomd
import hoomd
import polygen

# TEST #
N_A = 16
N_B = 16
M_A = 32
M_B = 32
rho = 0.85
l = 0.97
epsAB = 2.5

strSize = "NA_{:04d}_NB_{:04d}_MA_{:04d}_MB_{:04d}".format(N_A,N_B,M_A,M_B)

# generate and save random phase separated initial state
Apolys, Bpolys, boxsize = polygen.mc_phase_separated_AB(N_A,M_A,N_B,M_B,rho,l)
snap_rand = polygen.gen_snapshot_AB(Apolys, Bpolys, boxsize)
fname = "struct/macrosep.rand.{:s}.gsd".format(strSize)
with gsd.hoomd.open(name=fname, mode='wb') as f:
    f.append(snap_rand)

# remove expected overlaps
iterations = 100000
prefactor_min = 1
prefactor_max = 120
kT = 1.0
cpu = hoomd.device.CPU()
gpu = hoomd.device.CPU()

fname = "struct/macrosep.init.{:s}.gsd".format(strSize)
state_init = sim_routines.remove_overlaps_AB(snap_rand, gpu, kT, 
                                            [prefactor_min, prefactor_max], iterations, fname)

# relax remaining overlaps with displacement capped NVE dynamics
iterations = 15000
fname = "struct/macrosep.relax.{:s}.gsd".format(strSize)
state_relax = sim_routines.relax_overlaps_AB(state_init, cpu, epsAB, iterations, fname)

# equilibrate NPT
fname = "struct/macrosep.npt.{:s}.gsd".format(strSize)
ftraj = "traj/macrosep.npt.{:s}.gsd".format(strSize)
iterations = 500000
P = 0
state_equil = sim_routines.npt_relaxbox_AB(state_relax, gpu, epsAB, kT, P, iterations, fname, ftraj)

# # equilibrate NVT
# fname = "struct/macrosep.equil.{:s}.gsd".format(strSize)
# ftraj = "traj/macrosep.equil.{:s}.gsd".format(strSize)
# iterations = 5000000
# state_equil = sim_routines.equilibrate_AB(state_relax, gpu, epsAB, kT, iterations, fname, ftraj)
