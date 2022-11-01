import hoomd
from polymerMD.simtools import sim_routines

def run(snap_init, equilItr, period, epsAB, strID, strSys, rootdir):
    
    # remove expected overlaps
    # iterations = 100000
    # prefactor_min = 1
    # prefactor_max = 120
    kT = 1.0
    cpu = hoomd.device.CPU()
    gpu = hoomd.device.GPU()

    # fname = rootdir + "struct/{:s}.0_init.{:s}.gsd".format(strID, strSys)
    # state_init = sim_routines.remove_overlaps_AB(snap_init, gpu, kT, 
    #                                            [prefactor_min, prefactor_max], iterations, fname)

    # relax remaining overlaps with displacement-capped NVE dynamics
    # iterations = 15000
    # fname = rootdir + "struct/{:s}.1_relax.{:s}.gsd".format(strID, strSys)
    # state_relax = sim_routines.relax_overlaps_AB(state_init, cpu, epsAB, iterations, fname)

    # npt box relaxation
    # fname = rootdir + "struct/{:s}.2_npt.{:s}.gsd".format(strID, strSys)
    # ftraj = rootdir + "traj/{:s}.2_npt.{:s}.traj.gsd".format(strID, strSys)
    # P=0
    # state_npt = sim_routines.npt_relaxbox_AB(snap_init, gpu, epsAB, kT, P, equilItr, fname, ftraj)

    # nvt equilibration
    fname = rootdir + "struct/{:s}.3_nvt.{:s}.gsd".format(strID, strSys)
    ftraj = rootdir + "traj/{:s}.3_nvt.{:s}.traj.gsd".format(strID, strSys)
    state_nvt = sim_routines.equilibrate_AB(snap_init, gpu, epsAB, kT, equilItr, period, fname, ftraj)
    
    return
