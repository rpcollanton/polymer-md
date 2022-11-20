import hoomd
from polymerMD.simtools import custom
import numpy as np
import itertools

def compute_mol_conformation(sim: hoomd.Simulation, period: int):

    cluster = custom.getBondedClusters(sim.state) # notice, this could be any set of clusters!
    conformation = custom.Conformation(cluster)
    updater = hoomd.write.CustomWriter(action=conformation.updater,
                                        trigger=hoomd.trigger.Periodic(period))
    sim.operations.writers.append(updater)

    return conformation

def add_write_state(sim: hoomd.Simulation, iter: int, fname: str):
    # write gsd at a certain iteration number
    write_gsd = hoomd.write.GSD(filename=fname,
                             trigger=hoomd.trigger.On(iter),
                             mode='wb')
    sim.operations.writers.append(write_gsd)
    return

def add_write_trajectory(sim: hoomd.Simulation, period: int, logger: None, ftraj: str):

    write_traj_gsd = hoomd.write.GSD(filename=ftraj, logger=logger,
                            trigger=hoomd.trigger.Periodic(period=period),
                            mode='wb')
    sim.operations.writers.append(write_traj_gsd)

    return

def basic_logger(sim: hoomd.Simulation, thermo: custom.Thermo):

    logger = hoomd.logging.Logger(categories=['scalar','string'])
    logger.add(sim, ['timestep'])

    logger.add(sim, ['tps','walltime'])
    # compute estimated time remaining
    status = custom.Status(sim)
    logger += status

    # log thermodynamics
    logger += thermo # won't add pressure tensor because only scalar/string

    return logger

def add_table_log(sim: hoomd.Simulation, period: int, logger: hoomd.logging.Logger):

    table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(period=period),logger=logger)
    sim.operations.writers.append(table)

    return

def add_spatial_thermo(sim: hoomd.Simulation, period: int, axis: int, nbins: int, flog: str, fedge: str):

    Lmin = -sim.state.box.L[axis]/2
    Lmax = +sim.state.box.L[axis]/2
    edges = np.linspace(Lmin, Lmax, nbins+1, endpoint=True)

    # create filters and add filter updater to simulation with period matching thermo log period
    filters1D = [custom.Slice1DFilter(axis,edges[i],edges[i+1]) for i in range(nbins)]
    trigger = hoomd.trigger.Periodic(period)
    updater = hoomd.update.FilterUpdater(trigger=trigger, filters=filters1D)
    sim.operations.updaters.append(updater)
    
    # create and add spatially discretized thermodynamic quantity computes to simulation
    spatialthermo = custom.Thermo1DSpatial(sim,filters1D)

    # create logger and store basic simulation information
    logger = hoomd.logging.Logger(categories=['sequence'])
    logger.add(sim, ['timestep'])
    
    # store all spatial thermo information
    logger[('Thermo1DSpatial', 'spatial_temperature')] = (spatialthermo, "spatial_temperature", 'sequence')
    logger[('Thermo1DSpatial', 'spatial_kinetic_energy')] = (spatialthermo, "spatial_kinetic_energy", 'sequence')
    logger[('Thermo1DSpatial', 'spatial_potential_energy')] = (spatialthermo, "spatial_potential_energy", 'sequence')
    logger[('Thermo1DSpatial', 'spatial_pressure_tensor')] = (spatialthermo, "spatial_pressure_tensor", 'sequence')

    # create spatial thermo gsd log file
    log_writer = hoomd.write.GSD(filename=flog, trigger=trigger, mode='wb', filter=hoomd.filter.Null())
    log_writer.log = logger
    sim.operations.writers.append(log_writer)

    # write edges to a file!
    # put edges on correct axis, 0s for other axis
    # assumes edges won't be changing throughout...
    alledges = np.zeros((len(edges),3))
    alledges[:,axis] = edges
    np.savetxt(fedge, alledges)

    return logger, alledges

def remove_overlaps(initial_state, device, kT, prefactor_range, iterations, fname):

    # bonded interactions
    feneParam = dict(k=60.0, r0=1.5, epsilon=0.0, sigma=1.0, delta=0.0)
    snap = run_GAUSSIAN_FENE(initial_state, device, kT, prefactor_range, feneParam, iterations, fname)

    return snap

def relax_overlaps(initial_state, device, iterations, fname=None):

    # overlap will still be significant given the 1/r dependence of the LJ potential

    # force field parameters
    ljParam = {('A','A'): dict(epsilon=1.0, sigma=1.0)}
    lj_rcut = 2**(1/6)
    feneParam = {'A-A': dict(k=30.0, r0=1.5, epsilon=1.0, sigma=1.0, delta=0.0)}

    # newtonian NVE dynamics with limit on displacement
    displ = hoomd.variant.Ramp(0.001,0.005,0,iterations)
    nveCapped = hoomd.md.methods.DisplacementCapped(filter=hoomd.filter.All(), maximum_displacement=displ)
    methods = [nveCapped]

    # update period
    period = 5000
    
    sim = setup_LJ_FENE(initial_state, device, iterations, period, ljParam, lj_rcut, feneParam, methods, fstruct=fname)
    sim.run(iterations)
    
    return sim.state.get_snapshot()

def relax_overlaps_AB(initial_state, device, epsAB, iterations, fname=None):

    # overlap will still be significant given the 1/r dependence of the LJ potential

    # force field parameters
    sameParam = dict(epsilon=1.0, sigma=1.0)
    diffParam = dict(epsilon=epsAB, sigma=1.0)
    ljParam = {('A','A'): sameParam, ('B','B'): sameParam, ('A','B'): diffParam}
    lj_rcut = 2**(1/6)
    bondParam = dict(k=30.0, r0=1.5, epsilon=1.0, sigma=1.0, delta=0.0)
    feneParam = {}
    for bondtype in initial_state.bonds.types:
        feneParam[bondtype] = bondParam

    # newtonian NVE dynamics with limit on displacement
    displ = hoomd.variant.Ramp(0.001,0.005,0,iterations)
    nveCapped = hoomd.md.methods.DisplacementCapped(filter=hoomd.filter.All(), maximum_displacement=displ)
    methods = [nveCapped]

    # update period
    period = 5000
    
    sim = setup_LJ_FENE(initial_state, device, iterations, period, ljParam, lj_rcut, feneParam, methods, fstruct=fname)
    sim.run(iterations)
    
    return sim.state.get_snapshot()

def equilibrate(initial_state, device, kT, iterations, fstruct, ftraj):

    # force field parameters
    ljParam = {('A','A'): dict(epsilon=1.0, sigma=1.0)}
    lj_rcut = 2**(1/6)
    feneParam = {'A-A': dict(k=30.0, r0=1.5, epsilon=1.0, sigma=1.0, delta=0.0)}

    # langevin thermostat and integrator
    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT = kT)
    methods = [langevin]

    # update period
    period = 5000
    
    sim = setup_LJ_FENE(initial_state, device, iterations, period, ljParam, lj_rcut, feneParam, methods, 
                            fstruct=fstruct, ftraj=ftraj)
    sim.run(iterations)
    
    return sim.state.get_snapshot()

def equilibrate_AB(initial_state, device, epsAB, kT, iterations, fstruct, ftraj=None):

    # force field parameters
    ljParam = {('A','A'): dict(epsilon=1.0, sigma=1.0),
               ('B','B'): dict(epsilon=1.0, sigma=1.0),
               ('A','B'): dict(epsilon=epsAB, sigma=1.0)}
    lj_rcut = 2**(1/6)
    bondParam = dict(k=30.0, r0=1.5, epsilon=1.0, sigma=1.0, delta=0.0)
    feneParam = {}
    for bondtype in initial_state.bonds.types:
        feneParam[bondtype] = bondParam

    # langevin thermostat and integrator
    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT = kT)
    methods = [langevin]

    # update period
    period = 5000
    
    sim = setup_LJ_FENE(initial_state, device, iterations, period, ljParam, lj_rcut, feneParam, methods, 
                            fstruct=fstruct, ftraj=ftraj)
    sim.run(iterations)
    return sim.state.get_snapshot()

def production(initial_state, device, epsAB, kT, iterations, period=None, fstruct=None, ftraj=None, fthermo=None, fedge=None, nBins = 40, axis=0):

    # force field parameters
    ljParam = {('A','A'): dict(epsilon=1.0, sigma=1.0),
               ('B','B'): dict(epsilon=1.0, sigma=1.0),
               ('A','B'): dict(epsilon=epsAB, sigma=1.0)}
    lj_rcut = 2**(1/6)
    bondParam = dict(k=30.0, r0=1.5, epsilon=1.0, sigma=1.0, delta=0.0)
    feneParam = {}
    for bondtype in initial_state.bonds.types:
        feneParam[bondtype] = bondParam

    # langevin thermostat and integrator
    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT = kT)
    methods = [langevin]

    # update period
    if period==None:
        period = 5000
    
    sim = setup_LJ_FENE(initial_state, device, iterations, period, ljParam, lj_rcut, feneParam, methods, 
                            fstruct=fstruct, ftraj=ftraj)
    
    # add momentum zeroer! freeze interface in place, no bulk motion allowed
    zeromomentum = hoomd.md.update.ZeroMomentum(hoomd.trigger.Periodic(2000))
    sim.operations.updaters.append(zeromomentum)

    if fthermo!=None:
        add_spatial_thermo(sim, period, axis, 40, fthermo, fedge)
    
    sim.run(iterations)

    return sim.state.get_snapshot()

def output_spatial_thermo(initial_state, epsAB, kT, axis, nbins, fthermo, fedge, ):

    device = hoomd.device.CPU()
    iterations = 1
    period = 1
    thermoPeriod = iterations

    # force field parameters
    ljParam = {('A','A'): dict(epsilon=1.0, sigma=1.0),
               ('B','B'): dict(epsilon=1.0, sigma=1.0),
               ('A','B'): dict(epsilon=epsAB, sigma=1.0)}
    lj_rcut = 2**(1/6)
    bondParam = dict(k=30.0, r0=1.5, epsilon=1.0, sigma=1.0, delta=0.0)
    feneParam = {}
    for bondtype in initial_state.bonds.types:
        feneParam[bondtype] = bondParam

    # langevin thermostat and integrator
    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT = kT)
    methods = [langevin]

    sim = setup_LJ_FENE(initial_state, device, iterations, period, ljParam, lj_rcut, feneParam, methods, 
                            fstruct=None, ftraj=None)
    add_spatial_thermo(sim, iterations, axis, nbins, fthermo, fedge)
    
    sim.run(iterations)

    return sim.state.get_snapshot()

def run_GAUSSIAN_FENE(initial_state, device, kT, prefactor_range, feneParam, iterations, fname=None):
    sim = hoomd.Simulation(device=device, seed=1)
    sim.create_state_from_snapshot(initial_state)

    # bonded interactions
    fenewca = hoomd.md.bond.FENEWCA()
    if feneParam['epsilon'] != 0.0:
        raise ValueError("WCA contribution must be zeroed out for the soft overlap removal.")
    for bondtype in initial_state.bonds.types:
        fenewca.params[bondtype] = feneParam 

    # langevin thermostat
    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT = kT)

    # gaussian pair potential, similar to KG soft potential when pre-factor multiplied by 2
    # and length scale multiplied by 2/5
    nlist = hoomd.md.nlist.Cell(buffer=0.5) # buffer impacts performance, not correctness, with default other settings!
    gaussian = hoomd.md.pair.Gauss(nlist, default_r_cut=2.5)

    # custom ramping of energetic prefactor because parameters don't accept hoomd.variant type
    rampsteps = 100
    prefactor = np.arange(prefactor_range[0], prefactor_range[1], (prefactor_range[1]-prefactor_range[0])/rampsteps)
    itr_per_ramp = int(iterations*0.75/rampsteps) # ramp only for hte first 75% of timesteps, then keep at max

    # integrator
    integrator = hoomd.md.Integrator(dt=0.005, methods=[langevin], forces=[fenewca, gaussian])
    sim.operations.integrator = integrator

    # add table logging
    thermo = custom.Thermo(sim)
    tablelogger = basic_logger(sim, thermo)
    add_table_log(sim, 5000, tablelogger)

    # write final state
    if fname!=None:
        add_write_state(sim, iterations, fname)

    # ramp up prefactor gradually
    typepairs = list(itertools.combinations_with_replacement(initial_state.particles.types,2))
    for eps in prefactor:
        param_nonbonded = dict(epsilon=eps, sigma=1.0)
        for typepair in typepairs:
            gaussian.params[typepair] = param_nonbonded
        sim.run(itr_per_ramp)
    
    sim.run(iterations-sim.timestep) # remaining iterations

    return sim.state.get_snapshot()

def setup_LJ_FENE(initial_state, device, iterations, period, ljParam, lj_rcut, feneParam, methods, fstruct=None, ftraj=None):
    sim = hoomd.Simulation(device=device, seed=1)
    sim.create_state_from_snapshot(initial_state)

    # FENE bonded interactions
    fenewca = hoomd.md.bond.FENEWCA()
    fenewca.params = feneParam

    # LJ non-bonded interactions
    nlist = hoomd.md.nlist.Cell(buffer=0.5) # buffer impacts performance, not correctness, with default other settings!
    lj = hoomd.md.pair.LJ(nlist, default_r_cut=lj_rcut)
    lj.params = ljParam
    
    # integrator
    integrator = hoomd.md.Integrator(dt=0.005, methods=methods, forces=[fenewca, lj])
    sim.operations.integrator = integrator

    # loggable computes
    thermo = custom.Thermo(sim)
    conformation = compute_mol_conformation(sim, period)

    # (mostly) complete log
    trajlog = hoomd.logging.Logger()
    trajlog += thermo
    trajlog += conformation
    trajlog.add(sim, ['timestep'])

    # full logger
    if ftraj!=None:
        # write trajectory
        add_write_trajectory(sim, period, ftraj, logger=trajlog)

    if fstruct!=None:
        # write final state
        add_write_state(sim, iterations, fstruct)

    # add table logging
    tablelogger = basic_logger(sim, thermo)
    add_table_log(sim, period, tablelogger)
    
    return sim
