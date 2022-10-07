import hoomd
import numpy as np
import datetime

class Status():

    def __init__(self, sim):
        self.sim = sim

    @property
    def seconds_remaining(self):
        try:
            return (self.sim.final_timestep - self.sim.timestep) / self.sim.tps
        except ZeroDivisionError:
            return 0

    @property
    def etr(self):
        return str(datetime.timedelta(seconds=self.seconds_remaining))

class Thermo():
    
    def __init__(self, sim):
        self.sim = sim
        self.quantities = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
        self.sim.operations.computes.append(self.quantities)

    @property
    def total_energy(self):
        if self.sim.timestep > 0:
            return self.quantities.kinetic_energy + self.quantities.potential_energy
        else:
            return 0

def __write_state(sim, iter, fname):

    # write gsd at end
    write_final_gsd = hoomd.write.GSD(filename=fname,
                             trigger=hoomd.trigger.On(iter),
                             mode='wb')
    sim.operations.writers.append(write_final_gsd)

    return sim

def __write_trajectory(sim, period, ftraj):

    write_traj_gsd = hoomd.write.GSD(filename=ftraj,
                            trigger=hoomd.trigger.Periodic(period=period),
                            mode='wb')
    sim.operations.writers.append(write_traj_gsd)

    return sim

def __table_log(sim, period, writeTiming, writeThermo):

    logger = hoomd.logging.Logger(categories=['scalar','string'])
    logger.add(sim, ['timestep'])

    if writeTiming:
        logger.add(sim, ['tps'])
        # compute estimated time remaining
        stat = Status(sim)
        logger[('Status', 'etr')] = (stat, 'etr', 'string')

    if writeThermo:
        # compute thermodynamics
        thermo = Thermo(sim)
        thermo_props = thermo.quantities
        logger.add(thermo_props, ['kinetic_temperature', 'pressure', 'kinetic_energy', 'potential_energy'], 'Thermo')
        logger[('Thermo', 'total_energy')] = (thermo, "total_energy", 'scalar')


    table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(period=period),logger=logger)
    sim.operations.writers.append(table)

    return sim

def remove_overlaps(initial_state, device, kT, prefactor_range, iterations, fname):

    sim = hoomd.Simulation(device=device, seed=1)
    sim.create_state_from_snapshot(initial_state)

    # bonded interactions
    fenewca = hoomd.md.bond.FENEWCA()
    param_bonded = dict(k=60.0, r0=1.5, epsilon=0.0, sigma=1.0, delta=0.0)
    fenewca.params['A-A'] = param_bonded

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

    # table logger
    sim = __table_log(sim, 5000, writeTiming=True, writeThermo=True)

    # write final state
    sim = __write_state(sim, iterations, fname)

    # ramp up prefactor gradually
    for eps in prefactor:
        param_nonbonded = dict(epsilon=eps, sigma=1.0)
        gaussian.params[('A','A')] = param_nonbonded
        sim.run(itr_per_ramp)
    
    sim.run(iterations-sim.timestep) # remaining iterations

    return sim.state.get_snapshot()

def remove_overlaps_AB(initial_state, device, kT, prefactor_range, iterations, fname=None):

    sim = hoomd.Simulation(device=device, seed=1)
    sim.create_state_from_snapshot(initial_state)

    # bonded interactions
    fenewca = hoomd.md.bond.FENEWCA()
    param_bonded = dict(k=60.0, r0=1.5, epsilon=0.0, sigma=1.0, delta=0.0)
    fenewca.params['A-A'] = param_bonded
    fenewca.params['B-B'] = param_bonded

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

    # table logger
    sim = __table_log(sim, 5000, writeTiming=True, writeThermo=True)

    # write final state
    if fname!=None:
        sim = __write_state(sim, iterations, fname)

    # ramp up prefactor gradually
    for eps in prefactor:
        param_nonbonded = dict(epsilon=eps, sigma=1.0)
        gaussian.params[('A','A')] = param_nonbonded
        gaussian.params[('B','B')] = param_nonbonded
        gaussian.params[('A','B')] = param_nonbonded
        sim.run(itr_per_ramp)
    
    sim.run(iterations-sim.timestep) # remaining iterations

    return sim.state.get_snapshot()

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
    
    snap = run_LJ_FENE(initial_state, device, iterations, period, ljParam, lj_rcut, feneParam, methods, fstruct=fname)

    return snap

def relax_overlaps_AB(initial_state, device, epsAB, iterations, fname=None):

    # overlap will still be significant given the 1/r dependence of the LJ potential

    # force field parameters
    sameParam = dict(epsilon=1.0, sigma=1.0)
    diffParam = dict(epsilon=epsAB, sigma=1.0)
    ljParam = {('A','A'): sameParam, ('B','B'): sameParam, ('A','B'): diffParam}
    lj_rcut = 2**(1/6)
    feneParam = {'A-A': dict(k=30.0, r0=1.5, epsilon=1.0, sigma=1.0, delta=0.0),
                 'B-B': dict(k=30.0, r0=1.5, epsilon=1.0, sigma=1.0, delta=0.0)}

    # newtonian NVE dynamics with limit on displacement
    displ = hoomd.variant.Ramp(0.001,0.005,0,iterations)
    nveCapped = hoomd.md.methods.DisplacementCapped(filter=hoomd.filter.All(), maximum_displacement=displ)
    methods = [nveCapped]

    # update period
    period = 5000
    
    snap = run_LJ_FENE(initial_state, device, iterations, period, ljParam, lj_rcut, feneParam, methods, fstruct=fname)

    return snap

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
    
    snap = run_LJ_FENE(initial_state, device, iterations, period, ljParam, lj_rcut, feneParam, methods, 
                            fstruct=fstruct, ftraj=ftraj)

    return snap

def equilibrate_AB(initial_state, device, epsAB, kT, iterations, fstruct, ftraj):

    # force field parameters
    ljParam = {('A','A'): dict(epsilon=1.0, sigma=1.0),
               ('B','B'): dict(epsilon=1.0, sigma=1.0),
               ('A','B'): dict(epsilon=epsAB, sigma=1.0)}
    lj_rcut = 2**(1/6)
    bondParam = dict(k=30.0, r0=1.5, epsilon=1.0, sigma=1.0, delta=0.0)
    feneParam = {'A-A': bondParam, 'B-B': bondParam}

    # langevin thermostat and integrator
    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT = kT)
    methods = [langevin]

    # update period
    period = 5000
    
    snap = run_LJ_FENE(initial_state, device, iterations, period, ljParam, lj_rcut, feneParam, methods, 
                            fstruct=fstruct, ftraj=ftraj)

    return snap

def npt_relaxbox(initial_state, device, kT, P, iterations, fstruct=None, ftraj=None):

    # force field parameters
    ljParam = {('A','A'): dict(epsilon=1.0, sigma=1.0)}
    lj_rcut = 2**(1/6)
    feneParam = {'A-A': dict(k=30.0, r0=1.5, epsilon=1.0, sigma=1.0, delta=0.0)}

    # langevin thermostat and integrator
    npt = hoomd.md.methods.NPT(filter=hoomd.filter.All(), kT=kT, tau=0.5, S=P, tauS=5, couple="xyz")
    methods = [npt]

    # update period
    period = 5000
    
    snap = run_LJ_FENE(initial_state, device, iterations, period, ljParam, lj_rcut, feneParam, methods, 
                            fstruct=fstruct, ftraj=ftraj)

    return snap

def npt_relaxbox_AB(initial_state, device, epsAB, kT, P, iterations, fstruct=None, ftraj=None):

    # force field parameters
    ljParam = {('A','A'): dict(epsilon=1.0, sigma=1.0),
               ('B','B'): dict(epsilon=1.0, sigma=1.0),
               ('A','B'): dict(epsilon=epsAB, sigma=1.0)}
    lj_rcut = 2**(1/6)
    bondParam = dict(k=30.0, r0=1.5, epsilon=1.0, sigma=1.0, delta=0.0)
    feneParam = {'A-A': bondParam, 'B-B': bondParam}

    # langevin thermostat and integrator
    npt = hoomd.md.methods.NPT(filter=hoomd.filter.All(), kT=kT, tau=0.5, S=P, tauS=5, couple="xyz")
    methods = [npt]

    # update period
    period = 5000
    
    snap = run_LJ_FENE(initial_state, device, iterations, period, ljParam, lj_rcut, feneParam, methods, 
                            fstruct=fstruct, ftraj=ftraj)

    return snap

def nvt_dpd(initial_state, device, aAB, kT, iterations, fstruct=None, ftraj=None):

    # force field parameters
    dpdParam = {('A','A'): dict(A=25.0, gamma=3.0),
               ('B','B'): dict(A=25.0, gamma=3.0),
               ('A','B'): dict(A=aAB, gamma=3.0)}
    dpd_rcut = 2.5
    bondParam = dict(k=30.0, r0=1.5, epsilon=0.0, sigma=1.0, delta=0.0)
    feneParam = {'A-A': bondParam, 'B-B': bondParam}

    # update period
    period = 5000
    
    snap = run_DPD_FENE(initial_state, device, iterations, period, dpdParam, dpd_rcut, feneParam, kT,
                            fstruct=fstruct, ftraj=ftraj)

    return snap

def run_LJ_FENE(initial_state, device, iterations, period, ljParam, lj_rcut, feneParam, methods, fstruct=None, ftraj=None):

    sim = hoomd.Simulation(device=device, seed=1)
    sim.create_state_from_snapshot(initial_state)

    # FENE bonded interactions
    fenewca = hoomd.md.bond.FENEWCA()
    fenewca.params = feneParam

    # LJ non-bonded interactions
    nlist = hoomd.md.nlist.Cell(buffer=0.5) # buffer impacts performance, not correctness, with default other settings!
    lj = hoomd.md.pair.LJ(nlist, default_r_cut=lj_rcut) # same cutoff as LJ in simulation
    lj.params = ljParam
    
    # integrator
    integrator = hoomd.md.Integrator(dt=0.005, methods=methods, forces=[fenewca, lj])
    sim.operations.integrator = integrator

    if ftraj!=None:
        # write trajectory
        sim = __write_trajectory(sim, period, ftraj)

    if fstruct!=None:
        # write final state
        sim = __write_state(sim, iterations, fstruct)

    # table logger
    sim = __table_log(sim, period, writeTiming=True, writeThermo=True)
    
    sim.run(iterations)

    return sim.state.get_snapshot()

def run_DPD_FENE(initial_state, device, iterations, period, dpdParam, dpd_rcut, feneParam, kT, fstruct=None, ftraj=None):

    sim = hoomd.Simulation(device=device, seed=1)
    sim.create_state_from_snapshot(initial_state)

    # FENE bonded interactions
    fenewca = hoomd.md.bond.FENEWCA()
    fenewca.params = feneParam

    # LJ non-bonded interactions
    nlist = hoomd.md.nlist.Cell(buffer=0.5) # buffer impacts performance, not correctness, with default other settings!
    dpd = hoomd.md.pair.DPD(nlist, kT=kT, default_r_cut=dpd_rcut) # same cutoff as LJ in simulation
    dpd.params = dpdParam
    
    # integrator
    nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
    integrator = hoomd.md.Integrator(dt=0.005, methods=nve, forces=[fenewca, dpd])
    sim.operations.integrator = integrator

    if ftraj!=None:
        # write trajectory
        sim = __write_trajectory(sim, period, ftraj)

    if fstruct!=None:
        # write final state
        sim = __write_state(sim, iterations, fstruct)

    # table logger
    sim = __table_log(sim, period, writeTiming=True, writeThermo=True)
    
    sim.run(iterations)

    return sim.state.get_snapshot()
