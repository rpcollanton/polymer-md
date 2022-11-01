import hoomd
import datetime

class Slice1DFilter(hoomd.filter.CustomFilter):

    def __init__(self,axis,min_coord,max_coord):
        # axis can be 0 1 2 or 'x' 'y' 'z'
        if isinstance(axis,str):
            if not axis.isnumeric():
                # convert to numeric
                conv = {'x':0, 'y':1, 'z':2}
                axis = conv[axis]
            else:
                axis = int(axis)
        self._axis = axis
        self._min = min_coord
        self._max = max_coord
    
    def __hash__(self):
        return hash((self._axis, self._min, self._max))
    
    def __eq__(self,other):
        return (isinstance(other,Slice1DFilter)
                and self._axis == other._axis
                and self._min == other._min
                and self._max == other._max)
    
    def __call__(self,state):
        with state.cpu_local_snapshot as snap:
            pos = snap.particles.position
            indices = (pos[self._axis,:] > self._min
                        & pos[self._axis,:] < self._max)
            return np.copy(snap.particles.tag[indices])

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
        return self.kinetic_energy + self.potential_energy
    
    @property
    def kinetic_energy(self):
        if self.sim.timestep > 0:
            return self.quantities.kinetic_energy
        else:
            return 0
    
    @property
    def potential_energy(self):
        if self.sim.timestep > 0:
            return self.quantities.potential_energy
        else:
            return 0

    @property
    def pressure(self):
        if self.sim.timestep > 0:
            return self.quantities.pressure
        else:
            return 0
    
    @property
    def temperature(self):
        if self.sim.timestep > 0:
            return self.quantities.kinetic_temperature
        else:
            return 0

class SpatialThermo():

    def __init__(self,):

        return