from typing import List
import hoomd
from hoomd.logging import log
import freud
import datetime
import numpy as np

def getBondedClusters(snapshot):
    cluster = freud.cluster.Cluster()
    # get bond indices
    idx_querypts = []
    idx_pts = []
    for bond in snapshot.bonds.group:
        idx_querypts.append(bond[0])
        idx_pts.append(bond[1])

    box = freud.box.Box.from_box(snapshot.configuration.box)
    dist = box.compute_distances(snapshot.particles.position[idx_querypts], 
                                    snapshot.particles.position[idx_pts])
    N = snapshot.particles.N
    bondedneighbors = freud.locality.NeighborList.from_arrays(N, N, idx_querypts, idx_pts, dist)
    cluster.compute(snapshot,neighbors=bondedneighbors)

    return cluster

# classes
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
            indices = np.logical_and(pos[:,self._axis] > self._min, pos[:,self._axis] < self._max)
            return np.copy(snap.particles.tag[indices])

class Status(metaclass=hoomd.logging.Loggable):

    def __init__(self, sim):
        self.sim = sim

    @property
    def seconds_remaining(self):
        try:
            return (self.sim.final_timestep - self.sim.timestep) / self.sim.tps
        except ZeroDivisionError:
            return 0

    @log
    def etr(self):
        return str(datetime.timedelta(seconds=self.seconds_remaining))

class Thermo(metaclass=hoomd.logging.Loggable):
    
    def __init__(self, sim: hoomd.Simulation):
        self.sim = sim
        self.thermocompute = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
        self.sim.operations.computes.append(self.thermocompute)

        return
    
    @log
    def temperature(self):
        if self.sim.timestep > 0:
            return self.thermocompute.kinetic_temperature
        else:
            return 0
    
    @log
    def pressure(self):
        if self.sim.timestep > 0:
            return self.thermocompute.pressure
        else:
            return 0

    @log
    def volume(self):
        if self.sim.timestep > 0:
            return self.thermocompute.volume
        else:
            return 0
    
    @log
    def kinetic_energy(self):
        if self.sim.timestep > 0:
            return self.thermocompute.kinetic_energy
        else:
            return 0
    
    @log
    def potential_energy(self):
        if self.sim.timestep > 0:
            return self.thermocompute.potential_energy
        else:
            return 0
    
    @log
    def total_energy(self):
        return self.kinetic_energy + self.potential_energy
     
    @log(category='sequence')
    def pressure_tensor(self):
        if self.sim.timestep > 0:
            return self.thermocompute.pressure_tensor
        else:
            return [0]*6

class Thermo1DSpatial(metaclass=hoomd.logging.Loggable):

    def __init__(self, sim: hoomd.Simulation, filters: List[Slice1DFilter]):
        self.sim = sim
        self.filters = filters # filter updaters should be created elsewhere, wherever the period of the logger is defined
        self.nslice = len(filters)

        # create the list of thermodynamic quantity computes
        self.thermos = []
        for f in filters:
            self.thermos.append(hoomd.md.compute.ThermodynamicQuantities(filter=f))
            self.sim.operations.computes.append(self.thermos[-1])

        # store the axis along which this is being computed
        self.axis = filters[0]._axis
        for f in filters:
            if f._axis != self.axis:
                ValueError("Axes in 1D filters do not match.")

        # store the left and right coordinates of each slice 
        # I don't love this because Thermo1DSpatial isn't managing the updating of these filters, and so in principle they
        # could be out of date!! Particles in each filter might no longer be in the slices...
        # one solution could be to make this class just deal with thermo properties for a list of corresponding filters,
        # and some other class (or function?) responsible for updating the filters and saving the coordinates
        # also we don't need to log the coordinates at each step??? 
        # hmmmmmmmmmmmmmmmmmmmmmmmmmmm yeah no.
        # self.edges = np.zeros((self.nslice,2))
        # for i,f in enumerate(filters):
        #     self.edges[i,0] = f._min
        #     self.edges[i,1] = f._max

        return
    
    @log(category='sequence')
    def spatial_pressure_xx(self):
        if self.sim.timestep == 0:
            return [0]*self.nslice
        return [t.pressure_tensor[0] for t in self.thermos] # xx component of pressure tensor
    
    @log(category='sequence')
    def spatial_pressure_yy(self):
        if self.sim.timestep == 0:
            return [0]*self.nslice
        return [t.pressure_tensor[3] for t in self.thermos] # yy component of pressure tensor
    
    @log(category='sequence')
    def spatial_pressure_zz(self):
        if self.sim.timestep == 0:
            return [0]*self.nslice
        return [t.pressure_tensor[5] for t in self.thermos] # zz component of pressure tensor

    @log(category='sequence')
    def spatial_pressure_tensor(self):
        p = np.zeros((self.nslice,6))
        if self.sim.timestep == 0:
            return p
        for i,t in enumerate(self.thermos):
            p[i,:] = np.array(t.pressure_tensor)
        return p
    
    @log(category='sequence')
    def spatial_potential_energy(self):
        if self.sim.timestep == 0:
            return [0]*self.nslice
        return [t.potential_energy for t in self.thermos]
    
    @log(category='sequence')
    def spatial_kinetic_energy(self):
        if self.sim.timestep == 0:
            return [0]*self.nslice
        return [t.kinetic_energy for t in self.thermos]
    
    @log(category='sequence')
    def spatial_temperature(self):
        if self.sim.timestep == 0:
            return [0]*self.nslice
        return [t.kinetic_temperature for t in self.thermos]

class ClusterPropertiesUpdater(hoomd.custom.Action):

    def __init__(self,cluster,clprops):
        self._clprops = clprops
        self._clidx = cluster.cluster_idx
        return
    
    def attach(self, simulation):
        super().attach(simulation)
        return
            
    def act(self, timestep):
        box = freud.box.Box.from_box(self._state.box)
        with self._state.cpu_local_snapshot as snap:
            self._clprops.compute((box, snap.particles.position),self._clidx)
        return

class Conformation(metaclass=hoomd.logging.Loggable):

    def __init__(self, cluster: freud.cluster.Cluster):
        # initialize cluster, which will be a group of particles (i.e. bonded particles)
        self._cl = cluster
        # create cluster properties object. 
        # This will be updated by a ClusterPropertiesUpdater
        self._clprop = freud.cluster.ClusterProperties()
        self.updater = ClusterPropertiesUpdater(self._cl, self._clprop)
        return
    
    @log
    def avgRg(self):
        return np.mean(self._clprop.radii_of_gyration)
