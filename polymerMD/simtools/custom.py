from typing import List
import hoomd
import freud
import datetime
import numpy as np

# basic functions
def getBondedClusters(state):
        cluster = freud.cluster.Cluster()
        # get bond indices
        idx_querypts = []
        idx_pts = []
        for bond in state.bonds.group:
            idx_querypts.append(bond[0])
            idx_pts.append(bond[1])

        box = freud.box.Box.from_box(state.configuration.box)
        dist = box.compute_distances(state.particles.position[idx_querypts], 
                                     state.particles.position[idx_pts])
        N = state.particles.N
        bondedneighbors = freud.locality.NeighborList.from_arrays(N, N, idx_querypts, idx_pts, dist)
        cluster.compute(state,neighbors=bondedneighbors)

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
    
    def __init__(self, sim: hoomd.Simulation):
        self.sim = sim
        self.quantities = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
        self.sim.operations.computes.append(self.quantities)

        return

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

class Thermo1DSpatial():

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
    
    @property
    def spatial_pressure_xx(self):
        if self.sim.timestep == 0:
            return [0]*self.nslice
        return [t.pressure_tensor[0] for t in self.thermos] # xx component of pressure tensor
    
    @property
    def spatial_pressure_yy(self):
        if self.sim.timestep == 0:
            return [0]*self.nslice
        return [t.pressure_tensor[3] for t in self.thermos] # yy component of pressure tensor
    
    @property
    def spatial_pressure_zz(self):
        if self.sim.timestep == 0:
            return [0]*self.nslice
        return [t.pressure_tensor[5] for t in self.thermos] # zz component of pressure tensor

    @property
    def spatial_pressure_tensor(self):
        p = np.zeros((self.nslice,5))
        if self.sim.timestep == 0:
            return p
        for i,t in enumerate(self.thermos):
            p[i,:] = np.array(t.pressure_tensor)
        return p
    
    @property
    def spatial_potential_energy(self):
        if self.sim.timestep == 0:
            return [0]*self.nslice
        return [t.potential_energy for t in self.thermos]
    
    @property
    def spatial_kinetic_energy(self):
        if self.sim.timestep == 0:
            return [0]*self.nslice
        return [t.kinetic_energy for t in self.thermos]
    
    @property
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
        box = freud.box.Box.from_box(self._state.configuration.box)
        self._clprops.compute((box,self._state.particles.position),self._clidx)
        return

class Conformation():

    def __init__(self, cluster: freud.cluster.Cluster):
        # initialize
        self._cl = cluster
        # create cluster properties object. 
        # This will be updated by a clusterpropertiesupdater
        self._clprop = freud.cluster.ClusterProperties()
        self.updater = ClusterPropertiesUpdater(self._cl, self._clprop)
        return
    
    @property
    def avgRgSq(self):
        return np.mean(self._clprop.radii_of_gyration)