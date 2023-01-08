from typing import List
import hoomd
from hoomd.logging import log
import freud
import numpy as np


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

class PressureSpatial(metaclass=hoomd.logging.Loggable):

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

class nListStager:

    def __init__(self):
        
        self._idx_query_pts = []
        self._idx_pts = []
        self._distances = []

        return
    
    def reset(self):
        # re-initialize :)
        return self.__init__()
    
    def build_nlist(self, n_points):
        # check that things aren't empty and are the correct length
        if len(self._idx_query_pts) == 0:
            ValueError("Empty pair lists in nListStager.")
        elif len(self._idx_query_pts) != len(self._idx_pts):
            ValueError("Lengths of pair point lists do not match.")
        
        nList = freud.locality.NeighborList.from_arrays(n_points, n_points, self._idx_query_pts, self._idx_pts, self._distances)
        return nList
    
    def add_pair(self,pair,distance):
        self._idx_query_pts.append(pair[0])
        self._idx_pts.append(pair[1])
        self._distances.append(distance)
        return

class BinsIK:

    def __init__(self, nBins: int, axis: int, box: hoomd.Box): 
        # nunber of bins and dimension along which bins are created
        self._nbins = nBins
        self._axis = axis

        # get dimensions of axis
        self._L = box.L[axis]

        # bin edge points
        self._edges = np.linspace(-self._L/2,+self._L/2,self._nBins+1)

        # create list of member and neighbor lists for each bin
        self._members = [[] for i in range(nBins)]
        self._nListStagers = [nListStager() for i in range(nBins)] # each list will contain pairs of particles that are neighbors 
        # CREATE class to set up for freud NeighborList from_arrays class method. make self._neighborArrays a list of those classes
        # this class should have a N_points, point indices, and distances according to what from_arrays needs
        # this class should also have a "build freud list" method that will correctly use the arrays to build the list
        # call from_arrays at the end of updateLists and then store the resulting neighborList objecvts in the _neighborlist member array
        # use list comprehension to do all this
        # give class a reset method to wipe out arrays, call these at beginning of updateLists
        self._nLists = [] # created once neighbor arrays are populated

        return

    def updateLists(self, positions: np.ndarray, nlist_arrays: hoomd.md.data.NeighborListLocalAccess):
        # positions is an array (N_particlesx3) of coordinates. Get from 
        # neighbors is a neighbor list in the hoomd format, class hoomd.md.data.NeighborListLocalAccess
        # NOTE: get neighbors from sim.operations.integrator.forces[**THE PAIR POTENTIAL**]
        # in my code the pair potential is always index 1, and the bonded potential index 0
        # NOTE: may need to consider the case (which would be weird) that the potential is 

        # is this a full or half neighbor list?
        isFull = not nlist_arrays.half_nlist

        # reset stored neighbor arrays
        for stager in self._nListStagers:
            stager.reset()
        
        # determine the bin each particle belongs to. Note, first bin has index 1 from np.digitize
        particle_bins = np.digitize(positions[:,self._axis], self._edges)

        # This loops over all particles and then each of their neighbors, and determines
        # what bin(s) the interaction of each pair contributes to 
        nlist_iter = zip(nlist_arrays.head_list, nlist_arrays.n_neigh)

        for i,(head,n_neigh) in enumerate(nlist_iter):
            bin_i = particle_bins[i]-1 # shift so that first bin has index of 0
            for j_idx in range(head,head+n_neigh):
                j = nlist_arrays.nlist[j_idx]
                # i and j are indices of two neighbors                
                # if full neighbor list, only do cases where i < j
                if isFull and not (i < j):
                    continue
                bin_j = particle_bins[j]-1 # shift so that first bin has index of 0           

                # calculate distance here. it will be faster to only calculate it once regardless of the number of bins!
                dist_ij = np.sqrt(np.sum(np.square(positions[j,:]-positions[i,:])))     
                
                # add pair to the neighbor list of every bin between the two bins
                # determine beginning and end of range of bins to add particle too
                # this pair is considered a neighbor by hoomd, so we don't need to check if they are close enough to interact!
                start = min((bin_i,bin_j))
                stop = max((bin_i,bin_j))+1
                temprange = list(range(stop,start))
                for bin_idx in temprange:
                    particle_bins[bin_idx].add_pair((i,j), dist_ij)
        
        # after adding all pairs, build neighbor lists
        self._nLists = [stager.build_nlist(len(positions)) for stager in self._nListStagers]

        return
    
    @property 
    def nlists(self):
        return self._nLists
    
    @property
    def edges(self):
        return self._edges
    