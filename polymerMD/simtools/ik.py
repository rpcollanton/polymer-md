from typing import List
import hoomd
from hoomd.logging import log
import freud
import numpy as np

# helper functions
def convertNeighborListHoomdToFreud(nlist: hoomd.md.data.NeighborListLocalAccess):
    # write this if necessary
    return

def convertNeighborListFreudToHoomd(nlist: freud.locality.NeighborList):
    # write this if necessary
    return

def add_thermo_ik(sim: hoomd.Simulation, period: int, axis: int, nbins: int, flog: str, fedge: str):
    # create ThermoIK compute object and associated updater
    thermoIK = ThermoIK(sim, nbins, axis)
    trigger = hoomd.trigger.Periodic(period)
    updater = hoomd.write.CustomWriter(action=thermoIK.updater, trigger=trigger)
    sim.operations.writers.append(updater)

    # create ik spatial pressure logger and store basic simulation info
    logger = hoomd.logging.Logger(categories=['scalar', 'sequence'])
    logger.add(sim, ['timestep'])

    # store spatial IK information
    logger[('ThermoIK', 'spatial_pressure_tensor')] = (thermoIK, "spatial_pressure_tensor", 'sequence')

    # create IK thermo gsd log file
    log_writer = hoomd.write.GSD(filename=flog, trigger=trigger, mode='wb', filter=hoomd.filter.Null())
    log_writer.log = logger
    sim.operations.writers.append(log_writer)

    # write edges to a file!
    # put edges on correct axis, 0s for other axis
    # assumes edges won't be changing throughout...
    alledges = np.zeros((nbins+1,3))
    alledges[:,axis] = thermoIK.BinnedNListPairs.edges
    np.savetxt(fedge, alledges)

    return thermoIK

# classes

class ThermoIKUpdater(hoomd.custom.Action): 

    def __init__(self, thermoIK, nlistPair):
        self._thermoIK = thermoIK
        self._nlistPair = nlistPair
        return
    
    def attach(self, simulation):
        super().attach(simulation)
        return
            
    def act(self, timestep):
        with self._state.cpu_local_snapshot as snap: 
            # update pair neighbor lists
            with self._nlistPair.cpu_local_nlist_arrays as nlist_arrays:
                self._thermoIK.BinnedNListPairs.updateBinnedLists(snap.particles.position, nlist_arrays)

            # update bond neighbor lists
            self._thermoIK.BinnedBonds.updateBondLists(snap.particles.position, snap.bonds.group)

            # re-compute spatial thermo
            self._thermoIK.compute()
        return

class ThermoIK(metaclass=hoomd.logging.Loggable):

    def __init__(self, sim: hoomd.Simulation, nbins: int, axis: int):
        # keep a reference to the simulation
        self._sim = sim

        # store binned axis index and number of bins along that axis
        self._nbins = nbins
        self._axis = axis

        # create binned neighbor list objects to be updated later
        self.BinnedNListPairs = BinnedNeighborLists(nbins, axis, self._sim.state.box)
        self.BinnedBonds = BinnedBonds(nbins, axis, self._sim.state.box)

        # create thermoIKUpdater, passing this object into it. assume pair force is the second element
        # This will be an action added to a custom writer and added to sim.operations.writers, see ik.add_thermo_ik
        nlistPair = sim.operations.integrator.forces[1].nlist
        self.updater = ThermoIKUpdater(self, nlistPair)

        # spatial pressure tensor, to be updated.
        self._spatial_pressure_tensor = np.zeros((self._nbins,6))

        # pre-compute things to reduce computation
        self._pre_compute()
        return
    
    def _pre_compute(self):
        self._divA = self._sim.state.box.L[self._axis]/np.product(self._sim.state.box.L)
        self._divbinVol = 1/np.product(self._sim.state.box.L)*self._nbins
        return
    
    def compute(self):
        # reset spatial pressure tensor
        self._spatial_pressure_tensor = np.zeros((self._nbins,6))

        # get kT from the simulation's current state
        kT = self._sim.operations.computes[0].kinetic_temperature

        # virial contribution from pair interaction (add bond later) for each bin
        for i in range(self.BinnedNListPairs._nbins):
            # Kinetic contribution to the pressure, proportional to temperature and local density
            numdens = self._divbinVol*self.BinnedNListPairs._bincount[i] # maybe gaussian smear this later?
            kinetic_p = kT*numdens # need to ensemble average this separately!!!?
            self._spatial_pressure_tensor[i,0] += kinetic_p
            self._spatial_pressure_tensor[i,3] += kinetic_p
            self._spatial_pressure_tensor[i,5] += kinetic_p

            edge0 = self.BinnedNListPairs.edges[i]
            edge1 = self.BinnedNListPairs.edges[i+1]

            # Bond and pair virial contributions to the pressure
            virPair = self._divA*self._sim.operations.integrator.forces[1].compute_virial_pressure_contribution(self.BinnedNListPairs.nlists[i], self._axis, edge0, edge1)
            virBond = self._divA*self._sim.operations.integrator.forces[0].compute_virial_pressure_contribution(self.BinnedBonds.bondlists[i], self._axis, edge0, edge1)
            # print("virPair: " + str(virPair))
            # print("virBond: " + str(virBond))

            self._spatial_pressure_tensor[i,:] += self._divA * (virPair + virBond)
            #print(self._spatial_pressure_tensor)

        # print averaged from my calculation at this frame
        print("Average IK: ", 1/self._nbins*np.sum(self._spatial_pressure_tensor,axis=0))
        print("HOOMD Result: ", self._sim.operations.computes[0].pressure_tensor)
        
        # smeared average density function from freud, to be evaluated at each box's midpoint
        # gaussian density?

        # virial stuff here
        # where to get force calculations? HOOMD... nvm, only gives force on each particle, not contribution from different particles
        # where to get positions? do we take these as input from updater or use reference to sim? 

        # for each bin i:
            # convert pair neighbor list to hoomd format lol
            # call Pair.compute_virial_pressure method that I wrote
            # pass binned bond list in good format :) 
            # call Bond.compute_virial_pressure method that needs to be written
            # add pair/bond contributions together, divide by cross-sectional area
            # for diagonal pressure tensor elements, add the kinetic/density contribution. (compute with freud.density.GaussianDensity) 
            # rho(z) * kT (get kT from somewhere...)
            # add and update bin i, self._spatial_pressure_tensor[i,:]!

        # updates self._spatial_pressure_tensor!! 
        return

    @log(category='sequence')
    def spatial_pressure_tensor(self):
        return self._spatial_pressure_tensor

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

class BinnedBonds:

    def __init__(self, nBins: int, axis: int, box: hoomd.Box):
        # nunber of bins and dimension along which bins are created
        self._nbins = nBins
        self._axis = axis

        # get dimensions of axis
        self._L = box.L[axis]

        # bin edge points
        self._edges = np.linspace(-self._L/2,+self._L/2,self._nbins+1)

        # create list of member and neighbor lists for each bin
        self._bondLists = [[] for i in range(self._nbins)] # populate with a list of bond tags for each bin
        return

    def updateBondLists(self, positions: np.ndarray, bonds: hoomd.data.array):

        # reset stored bond arrays
        self._bondLists = [[] for i in range(self._nbins)]

        # determine the bin each particle belongs to. Note, first bin has index 1 from np.digitize
        particle_bins = np.digitize(positions[:,self._axis], self._edges)

        # loop over bonds
        for bondtag,bondedpair in enumerate(bonds):
            # indices of particles participating in this bond
            i = bondedpair[0]
            j = bondedpair[1]
            # indices of bins that these two particles are in
            bin_i = particle_bins[i]-1
            bin_j = particle_bins[j]-1
            # add pair to each bin between the two bins
            for bin_idx in self._getBinsBetween(bin_i, bin_j, self._nbins):
                self._bondLists[bin_idx].append(bondtag)

        return
    
    def _getBinsBetween(self, i: int, j: int, n: int):
        # i and j are bin indices, could be greater or lesser, or same
        # n is the number of bins
        l = min((i,j)) # lower bin
        u = max((i,j)) # upper bin
        if l == u:
            binsbtwn = [l]
        elif (u-l) < (l+n-u):
            binsbtwn = list(range(l,u+1))
        else:
            binsbtwn = list(range(u,n)) + list(range(0,l+1))
        return binsbtwn
    
    @property 
    def bondlists(self):
        return self._bondLists
    
    @property
    def edges(self):
        return self._edges
    
class BinnedNeighborLists:

    def __init__(self, nBins: int, axis: int, box: hoomd.Box): 
        # nunber of bins and dimension along which bins are created
        self._nbins = nBins
        self._axis = axis

        # get dimensions of axis
        self._L = box.L[axis]

        # bin edge points
        self._edges = np.linspace(-self._L/2,+self._L/2,self._nbins+1)

        # number of particles physically in each bin
        self._bincount = np.zeros((self._nbins))

        # create list of member and neighbor lists for each bin
        self._nListStagers = [nListStager() for i in range(nBins)] 
        self._nLists = [] # created once neighbor arrays are populated
        return

    def updateBinnedLists(self, positions: np.ndarray, nlist):
        # choose which function to call based on the type of inputted neighbor list to be binned
        if isinstance(nlist, hoomd.md.data.NeighborListLocalAccess):
            self._updateLists_hoomd(positions, nlist)
        elif isinstance(nlist, freud.locality.NeighborList):
            self._updateLists_freud(positions, nlist)
        return

    def _updateLists_hoomd(self, positions: np.ndarray, nlist_arrays: hoomd.md.data.NeighborListLocalAccess):
        # positions - an array (N_particlesx3) of coordinates. Get from 
        # neighbors - a neighbor list in the hoomd format, class hoomd.md.data.NeighborListLocalAccess
        # NOTE: get neighbors from sim.operations.integrator.forces[**THE PAIR POTENTIAL**]
        # in my code the pair potential is always index 1, and the bonded potential index 0

        # is this a full or half neighbor list?
        isFull = not nlist_arrays.half_nlist

        # reset stored neighbor arrays
        for stager in self._nListStagers:
            stager.reset()
        
        # determine the bin each particle belongs to. Note, first bin has index 1 from np.digitize
        particle_bins = np.digitize(positions[:,self._axis], self._edges)
        self._bincount = np.histogram(positions[:,self._axis], bins=self._edges, density=False)[0]

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
                for bin_idx in self._getBinsBetween(bin_i, bin_j, self._nbins):
                    self._nListStagers[bin_idx].add_pair((i,j), dist_ij)
        
        # after adding all pairs, build neighbor lists
        self._nLists = [stager.build_nlist(len(positions)) for stager in self._nListStagers]

        return
    
    def _updateLists_freud(self, positions: np.ndarray, nlist: freud.locality.NeighborList):
        # TO BE WRITTEN! don't need it yet
        return
    
    def _getBinsBetween(self, i: int, j: int, n: int):
        # i and j are bin indices, could be greater or lesser, or same
        # n is the number of bins
        l = min((i,j)) # lower bin
        u = max((i,j)) # upper bin
        if l == u:
            binsbtwn = [l]
        elif (u-l) < (l+n-u):
            binsbtwn = list(range(l,u+1))
        else:
            binsbtwn = list(range(u,n)) + list(range(0,l+1))
        return binsbtwn

    @property 
    def nlists(self):
        return self._nLists
    
    @property
    def edges(self):
        return self._edges
    