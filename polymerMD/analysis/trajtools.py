import gsd.hoomd
import gsd.pygsd
import numpy as np
from polymerMD.analysis import utility, structure
from polymerMD.structure import systemspec
import freud

# analysis functions
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

def density_system(f):
    # f is a trajectory or trajectory frame
    
    if isinstance(f, gsd.hoomd.HOOMDTrajectory):
        ts = [f[i].configuration.step for i in range(len(f))]
        return ts, list(map(density_system, f))
    
    box = f.configuration.box
    V = box[0]*box[1]*box[2]
    N = f.particles.N

    return N/V

def density_1D_monomers(f, nBins=100, axis=0, method='smoothed'):
    # f is a trajectory or trajectory frame
    # axis is the axis to plot the density along. averaged over other two

    if isinstance(f, gsd.hoomd.HOOMDTrajectory):
        ts = [f[i].configuration.step for i in range(len(f))]
        func = lambda t: density_1D_monomers(t, nBins=nBins, axis=axis)
        return ts, list(map(func, f))

    box = f.configuration.box[0:3]
    particleCoord = f.particles.position
    particleTypeID = f.particles.typeid
    types = f.particles.types

    hists = {}
    for i,type in enumerate(types):
        mask = particleTypeID==i
        coords = particleCoord[mask,:]
        if method=='smoothed':
            hists[type] = utility.smoothed_density_1D(coords, box, axis, nBins)
        elif method=='binned':
            hists[type] = utility.binned_density_1D(coords, box, axis, nBins)

    # modify histograms so that sum over species in each bin is 1. IE: convert to vol frac
    hists = utility.count_to_volfrac(hists)

    return hists

def density_1D_species(f, system: systemspec.System, nBins=100, axis=0, method='smoothed'):
    # f is a trajectory or a trajectory frame
    # system is a SystemSpec object describing the topology and composition of the system
    # nBins is the number of bins to use to compute the density
    # axis is the axis to plot the density along. Density effectively averaged over the other two.

    if isinstance(f, gsd.hoomd.HOOMDTrajectory):
        ts = [f[i].configuration.step for i in range(len(f))]
        func = lambda t: density_1D_species(t, system, nBins=nBins, axis=axis)
        return ts, list(map(func, f))

    # get geometric information
    box = f.configuration.box[0:3]

    # note: should optimize in future because this does not take advantage of the fact that system
    # topology is not changing from frame to frame! will recompute system labels/indices every time
    types = list(set(system.componentlabels)) # some components might have same label. Treat them as identical
    particleSpeciesTypes = np.array([types.index(type) for type in system.particleSpeciesTypes()])

    hists = {}
    for i,type in enumerate(types):
        mask = particleSpeciesTypes==i
        coords = f.particles.position[mask,:]
        if method=='smoothed':
            hists[type] = utility.smoothed_density_1D(coords, box, axis, nBins)
        elif method=='binned':
            hists[type] = utility.binned_density_1D(coords, box, axis, nBins)
    
    # modify histograms so that sum over species in each bin is 1. IE: convert to vol frac
    hists = utility.count_to_volfrac(hists)
    
    return hists

def internaldistances_all(f):

    if isinstance(f, gsd.hoomd.HOOMDTrajectory):
        ts = [f[i].configuration.step for i in range(len(f))]
        func = lambda t: internaldistances_all(t)
        return ts, list(map(func, f))
    
    # get box information
    box = freud.box.Box.from_box(f.configuration.box)
    
    # get cluster
    cluster = getBondedClusters(f)

    # compute and return
    n,avgRsq = structure.meanSqInternalDist(f.particles.position, cluster.cluster_keys, box)

    return n, avgRsq

def internaldistances_species(f, system: systemspec.System):

    if isinstance(f, gsd.hoomd.HOOMDTrajectory):
        ts = [f[i].configuration.step for i in range(len(f))]
        func = lambda t: internaldistances_species(t, system)
        return ts, list(map(func, f))
    
    # get box information
    box = freud.box.Box.from_box(f.configuration.box)

    # get information about which molecule is which type
    types = list(set(system.componentlabels)) # some components might have same label. Treat them as identical
    molSpeciesTypes = np.array([types.index(type) for type in system.speciesTypes()])
    molIndices = system.indicesByMolecule()

    speciesRsq = {}
    for i,type in enumerate(types):
        # pull out the molecules just of this type
        mask = molSpeciesTypes==i
        molOfType = [molIndices[idx] for idx,isType in enumerate(mask) if isType]
        # compute avg internal distances over these molecules. pass all position because indices will be 
        # maintained as relative to the entire system, so can't slice position without changing mol indices
        n,avgRsq = structure.meanSqInternalDist(f.particles.position, molOfType, box)
        speciesRsq[type] = (n,avgRsq)

    return speciesRsq

def volfrac_fields(f, nBins=None):

    if isinstance(f, gsd.hoomd.HOOMDTrajectory):
        ts = [f[i].configuration.step for i in range(len(f))]
        func = lambda t: volfrac_fields(t, nBins=nBins) # to pass non-iterable argument
        return ts, list(map(func, f))

    # f is a frame of a trajectory (a snapshot)
    box = f.configuration.box[0:3]
    particleCoord = f.particles.position
    particleTypeID = f.particles.typeid
    types = f.particles.types

    if nBins == None:
        # determine number of bins based on number of particles
        nParticles = f.particles.N
        nBins = int(0.5 * nParticles**(1/3))

    # compute 3D binned density functions for each particle type
    hists = {}
    for i,type in enumerate(types):
        mask = particleTypeID==i
        coords = particleCoord[mask,:]
        hists[type] = utility.binned_density_ND(coords, box, N=3, nBins=nBins)

    # convert to "volume fractions"
    volfracs = utility.count_to_volfrac(hists)

    return volfracs

def exchange_average(f, nBins=None):

    if isinstance(f, gsd.hoomd.HOOMDTrajectory):
        ts = [f[i].configuration.step for i in range(len(f))]
        func = lambda t: exchange_average(t, nBins=nBins) # to pass non-iterable argument
        return ts, list(map(func, f))

    # f is a frame of a trajectory (a snapshot)
    volfracs = volfrac_fields(f, nBins)

    # Specific to an A-B System! Exchange field, psi order parameter in Kremer/Grest 1996
    exchange = volfracs['A'][0] - volfracs['B'][0]
    
    # Take average of absolute value of exchange field
    avg_exchange = np.mean(np.absolute(exchange))

    return avg_exchange

def overlap_integral(f, nBins=None):

    if isinstance(f, gsd.hoomd.HOOMDTrajectory):
        ts = [f[i].configuration.step for i in range(len(f))]
        func = lambda t: overlap_integral(t, nBins=nBins) # to pass non-iterable argument
        return ts, list(map(func, f))

    # f is a frame of a trajectory (a snapshot)
    volfracs = volfrac_fields(f, nBins)
    types = list(volfracs.keys())
    nTypes = len(types)

    # for each function, compute overlap integral.  
    x = volfracs[types[0]][1] # get coordinates of samples. Should be same for different particle types in same frame with same number of bins
    overlaps = np.zeros((nTypes,nTypes))
    for i in range(nTypes):
        for j in range(i, nTypes):
            dat = np.multiply(volfracs[types[i]][0], volfracs[types[j]][0])
            overlaps[i,j] = utility.integral_ND( dat, x, N=3 )
            overlaps[j,i] = overlaps[i,j]

    return overlaps

def junction_RDF(f, system:systemspec.System, nBins=40, rmax = 5):
    # get junction centers
    junctions = system.junctions()
    pos = np.array([1/2*np.sum(f.particles.position[junc,:],axis=0) for junc in junctions])  

    # compute rdf
    rdf = freud.density.RDF(nBins,rmax)
    rdf.compute(f,pos)

    # return bin centers and histogram
    return rdf.bin_centers, rdf.rdf

def interfacial_tension_IK(dat, edges, axis):

    # here, dat is a HOOMDTrajectory/frame containing log data 
    # edges is a numpy array

    if isinstance(dat, gsd.hoomd.HOOMDTrajectory):
        ts = [dat[i].log["Simulation/timestep"] for i in range(len(dat))]
        func = lambda t: interfacial_tension_IK(t, edges=edges,axis=axis) # to pass non-iterable argument
        return ts, list(map(func, dat))
    
    # gamma is interfacial tension and will be computed via integration
    p_tensor = dat.log['ThermoIK/spatial_pressure_tensor']
    pT_indices = [0, 3, 5]
    pN_idx = pT_indices.pop(axis)
    integrand = p_tensor[:,pN_idx] - 1/2 * np.sum(p_tensor[:,pT_indices],axis=1)
    gamma = np.trapz(integrand,edges[:-1])

    return gamma

def interfacial_tension_global(dat, axis, L):

    # dat is a hoomdtrajectory or frame containing log data
    # axis is the axis normal to the interface(s)

    if isinstance(dat, gsd.hoomd.HOOMDTrajectory):
        ts = [dat[i].log["Simulation/timestep"] for i in range(len(dat))]
        func = lambda t: interfacial_tension_global(t,axis=axis, L=L) # to pass non-iterable argument
        return ts, list(map(func, dat))
    
    # gamma is interfacial tension and will be computed via integration
    p_tensor = dat.log['polymerMD/simtools/custom/Thermo/pressure_tensor'] # ew.
    pT_indices = [0, 3, 5]
    pN_idx = pT_indices.pop(axis)
    pdiff = p_tensor[pN_idx] - 1/2 * np.sum(p_tensor[pT_indices])
    gamma = L * pdiff / 2 # Assume there are two interfaces

    return gamma

def ensemble_average_log(dat):
    avglog = dat[0].log
    nframe = len(dat)

    for frame in dat[1:]:
        for key,val in frame.log.items():
            avglog[key] += val
    
    for key,val in avglog.items():
        avglog[key] = val/nframe

    # remove temporal things
    # avglog.pop("Simulation/timestep",0) # remove in a way that is safe if it doesn't exist? 

    return avglog

