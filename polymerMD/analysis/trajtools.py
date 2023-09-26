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

def lineardistancesfromjunctions(f, system: systemspec.System):

    if isinstance(f, gsd.hoomd.HOOMDTrajectory):
        ts = [f[i].configuration.step for i in range(len(f))]
        func = lambda t: lineardistancesfromjunctions(t, system)
        return ts, list(map(func, f))
    
    # get box information
    box = freud.box.Box.from_box(f.configuration.box)

    # get block particle indices and junctions for each copolymer
    particleindices = [mol for mol in system.indicesByBlockByMolecule() if len(mol) > 1]
    junctionindices = [mol for mol in system.junctionsByMolecule() if len(mol) > 0]
    # structure is [ [[],[]], [[],[]], ... ] where outer is whole system, 2nd level is molecules, 3rd level is blocks
    # so.... particle[0][1][2] gets the index of the third particle in the second block in the first molecule
    # and... junction[0][1] gets the list [a,b] for the second junction of the first molecule, 
    # where a is the first atom in the bond and b is the second
    
    # compute junction coordinates and store in same organization
    junctioncoordinates = []
    for moljunctionindices in junctionindices:
        moljunctioncoordinates = []
        for junction in moljunctionindices:
            coord0 = f.particles.position[[junction[0]],:]
            coord1 = f.particles.position[[junction[1]],:]
            jxncoord = utility.get_midpoint(coord0,coord1,box.L).reshape(-1)
            moljunctioncoordinates.append(jxncoord)
        junctioncoordinates.append(np.array(moljunctioncoordinates))

    # split into endbloocks and midblocks
    endblocks = []
    endblockjunctions = []
    midblocks = []
    midblockjunctions = []
    for molparticleindices, moljunctioncoordinates in zip(particleindices,junctioncoordinates):
        # order indices by nearest junction to furthest from junction
        endblocks.append(molparticleindices[0][::-1]) # [::-1] reverses the order of the list by taking every "-1th" element of list
        endblocks.append(molparticleindices[-1]) 
        endblockjunctions.append(moljunctioncoordinates[0,:])
        endblockjunctions.append(moljunctioncoordinates[-1,:])

        # check if this is longer than a diblock (ie if there are midblocks)
        if len(molparticleindices) > 2: # if more than 2 blocks
            # for midblocks, split down the middle associate with correct junction
            for i, midblock in enumerate(molparticleindices[1:-1]):
                blocklength = len(midblock)
                # first half of block
                midblocks.append(midblock[0:int(blocklength/2)])
                midblockjunctions.append(moljunctioncoordinates[i,:])
                # other half of block
                midblocks.append(midblock[int(blocklength/2):][::-1])
                midblockjunctions.append(moljunctioncoordinates[i+1,:])

    endblockjunctions = np.array(endblockjunctions)
    midblockjunctions = np.array(midblockjunctions)

    #print("endblock indices")
    #print(endblocks[9])
    #print("diff between coordinate 0 and junction coordinate ")
    #for i,block in enumerate(endblocks):
    #    print(f.particles.position[block[0],:]-endblockjunctions[i,:])


    # get Rsq vs n for endblocks and midblocks where n is distance from junction
    blockdata = {}
    endblockAvgRsq, endblockN = structure.meanSqDistanceFromJunction(f.particles.position, endblocks, endblockjunctions, box)
    blockdata['endblocks'] = (endblockN, endblockAvgRsq)
    
    if len(midblocks) > 0:
        midblockAvgRsq, midblockN = structure.meanSqDistanceFromJunction(f.particles.position, midblocks, midblockjunctions, box)
        blockdata['midblocks'] = (midblockN, midblockAvgRsq)
    
    return blockdata

def volfrac_fields(f, nBins=None, density_type='binned'):

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
        if density_type=='binned':
            hists[type] = utility.binned_density_ND(coords, box, N=3, nBins=nBins)
        elif density_type=='gaussian':
            hists[type] = utility.gaussian_density_ND(coords, box, N=3, nBins=nBins)

    # convert to "volume fractions"
    volfracs = utility.count_to_volfrac(hists)

    return volfracs

def exchange_average(f, nBins=None,density_type='binned'):

    if isinstance(f, gsd.hoomd.HOOMDTrajectory):
        ts = [f[i].configuration.step for i in range(len(f))]
        func = lambda t: exchange_average(t, nBins=nBins) # to pass non-iterable argument
        return ts, list(map(func, f))

    # f is a frame of a trajectory (a snapshot)
    volfracs = volfrac_fields(f, nBins,density_type=density_type)

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

def junction_RDF(f, system:systemspec.System, axis, nBins=40, rmax = 5):
    # get junction centers
    junctions = system.junctions()
    pos = np.array([utility.get_midpoint(
        f.particles.position[[junc[0]],:],
        f.particles.position[[junc[1]],:],
        f.configuration.box
    ).reshape(-1) for junc in junctions])  

    # need to compute and accumulate rdf for two separate interfaces, in 2D.
    axis_indices = [i for i in range(3) if i!=axis]
    
    pos2D_left = pos[np.where(pos[:,axis] < 0)[0],:]
    pos2D_right = pos[np.where(pos[:,axis] > 0)[0],:]
    pos2D_left[:,axis] = 0 # set to 0 per freud requirement for 2D boxes
    pos2D_right[:,axis] = 0

    # swap columns so that coordinates in plane of interface are in x,y positions
    for i in range(len(axis_indices)):
        pos2D_left[:,[i,axis_indices[i]]] = pos2D_left[:,[axis_indices[i],i]]
        pos2D_right[:,[i,axis_indices[i]]] = pos2D_right[:,[axis_indices[i],i]]

    box2D = [f.configuration.box[i] for i in axis_indices]
    # compute rdf
    rdf = freud.density.RDF(nBins,rmax)
    rdf.compute((box2D,pos2D_left),reset=False)
    rdf.compute((box2D,pos2D_right),reset=False)

    # return bin centers and histogram
    return rdf.bin_centers, rdf.rdf

def junction_RDF_accumulate(fs, systems, axis, nBins=40, rmax = 5):
    
    # create rdf object
    rdf = freud.density.RDF(nBins,rmax)

    # need to compute and accumulate rdf for two separate interfaces, in 2D.
    axis_indices = [i for i in range(3) if i!=axis]
   
    # for each system to be accumulated into final rdf
    for f,system in zip(fs,systems):
        # get junction centers
        junctions = system.junctions()
        if not junctions:
            ValueError("Can not calculate junction RDF for system with no copolymers.")
        pos = np.array([utility.get_midpoint(
            f.particles.position[[junc[0]],:],
            f.particles.position[[junc[1]],:],
            f.configuration.box
        ).reshape(-1) for junc in junctions])
        pos2D_left = pos[np.where(pos[:,axis] < 0)[0],:]
        pos2D_right = pos[np.where(pos[:,axis] > 0)[0],:]
        pos2D_left[:,axis] = 0 # set to 0 per freud requirement for 2D boxes
        pos2D_right[:,axis] = 0

        # swap columns so that coordinates in plane of interface are in x,y positions
        for i in range(len(axis_indices)):
            pos2D_left[:,[i,axis_indices[i]]] = pos2D_left[:,[axis_indices[i],i]]
            pos2D_right[:,[i,axis_indices[i]]] = pos2D_right[:,[axis_indices[i],i]]

        box2D = [f.configuration.box[i] for i in axis_indices]
    
        rdf.compute((box2D,pos2D_left),reset=False)
        rdf.compute((box2D,pos2D_right),reset=False)

    # return rdf object
    return rdf

def junction_density_smeared(f, system: systemspec.System, axis, nBins=500, sigma=4.0):
    
    # get system information
    axis_indices = [i for i in range(3) if i!=axis]
    box2D = [f.configuration.box[i] for i in axis_indices]

    # get junction positions and convert to 2D for left and right
    junctions = system.junctions()
    pos = np.array([utility.get_midpoint(
        f.particles.position[[junc[0]],:],
        f.particles.position[[junc[1]],:],
        f.configuration.box
    ).reshape(-1) for junc in junctions])
    pos2D_left = pos[np.where(pos[:,axis] < 0)[0],:]
    pos2D_right = pos[np.where(pos[:,axis] > 0)[0],:]
    pos2D_left[:,axis] = 0 # set to 0 per freud requirement for 2D boxes
    pos2D_right[:,axis] = 0

    # swap columns so that coordinates in plane of interface are in x,y positions
    for i in range(len(axis_indices)):
        pos2D_left[:,[i,axis_indices[i]]] = pos2D_left[:,[axis_indices[i],i]]
        pos2D_right[:,[i,axis_indices[i]]] = pos2D_right[:,[axis_indices[i],i]]
    
    aq_left  = freud.AABBQuery(box2D, pos2D_left)
    aq_right = freud.AABBQuery(box2D, pos2D_right)
    gd_left  = freud.density.GaussianDensity(nBins, box2D[0] / 3, sigma)
    gd_right = freud.density.GaussianDensity(nBins, box2D[0] / 3, sigma)
    gd_left.compute(aq_left)
    gd_right.compute(aq_right)

    return gd_left, gd_right

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

def interfacial_tension_global(dat, axis, L=None):

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
    # get L from current frame if L = none. Useful if L can change
    if L==None:
        L = dat.configuration.box[axis]
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

