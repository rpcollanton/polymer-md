import gsd.hoomd
import gsd.pygsd
import numpy as np
import scipy as sp
from polymerMD.analysis import utility

# analysis functions
def density_system(f):
    # f is a trajectory or trajectory frame
    
    if isinstance(f, gsd.hoomd.HOOMDTrajectory):
        ts = [f[i].configuration.step for i in range(len(f))]
        return ts, list(map(density_system, f))
    
    box = f.configuration.box
    V = box[0]*box[1]*box[2]
    N = f.particles.N

    return N/V

def density_profile_1D(f, nBins=100, axis=0):
    # f is a trajectory or trajectory frame
    # axis is the axis to plot the density along. averaged over other two

    if isinstance(f, gsd.hoomd.HOOMDTrajectory):
        ts = [f[i].configuration.step for i in range(len(f))]
        return ts, list(map(density_profile_1D, f))

    box = f.configuration.box[0:3]
    particleCoord = f.particles.position
    particleTypeID = f.particles.typeid
    types = f.particles.types

    hists = {}
    for i,type in enumerate(types):
        mask = particleTypeID==i
        coords = particleCoord[mask,:]
        hists[type] = utility.binned_density_1D(coords, box, axis, nBins)

    # modify histograms so that sum over species in each bin is 1. IE: convert to vol frac
    hists = utility.count_to_volfrac(hists)

    return hists

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

def interfacial_tension_IK(dat, edges, axis):

    # here, dat is a HOOMDTrajectory/frame containing log data 
    # edges is a numpy array

    if isinstance(dat, gsd.hoomd.HOOMDTrajectory):
        ts = [dat[i].log["Simulation/timestep"] for i in range(len(dat))]
        func = lambda t: overlap_integral(t, edges=edges,axis=axis) # to pass non-iterable argument
        return ts, list(map(func, dat))
    
    # gamma is interfacial tension and will be computed via integration
    p_tensor = dat.log['Thermo1DSpatial/spatial_pressure_tensor']
    pT_indices = [0, 3, 5]
    pN_idx = pT_indices.pop(axis)
    integrand = p_tensor[:,pN_idx] - 1/2 * np.sum(p_tensor[:,pT_indices],axis=1)

    gamma = np.trapz(integrand,edges[:-1,axis])

    return gamma




    
