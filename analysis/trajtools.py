import gsd.hoomd
import gsd.pygsd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# utility functions
def read_traj_from_gsd(fname):

    f = gsd.pygsd.GSDFile(open(fname, 'rb'))
    t = gsd.hoomd.HOOMDTrajectory(f)

    return t 

def read_snapshot_from_gsd(fname):
    return read_traj_from_gsd(fname)[-1] # return the last snapshot/frame!

def binned_density_1D(coord, box, axis, nBins):
    # given a set of coordinates (and a box that those coordinates should all fall within, centered on origin),
    # compute the binned density along the specified axis!

    lmin = 0 - box[axis]/2
    lmax = 0 + box[axis]/2

    h = np.histogram(coord[:,axis], nBins, range=(lmin,lmax))

    binvol = box[0]*box[1]*box[2] / nBins
    h = (h[0] / binvol, h[1])

    return h

def binned_density_ND(coord, box, N, nBins):

    boxrange = [(0-box[d]/2, 0+box[d]/2) for d in range(N)] # centered on 0. Specific to how I set up my simulations... and maybe HOOMD convention?

    h = np.histogramdd(coord, nBins, boxrange, density=False)
    
    totalbins = np.product(h[0].shape)
    binvol = box[0]*box[1]*box[2] / totalbins
    h = (h[0] / binvol, h[1])

    return h

def count_to_volfrac(hists):

    # takes a dict of numpy histograms, assumed to each be a count of a different species with the same bins,
    # and converts them to volume fractions such that the sum of each histogram sums to 1 at each bin. 
    # Note: only using np histograms for convenience, these are not proper histograms once they've been rescaled
    # differently at each bin like this!! 

    types = list(hists.keys())
    totcount = np.zeros_like(hists[types[0]][0])
    for hist in hists.values():
        totcount += hist[0]
    for type,hist in hists.items():
        hists[type] = (hists[type][0]/totcount, hists[type][1])

    return hists    

def integral_ND(dat, x, N):

    # dat should be an N-dimension numpy array that is a sample of the function to be integrated
    # x should be a tuple of the sample points of the data in dat in each dimension
    # N should be the number of dimensions

    if len(x) != N or dat.ndim != N:
        # data not correct dimension 
        raise ValueError("Data passed to integral_ND does not match inputted dimension N.")
    
    # check length of arrays in x. If 1 greater than number of data points in that direction, assume these
    # are bin edges, and drop the last one. 
    for d in range(N):
        if len(x[d]) == dat.shape[d]+1:
            x[d] = x[d][:-1]

    I = np.trapz(dat, x = x[0], axis=0)
    for d in range(1,N):
        I = np.trapz(I, x=x[d], axis=0) # always axis 0 because it keeps getting reduced! 

    return I # should be a scalar now!

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
        hists[type] = binned_density_1D(coords, box, axis, nBins)

    # modify histograms so that sum over species in each bin is 1. IE: convert to vol frac
    hists = count_to_volfrac(hists)

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
        hists[type] = binned_density_ND(coords, box, N=3, nBins=nBins)

    # convert to "volume fractions"
    volfracs = count_to_volfrac(hists)

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
            overlaps[i,j] = integral_ND( dat, x, N=3 )
            overlaps[j,i] = overlaps[i,j]

    return overlaps
