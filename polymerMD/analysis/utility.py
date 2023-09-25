import gsd.hoomd
import gsd.pygsd
import numpy as np
import freud
from scipy import stats

# utility functions
def read_traj_from_gsd(fname):

    f = gsd.pygsd.GSDFile(open(fname, 'rb'))
    t = gsd.hoomd.HOOMDTrajectory(f)

    return t 

def read_snapshot_from_gsd(fname):
    return read_traj_from_gsd(fname)[-1] # return the last snapshot/frame!

def write_gsd_from_snapshot(snapshot, fname):
    with gsd.hoomd.open(name=fname, mode='wb') as f:
            f.append(snapshot)
    return

def wrap_coords(coords,boxsize):

    # wrap coordinates into a rectangular box with side lengths given by boxsize

    dims = len(boxsize)
    if dims > 3:
        dims = 3
        boxsize = boxsize[0:3]
    
    wrapped = np.zeros_like(coords)
    for i in range(dims):
        wrapped[:,i] = coords[:,i] - boxsize[i] * np.rint(coords[:,i]/boxsize[i])

    return wrapped 

def get_midpoint(coord0,coord1,box):
    
    diff = wrap_coords(coord1-coord0,box)
    midpt = wrap_coords(coord0 + diff/2,box)

    return midpt

def shift_pbc(coords, box, shift):
    # dimensions need to match
    # coords N x d, box 1 x d, shift 1 x d

    coords = coords+np.multiply(shift,box)
    coords = wrap_coords(coords,box)

    return coords

def binned_density_1D(coord, box, axis, nBins):
    # given a set of coordinates (and a box that those coordinates should all fall within, centered on origin),
    # compute the binned density along the specified axis!

    lmin = 0 - box[axis]/2
    lmax = 0 + box[axis]/2

    h = np.histogram(coord[:,axis], nBins, range=(lmin,lmax))

    binvol = box[0]*box[1]*box[2] / nBins
    h = (h[0] / binvol, h[1])

    return h

def smoothed_density_1D(coord, box, axis, nBins):
    # given a set of coordinates (and a box that those coordinates should all fall within, centered on origin),
    # compute the gaussian smoothed density along the specified axis!

    lmin = 0 - box[axis]/2
    lmax = 0 + box[axis]/2

    locs = coord[:,axis]
    nparticles = np.shape(locs)[0]
    scale = (lmax-lmin)/75 # might need to adjust 
    dists = [stats.norm(loc = loc, scale = scale) for loc in locs]

    nedges = nBins+1
    xbins = np.linspace(lmin,lmax,nedges)
    dens = np.zeros_like(xbins[:-1])
    for dist in dists:
        dens += dist.pdf(xbins[:-1])/nparticles

    # account for periodic boundary conditions!
    xpbc = np.zeros_like(xbins)
    for i in range(nedges):
        if xbins[i] > 0:
            xpbc[i] = xbins[i]-box[axis]
        elif xbins[i] <= 0:
            xpbc[i] = xbins[i]+box[axis]
    for dist in dists:
        dens += dist.pdf(xpbc[:-1])/nparticles
        

    return (dens,xbins) # tuple to be consistent with binned density histograms

def binned_density_ND(coord, box, N, nBins):

    boxrange = [(0-box[d]/2, 0+box[d]/2) for d in range(N)] # centered on 0. Specific to how I set up my simulations... and maybe HOOMD convention?

    h = np.histogramdd(coord, nBins, boxrange, density=False)
    
    totalbins = np.product(h[0].shape)
    binvol = box[0]*box[1]*box[2] / totalbins
    h = (h[0] / binvol, h[1])

    return h

def gaussian_density_ND(coord, box, N, nBins,sigma=2**(1/6)):
    cutoff = np.amax(box)/3
    gd = freud.density.GaussianDensity(nBins,cutoff,sigma)
    gd.compute((box,coord))

    boxrange = [(0-box[d]/2, 0+box[d]/2) for d in range(N)]
    bins = [np.linspace(boxrange[d][0],boxrange[d][1],nBins+1) for d in range(N)]
    h = (gd.density, bins)

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

def findInterfaceAxis(snapshot, species):

    # finds the most likely axis of the interface
    # computes 1D binned densities of each species in species along each axis
    # the axis with the highest range is likely the one perpendicular to the interface

    return
