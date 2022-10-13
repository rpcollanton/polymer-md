import numpy as np
import gsd.hoomd
import gsd.pygsd

def read_snapshot_from_gsd(fname):

    f = gsd.pygsd.GSDFile(open(fname, 'rb'))
    t = gsd.hoomd.HOOMDTrajectory(f)

    return t[-1] # return the last snapshot/frame!

def write_gsd_from_snapshot(snapshot, fname):
    with gsd.hoomd.open(name=fname, mode='wb') as f:
            f.append(snapshot)
    return

def wrap_coords(coords,boxsize):

    # wrap coordinates into a rectangular box with side lengths given by boxsize

    dims = len(boxsize)
    wrapped = np.zeros_like(coords)
    for i in range(dims):
        wrapped[:,i] = coords[:,i] - boxsize[i] * np.rint(coords[:,i]/boxsize[i])

    return wrapped 

def translate_coords(coords, box, shift):

    # shift coord[:] to [coord[0] + x, coord[1] + y, coord[2] + z]

    newpos = coords + np.array(shift)
    newpos = wrap_coords(newpos, box[0:3])

    return newpos

def center_box_on_type(snapshot, type='B'):

    # box details
    box = snapshot.configuration.box
    pos = snapshot.particles.position

    # compute the "center of mass" of the particle of type "type"
    particleTypes = snapshot.particles.types
    particleTypeID =  snapshot.particles.typeid
    if type not in particleTypes:
        raise ValueError("Invalid type specified.")
    
    mask = particleTypeID==particleTypes.index(type)
    com = np.average(pos[mask,:],axis=0)

    # shift the box such that the origin is at this center of mass
    pos = translate_coords(pos, box, -1*com)
    snapshot.particles.position = pos

    return snapshot

def center_snapshot_on_type(fold,fnew,type='B'):
    snapshot = read_snapshot_from_gsd(fold)
    snapshot = center_box_on_type(snapshot, type)
    write_gsd_from_snapshot(snapshot, fnew)
    return



