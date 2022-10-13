import systemspec
import numpy as np
import systemgen
import gsd.hoomd

def write_gsd_from_snapshot(snapshot, fname):
    with gsd.hoomd.open(name=fname, mode='wb') as f:
            f.append(snapshot)
    return

# System parameters
l = 1
Alength = [64]
M_A = 1063
Blength = [64]
M_B = 1063
CPlength = [64,64,64]
CPtype = ['A','B','A']
M_CP = 34
x = 81.798734192491
area = 2050.829952
y = np.sqrt(area)
z = y
boxsize = [x,y,z,np.pi/2,np.pi/2,np.pi/2]

A = systemspec.LinearPolymerSpec(1, ['A'], Alength, [l])
B = systemspec.LinearPolymerSpec(1, ['B'], Blength, [l])
CP = systemspec.LinearPolymerSpec(3, CPtype, CPlength, [l,l,l])

# make system
system = systemspec.System()
system.box = boxsize
system.nComponents = 3
system.component[0] = A
system.component[1] = B
system.component[2] = CP

#snap = systemgen.build_snapshot(system)
root = "/Users/ryancollanton/Desktop/"
stem = "A{:03d}_{:04d}_B{:03d}_{:04d}.A{:03d}_B{:03d}_A{:03d}_{:04d}.init.gsd".format(Alength[0], M_A, Blength[0], M_B, CPlength[0], CPlength[1], CPlength[2], M_CP)
print(stem)
fname = root + stem
#write_gsd_from_snapshot(fname)