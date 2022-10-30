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
N_A = [64]
M_A = 1063
N_B = [64]
M_B = 1063
N_CP = [64,64,64]
M_CP = 34

x = 81.798734192491
area = 2050.829952
y = np.sqrt(area)
z = y
boxsize = [x,y,z]

# make system
system = systemspec.System()
system.box = boxsize
A = system.addMonomer('A',l)
B = system.addMonomer('B',l)
poly_A = systemspec.LinearPolymerSpec([A], N_A,)
poly_B = systemspec.LinearPolymerSpec([B], N_B)
poly_ABA = systemspec.LinearPolymerSpec([A,B,A], N_CP)
system.addComponent(poly_A, M_A)
system.addComponent(poly_B, M_B)
system.addComponent(poly_ABA, M_CP)

snap = systemgen.build_snapshot(system)
root = "init/"
stem = "A{:03d}_{:04d}_B{:03d}_{:04d}.A{:03d}_B{:03d}_A{:03d}_{:04d}.init.gsd".format(N_A[0], M_A, N_B[0], M_B, N_CP[0], N_CP[1], N_CP[2], M_CP)
fname = root + stem
write_gsd_from_snapshot(snap, fname)
