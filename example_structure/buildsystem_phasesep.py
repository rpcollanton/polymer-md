import polymerMD.structure.systemspec as systemspec
import polymerMD.structure.systemgen as systemgen
import numpy as np
import gsd.hoomd

def write_gsd_from_snapshot(snapshot, fname):
    with gsd.hoomd.open(name=fname, mode='wb') as f:
            f.append(snapshot)
    return

# System parameters
l = 1
N_A = 64
M_A = 1063
N_B = 64
M_B = 1063
N_CP = [64,64,64]
M_CP = 34

x = 81.798734192491
area = 2050.829952
y = np.sqrt(area)
z = y
boxsize = [x,y,z]

# phase separated regions
xratio = (N_A*M_A)**(1/3)/(N_A*M_B)**(1/3)
x_A = xratio/(1+xratio)*x
x_B = 1/(1+xratio)*x

reg_A = [x_A,y,z]
regcenter_A = np.array([-x_A/2 - x_B/2, 0, 0])
reg_B = [x_B,y,z]
regcenter_B = np.array([0, 0, 0])
reg_ABA = [x,y,z]
regcenter_ABA = np.array([0, 0, 0])

regions = [reg_A, reg_B, reg_ABA]
regioncenters = [regcenter_A, regcenter_B, regcenter_ABA]

# make system
system = systemspec.System()
system.box = boxsize
A = system.addMonomer('A',l)
B = system.addMonomer('B',l)
poly_A = systemspec.LinearPolymerSpec([A], [N_A])
poly_B = systemspec.LinearPolymerSpec([B], [N_B])
poly_ABA = systemspec.LinearPolymerSpec([A,B,A], N_CP)
system.addComponent(poly_A, M_A)
system.addComponent(poly_B, M_B)
system.addComponent(poly_ABA, M_CP)


snap = systemgen.build_snapshot(system,'boxregions',regions,regioncenters)
root = "/Users/ryancollanton/Desktop/"
stem = "A{:03d}_{:04d}_B{:03d}_{:04d}.A{:03d}_B{:03d}_A{:03d}_{:04d}.init.gsd".format(N_A, M_A, N_B, M_B, N_CP[0], N_CP[1], N_CP[2], M_CP)
fname = root + stem
write_gsd_from_snapshot(snap, fname)