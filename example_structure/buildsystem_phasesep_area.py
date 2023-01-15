from polymerMD.structure import systemspec
from polymerMD.structure import systemgen
import numpy as np
import gsd.hoomd
import sys

def write_gsd_from_snapshot(snapshot, fname):
    with gsd.hoomd.open(name=fname, mode='wb') as f:
            f.append(snapshot)
    return

# input from command line
if __name__=="__main__":
    ncp = int(sys.argv[1]) # one length for each block of copolymer
    beadscale = float(sys.argv[2])

# system parameters
rho = 0.85
N_A = 64
N_B = 64
N_CP = [ncp,ncp,ncp]
M_CP = 34
l = 1 # monomer size of some sort... unclear currently

# scaled dimensions
M_orig = 1063
x_orig = 81.798734192491
n_bead_orig = M_orig*(N_A + N_B) + M_CP*sum(N_CP)
area_orig = n_bead_orig/rho/x_orig

M_A = int(M_orig*beadscale)
M_B = int(M_orig*beadscale)
n_beads = N_A*M_A + N_B*M_B + M_CP*sum(N_CP)
x = x_orig
area = n_beads/rho/x
y = np.sqrt(area)
z = y
boxsize = [x,y,z]
print("Resulting area scale: {:f}".format(area/area_orig))

# phase separated regions
xratio = (N_A*M_A)**(1/3)/(N_B*M_B)**(1/3)
x_A = xratio/(1+xratio)*x
x_B = 1/(1+xratio)*x

reg_A = [x_A,y,z]
regcenter_A = np.array([-x_A/2 - x_B/2, 0, 0])
reg_B = [x_B,y,z]
regcenter_B = np.array([0, 0, 0])
delta = 0.001
reg_ABA = [delta,y,z]
regcenter_ABA_1 = np.array([-x_B/2, 0, 0])
regcenter_ABA_2 = np.array([+x_B/2, 0, 0])


regions = [reg_A, reg_B]
regioncenters = [regcenter_A, regcenter_B]

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
if M_CP != 0 and ncp != 0:
    regions += [reg_ABA, reg_ABA]
    regioncenters += [regcenter_ABA_1, regcenter_ABA_2]
    system.addComponent(poly_ABA, int(M_CP/2))
    system.addComponent(poly_ABA, int(M_CP/2)) # two groups for two different regions...


snap = systemgen.build_snapshot(system,'boxregions',regions,regioncenters)
root = "init/"
stem = "A{:03d}_{:04d}_B{:03d}_{:04d}.A{:03d}_B{:03d}_A{:03d}_{:04d}.init_phasesep.gsd".format(N_A, M_A, N_B, M_B, N_CP[0], N_CP[1], N_CP[2], M_CP)
fname = root + stem
write_gsd_from_snapshot(snap, fname)
