import sys
import md_blend
import gsd.hoomd
import gsd.pygsd
from pathlib import Path

def read_snapshot_from_gsd(fname):

    f = gsd.pygsd.GSDFile(open(fname, 'rb'))
    t = gsd.hoomd.HOOMDTrajectory(f)

    return t[-1] # return the last snapshot/frame!

def main(argv):

    f_in = argv[0]
    eps_AB = float(argv[1])

    snap_initial = read_snapshot_from_gsd(f_in)

    # create directory
    Path("dat/struct").mkdir(parents=True, exist_ok=True)
    Path("dat/traj").mkdir(parents=True, exist_ok=True)

    equil_itr = 40000000
    period = 10000
    strID = Path(f_in).stem.split(".")[1]
    strSys = "eps_{:04.2f}".format(eps_AB)
    
    md_blend.run(snap_initial, equil_itr, period, eps_AB, strID, strSys, "dat/")

    return

if __name__ == "__main__":
   main(sys.argv[1:])
