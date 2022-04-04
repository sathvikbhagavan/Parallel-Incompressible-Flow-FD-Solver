import parameters as para
import grid as gd
import h5py as hp
from glob_ import *


def writeSoln_RBC(Vx, Vy, Vz, Press, time):

    fName = "Soln_" + "{0:09.5f}.h5".format(time)
    #print("#Writing solution file: ", fName)        
    f = hp.File(fName, "w")

    dset = f.create_dataset("Vx", data = Vx[x0, y0, z0])
    dset = f.create_dataset("Vy", data = Vy[x0, y0, z0])
    dset = f.create_dataset("Vz", data = Vz[x0, y0, z0])
    dset = f.create_dataset("P", data = Press[x0, y0, z0])
    dset = f.create_dataset("X", data = gd.x)
    dset = f.create_dataset("Y", data = gd.y)
    dset = f.create_dataset("Z", data = gd.z)
    f.close()


def writeRestart_RBC(Vx, Vy, Vz, Press, time):

    fName = "Restart.h5"
    #print("#Writing Restart file: ", fName)        
    f = hp.File(fName, "w")

    dset = f.create_dataset("Vx", data = Vx[x0, y0, z0])
    dset = f.create_dataset("Vy", data = Vy[x0, y0, z0])
    dset = f.create_dataset("Vz", data = Vz[x0, y0, z0])
    dset = f.create_dataset("P", data = Press[x0, y0, z0])
    dset = f.create_dataset("Time", data = timers)
    dset = f.create_dataset("X", data = gd.x)
    dset = f.create_dataset("Y", data = gd.y)
    dset = f.create_dataset("Z", data = gd.z)
    f.close()

