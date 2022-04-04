import numpy as np
import parameters as para
import boundary as BC
import h5py as hp
import grid as gd
from glob_ import *



def initFields():
    #U, V, W, P, T = deadstate()
    U, V, W, P = TGV()
    
    if para.restart: U[x0, y0, z0], V[x0, y0, z0], W[x0, y0, z0], P[x0, y0, z0], time = restartrun()
        
    fwTime = para.fwInt
    rwTime = para.rwInt

    time = para.time

    if para.restart:
        fwTime = time + para.fwInt
        rwTime = time + para.rwInt

    # BC.imposeUBCs(U), BC.imposeVBCs(V), BC.imposeWBCs(W), BC.imposePBCs(P)
    U[0, :, :], U[-1, :, :] = -U[1, :, :], -U[-2, :, :]
    U[:, 0, :], U[:, -1, :] = -U[:, 1, :], -U[:, -2, :]
    U[:, :, 0], U[:, :, -1] = -U[:, :, 1], -U[:, :, -2]

    V[0, :, :], V[-1, :, :] = -V[1, :, :], -V[-2, :, :]
    V[:, 0, :], V[:, -1, :] = -V[:, 1, :], -V[:, -2, :]
    V[:, :, 0], V[:, :, -1] = -V[:, :, 1], -V[:, :, -2]

    W[0, :, :], W[-1, :, :] = -W[1, :, :], -W[-2, :, :]
    W[:, 0, :], W[:, -1, :] = -W[:, 1, :], -W[:, -2, :]
    W[:, :, 0], W[:, :, -1] = -W[:, :, 1], -W[:, :, -2]
    
    
    P[0, :, :], P[-1, :, :] = -P[1, :, :], -P[-2, :, :]
    P[:, 0, :], P[:, -1, :] = -P[:, 1, :], -P[:, -2, :]
    P[:, :, 0], P[:, :, -1] = -P[:, :, 1], -P[:, :, -2]


    return U, V, W, P, time, fwTime, rwTime


def restartrun():
    filename = "Restart.h5"
    def hdf5_reader(filename,dataset):
        file_V1_read = hp.File(filename, "r")
        dataset_V1_read = file_V1_read["/"+dataset]
        V1=dataset_V1_read[:,:,:]
        return V1
    
    U = hdf5_reader(filename, "Vx")
    V = hdf5_reader(filename, "Vy")
    W = hdf5_reader(filename, "Vz")
    P = hdf5_reader(filename, "P")

    time = para.time

    return U, V, W, P, T, time 


def deadstate():
    U = np.zeros([gd.Nx+2, gd.Ny+2, gd.Nz+2])   
    V = np.zeros([gd.Nx+2, gd.Ny+2, gd.Nz+2]) 
    W = np.zeros([gd.Nx+2, gd.Ny+2, gd.Nz+2])
    P = np.zeros([gd.Nx+2, gd.Ny+2, gd.Nz+2])

    return U, V, W, P


def TGV():
    U = np.zeros(([gd.Nx+2, gd.Ny+2, gd.Nz+2]))
    V = np.zeros(([gd.Nx+2, gd.Ny+2, gd.Nz+2]))
    W = np.zeros(([gd.Nx+2, gd.Ny+2, gd.Nz+2]))
    P = np.zeros(([gd.Nx+2, gd.Ny+2, gd.Nz+2]))

    xu = np.linspace(0, para.Lx + gd.dx, gd.Nx + 2, endpoint=True) - gd.dx/2
    yu = np.linspace(0, para.Ly + gd.dy, gd.Ny + 2, endpoint=True) - gd.dy/2
    zu = np.linspace(0, para.Lz + gd.dz, gd.Nz + 2, endpoint=True) - gd.dz/2

    xu = xu[:, np.newaxis, np.newaxis]
    yu = yu[:, np.newaxis]

    V[x0, y0, z0] = np.sin(-((2*np.pi)/para.Lx)*xu[x0]) * np.cos(((2*np.pi)/para.Ly)*yu[y0]) * np.sin(((2*np.pi)/para.Lz)*zu[z0])
    U[x0, y0, z0] = np.cos(-((2*np.pi)/para.Lx)*xu[x0]) * np.sin(((2*np.pi)/para.Ly)*yu[y0]) * np.sin(((2*np.pi)/para.Lz)*zu[z0])
    W[x0, y0, z0] = 0.0


    return U, V, W, P
