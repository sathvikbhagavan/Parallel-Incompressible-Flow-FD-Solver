import numpy as np
import parameters as para


sLst = [2**x for x in range(12)]

N = [(sLst[x[0]], sLst[x[1]], sLst[x[2]]) for x in [para.sInd - y for y in range(para.VDepth + 1)]]

Nx, Ny, Nz = sLst[para.sInd[0]], sLst[para.sInd[1]], sLst[para.sInd[2]]

dx, dy, dz = para.Lx/Nx, para.Ly/Ny, para.Lz/Nz

i2dx, i2dy, i2dz = 1.0/(2.0*dx), 1.0/(2.0*dy), 1.0/(2.0*dz)

dx2, dy2, dz2 = dx*dx, dy*dy, dz*dz

idx2, idy2, idz2 = 1.0/dx2, 1.0/dy2, 1.0/dz2

x = np.linspace(0, para.Lx + dx, Nx + 2, endpoint=True) - dx/2.0
y = np.linspace(0, para.Ly + dy, Ny + 2, endpoint=True) - dy/2.0
z = np.linspace(0, para.Lz + dz, Nz + 2, endpoint=True) - dz/2.0