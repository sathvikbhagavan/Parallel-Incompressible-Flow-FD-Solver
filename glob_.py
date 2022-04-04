import numpy as np
import parameters as para
import grid as gd

xSt, ySt, zSt = 1, 1, 1
xEn, yEn, zEn = gd.Nx+1, gd.Ny+1, gd.Nz+1

x0 = slice(xSt, xEn)
xm1 = slice(xSt-1, xEn-1)
xp1 = slice(xSt+1, xEn+1)

y0 = slice(ySt, yEn)
ym1 = slice(ySt-1, yEn-1)
yp1 = slice(ySt+1, yEn+1)

z0 = slice(zSt, zEn)
zm1 = slice(zSt-1, zEn-1)
zp1 = slice(zSt+1, zEn+1)

tmp = np.zeros([gd.Nx+2, gd.Ny+2, gd.Nz+2])
rhs = np.zeros([gd.Nx+2, gd.Ny+2, gd.Nz+2])
Pp = np.zeros([gd.Nx+2, gd.Ny+2, gd.Nz+2])
tmp2 = np.zeros([gd.Nx, gd.Ny, gd.Nz])
	
