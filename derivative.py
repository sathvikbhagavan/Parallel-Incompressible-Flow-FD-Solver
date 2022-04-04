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

def dfx(F, x_pm, x_pp):
    return (F[x_pp, y0, z0] - F[x_pm, y0, z0]) * gd.i2dx

def dfy(F, x_p0):
    return (F[x_p0, yp1, z0] - F[x_p0, ym1, z0]) * gd.i2dy

def dfz(F, x_p0):
    return (F[x_p0, y0, zp1] - F[x_p0, y0, zm1]) * gd.i2dz

def d2fxx(F, x_p0, x_pm, x_pp):
    return (F[x_pp, y0, z0] - 2.0*F[x_p0, y0, z0] + F[x_pm, y0, z0]) * gd.idx2

def d2fyy(F, x_p0):
    return (F[x_p0, yp1, z0] - 2.0*F[x_p0, y0, z0] + F[x_p0, ym1, z0]) * gd.idy2

def d2fzz(F, x_p0):
    return (F[x_p0, y0, zp1] - 2.0*F[x_p0, y0, z0] + F[x_p0, y0, zm1]) * gd.idz2


