import numpy as np
import grid as gd
import boundary as BC
import derivative as df
import parameters as para
import simpsons as integ
from glob_ import *

xSt, ySt, zSt = 1, 1, 1
xEn, yEn, zEn = gd.Nx+1, gd.Ny+1, gd.Nz+1

x0 = slice(xSt, xEn)
y0 = slice(ySt, yEn)
z0 = slice(zSt, zEn)

xm1 = slice(xSt-1, xEn-1)
xp1 = slice(xSt+1, xEn+1)


def energy(U, V, W):
    tmp2 = U[x0, y0, z0]**2.0 + V[x0, y0, z0]**2.0 + W[x0, y0, z0]**2.0
    u2Int = integ.simps(integ.simps(integ.simps(tmp2, gd.z[z0]), gd.y[y0]), gd.x[x0])/para.Volume
    Energy = 0.5*u2Int

    return Energy


def getDiv(U, V, W):

    tmp2 = df.dfx(U, xm1, xp1) + df.dfy(V, x0) + df.dfz(W, x0)

    maxDiv = np.max(np.abs(tmp2))
    #return np.unravel_index(divNyat.argmax(), divMat.shape), np.mean(divMat)
    return maxDiv