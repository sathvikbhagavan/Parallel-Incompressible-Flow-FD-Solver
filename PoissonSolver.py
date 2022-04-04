import numpy as np
import grid as gd
import parameters as para
import boundary as BC
import numba
from glob_ import *


def Poisson_Jacobi(rho):
    jCnt = 0
    # Pp.fill(0.0)
    while True:

        Pp[x0, y0, z0] = (1.0/(-2.0*(gd.idx2 + gd.idy2 + gd.idz2))) * (rho[x0, y0, z0] -
                                                                       gd.idx2*(Pp[xm1, y0, z0] + Pp[xp1, y0, z0]) -
                                                                       gd.idy2*(Pp[x0, ym1, z0] + Pp[x0, yp1, z0]) -
                                                                       gd.idz2*(Pp[x0, y0, zm1] + Pp[x0, y0, zp1]))
        BC.imposePpBCs(Pp)

        maxErr = np.amax(np.abs(rho[x0, y0, z0] - ((
                        (Pp[xm1, y0, z0] - 2.0*Pp[x0, y0, z0] + Pp[xp1, y0, z0])*gd.idx2 +
                        (Pp[x0, ym1, z0] - 2.0*Pp[x0, y0, z0] + Pp[x0, yp1, z0])*gd.idy2 +
                        (Pp[x0, y0, zm1] - 2.0*Pp[x0, y0, z0] + Pp[x0, y0, zp1])*gd.idz2))))

        if (jCnt % 100 == 0):
            print(jCnt, maxErr)
        jCnt += 1

        if maxErr < para.PoissonTolerance:
            # print(jCnt)
            break

        if jCnt > para.maxiteration:
            print("ERROR: Jacobi Poisson solver not converging. Aborting")
            quit()

    return Pp


def initMG():
    global N, Nx, Ny, Nz
    global pData, rData, sData, iTemp
    global hyhz, hzhx, hxhy, hxhyhz, gsFactor, maxCount, nList, VDepth
    global mghx2, mghy2, mghz2, mghx, mghy, mghz

    Lx, Ly, Lz = para.Lx, para.Ly, para.Lz

    sInd = para.sInd

    VDepth = para.VDepth

    # N should be of the form 2^n
    # Then there will be 2^n + 2 points, including two ghost points
    sLst = [2**x for x in range(12)]

    Nx, Ny, Nz = sLst[sInd[0]], sLst[sInd[1]], sLst[sInd[2]]

    #############################################################

    # Get array of grid sizes are tuples corresponding to each level of V-Cycle
    N = [(sLst[x[0]], sLst[x[1]], sLst[x[2]])
         for x in [sInd - y for y in range(VDepth + 1)]]

    # Define array of grid spacings along X
    h0 = Lx/(N[0][0])
    mghx = [h0*(2**x) for x in range(VDepth+1)]

    # Define array of grid spacings along Y
    h0 = Ly/(N[0][1])
    mghy = [h0*(2**x) for x in range(VDepth+1)]

    # Define array of grid spacings along Z
    h0 = Lz/(N[0][2])
    mghz = [h0*(2**x) for x in range(VDepth+1)]

    # Square of hx, used in finite difference formulae
    mghx2 = [x*x for x in mghx]

    # Square of hy, used in finite difference formulae
    mghy2 = [x*x for x in mghy]

    # Square of hz, used in finite difference formulae
    mghz2 = [x*x for x in mghz]

    # Cross product of hy and hz, used in finite difference formulae
    hyhz = [mghy2[i]*mghz2[i] for i in range(VDepth + 1)]

    # Cross product of hx and hz, used in finite difference formulae
    hzhx = [mghx2[i]*mghz2[i] for i in range(VDepth + 1)]

    # Cross product of hx and hy, used in finite difference formulae
    hxhy = [mghx2[i]*mghy2[i] for i in range(VDepth + 1)]

    # Cross product of hx, hy and hz used in finite difference formulae
    hxhyhz = [mghx2[i]*mghy2[i]*mghz2[i] for i in range(VDepth + 1)]

    # Factor in denominator of Gauss-Seidel iterations
    gsFactor = [1.0/(2.0*(hyhz[i] + hzhx[i] + hxhy[i]))
                for i in range(VDepth + 1)]

    # Maximum number of iterations while solving at coarsest level
    maxCount = 10*N[-1][0]*N[-1][1]*N[-1][2]

    # Integer specifying the level of V-cycle at any point while solving
    vLev = 0

    nList = np.array(N)

    pData = [np.zeros(tuple(x)) for x in nList + 2]
    rData = [np.zeros_like(x) for x in pData]
    sData = [np.zeros_like(x) for x in pData]
    iTemp = [np.zeros_like(x) for x in pData]


############################## MULTI-GRID SOLVER ###############################

# The root function of MG-solver. And H is the RHS
def Poisson_MG(H, iCnt):
    global N, Nx, Ny, Nz
    global pData, rData, sData, iTemp
    global hyhz, hzhx, hxhy, hxhyhz, gsFactor, maxCount, nList, VDepth
    global mghx2, mghy2, mghz2, mghx, mghy, mghz

    if iCnt == 0:
        initMG()

    rData[0] = H

    vcnt = 0
    for i in range(para.vcCnt):
        v_cycle(para.preSm, para.pstSm)

        resVal = float(
            np.amax(np.abs(H[1:-1, 1:-1, 1:-1] - laplace(pData[0]))))

        vcnt += 1

        #print("vcnt", vcnt, resVal)

        if resVal < para.PoissonTolerance:
            #print("multigrid v-cycles:", vcnt)
            break

    return pData[0]


# Multigrid V-cycle without the use of recursion
def v_cycle(preSm, pstSm):
    global VDepth
    global vLev
    #global pstSm, preSm

    vLev = 0
    zeroBC = False

    # Pre-smoothing
    smooth(para.preSm)

    zeroBC = True
    for i in range(para.VDepth):
        # Compute residual
        calcResidual()

        # Copy smoothed pressure for later use
        sData[vLev] = np.copy(pData[vLev])

        # Restrict to coarser level
        restrict()

        # Reinitialize pressure at coarser level to 0 - this is critical!
        pData[vLev].fill(0.0)

        # If the coarsest level is reached, solve. Otherwise, keep smoothing!
        if vLev == para.VDepth:
            # solve()
            smooth(preSm)
        else:
            smooth(preSm)

    # Prolongation operations
    for i in range(para.VDepth):
        # Prolong pressure to next finer level
        prolong()

        # Add previously stored smoothed data
        pData[vLev] += sData[vLev]

        # Post-smoothing
        smooth(pstSm)


# Smoothens the solution sCount times using Gauss-Seidel smoother
def smooth(sCount):
    global N
    global vLev
    global gsFactor
    global rData, pData
    global hyhz, hzhx, hxhy, hxhyhz

    n = N[vLev]
    for iCnt in range(sCount):
        imposePpBCs(pData[vLev])
        # Jacobi
        '''
        pData[vLev][1:-1, 1:-1, 1:-1] = (hyhz[vLev]*(pData[vLev][2:, 1:-1, 1:-1] + pData[vLev][:-2, 1:-1, 1:-1]) +
                                                   hzhx[vLev]*(pData[vLev][1:-1, 2:, 1:-1] + pData[vLev][1:-1, :-2, 1:-1]) +
                                                   hxhy[vLev]*(pData[vLev][1:-1, 1:-1, 2:] + pData[vLev][1:-1, 1:-1, :-2]) -
                                                 hxhyhz[vLev]*rData[vLev][1:-1, 1:-1, 1:-1]) * gsFactor[vLev]
        
        '''
        # RBGS
        pData[vLev][1:-1:2, 1:-1:2, 1:-1:2] = (hyhz[vLev]*(pData[vLev][2::2, 1:-1:2, 1:-1:2] + pData[vLev][:-2:2, 1:-1:2, 1:-1:2]) +
                                               hzhx[vLev]*(pData[vLev][1:-1:2, 2::2, 1:-1:2] + pData[vLev][1:-1:2, :-2:2, 1:-1:2]) +
                                               hxhy[vLev]*(pData[vLev][1:-1:2, 1:-1:2, 2::2] + pData[vLev][1:-1:2, 1:-1:2, :-2:2]) -
                                               hxhyhz[vLev]*rData[vLev][1:-1:2, 1:-1:2, 1:-1:2]) * gsFactor[vLev]

        # 1, 1, 0 configuration
        pData[vLev][2::2, 2::2, 1:-1:2] = (hyhz[vLev]*(pData[vLev][3::2, 2::2, 1:-1:2] + pData[vLev][1:-1:2, 2::2, 1:-1:2]) +
                                           hzhx[vLev]*(pData[vLev][2::2, 3::2, 1:-1:2] + pData[vLev][2::2, 1:-1:2, 1:-1:2]) +
                                           hxhy[vLev]*(pData[vLev][2::2, 2::2, 2::2] + pData[vLev][2::2, 2::2, :-2:2]) -
                                           hxhyhz[vLev]*rData[vLev][2::2, 2::2, 1:-1:2]) * gsFactor[vLev]

        # 1, 0, 1 configuration
        pData[vLev][2::2, 1:-1:2, 2::2] = (hyhz[vLev]*(pData[vLev][3::2, 1:-1:2, 2::2] + pData[vLev][1:-1:2, 1:-1:2, 2::2]) +
                                           hzhx[vLev]*(pData[vLev][2::2, 2::2, 2::2] + pData[vLev][2::2, :-2:2, 2::2]) +
                                           hxhy[vLev]*(pData[vLev][2::2, 1:-1:2, 3::2] + pData[vLev][2::2, 1:-1:2, 1:-1:2]) -
                                           hxhyhz[vLev]*rData[vLev][2::2, 1:-1:2, 2::2]) * gsFactor[vLev]

        # 0, 1, 1 configuration
        pData[vLev][1:-1:2, 2::2, 2::2] = (hyhz[vLev]*(pData[vLev][2::2, 2::2, 2::2] + pData[vLev][:-2:2, 2::2, 2::2]) +
                                           hzhx[vLev]*(pData[vLev][1:-1:2, 3::2, 2::2] + pData[vLev][1:-1:2, 1:-1:2, 2::2]) +
                                           hxhy[vLev]*(pData[vLev][1:-1:2, 2::2, 3::2] + pData[vLev][1:-1:2, 2::2, 1:-1:2]) -
                                           hxhyhz[vLev]*rData[vLev][1:-1:2, 2::2, 2::2]) * gsFactor[vLev]

        # Update black cells
        # 1, 0, 0 configuration
        pData[vLev][2::2, 1:-1:2, 1:-1:2] = (hyhz[vLev]*(pData[vLev][3::2, 1:-1:2, 1:-1:2] + pData[vLev][1:-1:2, 1:-1:2, 1:-1:2]) +
                                             hzhx[vLev]*(pData[vLev][2::2, 2::2, 1:-1:2] + pData[vLev][2::2, :-2:2, 1:-1:2]) +
                                             hxhy[vLev]*(pData[vLev][2::2, 1:-1:2, 2::2] + pData[vLev][2::2, 1:-1:2, :-2:2]) -
                                             hxhyhz[vLev]*rData[vLev][2::2, 1:-1:2, 1:-1:2]) * gsFactor[vLev]

        # 0, 1, 0 configuration
        pData[vLev][1:-1:2, 2::2, 1:-1:2] = (hyhz[vLev]*(pData[vLev][2::2, 2::2, 1:-1:2] + pData[vLev][:-2:2, 2::2, 1:-1:2]) +
                                             hzhx[vLev]*(pData[vLev][1:-1:2, 3::2, 1:-1:2] + pData[vLev][1:-1:2, 1:-1:2, 1:-1:2]) +
                                             hxhy[vLev]*(pData[vLev][1:-1:2, 2::2, 2::2] + pData[vLev][1:-1:2, 2::2, :-2:2]) -
                                             hxhyhz[vLev]*rData[vLev][1:-1:2, 2::2, 1:-1:2]) * gsFactor[vLev]

        # 0, 0, 1 configuration
        pData[vLev][1:-1:2, 1:-1:2, 2::2] = (hyhz[vLev]*(pData[vLev][2::2, 1:-1:2, 2::2] + pData[vLev][:-2:2, 1:-1:2, 2::2]) +
                                             hzhx[vLev]*(pData[vLev][1:-1:2, 2::2, 2::2] + pData[vLev][1:-1:2, :-2:2, 2::2]) +
                                             hxhy[vLev]*(pData[vLev][1:-1:2, 1:-1:2, 3::2] + pData[vLev][1:-1:2, 1:-1:2, 1:-1:2]) -
                                             hxhyhz[vLev]*rData[vLev][1:-1:2, 1:-1:2, 2::2]) * gsFactor[vLev]

        # 1, 1, 1 configuration
        pData[vLev][2::2, 2::2, 2::2] = (hyhz[vLev]*(pData[vLev][3::2, 2::2, 2::2] + pData[vLev][1:-1:2, 2::2, 2::2]) +
                                         hzhx[vLev]*(pData[vLev][2::2, 3::2, 2::2] + pData[vLev][2::2, 1:-1:2, 2::2]) +
                                         hxhy[vLev]*(pData[vLev][2::2, 2::2, 3::2] + pData[vLev][2::2, 2::2, 1:-1:2]) -
                                         hxhyhz[vLev]*rData[vLev][2::2, 2::2, 2::2]) * gsFactor[vLev]

    imposePpBCs(pData[vLev])


# Compute the residual and store it into iTemp array
def calcResidual():
    global vLev
    global iTemp, rData, pData

    iTemp[vLev].fill(0.0)
    iTemp[vLev][1:-1, 1:-1, 1:-1] = rData[vLev][1:-
                                                1, 1:-1, 1:-1] - laplace(pData[vLev])


# Restricts the data from an array of size 2^n to a smaller array of size 2^(n - 1)
def restrict():
    global N
    global vLev
    global iTemp, rData

    pLev = vLev
    vLev += 1

    n = N[vLev]
    rData[vLev][1:-1, 1:-1, 1:-1] = (iTemp[pLev][1:-1:2, 1:-1:2, 1:-1:2] + iTemp[pLev][2::2, 2::2, 2::2] +
                                     iTemp[pLev][1:-1:2, 1:-1:2, 2::2] + iTemp[pLev][2::2, 2::2, 1:-1:2] +
                                     iTemp[pLev][1:-1:2, 2::2, 1:-1:2] + iTemp[pLev][2::2, 1:-1:2, 2::2] +
                                     iTemp[pLev][2::2, 1:-1:2, 1:-1:2] + iTemp[pLev][1:-1:2, 2::2, 2::2])/8


# Solves at coarsest level using the Gauss-Seidel iterative solver
def solve():
    global N, vLev
    global gsFactor
    global maxCount
    global tolerance
    global pData, rData
    global hyhz, hzhx, hxhy, hxhyhz

    n = N[vLev]

    jCnt = 0
    while True:
        imposePpBCs(pData[vLev])

        pData[vLev][1:-1, 1:-1, 1:-1] = (hyhz[vLev]*(pData[vLev][2:, 1:-1, 1:-1] + pData[vLev][:-2, 1:-1, 1:-1]) +
                                         hzhx[vLev]*(pData[vLev][1:-1, 2:, 1:-1] + pData[vLev][1:-1, :-2, 1:-1]) +
                                         hxhy[vLev]*(pData[vLev][1:-1, 1:-1, 2:] + pData[vLev][1:-1, 1:-1, :-2]) -
                                         hxhyhz[vLev]*rData[vLev][1:-1, 1:-1, 1:-1]) * gsFactor[vLev]

        maxErr = np.amax(
            np.abs(rData[vLev][1:-1, 1:-1, 1:-1] - laplace(pData[vLev])))

        if maxErr < tolerance:
            break

        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Poisson solver not converging at the coarsest level. Aborting")
            quit()

    imposePpBCs(pData[vLev])


# Interpolates the data from an array of size 2^n to a larger array of size 2^(n + 1)
def prolong():
    global vLev
    global pData

    pLev = vLev
    vLev -= 1

    pData[vLev][1:-1:2, 1:-1:2, 1:-1:2] = pData[vLev][2::2, 1:-1:2, 1:-1:2] = pData[vLev][1:-1:2, 2::2, 1:-1:2] = pData[vLev][1:-1:2, 1:-1:2, 2::2] = \
        pData[vLev][2::2, 2::2, 1:-1:2] = pData[vLev][1:-1:2, 2::2, 2::2] = pData[vLev][2::2,
                                                                                        1:-1:2, 2::2] = pData[vLev][2::2, 2::2, 2::2] = pData[pLev][1:-1, 1:-1, 1:-1]


# Computes the 3D laplacian of function
def laplace(function):

    laplacian = ((function[:-2, 1:-1, 1:-1] - 2.0*function[1:-1, 1:-1, 1:-1] + function[2:, 1:-1, 1:-1])/mghx2[vLev] +
                 (function[1:-1, :-2, 1:-1] - 2.0*function[1:-1, 1:-1, 1:-1] + function[1:-1, 2:, 1:-1])/mghy2[vLev] +
                 (function[1:-1, 1:-1, :-2] - 2.0*function[1:-1, 1:-1, 1:-1] + function[1:-1, 1:-1, 2:])/mghz2[vLev])

    return laplacian


def imposePpBCs(Pp):
    Pp[0, :, :], Pp[-1, :, :] = -Pp[1, :, :], -Pp[-2, :, :]
    Pp[:, 0, :], Pp[:, -1, :] = -Pp[:, 1, :], -Pp[:, -2, :]
    Pp[:, :, 0], Pp[:, :, -1] = -Pp[:, :, 1], -Pp[:, :, -2]


@numba.njit
def Poisson_GS(Pp, tmp, rho):

    jCnt = 0
    # Pp.fill(0.0)
    while True:

        for i in range(1, gd.Nx+1):
            for j in range(1, gd.Ny+1):
                for k in range(1, gd.Nz+1):
                    Pp[i, j, k] = (1-para.gssorPp)*Pp[i, j, k] + para.gssorPp*(1.0/(-2.0*(gd.idx2 + gd.idy2 + gd.idz2))) * (rho[i, j, k] -
                                                                                                                            gd.idx2*(Pp[i+1, j, k] + Pp[i-1, j, k]) -
                                                                                                                            gd.idy2*(Pp[i, j+1, k] + Pp[i, j-1, k]) -
                                                                                                                            gd.idz2*(Pp[i, j, k+1] + Pp[i, j, k-1]))

        # BC.imposePpBCs(Pp)
        Pp[0, :, :], Pp[-1, :, :] = Pp[1, :, :], Pp[-2, :, :]
        Pp[:, 0, :], Pp[:, -1, :] = Pp[:, 1, :], Pp[:, -2, :]
        Pp[:, :, 0], Pp[:, :, -1] = Pp[:, :, 1], Pp[:, :, -2]

        for i in range(1, gd.Nx+1):
            for j in range(1, gd.Ny+1):
                for k in range(1, gd.Nz+1):
                    tmp[i, j, k] = rho[i, j, k] - ((
                        (Pp[i+1, j, k] - 2.0*Pp[i, j, k] + Pp[i-1, j, k])*gd.idx2 +
                        (Pp[i, j+1, k] - 2.0*Pp[i, j, k] + Pp[i, j-1, k])*gd.idy2 +
                        (Pp[i, j, k+1] - 2.0*Pp[i, j, k] + Pp[i, j, k-1])*gd.idz2))

        maxErr = np.amax(np.abs(tmp[1:-1, 1:-1, 1:-1]))

        if (jCnt % 100 == 0):
            print(jCnt, maxErr)

        jCnt += 1

        if maxErr < para.PoissonTolerance:
            print(jCnt)
            break

        if jCnt > para.maxiteration:
            print("ERROR: Poisson solver not converging. Aborting")
            break

    return Pp
