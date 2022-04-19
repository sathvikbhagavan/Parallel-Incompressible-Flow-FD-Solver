import numpy as np
import grid as gd
import parameters as para
import boundary as BC
import numba
from glob_ import *
from mpi4py import MPI


def Poisson_Jacobi(rho, F, comm, x_p0, x_pm, x_pp):
    jCnt = 0
    rank = comm.Get_rank()
    procs = comm.Get_size()
    
    iterations = 1000
    for i in range(iterations):
    # while True:

        F[x_p0, y0, z0] = (1.0/(-2.0*(gd.idx2 + gd.idy2 + gd.idz2))) * (rho[x_p0, y0, z0] -
                                                                       gd.idx2*(F[x_pm, y0, z0] + F[x_pp, y0, z0]) -
                                                                       gd.idy2*(F[x_p0, ym1, z0] + F[x_p0, yp1, z0]) -
                                                                       gd.idz2*(F[x_p0, y0, zm1] + F[x_p0, y0, zp1]))
        

        if rank == 0:
            F[:, -1, :] = F[:, -2, :]
            F[:, :, -1] = F[:, :, -2]

            F[0, :, :] = F[1, :, :]
            F[:, 0, :] = F[:, 1, :] 
            F[:, :, 0] = F[:, :, 1]
        
        if rank == procs-1:
            F[:, 0, :] = F[:, 1, :] 
            F[:, :, 0] = F[:, :, 1]

            F[-1, :, :] = F[-2, :, :]
            F[:, -1, :] = F[:, -2, :]
            F[:, :, -1] = F[:, :, -2]

        else:
            F[:, -1, :] = F[:, -2, :]
            F[:, :, -1] = F[:, :, -2]
            F[:, 0, :] = F[:, 1, :] 
            F[:, :, 0] = F[:, :, 1]

        maxErr = np.amax(np.abs(rho[x_p0, y0, z0] - ((
                        (F[x_pm, y0, z0] - 2.0*F[x_p0, y0, z0] + F[x_pp, y0, z0])*gd.idx2 +
                        (F[x_p0, ym1, z0] - 2.0*F[x_p0, y0, z0] + F[x_p0, yp1, z0])*gd.idy2 +
                        (F[x_p0, y0, zm1] - 2.0*F[x_p0, y0, z0] + F[x_p0, y0, zp1])*gd.idz2))))

        # totalMaxErr = comm.allreduce(maxErr, op=MPI.MIN)

        jCnt += 1
        # if totalMaxErr < para.PoissonTolerance:
        #     break


        if jCnt > para.maxiteration:
            print("ERROR: Jacobi Poisson solver not converging. Aborting")
            quit()

        if rank == 0:
            if procs != 1:
                F[x_p0.stop, :, :] = comm.recv(source=rank+1)
                comm.send(F[x_p0.stop-1, :, :], dest=rank+1)

        elif rank == procs-1:

            comm.send(F[x_p0.start, :, :], dest=rank-1)
            F[x_p0.start-1, :, :] = comm.recv(source=rank-1)
        
        else:
            if rank % 2 == 0:
                
                F[x_p0.stop, :, :] = comm.recv(source=rank+1)
                F[x_p0.start-1, :, :] = comm.recv(source=rank-1)

                comm.send(F[x_p0.stop-1, :, :], dest=rank+1)
                comm.send(F[x_p0.start, :, :], dest=rank-1)
                
 
            else:

                comm.send(F[x_p0.start, :, :], dest=rank-1)
                comm.send(F[x_p0.stop-1, :, :], dest=rank+1)

                F[x_p0.stop, :, :] = comm.recv(source=rank+1)
                F[x_p0.start-1, :, :] = comm.recv(source=rank-1)
        
        comm.Barrier()

    return F