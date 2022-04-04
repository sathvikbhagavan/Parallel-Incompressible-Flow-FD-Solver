import parameters as para
import numpy as np
import grid as gd
import boundary as BC
from glob_ import *
from mpi4py import MPI

def predictor(Field, F, rho, const, comm, x_p0, x_pm, x_pp):

    jCnt = 0
    rank = comm.Get_rank()

    while True:
        
        temp = (1+const*para.dt*(gd.idx2 + gd.idy2 + gd.idz2))
        # F[x_p0, y0, z0] =(1.0/(1+const*para.dt*(gd.idx2 + gd.idy2 + gd.idz2))) * (rho[x_p0, y0, z0] + 
        #                                0.5*const*para.dt*gd.idx2*(F[x_pm, y0, z0] + F[x_pp, y0, z0]) +
        #                                0.5*const*para.dt*gd.idy2*(F[x_p0, ym1, z0] + F[x_p0, yp1, z0]) +
        #                                0.5*const*para.dt*gd.idz2*(F[x_p0, y0, zm1] + F[x_p0, y0, zp1]))   

        # if rank == 1:
        #     print(F[x_pm, 1, 1])

        F[x_p0, y0, z0] =(1.0/temp) * (rho[x_p0, y0, z0] + 
                                       0.5*const*para.dt*gd.idx2*(F[x_pm, y0, z0] + F[x_pp, y0, z0]) +
                                       0.5*const*para.dt*gd.idy2*(F[x_p0, ym1, z0] + F[x_p0, yp1, z0]) +
                                       0.5*const*para.dt*gd.idz2*(F[x_p0, y0, zm1] + F[x_p0, y0, zp1]))  
        # if Field == "Vx": BC.imposeUBCs(F, comm)
        # if Field == "Vy": BC.imposeVBCs(F, comm)
        # if Field == "Vz": BC.imposeWBCs(F, comm)
        # if Field == "T": BC.imposeTBCs(F, comm)
        
        if rank == 0:
            F[:, -1, :] = -F[:, -2, :]
            F[:, :, -1] = -F[:, :, -2]

            F[0, :, :] = -F[1, :, :]
            F[:, 0, :] = -F[:, 1, :] 
            F[:, :, 0] = -F[:, :, 1]
        
        elif rank == 1:
            F[:, 0, :] = -F[:, 1, :] 
            F[:, :, 0] = -F[:, :, 1]

            F[-1, :, :] = -F[-2, :, :]
            F[:, -1, :] = -F[:, -2, :]
            F[:, :, -1] = -F[:, :, -2]
            # print(F[17, ym1, 1])


        maxErr = np.amax(np.abs(rho[x_p0, y0, z0] - (F[x_p0, y0, z0] - 0.5*const*para.dt*(
                            (F[x_pm, y0, z0] - 2.0*F[x_p0, y0, z0] + F[x_pp, y0, z0])*gd.idx2 +
                            (F[x_p0, ym1, z0] - 2.0*F[x_p0, y0, z0] + F[x_p0, yp1, z0])*gd.idy2 +
                            (F[x_p0, y0, zm1] - 2.0*F[x_p0, y0, z0] + F[x_p0, y0, zp1])*gd.idz2))))
        
        totalMaxErr = comm.allreduce(maxErr, op=MPI.MIN)
        
        jCnt += 1

        if rank == 0:
            req = comm.irecv(source=1)
            F[17, :, :] = req.wait()
            req = comm.isend(F[16, :, :], dest=1)
            req.wait()
        
        else:
            req = comm.isend(F[17, :, :], dest=0)
            req.wait()
            req = comm.irecv(source=0)
            F[16, :, :] = req.wait()

        if totalMaxErr < para.FpTolerance:
            break
        
        if jCnt > para.maxiteration:
                print("ERROR: predictor not converging in", Field,". Aborting")
                quit()
        

        # if rank == 0:
        #     req = comm.irecv(source=1)
        #     F[17, :, :] = req.wait()
        #     req = comm.isend(F[16, :, :], dest=1)
        #     req.wait()
        
        # else:
        #     req = comm.isend(F[17, :, :], dest=0)
        #     req.wait()
        #     req = comm.irecv(source=0)
        #     F[16, :, :] = req.wait()
        
        # break
        if rank == 0:
            # print(f'rank:{rank}, {F[17, 2, 2]}')
            ...
        else:
            ...
            # print(ym1.start, ym1.stop)
            # print(F[17, ym1, 1])
            # print(f'rank:{rank}, {F[17:20, 1, 1]}')

        comm.Barrier()

    return F      









