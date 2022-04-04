import numpy as np
import parameters as para
import grid as gd
import derivative as df
import predictor as predict
import PoissonSolver as pSolve
import boundary as BC
import write_field as wf
import timeseries as tmsr
from glob_ import *


def solveHydro(Vx, Vy, Vz, Pa, time, iCnt, fwTime, rwTime, U, P, comm):
    rank = comm.Get_rank()
    
    while time<=para.tMax:
        
        if rank == 0:
            x_p0 = slice(1, 17)
            x_pm = slice(0, 16)
            x_pp = slice(2, 18)

            comm.send(U.Vx[16, :, :], dest=1)
            comm.send(U.Vy[16, :, :], dest=1)
            comm.send(U.Vz[16, :, :], dest=1)

            U.Vx[17, :, :] = comm.recv(source=1)
            U.Vy[17:, :, :] = comm.recv(source=1)
            U.Vz[17:, :, :] = comm.recv(source=1)
            

        else:
            U.Vx[16, :, :] = comm.recv(source=0)
            U.Vy[16, :, :] = comm.recv(source=0)
            U.Vz[16, :, :] = comm.recv(source=0)

            comm.send(U.Vx[17, :, :], dest=0)
            comm.send(U.Vy[17, :, :], dest=0)
            comm.send(U.Vz[17, :, :], dest=0)

            x_p0 = slice(17, 33)
            x_pm = slice(16, 32)
            x_pp = slice(18, 34)
        
        U.nonlinear(x_p0, x_pm, x_pp)
        U.diffusion(x_p0, x_pm, x_pp)

        U.tmp = U.diffusionX*0.5*para.nu - U.nlinX  
        U.tmp[x_p0, y0, z0] = U.Vx[x_p0, y0, z0] + para.dt*(U.tmp[x_p0, y0, z0] - df.dfx(P.Pa, x_pm, x_pp))        
        U.Vx = predict.predictor("Vx", U.Vx, U.tmp, para.nu, comm, x_p0, x_pm, x_pp)
        
        U.tmp = U.diffusionY*0.5*para.nu - U.nlinY 
        U.tmp[x_p0, y0, z0] = U.Vy[x_p0, y0, z0] + para.dt*(U.tmp[x_p0, y0, z0] - df.dfy(P.Pa, x_p0))
        U.Vy = predict.predictor("Vy", U.Vy, U.tmp, para.nu, comm, x_p0, x_pm, x_pp)

        U.tmp = U.diffusionZ*0.5*para.nu - U.nlinZ 
        U.tmp[x_p0, y0, z0] = U.Vz[x_p0, y0, z0] + para.dt*(U.tmp[x_p0, y0, z0] - df.dfz(P.Pa, x_p0))
        U.Vz = predict.predictor("Vz", U.Vz, U.tmp, para.nu, comm, x_p0, x_pm, x_pp)
        
        U.tmp[x_p0, y0, z0] = (df.dfx(U.Vx, x_pm, x_pp) + df.dfy(U.Vy, x_p0) + df.dfz(U.Vz, x_p0))/para.dt
        
        
        P.Pp = pSolve.Poisson_Jacobi(U.tmp, P.Pp, comm, x_p0, x_pm, x_pp)

        U.Vx[x_p0, y0, z0] = U.Vx[x_p0, y0, z0] - para.dt*df.dfx(P.Pp, x_pm, x_pp)
        U.Vy[x_p0, y0, z0] = U.Vy[x_p0, y0, z0] - para.dt*df.dfy(P.Pp, x_p0)
        U.Vz[x_p0, y0, z0] = U.Vz[x_p0, y0, z0] - para.dt*df.dfz(P.Pp, x_p0)
        

        if rank == 0:
            U.Vx[:, -1, :] = -U.Vx[:, -2, :]
            U.Vx[:, :, -1] = -U.Vx[:, :, -2]
            U.Vx[0, :, :] = -U.Vx[1, :, :]
            U.Vx[:, 0, :] = -U.Vx[:, 1, :] 
            U.Vx[:, :, 0] = -U.Vx[:, :, 1]

            U.Vy[:, -1, :] = -U.Vy[:, -2, :]
            U.Vy[:, :, -1] = -U.Vy[:, :, -2]
            U.Vy[0, :, :] = -U.Vy[1, :, :]
            U.Vy[:, 0, :] = -U.Vy[:, 1, :] 
            U.Vy[:, :, 0] = -U.Vy[:, :, 1]

            U.Vz[:, -1, :] = -U.Vz[:, -2, :]
            U.Vz[:, :, -1] = -U.Vz[:, :, -2]
            U.Vz[0, :, :] = -U.Vz[1, :, :]
            U.Vz[:, 0, :] = -U.Vz[:, 1, :] 
            U.Vz[:, :, 0] = -U.Vz[:, :, 1]

            P.Pa[:, -1, :] = P.Pa[:, -2, :]
            P.Pa[:, :, -1] = P.Pa[:, :, -2]
            P.Pa[0, :, :] = P.Pa[1, :, :]
            P.Pa[:, 0, :] = P.Pa[:, 1, :] 
            P.Pa[:, :, 0] = P.Pa[:, :, 1]
        
        elif rank == 1:
            U.Vx[:, 0, :] = -U.Vx[:, 1, :] 
            U.Vx[:, :, 0] = -U.Vx[:, :, 1]
            U.Vx[-1, :, :] = -U.Vx[-2, :, :]
            U.Vx[:, -1, :] = -U.Vx[:, -2, :]
            U.Vx[:, :, -1] = -U.Vx[:, :, -2]

            U.Vy[:, 0, :] = -U.Vy[:, 1, :] 
            U.Vy[:, :, 0] = -U.Vy[:, :, 1]
            U.Vy[-1, :, :] = -U.Vy[-2, :, :]
            U.Vy[:, -1, :] = -U.Vy[:, -2, :]
            U.Vy[:, :, -1] = -U.Vy[:, :, -2]

            U.Vz[:, 0, :] = -U.Vz[:, 1, :] 
            U.Vz[:, :, 0] = -U.Vz[:, :, 1]
            U.Vz[-1, :, :] = -U.Vz[-2, :, :]
            U.Vz[:, -1, :] = -U.Vz[:, -2, :]
            U.Vz[:, :, -1] = -U.Vz[:, :, -2]

            P.Pa[:, 0, :] = P.Pa[:, 1, :] 
            P.Pa[:, :, 0] = P.Pa[:, :, 1]
            P.Pa[-1, :, :] = P.Pa[-2, :, :]
            P.Pa[:, -1, :] = P.Pa[:, -2, :]
            P.Pa[:, :, -1] = P.Pa[:, :, -2]


        # if rank == 1:
        #     comm.send(U.Vx[17:, :, :], dest=0)
        #     comm.send(U.Vy[17:, :, :], dest=0)
        #     comm.send(U.Vz[17:, :, :], dest=0)


        # if rank == 0:
        #     U.Vx[17:, :, :] = comm.recv(source=1)
        #     U.Vy[17:, :, :] = comm.recv(source=1)
        #     U.Vz[17:, :, :] = comm.recv(source=1)
            

            # if abs(rwTime - time) < 0.5*para.dt:
            #     rwTime = rwTime + para.rwInt
            #     if iCnt > 1: wf.writeRestart_RBC(U.Vx, U.Vy, U.Vz, P.Pa, time)                   
                        
            # if abs(fwTime - time) < 0.5*para.dt:
            #     fwTime = fwTime + para.fwInt
            #     wf.writeSoln_RBC(U.Vx, U.Vy, U.Vz, P.Pa, time)
        time, iCnt = time + para.dt, iCnt + 1

        if rank == 0:
            print(time)
            # if iCnt % para.opInt == 0: 

            #     energy = tmsr.energy(U.Vx, U.Vy, U.Vz)
        
            #     maxDiv = tmsr.getDiv(U.Vx, U.Vy, U.Vz)

            #     # f = open('TimeSeries.dat', 'a')
            #     # f.write("%.5f\t%.5e\t%.5e\t%.5e \n" %(time, energy, maxDiv, para.dt))
            #     # f.close()

            #     print("%.5f    %.5e    %.5e" %(time, energy, maxDiv))

            #     if abs(maxDiv)>1e4: 
            #         print("Error! Simulation blowing up. Aborting!")
            #         quit()
            ...

        comm.Barrier()