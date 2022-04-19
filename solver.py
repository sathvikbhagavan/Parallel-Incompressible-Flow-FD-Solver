from re import X
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
    procs = comm.Get_size()

    x_p0 = slice(rank*gd.Nx//procs+1, (rank+1)*gd.Nx//procs+1)
    x_pm = slice(x_p0.start-1, x_p0.stop-1)
    x_pp = slice(x_p0.start+1, x_p0.stop+1)
    
    while time <= para.tMax:

        if rank == 0:

            if procs != 1:
                U.Vx[x_p0.stop, :, :] = comm.recv(source=rank+1)
                U.Vy[x_p0.stop, :, :] = comm.recv(source=rank+1)
                U.Vz[x_p0.stop, :, :] = comm.recv(source=rank+1)

                comm.send(U.Vx[x_p0.stop-1, :, :], dest=rank+1)
                comm.send(U.Vy[x_p0.stop-1, :, :], dest=rank+1)
                comm.send(U.Vz[x_p0.stop-1, :, :], dest=rank+1)

        elif rank == procs-1:
            
            comm.send(U.Vx[x_p0.start, :, :], dest=rank-1)
            comm.send(U.Vy[x_p0.start, :, :], dest=rank-1)
            comm.send(U.Vz[x_p0.start, :, :], dest=rank-1)

            U.Vx[x_p0.start-1, :, :] = comm.recv(source=rank-1)
            U.Vy[x_p0.start-1, :, :] = comm.recv(source=rank-1)
            U.Vz[x_p0.start-1, :, :] = comm.recv(source=rank-1)
        else:
            if rank % 2 == 0:
                U.Vx[x_p0.stop, :, :] = comm.recv(source=rank+1)
                U.Vy[x_p0.stop, :, :] = comm.recv(source=rank+1)
                U.Vz[x_p0.stop, :, :] = comm.recv(source=rank+1)
                
                U.Vx[x_p0.start-1, :, :] = comm.recv(source=rank-1)
                U.Vy[x_p0.start-1, :, :] = comm.recv(source=rank-1)
                U.Vz[x_p0.start-1, :, :] = comm.recv(source=rank-1)

                comm.send(U.Vx[x_p0.stop-1, :, :], dest=rank+1)
                comm.send(U.Vy[x_p0.stop-1, :, :], dest=rank+1)
                comm.send(U.Vz[x_p0.stop-1, :, :], dest=rank+1)
                
                comm.send(U.Vx[x_p0.start, :, :], dest=rank-1)
                comm.send(U.Vy[x_p0.start, :, :], dest=rank-1)
                comm.send(U.Vz[x_p0.start, :, :], dest=rank-1)

            else:
                comm.send(U.Vx[x_p0.start, :, :], dest=rank-1)
                comm.send(U.Vy[x_p0.start, :, :], dest=rank-1)
                comm.send(U.Vz[x_p0.start, :, :], dest=rank-1)
                
                comm.send(U.Vx[x_p0.stop-1, :, :], dest=rank+1)
                comm.send(U.Vy[x_p0.stop-1, :, :], dest=rank+1)
                comm.send(U.Vz[x_p0.stop-1, :, :], dest=rank+1)
                
                U.Vx[x_p0.stop, :, :] = comm.recv(source=rank+1)
                U.Vy[x_p0.stop, :, :] = comm.recv(source=rank+1)
                U.Vz[x_p0.stop, :, :] = comm.recv(source=rank+1)

                U.Vx[x_p0.start-1, :, :] = comm.recv(source=rank-1)
                U.Vy[x_p0.start-1, :, :] = comm.recv(source=rank-1)
                U.Vz[x_p0.start-1, :, :] = comm.recv(source=rank-1)

        comm.Barrier()
        
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
        
        if rank == procs-1:
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
        
        else:
            U.Vx[:, -1, :] = -U.Vx[:, -2, :]
            U.Vx[:, :, -1] = -U.Vx[:, :, -2]
            U.Vx[:, 0, :] = -U.Vx[:, 1, :] 
            U.Vx[:, :, 0] = -U.Vx[:, :, 1]

            U.Vy[:, -1, :] = -U.Vy[:, -2, :]
            U.Vy[:, :, -1] = -U.Vy[:, :, -2]
            U.Vy[:, 0, :] = -U.Vy[:, 1, :] 
            U.Vy[:, :, 0] = -U.Vy[:, :, 1]

            U.Vz[:, -1, :] = -U.Vz[:, -2, :]
            U.Vz[:, :, -1] = -U.Vz[:, :, -2]
            U.Vz[:, 0, :] = -U.Vz[:, 1, :] 
            U.Vz[:, :, 0] = -U.Vz[:, :, 1]

            P.Pa[:, -1, :] = P.Pa[:, -2, :]
            P.Pa[:, :, -1] = P.Pa[:, :, -2]
            P.Pa[:, 0, :] = P.Pa[:, 1, :] 
            P.Pa[:, :, 0] = P.Pa[:, :, 1]

        if rank != 0:
            if rank == procs - 1:
                k = slice(rank*gd.Nx//procs+1, (rank+1)*gd.Nx//procs+2)
                comm.send(U.Vx[k, :, :], dest=0)
                comm.send(U.Vy[k, :, :], dest=0)
                comm.send(U.Vz[k, :, :], dest=0)
            else:
                comm.send(U.Vx[x_p0, :, :], dest=0)
                comm.send(U.Vy[x_p0, :, :], dest=0)
                comm.send(U.Vz[x_p0, :, :], dest=0)
        
        else:
            for i in range(1, procs):
                if i == procs - 1:
                    k = slice(i*gd.Nx//procs+1, (i+1)*gd.Nx//procs+2)
                else:
                    k = slice(i*gd.Nx//procs+1, (i+1)*gd.Nx//procs+1)
                U.Vx[k, :, :] = comm.recv(source=i)
                U.Vy[k, :, :] = comm.recv(source=i)
                U.Vz[k, :, :] = comm.recv(source=i)

        time, iCnt = time + para.dt, iCnt + 1

        if rank == 0:
            if iCnt % para.opInt == 0: 

                energy = tmsr.energy(U.Vx, U.Vy, U.Vz)
        
                maxDiv = tmsr.getDiv(U.Vx, U.Vy, U.Vz)

                # f = open('TimeSeries.dat', 'a')
                # f.write("%.5f\t%.5e\t%.5e\t%.5e \n" %(time, energy, maxDiv, para.dt))
                # f.close()

                print("%.5f    %.5e    %.5e" %(time, energy, maxDiv))

                if abs(maxDiv)>1e4: 
                    print("Error! Simulation blowing up. Aborting!")
                    quit()

        comm.Barrier()