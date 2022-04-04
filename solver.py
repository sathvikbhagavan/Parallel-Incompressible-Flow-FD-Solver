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

            comm.send(U.Vx[16:, :, :], dest=1)
            comm.send(U.Vy[16:, :, :], dest=1)
            comm.send(U.Vz[16:, :, :], dest=1)
            

        else:
            # x_p = slice(rank*xEn//procs, (rank+1)*xEn//procs+1)
            # x_p0 = slice(x_p.start+1, x_p.stop)
            
            
            # x_pm = slice(x_p.start-1, x_p.stop-1)
            # x_pp = slice(x_p.start+1, x_p.stop+1)
            U.Vx[16:, :, :] = comm.recv(source=0)
            U.Vy[16:, :, :] = comm.recv(source=0)
            U.Vz[16:, :, :] = comm.recv(source=0)

            x_p0 = slice(17, 33)
            x_pm = slice(16, 32)
            x_pp = slice(18, 34)
        
        U.nonlinear(x_p0, x_pm, x_pp)
        U.diffusion(x_p0, x_pm, x_pp)

        U.tmp = U.diffusionX*0.5*para.nu - U.nlinX  
        U.tmp[x_p0, y0, z0] = U.Vx[x_p0, y0, z0] + para.dt*(U.tmp[x_p0, y0, z0] - df.dfx(P.Pa, x_pm, x_pp))        
        U.Vx = predict.predictor("Vx", U.Vx, U.tmp, para.nu, comm, x_p0, x_pm, x_pp)
        # break
        
        comm.Barrier()
        

        U.tmp = U.diffusionY*0.5*para.nu - U.nlinY 
        U.tmp[x_p0, y0, z0] = U.Vy[x_p0, y0, z0] + para.dt*(U.tmp[x_p0, y0, z0] - df.dfy(P.Pa, x_p0))
        U.Vy = predict.predictor("Vy", U.Vy, U.tmp, para.nu, comm, x_p0, x_pm, x_pp)

        U.tmp = U.diffusionZ*0.5*para.nu - U.nlinZ 
        U.tmp[x_p0, y0, z0] = U.Vz[x_p0, y0, z0] + para.dt*(U.tmp[x_p0, y0, z0] - df.dfz(P.Pa, x_p0))
        U.Vz = predict.predictor("Vz", U.Vz, U.tmp, para.nu, comm, x_p0, x_pm, x_pp)
        

        comm.Barrier()

        if rank == 1:
            # for i in range(1, procs):
            comm.send(U.Vx[17:, :, :], dest=0)
            comm.send(U.Vy[17:, :, :], dest=0)
            comm.send(U.Vz[17:, :, :], dest=0)
            # comm.send(U.tmp[17:, :, :], dest=0)


        if rank == 0:
            # for i in range(1, procs):
            U.Vx[17:, :, :] = comm.recv(source=1)
            U.Vy[17:, :, :] = comm.recv(source=1)
            U.Vz[17:, :, :] = comm.recv(source=1)
            # U.tmp[17:, :, :] = comm.recv(source=1)

            U.tmp[x0, y0, z0] = (df.dfx(U.Vx, xm1, xp1) + df.dfy(U.Vy, x0) + df.dfz(U.Vz, x0))/para.dt

            # print(U.tmp[:, 1, 1])

            # print('\n\nCombined:')
            # U.Vx[0, :, :], U.Vx[-1, :, :] = -U.Vx[1, :, :], -U.Vx[-2, :, :]
            # U.Vx[:, 0, :], U.Vx[:, -1, :] = -U.Vx[:, 1, :], -U.Vx[:, -2, :]
            # U.Vx[:, :, 0], U.Vx[:, :, -1] = -U.Vx[:, :, 1], -U.Vx[:, :, -2]

            # U.Vy[0, :, :], U.Vy[-1, :, :] = -U.Vy[1, :, :], -U.Vy[-2, :, :]
            # U.Vy[:, 0, :], U.Vy[:, -1, :] = -U.Vy[:, 1, :], -U.Vy[:, -2, :]
            # U.Vy[:, :, 0], U.Vy[:, :, -1] = -U.Vy[:, :, 1], -U.Vy[:, :, -2]

            # U.Vz[0, :, :], U.Vz[-1, :, :] = -U.Vz[1, :, :], -U.Vz[-2, :, :]
            # U.Vz[:, 0, :], U.Vz[:, -1, :] = -U.Vz[:, 1, :], -U.Vz[:, -2, :]
            # U.Vz[:, :, 0], U.Vz[:, :, -1] = -U.Vz[:, :, 1], -U.Vz[:, :, -2]
            
            
            # P.Pa[0, :, :], P.Pa[-1, :, :] = -P.Pa[1, :, :], -P.Pa[-2, :, :]
            # P.Pa[:, 0, :], P.Pa[:, -1, :] = -P.Pa[:, 1, :], -P.Pa[:, -2, :]
            # P.Pa[:, :, 0], P.Pa[:, :, -1] = -P.Pa[:, :, 1], -P.Pa[:, :, -2]
            # print(U.Vx[x0, 1, 1])

            P.Pp = pSolve.Poisson_MG(U.tmp, iCnt)
            U.Vx[x0, y0, z0] = U.Vx[x0, y0, z0] - para.dt*df.dfx(P.Pp, xm1, xp1)
            U.Vy[x0, y0, z0] = U.Vy[x0, y0, z0] - para.dt*df.dfy(P.Pp, x0)
            U.Vz[x0, y0, z0] = U.Vz[x0, y0, z0] - para.dt*df.dfz(P.Pp, x0)

            
           

            # BC.imposeUBCs(U.Vx, comm)          
            # BC.imposeUBCs(U.Vy, comm)          
            # BC.imposeUBCs(U.Vz, comm)          
            # BC.imposePBCs(P.Pa, comm)
            # break

            U.Vx[0, :, :], U.Vx[-1, :, :] = -U.Vx[1, :, :], -U.Vx[-2, :, :]
            U.Vx[:, 0, :], U.Vx[:, -1, :] = -U.Vx[:, 1, :], -U.Vx[:, -2, :]
            U.Vx[:, :, 0], U.Vx[:, :, -1] = -U.Vx[:, :, 1], -U.Vx[:, :, -2]

            U.Vy[0, :, :], U.Vy[-1, :, :] = -U.Vy[1, :, :], -U.Vy[-2, :, :]
            U.Vy[:, 0, :], U.Vy[:, -1, :] = -U.Vy[:, 1, :], -U.Vy[:, -2, :]
            U.Vy[:, :, 0], U.Vy[:, :, -1] = -U.Vy[:, :, 1], -U.Vy[:, :, -2]

            U.Vz[0, :, :], U.Vz[-1, :, :] = -U.Vz[1, :, :], -U.Vz[-2, :, :]
            U.Vz[:, 0, :], U.Vz[:, -1, :] = -U.Vz[:, 1, :], -U.Vz[:, -2, :]
            U.Vz[:, :, 0], U.Vz[:, :, -1] = -U.Vz[:, :, 1], -U.Vz[:, :, -2]
            
            
            P.Pa[0, :, :], P.Pa[-1, :, :] = -P.Pa[1, :, :], -P.Pa[-2, :, :]
            P.Pa[:, 0, :], P.Pa[:, -1, :] = -P.Pa[:, 1, :], -P.Pa[:, -2, :]
            P.Pa[:, :, 0], P.Pa[:, :, -1] = -P.Pa[:, :, 1], -P.Pa[:, :, -2]
            
            # print(U.Vx[:, 2, 2])
            # break

            # U.Vxold[:, :, :] = U.Vx
            # U.Vyold[:, :, :] = U.Vy
            # U.Vzold[:, :, :] = U.Vz
            # print(U.Vx[:, 1, 1])
            # break
            
            # if abs(rwTime - time) < 0.5*para.dt:
            #     rwTime = rwTime + para.rwInt
            #     if iCnt > 1: wf.writeRestart_RBC(U.Vx, U.Vy, U.Vz, P.Pa, time)                   
                        
            # if abs(fwTime - time) < 0.5*para.dt:
            #     fwTime = fwTime + para.fwInt
            #     wf.writeSoln_RBC(U.Vx, U.Vy, U.Vz, P.Pa, time)
                    
            if iCnt % para.opInt == 0: 

                energy = tmsr.energy(U.Vx, U.Vy, U.Vz)
        
                maxDiv = tmsr.getDiv(U.Vx, U.Vy, U.Vz)

                # f = open('TimeSeries.dat', 'a')
                # f.write("%.5f\t%.5e\t%.5e\t%.5e \n" %(time, energy, maxDiv, para.dt))
                # f.close()

                print("%.5f    %.5e    %.5e" %(time, energy, maxDiv))

            #     if abs(maxDiv)>1e4: 
            #         print("Error! Simulation blowing up. Aborting!")
            #         quit()
        time, iCnt = time + para.dt, iCnt + 1
        comm.Barrier()
        # print(f'{rank}: {time}')
        # break