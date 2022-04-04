import numpy as np
import parameters as para
import grid as gd
import timeseries as tmsr
import initial as initC
import write_field as wf
import solver as solve
from vector_field import VectField
from pressure_field import Pressure
from datetime import datetime
from mpi4py import MPI

comm = MPI.COMM_WORLD

def main():

    U = VectField()
    
    P = Pressure()

    U.initialize_arrays()
    P.initialize_arrays()

    Vx, Vy, Vz, Pa, time, fwTime, rwTime = initC.initFields()

    U.set_initcond(Vx, Vy, Vz)
    P.set_initcond(Pa)

    energy = tmsr.energy(Vx, Vy, Vz)

    maxDiv = tmsr.getDiv(Vx, Vy, Vz)

    # f = open('TimeSeries.dat', 'a')
    # f.write("# nu = %.3e \t  Nx = %i \t Ny = %i \t Nz = %i \n" %(para.nu, gd.Nx, gd.Ny, gd.Nz)) 
    # f.write('# time, energy, Divergence, dt \n')
    # f.write("%.5e\t%.5e\t%.5e\t%.5e \n" %(time, energy, maxDiv, para.dt))

    # f.close()
    
    # print('# time, energy, Divergence')
    # print("%.5f    %.2e    %.3e" %(time, energy, maxDiv))

    # wf.writeSoln_RBC(Vx, Vy, Vz, Pa, time)

    # if abs(time - para.tMax) < 0.5*para.dt or time >= para.tMax:
    #     print("# Error! Final time is less than the start time")
    #     quit()

    iCnt = 0
    ts1 = datetime.now()
    solve.solveHydro(Vx, Vy, Vz, P, time, iCnt, fwTime, rwTime, U, P, comm)

    ts2 = datetime.now()
    if comm.Get_rank() == 0:
        print("\nSimulation completed!")
        print("Simulation time",ts2-ts1)

main()








