import numpy as np
import parameters as para
import grid as gd
import derivative as df
from glob_ import *


class VectField:
    def __init__(self):
        self.Vx = []
        self.Vy = []
        self.Vz = []
        self.nlinX = []
        self.nlinY = []
        self.nlinZ = []
        self.diffusionX = []
        self.diffusionY = []
        self.diffusionZ = []
        self.tmp = []
        

    def set_initcond(self, Vx, Vy, Vz):
        self.Vx = Vx 
        self.Vz = Vz 
        self.Vy = Vy

        return

    def initialize_arrays(self): 
        self.Vx = np.zeros([gd.Nx+2, gd.Ny+2, gd.Nz+2])
        self.Vy = np.zeros([gd.Nx+2, gd.Ny+2, gd.Nz+2])
        self.Vz = np.zeros([gd.Nx+2, gd.Ny+2, gd.Nz+2])
        self.nlinX = np.zeros([gd.Nx+2, gd.Ny+2, gd.Nz+2])
        self.nlinY = np.zeros([gd.Nx+2, gd.Ny+2, gd.Nz+2])
        self.nlinZ = np.zeros([gd.Nx+2, gd.Ny+2, gd.Nz+2]) 
        self.diffusionX = np.zeros([gd.Nx+2, gd.Ny+2, gd.Nz+2])
        self.diffusionY = np.zeros([gd.Nx+2, gd.Ny+2, gd.Nz+2])
        self.diffusionZ = np.zeros([gd.Nx+2, gd.Ny+2, gd.Nz+2]) 
        self.tmp = np.zeros([gd.Nx+2, gd.Ny+2, gd.Nz+2]) 
        return    


    def diffusion(self, x_p0, x_pm, x_pp):
        self.diffusionX[x_p0, y0, z0] = df.d2fxx(self.Vx, x_p0, x_pm, x_pp) + df.d2fyy(self.Vx, x_p0) + df.d2fzz(self.Vx, x_p0)
        self.diffusionY[x_p0, y0, z0] = df.d2fxx(self.Vy, x_p0, x_pm, x_pp) + df.d2fyy(self.Vy, x_p0) + df.d2fzz(self.Vy, x_p0)
        self.diffusionZ[x_p0, y0, z0] = df.d2fxx(self.Vz, x_p0, x_pm, x_pp) + df.d2fyy(self.Vz, x_p0) + df.d2fzz(self.Vz, x_p0)
        return
                        

    def nonlinear(self, x_p0, x_pm, x_pp):
        self.nlinX[x_p0, y0, z0] = self.Vx[x_p0, y0, z0]*df.dfx(self.Vx, x_pm, x_pp) + self.Vy[x_p0, y0, z0]*df.dfy(self.Vx, x_p0) + self.Vz[x_p0, y0, z0]*df.dfz(self.Vx, x_p0)
        self.nlinY[x_p0, y0, z0] = self.Vx[x_p0, y0, z0]*df.dfx(self.Vy, x_pm, x_pp) + self.Vy[x_p0, y0, z0]*df.dfy(self.Vy, x_p0) + self.Vz[x_p0, y0, z0]*df.dfz(self.Vy, x_p0)
        self.nlinZ[x_p0, y0, z0] = self.Vx[x_p0, y0, z0]*df.dfx(self.Vz, x_pm, x_pp) + self.Vy[x_p0, y0, z0]*df.dfy(self.Vz, x_p0) + self.Vz[x_p0, y0, z0]*df.dfz(self.Vz, x_p0)
        return

    def divergence(self):
        self.tmp = df.dfx(self.Vx) + df.dfy(self.Vy) + df.dfz(self.Vz)
        maxDiv = np.amax(self.tmp)
        return maxDiv

    

