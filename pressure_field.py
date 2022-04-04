import numpy as np
import parameters as para
import grid as gd
import derivative as df
#import pyfftw.interfaces.numpy_fft as fp

class Pressure:
    def __init__(self):
        self.Pa = []
        self.Pp = []
        
    def set_initcond(self, Pa):
        self.Pa = Pa 
        
        return

    def initialize_arrays(self): 
        self.Pa = np.zeros([gd.Nx+2, gd.Ny+2, gd.Nz+2])
        self.Pp = np.zeros([gd.Nx+2, gd.Ny+2, gd.Nz+2])
         
        return    

    
                        


    
    

