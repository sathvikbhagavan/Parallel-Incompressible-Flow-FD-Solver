import numpy as np

nu = 0.1

Lx, Ly, Lz = 1.0, 1.0, 1.0

# Size index: 0 1 2 3  4  5  6  7   8   9   10   11   12   13    14
# Grid sizes: 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384
sInd = np.array([6, 6, 6])

restart = False   

time = 0    

dt = 0.01

tMax = 0.1

Cn = 0.5 

opInt = 1

fwInt = 100

rwInt = 5

FpTolerance = 1.0e-3

PoissonTolerance = 1.0e-3

gssorPp = 1.0

gssorT = 1.0

gssorWp = 1.0

maxiteration = 10000


VDepth = min(sInd) - 1   

preSm = 5   

pstSm = 5   

vcCnt = 10  

Volume = Lx*Ly*Lz   

if (int(tMax/fwInt)>20):
    print("# Warning! File writing exceeding limit. New fwInt is set")
    fwInt = int(tMax/10)











