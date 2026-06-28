import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
from psc.lib.g_space import g
from psc.lib.x3Dlinearization import linearizenD_EPA
from psc.lib.x3Drepetition import getpolytope_EPA  


# ------------------------------------------------------------------------------------------------
# --->  Generate required information such as atomic coordinate to be solved
#       artificial atomic scattering factors. 'j' fixes the direction of third atomic coordinate
# ------------------------------------------------------------------------------------------------

coordinate = np.array([0.349, 0.362, 0.1615, 0.1615])
f    = [1.0]*len(coordinate)
j    = len(coordinate)-1


# ------------------------------------------------------------------------------------------------
# ---> Apply origin fixing rule. The origin is always fixed at [0, 0, ....]
# ------------------------------------------------------------------------------------------------
l = 1
coordinate = np.sort(coordinate)[::-1]  if (np.sign(g(l, coordinate, f))>0) else np.sort(0.5-coordinate)[::-1]

# ------------------------------------------------------------------------------------------------
# ---> Start to solve given atomic structure using first 4 number of reflections
# ------------------------------------------------------------------------------------------------

h  = 2
info, plist = [], []
IorG='intensity'

for l in range(1,h+1):
        
    # ===> 1. initilization
    k  = 2*np.pi*l
    gi = np.abs(g(l, coordinate, f))
    amplitudesign = np.sign(g(l, coordinate, f))
    
    # ===> 2. linearization
    normal, distance, boundarypoints = linearizenD_EPA(l, f, gi)
    
    ST = time.time()
    # ===> 3. get all polytope
    p = getpolytope_EPA( l, normal, distance, amplitudesign, IorG, imax=0.5)
    plist.append(p)
    ET = time.time()
    print(f'===> Time taken for RO {l} is {ET-ST}')
    info.append([l, normal, distance])
    
    # ===> 4. check linearization
    #checklinear(l, f, gi, normal, distance, j=len(f)-1, n=50, s=1, testiso=True)
###


