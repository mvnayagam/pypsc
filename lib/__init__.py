# ----------------------------------------------------------
# Dt: 28.06.2026 by: Muthu
# importing all functions for general usage
# ----------------------------------------------------------

# ----------------------------------------------------------
# Call some tools from lib for structure solving 
# ----------------------------------------------------------
# from .MCin2DPS_EPA  import MC2DPS_EPA
# from .MCin2DPS_nEPA import MC2DPS_nEPA
# from .MCinNDPS_EPA  import isosurfs_EPA, MCNDPS_EPA

# ----------------------------------------------------------
# Main library for psc structure solving
# ----------------------------------------------------------
# from .x2Dlinearizer2D  import find_interception, findpx, findpy, fn_solveforx_v2
# from .x2Dlinearizer2D  import double_segment_EPA, single_segment_EPA, single_segment_nEPA, single_segment_nEPA, double_segment_nEPA
# from .x2Dpolygon        import multistrip, getploygons_EPA_SS, getploygons_EPA_DS, getploygons_nEPA, polyintersect, polyintersect_MC
# from .x2Drepetition     import repeat2D, linrep_DS, linrep_SS, writedata
# from .x2Dwritesolution  import writepolygons, isInside, get_error, get_error_v3a, pseudosolution, realsolution, analyzesolution
# from .xlinearizationtools   import radius_from_volume
# from .checklinearizer   import checklinear, checklinear_I, getpoly_mitd

from .gspacer import g, F, grad_F, grad_g, hsurf_F, hsurf_F2, hsurf_g
from .intersector       import find_intersection 
from .linearizer        import linearizenD_EPA, linearizenD_nEPA
from .solutionIO        import wrtdata, wrtcoor, wrtvolume, wrtallsolution, readoldsolution, readh5file, readh5file_v2
from .tessellator       import getmesh, getsigncombination, getpolytope, getpolytope_EPA, getpolytope_nEPA


from psc.config.logconfig import setup_logging

from .gspacer import GSpacer
from .linearizer import EPALinearizer
from .tessellator import Tessellator
from .intersector import Intersector
from .solutionmerger import SolutionMerger
from .checklinearizer import CheckLinearizer
from .solutionIO import SolutionIO
from .choiceoforigin import ChoiceOfOrigin
from .pscsolver import PSCSolver


