import numpy as np
import polytope as pc
from .gspacer import hsurf_g, hsurf_F, hsurf_F2
from .tessellator import Tessellator
from scipy.optimize import minimize


import logging
# logger = logging.getLogger(__name__)


class CheckLinearizer:
    """
    Check isosurface linearization using either:
    1. Minimization-based approach (default)
    2. Grid-based approach

    Parameters
    ----------
    reflection : int
    structurefactor : array-like
    amplitude : float
    normal : array-like
    distance : list or tuple
        [dmin, dmax]
    """

    def __init__(self, reflection: int, structurefactor: list, amplitude: float, normal: list, distance: list, variant: str='EPA', logger=None):
        
        self.logger = logger or logging.getLogger("psc.checklinear")
        
        self.reflection = reflection
        self.structurefactor = np.asarray(structurefactor, dtype=float)
        self.amplitude = amplitude

        # Ensure normal is a unit vector from the start
        normal_arr = np.asarray(normal, dtype=float)
        norm = np.linalg.norm(normal_arr)
        if norm < 1e-12:
            raise ValueError("Normal vector cannot be zero")
        self.normal = normal_arr / norm
                
        self.distance = np.asarray(distance, dtype=float) # list(distance)
        self.dim = len(self.structurefactor)
        self.k = 2 * np.pi * reflection
        self.variant=variant
        
    # =========================================================
    # Check for orthogonality between normal and gradient of isosurface
    # =========================================================
    def normal_alignment(self, x0, tol: float = 1e-3):
        """Test whether n is the normal to the isosurface at x0."""
        x0 = np.asarray(x0, dtype=float)
        
        # Gradient components
        grad = -self.k * self.structurefactor * np.sin(self.k * x0)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-12:
            return {"gradient": grad, "cos_angle": 0.0, "is_normal": False}
            
        grad_unit = grad / grad_norm
        cosang = np.abs(np.dot(grad_unit, self.normal))
        
        return {"gradient": grad, "cos_angle": cosang, "is_normal": cosang > 1 - tol}

    # =========================================================
    # Core cosine function
    # =========================================================
    def gfun(self, x):
        return np.sum(self.structurefactor * np.cos(self.k * x))

    def gfun_NEPA(self, x):
        return np.sum(self.structurefactor * np.cos(self.k * x))**2

    # =========================================================
    # Grid-based method
    # =========================================================
    
    def gridbased(self, gridsize: int = 20, sign: int = 1, testisosurface: bool = True):
        
        imax = 1.0 / (2.0 * self.reflection)
        lspace = np.linspace(0.0, imax, gridsize)

        # Generate meshgrid dynamically based on dimensions
        if self.dim == 1:
            kz = [lspace]
        else:
            kz = np.meshgrid(*([lspace] * (self.dim - 1)), indexing='ij')
        
        # Core isosurface evaluation
        if self.variant=='NEPA': 
            gzp = hsurf_F2(self.reflection, kz, self.structurefactor, self.amplitude, j=self.dim - 1, s=sign, s2=sign)
        else:
            gzp = hsurf_g(self.reflection, kz, self.structurefactor, self.amplitude, j=self.dim - 1, s=sign)
        
        valid_mask = ~np.isnan(gzp)
        if not np.any(valid_mask):
            return {"status": False, "message": "No valid isosurface points found"}
        
        # Gather points into an (M, dim) array
        coords = [k[valid_mask] for k in kz]
        coords.append(gzp[valid_mask])
        
        iso_pts = np.column_stack(coords)
        inside = np.all((iso_pts >= 0) & (iso_pts <= imax), axis=1)
        iso_pts = iso_pts[inside]
        
        # Fast vector projection check (Always computed)
        dx = iso_pts @ self.normal
        dmin_calc, dmax_calc = float(dx.min()), float(dx.max())
        
        dmin_old, dmax_old = self.distance
        # dx_passed = (dmin_calc >= dmin_old) and (dmax_calc <= dmax_old)
        # dx_passed = (dmin_calc >= dmin_old-1e-12) and (dmax_calc <= dmax_old+1e-12)
        # dx_passed = (dmin_calc <= dmin_old-1e-12) and (dmax_calc >= dmax_old+1e-12)
        dx_passed = (dmin_calc >= dmin_old-1e-12) and (dmax_calc <= dmax_old+1e-12)
        
        # print(f'dmin_old: {dmin_old}, dmax_old: {dmax_old} dmin_calc:{dmin_calc} dmax_calc:{dmax_calc} dx_passed:{dx_passed}')
        
        result = {
            "xmin": iso_pts[dx.argmin()],
            "xmax": iso_pts[dx.argmax()],
            "dmin": dmin_calc,
            "dmax": dmax_calc,
            "boundary": {'innerdistance':dmin_calc, 'outerdistance': dmax_calc},
            "dist_new": np.array([dmin_calc, dmax_calc]),
            "dist_old": self.distance
            }

        if testisosurface:
            
            # Full physical boundary / polytope evaluation
            obj = Tessellator(self.reflection, self.normal, self.distance, IorG='intensity')
            P = obj.getpolytope_dfixed(dfixed=np.array([[0]*self.dim]))

            # P = getpoly_mitd(self.reflection, self.normal, self.distance, scom=np.ones((1, self.dim)), dlist=np.zeros((1, self.dim)), imax=imax)
            
            poly_inside = np.all(P[0].A @ iso_pts.T <= P[0].b[:, None] + 1e-12, axis=0)
            poly_passed = np.all(poly_inside)
            
            # For ultimate safety, both the projection boundary AND lateral walls must match
            status = bool(dx_passed and poly_passed)
            
            result.update({"status": status, "method": "checking points in polytope used",
                           "message": "Linearization successful (polytope used)" if status else "Failed. Out of polytope bounds or distance mismatch"
                           })
        else:
            # Light mode: Rely completely on the dx vector boundaries
            result.update({"status": dx_passed, "method": "gridbased and distance comparision", 
                           "message": "successful. CheckLinearizer is done by distance check" if dx_passed else "Failed. Use dist_new"})
        return result
    
    # =========================================================
    # Minimization-based method (With Iterative Loop Inside)
    # =========================================================
    def minimizebased(self, x0, tol: float = 1e-8, max_iter: int = 50):
        if x0 is None:
            raise ValueError("An initial guess 'x0' is required for the minimization method.")
        
        x0 = np.asarray(x0, dtype=float)
        dmin, dmax = self.distance
        success_streak = 0
        last_result = None

        # Pre-build optimization configurations once to save overhead cost
        if self.variant=='NEPA':
            cons = {'type': 'eq', 'fun': lambda x: self.gfun_NEPA(x) - self.amplitude}
        else:
            cons = {'type': 'eq', 'fun': lambda x: self.gfun(x) - self.amplitude}
        
        bounds = [(0, 1 / (2 * self.reflection))] * self.dim
        
        obj_max = lambda x: -np.dot(self.normal, x)
        obj_min = lambda x: np.dot(self.normal, x)
        
        x0_max, x0_min = x0.copy(), x0.copy()
        for it in range(max_iter):
            res_max = minimize(obj_max, x0=x0_max, constraints=cons, bounds=bounds, method='SLSQP')
            res_min = minimize(obj_min, x0=x0_min, constraints=cons, bounds=bounds, method='SLSQP')
            
            if not res_max.success or not res_min.success:
                return {"status":False,"message":"Optimization failed","max":res_max.message,"min":res_min.message}
            
            xmax, xmin = res_max.x, res_min.x
            
            x0_max, x0_min = xmax, xmin
            
            proj_max = np.dot(self.normal, xmax)
            proj_min = np.dot(self.normal, xmin)
            
            status = (proj_min >= dmin - tol and proj_max <= dmax + tol)
            
            # Adaptive bound adjustment
            dmin_new = min(dmin, proj_min)
            dmax_new = max(dmax, proj_max)
            converged_bounds = np.isclose(dmin_new, dmin, atol=tol) and np.isclose(dmax_new, dmax, atol=tol)
            
            dmin, dmax = dmin_new, dmax_new
            self.distance = [dmin, dmax]  # Mutate state tracker
            
            success_streak = (success_streak + 1) if status else 0
            
            last_result = {
                "iter": it,
                "dmin": dmin,
                "dmax": dmax,
                "proj_min": proj_min,
                "proj_max": proj_max,
                "xmin": xmin,
                "xmax": xmax,
                "status": status,
                "success_streak": success_streak,
                "method": "minimizebased"
            }

            # Break rules for performance efficiency
            if success_streak >= 2:
                break
            if status and converged_bounds and success_streak >= 1:
                break
            
            # Use current optimum coordinates as the hot-start guess for the next iteration
            x0 = xmin 

        return last_result

    # =========================================================
    # Unified Router Execution Entry Point
    # =========================================================
    def run(self, method: str, x0: list = None, tol: float = 1e-8, max_iter: int = 50, **kwargs):
        """
       runner interface for linearization checking
        
        Parameters:
            method : str ("gridbased", "minimizebased", etc.)
            x0  : list/array (Optional for grid-based, required for minimization)
            tol : float
            max_iter : int
            **kwargs : Supplementary args passed directly down to target execution functions
        """
        method = method.strip().lower()

        if method == "gridbased":
            # Extract specific parameters meant for grid calculation if present in kwargs
            gridsize = kwargs.get('gridsize', 20)
            sign = kwargs.get('sign', 1)
            testisosurface = kwargs.get('testisosurface', False)
            
            return self.gridbased(gridsize=gridsize, sign=sign, testisosurface=testisosurface)
        
        elif method == "minimizebased":
            return self.minimizebased(x0=x0, tol=tol, max_iter=max_iter)
        
        # Scalability: Add future techniques seamlessly here
        # elif method == "analytical":
        #     return self.analyticalbased(...)
        
        else:
            raise ValueError(f"Method '{method}' is unrecognized. Choose 'gridbased' or 'minimizebased'.")

# # How to use this class :
# # Define input

# h=2; f=np.array([10, 10,  8]); I=np.sqrt(11.406637303872357); n=np.array([0.603801, 0.603801, 0.52043 ])
# dmin, dmax = 0.1733642676273733, 0.22290624158852237
# o = CheckLinearizer(reflection=h, structurefactor=f, amplitude=I, normal=n, distance=[dmin, dmax] )
# o.run(method='gridbased')
# To activate checking point in polytope option 
#   o.run(method='gridbased', testisosurface=True)

#  To use minimization method, define a initial search point
# x0 = np.array([0., 0.2, 0.1 ])
# o.run(method='minimizebased', x0=x0)


def getpoly_mitd( l, normal, distance, scom, dlist, imax=1/6):
    
    polylist = []
    
    gpsc  = np.identity(len(normal))
    Apsc  = np.array(np.vstack([-gpsc, gpsc]))
    bpsc  = np.array([0]*len(normal) + [imax]*len(normal))
    psc   = pc.Polytope(Apsc, bpsc)
    
    aa    = np.array(normal)
    bb    = np.array(distance)
    
    for d in dlist:
        d  = np.array(d)
        oo = np.cos(2*np.pi*l*d)
        if (np.all(np.sign(oo) == 1) or np.all(np.sign(oo) == -1)):
            for i in scom:
                
                A = []
                A.append(-i*aa)
                A.append( i*aa)
                
                if i[len(normal)-1]>0:
                    b=np.array(np.array([-i[len(normal)-1], i[len(normal)-1]])*(bb + np.sum([i[kk]*aa[kk]*d[kk] for kk in range(len(d))])))
                
                else:
                    b=np.array(np.array([i[len(normal)-1], -i[len(normal)-1]])*(bb + np.sum([i[kk]*aa[kk]*d[kk] for kk in range(len(d))])))
                
                # ---> inner
                iden = np.identity(len(normal))
                for k in range(len(normal)):
                    A=np.vstack([A,-1*iden[k]])
                
                de = d + (i-1)*(1/(4*l))
                b=np.append(b, -de)
                
                # ---> outter
                for k in range(len(normal)):
                    A=np.vstack([A,iden[k]])
                    
                de = d + 1*(i+1)*(1/(4*l))
                b=np.append(b, de)
                
                w=pc.Polytope(np.array(A),np.array(b))
                
                if w.chebXc is not None:
                    if (w.chebXc in psc):
                        polylist.append(w)
                
    return pc.Region(polylist)

# def checklinear(l: int, f: list, I: float, normal: list, distance: list, j: int=2, n: int=20, s:int =1, testiso: bool=True):
        
#     """_Checks the quality of linearization. This is for EPA model not for non-EPA_
#     Args:
#         l (int)          : _The reflection order to be processed_
#         xcoor (list)     : _ Given atomic structure_
#         f (list)         : _atomic scattering factors it is actually [1.0]*len(xcoor)_
#         normal (list)    : _Found normal vector of isosurface of l_
#         distance (list)  : _Distance of inner and outer boundaries_
#         j (int, optional): _The atom index along last axis_. Defaults to 2.
#         n (int, optional): _Number of points to create isosurface_. Defaults to 20.
#         s (int, optional): _sign of amplitude_. Defaults to 1.
#         testiso (bool, optional): _Testing the isosurface_. Defaults to True.
#     """
    
#     # Create a linearly spaced array for the polytope's boundary
#     lspace = np.linspace(0, 1 / (2 * l), n)

#     # Generate the meshgrid for k-space dimensions
#     kz = np.meshgrid(*([lspace] * (len(f) - 1))) ; kz = list(kz)

#     # Compute the surface and isosurface points
#     gi = I #np.abs(g(l, xcoor, f))
#     gzp = hsurf_g(l, kz, f, gi, j=len(f)-1, s=s)
    
#     #; print(f'I={I} f={f} l={l} normal: {normal}')
    
#     # Calculate the polytope
#     o = getpoly_mitd(l, normal, distance, scom=np.ones((1, len(f))), dlist=np.zeros((1, len(f))), imax=lspace.max())
    
#     # Flatten each 3D array in kz and the gzp array
#     kz_flattened = [kz_i.flatten() for kz_i in kz]
#     gzp_flattened = gzp.flatten()
    
#     # Stack the flattened arrays column-wise to get the desired 2D array
#     iso_grid = np.vstack(kz_flattened + [gzp_flattened]).T
#     valid_iso_grid = iso_grid[~np.isnan(iso_grid).any(axis=1)]
    
#     dx=np.dot(valid_iso_grid, normal)
    
#     #print(f'from checklineara d_min & d_max : {np.min(dx)} {np.max(dx)}')
#     #print(f'from checklineara location are  : {valid_iso_grid[np.where(dx<=np.min(dx))][0]} {valid_iso_grid[np.where(dx==np.max(dx))]}\n--------------------')
    
#     # Check for isosurface containment within the polytope if required
#     if testiso:
#         # Check if any point lies outside the polytope
#         outside_points = np.array([ti for ti in valid_iso_grid if ti not in o])
        
#         if len(outside_points)!=0:
#             #ds=np.dot(outside_points, normal)
#             #print(f"\n\x1b[1;31m--> Checking the quality of linearization process")
#             #print(f"--> Found isosurface outside for the point at {len(outside_points)} locations.")# ds are ---> {np.min(dx)} {np.max(dx)}")
#             #print("\x1b[1;31m--> Check the linearization step <--\x1b[0m")
#             dt_corrected = [np.min(dx), np.max(dx)]
#             status = False
#             #raise ValueError("\x1b[1;32m--> Exiting: Linearization failed as isosurface points are outside the polytope\x1b[0m")
#         else:
#             print("\x1b[1;32m--> Polytope contains complete isosurface. Successful Linearization for \x1b[1;31mRO = %g\x1b[0m" % l)
#             dt_corrected = []
#             status = True
#     return status, dt_corrected

def checklinear_I(l, I, f, normal, distance, n=100, s=1):
    
    j = len(f)-1
    lspace  = np.linspace(0, 1/(2*l), n)
    kj = [lspace]*(len(f)-1)
    kz = np.meshgrid(*kj)
    #gz = np.zeros_like(kz[0])
    
    gzp = hsurf_F(l, [*kz], f, I, j, s=1, s2=1)
    o   = getpoly_mitd(l, normal, distance, scom=np.ones((1, len(f))), dlist=np.zeros((1, len(f))), imax=lspace.max() )
    
    # Flatten each 3D array in kz and the gzp array
    kz_flattened = [kz_i.flatten() for kz_i in kz]
    gzp_flattened = gzp.flatten()
    
    # Stack the flattened arrays column-wise to get the desired 2D array
    iso_grid = np.vstack(kz_flattened + [gzp_flattened]).T
    valid_iso_grid = iso_grid[~np.isnan(iso_grid).any(axis=1)]
    
    check=[i in o for i in valid_iso_grid]
        
    if not np.all(check):
        index = np.where(~np.array(check))[0]
        dr  = [np.dot(normal,valid_iso_grid[inx]) for inx in index]
        return False, [np.min(dr), np.max(dr)]
    else:
        return True, []

