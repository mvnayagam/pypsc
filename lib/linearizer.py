import logging
import numpy as np
from itertools import combinations, product

from psc.lib.gspacer import hsurf_g, hsurf_F2, hsurf_F
from psc.lib.checklinearizer import CheckLinearizer
# from psc.lib.checklinearizer import getpoly_mitd, checklinear

from scipy.optimize import minimize, least_squares


# -------------------------------------------------------
# Class for linearization
# -------------------------------------------------------

class EPALinearizer:
    """
    EPA hypersurface linearizer
    """

    # ==========================================================
    # init
    # ==========================================================
    def __init__( self, reflection, structurefactor, amplitude=None, structure=None, imax=0.5, logger=None):
        
        self.logger = logger or logging.getLogger("psc.linearizer")
        
        self.reflection = reflection
        self.structurefactor = np.asarray(structurefactor, dtype=float)
        
        self.dim = len(self.structurefactor)
        
        self.k = 2 * np.pi * reflection
        
        self.indices = np.arange(self.dim)
        
        self.imax = imax
        
        # ------------------------------------------------------
        # amplitude
        # ------------------------------------------------------
        if amplitude is None:
            if structure is None:
                raise ValueError("Either amplitude or structure must be provided.")
            self.amplitude = np.abs(self.g(structure))
        else:
            self.amplitude = np.abs(amplitude)
        
        # ------------------------------------------------------
        # cached diagonal point
        # ------------------------------------------------------
        self.xp_scalar = (1 / self.k) * np.arccos(np.abs(self.amplitude) / np.sum(self.structurefactor))
        self.xp = np.full(self.dim, self.xp_scalar)
    
    # ==========================================================
    # structure factor
    # ==========================================================
    def g(self, processatoms):
        return np.sum(self.structurefactor* np.cos(self.k * processatoms))
    
    # ==========================================================
    # SVD helper
    # ==========================================================
    @staticmethod
    def _svd_normal(points):
        centroid = points.mean(axis=0)
        _, _, v = np.linalg.svd( points - centroid, full_matrices=False)
        normal = v[-1]
        
        # orientation preserving
        if np.dot(normal, centroid) < 0:
            normal *= -1
        return normal, centroid

    # ==========================================================
    # isosurface solver
    # ==========================================================
    def hsurf_g(self, modifiedatom, atomindex,s=1):

        mask = np.ones(self.dim, dtype=bool)
        mask[atomindex] = False
        
        cos_terms = np.cos(self.k * modifiedatom[mask])
        
        argm = ((s* self.amplitude/ self.structurefactor[atomindex]) - 
                np.sum((self.structurefactor[mask]/ self.structurefactor[atomindex])* cos_terms))
                
        # numercal safty
        # argm = argm-np.ceil(argm) if argm <0 else argm-np.floor(argm)
        # argm = np.clip(argm, -1.0, 1.0)
        
        return np.arccos(argm) / self.k
    
    # ==========================================================
    # merged edge point generator
    # ==========================================================
    def get_edgepoints( self, fallback=False, xinit=0):
        
        pts = np.zeros((self.dim, self.dim))

        # ------------------------------------------------------
        # primary edge points
        # ------------------------------------------------------
        for atomindexi in range(self.dim):
            
            x = np.zeros(self.dim)
            x[atomindexi] = self.hsurf_g(x,atomindexi,s=1)
            pts[atomindexi] = x

        # ------------------------------------------------------
        # fallback reconstruction
        # ------------------------------------------------------
        if fallback:
            nanmask = np.isnan(pts)
            if nanmask.any():
                bad = np.where(nanmask.any(axis=1))[0]
                for iw in bad:
                    x = np.zeros(self.dim)
                    x[iw] = (xinit / self.reflection)
                    mlist = np.delete(self.indices,iw)
                    numerator = self.amplitude - self.structurefactor[iw]* np.cos(self.k * x[iw])
                    denominator = np.sum(self.structurefactor[mlist]* np.cos(self.k * x[mlist]))
                    arg = numerator / denominator
                    xval = (1 / self.k)* np.arccos(arg)
                    x[iw + 1:] = xval
                    x[:iw] = xval
                    pts[iw] = x
        # ------------------------------------------------------
        # add diagonal point
        # ------------------------------------------------------
        # if fallback:
            # pts = np.vstack([ pts, self.xp ])
        
        return np.array(pts)

    # ==========================================================
    # optimized face point generator
    # ==========================================================
    def pointonface(self):

        kp = []
        
        for i in range(self.dim - 1):
            z = np.zeros(self.dim)
            up = np.delete(self.indices,np.arange(i, self.dim))
            
            down = np.arange(i, self.dim)
            numerator   = self.amplitude-np.sum(self.structurefactor[up]* np.cos(self.k * z[up]))
            denominator = np.sum(self.structurefactor[down])
            
            gg = (1 / self.k)*np.arccos(numerator / denominator)
            
            if np.isnan(gg):
                gg = 1/(2 * self.reflection)
            
            # --------------------------------------------------
            # combinations instead of permutations
            # --------------------------------------------------
            n_fill = len(down)

            for comb in combinations( self.indices, n_fill):
                x = np.zeros(self.dim)
                x[list(comb)] = gg
                kp.append(x)

        return np.array(kp)

    # ==========================================================
    # main linearizer
    # ==========================================================
    def linearize(self):

        edgepoints = self.get_edgepoints()

        nanmask = np.isnan(edgepoints)
        
        # ======================================================
        # CASE 1: easy - isosurface intersects all axis 
        # ======================================================
        if not nanmask.any():
            
            self.logger.info(f"for l : {self.reflection} - No edgepoints missing")
            
            normal, centroid = self._svd_normal(edgepoints)
            
            dinner = np.dot(normal,centroid)
            
            douter = np.dot(normal,self.xp)

            # --------------------------------------------------
            # safe divide
            # --------------------------------------------------
            eps = 1e-12

            aps = douter / np.where(np.abs(normal) < eps, eps, normal)

            outerpoint = (np.eye(self.dim) * aps)

            completepoint = np.vstack([edgepoints, outerpoint, self.xp])
            
            boundary = {"innerdistance": dinner, "outerdistance": douter}
            
            # checkstatus, checked_dist = checklinear(self.reflection, self.structurefactor, np.abs(self.amplitude), normal, np.array([dinner, douter]), j=self.dim - 1)
            oo = CheckLinearizer(reflection=self.reflection, structurefactor=self.structurefactor, amplitude=np.abs(self.amplitude), normal=normal,distance=np.array([dinner, douter]), logger=self.logger.getChild("checklinear"))
                        
            o = oo.run(method='gridbased')
                
            if o['status']:
                self.logger.info(f"! Status : {o['status']} Linearization for reflection {self.reflection} is {o['message']}")
            else:
                self.logger.warning(f"! Status : {o['status']} Linearization for reflection {self.reflection} is {o['message']}")
            
            # self.logger.info(f'checkstatus: {checkstatus} checked_dist: {checked_dist} old_dist {[dinner, douter]}')
            
        # ======================================================
        # CASE 2: hard - isosurface does not intersect one or many axis 
        # ======================================================
        else:

            self.logger.info(f"NaN detected: {nanmask.sum()} edgepoints missing - fallback mode")

            facepoint = self.pointonface()

            innerpoint = self.get_edgepoints(fallback=True, xinit=0 )

            outerpoint = self.get_edgepoints(fallback=True, xinit=0.5)

            completepoint = np.vstack([innerpoint, outerpoint, facepoint, self.xp])
            
            # optional deduplication
            completepoint = np.unique(completepoint, axis=0 )

            normal, _ = self._svd_normal( outerpoint)

            projected_dist = completepoint @ normal
            
            d_all = [projected_dist.min(), projected_dist.max()]

            # checkstatus, checked_dist = checklinear(self.reflection, self.structurefactor, self.amplitude, normal, d_all, j=self.dim - 1)
            oo = CheckLinearizer(reflection=self.reflection, structurefactor=self.structurefactor, amplitude=np.abs(self.amplitude), normal=normal,distance=d_all, logger=self.logger.getChild("checklinear"))
            
            o = oo.run(method='gridbased')
                        
            self.logger.info(f'checkstatus: {o['status']} New dist:{ np.array([o['boundary']['innerdistance'], o['boundary']['outerdistance'] ]) }')
            
            if not o['status']:

                d_all = np.array([o['boundary']['innerdistance'], o['boundary']['outerdistance'] ])

                # checkstatus, checked_dist = checklinear( self.reflection, self.structurefactor, self.amplitude, normal, d_all, j=self.dim - 1)
                
                oo = CheckLinearizer(reflection=self.reflection, structurefactor=self.structurefactor, amplitude=np.abs(self.amplitude), normal=normal,distance=d_all, logger=self.logger.getChild("checklinear"))
                
            boundary = { 
                        "innerdistance": (np.floor(d_all[0] * 1e6) - 1) / 1e6,
                        "outerdistance": ( np.floor(d_all[1] * 1e6) + 1) / 1e6
                        }

        self.logger.info(f"Normal = {normal}")
        self.logger.info(f"Boundary distance = {boundary}")
        
        return {"normal": normal, "boundary": boundary, "polytope_points": completepoint}

# # How to use this class:
# normal=[0.577350, 0.577350, 0.577350]; ds=[0.157692, 0.194407]
# l=1; atom=np.array([0.47512, 0.35034, 0.11992]); f=np.array([1]*3)
# amp = 0.8480055814202209 # np.abs(g2(l, atom, f))
# epa = EPALinearizer(reflection=l, structurefactor=f,  atoms=atom, imax=0.5)
# out = epa.linearize()
# print(out) - out is dict of form :  {"normal": normal, "boundary": boundary, "polytope points": completepoint}


class NEPALinearizer:
    
    def __init__( self, reflection, structurefactor, intensity=None, structure=None, imax=0.5, logger=None):
        
        self.logger = logger or logging.getLogger("psc.linearizer")
        
        self.reflection = reflection
        self.structurefactor = np.asarray(structurefactor, dtype=float)
        self.dim = len(self.structurefactor)
        self.k = 2 * np.pi * reflection
        self.indices = np.arange(self.dim)
        self.imax = imax
        self.period  = [0, 1/(2*reflection)]
        
        # ------------------------------------------------------
        # Intensity
        # ------------------------------------------------------
        if intensity is None:
            if structure is None:
                raise ValueError("Either intensity or structure must be provided.")
            self.intensity = self.F(structure)**2
        else:
            self.intensity = float(np.abs(intensity))
        
        arg = np.sqrt(self.intensity) / np.sum(self.structurefactor)
        arg = np.clip(arg, -1.0, 1.0)
        
        self.xp_scalar = np.arccos(arg) / self.k
        self.xp = np.full(self.dim, self.xp_scalar)
        
    def gradFs(self, coordinates):
        return -self.k * self.structurefactor * np.sin(self.k * coordinates)
    
    def F(self, zcoordinate):
        zcoordinate = np.asarray(zcoordinate, dtype=float)
        return np.sum(self.structurefactor * np.cos(self.k * zcoordinate))
    
    @staticmethod
    def findnormal(ps):

        ps = np.asarray(ps)

        centroid = ps.mean(axis=0)

        _, _, vh = np.linalg.svd(ps - centroid)

        normal = vh[-1]

        if normal[-1] < 0:
            normal *= -1

        ds = np.min(np.abs(ps @ normal))

        return normal, ds
    
    @staticmethod
    def angle_between(v1, v2):
        dot_pr = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(np.clip(dot_pr, -1.0, 1.0))
        
    def get_edgepoints(self): # getxyz
        
        edgepoints = []
        
        for j in range(self.dim):
            
            others = np.delete(self.indices, j)
            
            for comb in product(self.period, repeat=len(others)):

                tmpcoordinate = np.zeros(self.dim)
                
                for idx, value in zip(others, comb):
                    tmpcoordinate[idx] = value
                    
                z = hsurf_F(self.reflection, tmpcoordinate, self.structurefactor, self.intensity, j=j, s=1, s2=1)
                
                if not np.isnan(z):
                    tmpcoordinate[j] = z
                    edgepoints.append(tmpcoordinate.copy())
                    
        return np.array(np.unique(edgepoints, axis=0))
    
    def get_axispoints(self):  # get_pnts
        tot_coor = np.zeros((self.dim, self.dim))
        
        for i in range(self.dim):
            tot_coor[i, i] = hsurf_F2(self.reflection, tot_coor[i], self.structurefactor, self.intensity, j=i, s=1, s2=1)
        return np.array(tot_coor)
        
    def ponitonplane(self):
        
        points = []
        
        for j in range(0, self.dim):
            inx = np.delete(np.arange(self.dim), j)
            denominator = np.sum(self.structurefactor[inx])
                
            for zj in self.period:
                temp = np.array([zj] * self.dim)
                
                # dr = (1/self.k) * np.arccos((np.sqrt(self.intensity) - self.structurefactor[j] * np.cos(self.k * zj)) / denominator)
                
                arg = ( np.sqrt(self.intensity) - self.structurefactor[j]*np.cos(self.k*zj) ) / denominator
                arg = np.clip(arg, -1.0, 1.0)
                
                dr = np.arccos(arg)/self.k
                
                temp = np.where(np.isin(np.arange(self.dim), inx), dr, temp)
                
                points.append(temp) if not np.any(np.isnan(temp)) else None
        
        return np.array(points)
    
    def tangentpoints(self, tangent_guess, normal):
        
        # Define global lists to store values at each iteration
        z_final, grad_norm, anglelist = [], [], []
        
        z_final_all, grad_norm_all, anglelist_all, tangentpoint = [], [], [], []
        
        # Defining the error function
        def errfun(z0, normal):
            
            my_scale  = 1E10
            # z0 = z0 if z0.ndim == 1 else z0[0]
            z0 = np.asarray(z0, dtype=float)
            
            z = z0.copy()
            z[self.dim-1] = hsurf_F2( self.reflection, z0, self.structurefactor, self.intensity, j=self.dim-1, s=1, s2=1, nan=False )
            
            I_new = self.F(z)**2
            
            grad_iso_norm = self.gradFs(z)
            
            dotproduct = np.dot(grad_iso_norm, normal)
            angle = self.angle_between(grad_iso_norm, normal)
            
            # Append values to the global lists
            z_final.append(z.copy())
            grad_norm.append(grad_iso_norm.copy())
            anglelist.append(angle.copy())
            
            return ( I_new - self.intensity)**2 - my_scale * (dotproduct + 1)**2
        
        # Define searching space by bound
        # bound=(np.array(np.min(tangent_guess, axis=0)), np.array(np.max(tangent_guess, axis=0)))
        bound=(np.array(np.min(tangent_guess, axis=0)), np.array(np.max(tangent_guess, axis=0)))
        ds=(bound[1]-bound[0])/100
        
        diagonalpnt=tangent_guess[-1]
        
        # Iterate over different z_2D points
        for z2D in tangent_guess:
            
            z_final, grad_norm, anglelist = [], [], [] 
            
            z2D = np.array(z2D)
            z_direction = np.sign(np.array(diagonalpnt - z2D))
            
            ds=(bound[1]-bound[0])/100
            
            trycount = 0
            
            while trycount <=10:
                try:
                    res = least_squares(errfun, z2D, jac='2-point', bounds=bound, method='trf', ftol=1e-30, xtol=1e-10,
                                        gtol=1e-16, x_scale=1, loss='linear', f_scale=1.0,diff_step=None, tr_solver=None,
                                        tr_options={}, jac_sparsity=None, max_nfev=1000,verbose=0,
                                        args=(normal, ))
                    
                    # Append the result to the list
                    tangentpoint.append(res.x)
                    
                    break
                except (ValueError, RuntimeError) as e:
                    logging.warning(f"Optimization failed for initial guess = {z2D, z_direction} with error: {e} ")
                    
                    z_2Dp = (z2D + ds * z_direction)
                    z_2Dn = (z2D - ds * z_direction)
                    z_2Dm = z_2Dn if np.all(z_2Dn <= bound[1]) or np.all(z_2Dn >= bound[0]) else z_2Dp
                    z2D = np.where(z_direction<=0, z_2Dm, z2D)
                    
                    trycount += 1
            else:
                logging.warning(f"Optimization failed for initial guess = {z2D} from else")
                
            z_final_all.append(np.array(z_final))
            grad_norm_all.append(np.array(grad_norm))
            anglelist_all.append(np.array(anglelist))
            
        return np.array(tangentpoint), z_final_all, grad_norm_all, anglelist_all
    
    def tangent_extreme_points(self, tangent_guess, normal):
        
        # Dictionaries to split your final outputs cleanly
        results = {'maximize': {}, 'minimize': {}}
        tangentcollection = {'maximize': {}, 'minimize': {}}
        
        PHYSICAL_MAX = 1.0 / (2.0 * self.reflection)   # replace with self.period[0]
        normal_unit = np.array(normal) / np.linalg.norm(normal)
        
        # --- SWEEP BOTH EXTREMES IN ONE GO ---
        for mode in ['maximize', 'minimize']:
            # Set the directional sign toggler
            direction_sign = -1.0 if mode == 'maximize' else 1.0
            
            mpcollection = []
             
            for inx, z_2D in enumerate(tangent_guess):
                z_2D = np.array(z_2D, dtype=float)
                z_2D_initial = z_2D.copy()
                
                def cost_and_gradient(z_eval, normal_vector, w_intensity):
                    if z_eval.ndim > 1:
                        z_eval = z_eval[0]
                        
                    # sqrt_I = np.sqrt(self.intensity)
                    sqrt_I = self.intensity
                    
                    # 1. --- DYNAMIC SCALAR COST ---
                    F_val = self.F( z_eval)
                    # g_z   = np.sqrt(F_val**2) - sqrt_I
                    g_z   = F_val**2 - self.intensity
                    err_intensity = g_z**2
                    
                    # Applies -1 for max tracking, +1 for min tracking
                    cost_distance = direction_sign * np.dot(normal_vector, z_eval) 
                    total_cost = cost_distance + (w_intensity * err_intensity)
                    
                    # 2. --- DYNAMIC ANALYTICAL GRADIENT ---
                    grad_distance = direction_sign * normal_vector
                    analytical_gradF = self.gradFs(z_eval) # gradF_analytical(h, f, z_eval) 
                    grad_intensity = 2.0 * w_intensity * g_z * analytical_gradF
                    
                    total_gradient = grad_distance + grad_intensity
                    
                    return total_cost, total_gradient
                
                bounds = [(0.0, PHYSICAL_MAX)] * self.dim
                step_chunks = [0.0, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.05, 0.065, 0.07, 0.075] 
                starting_points = [z_2D_initial + (chunk * PHYSICAL_MAX) for chunk in step_chunks]
                
                trycount = 0
                success = False
                
                while trycount < len(starting_points):
                    z_search = np.clip(starting_points[trycount], 0.0, PHYSICAL_MAX)
                    w_intensity = 1E5
                    
                    # res = minimize(cost_and_gradient, z_search, method='L-BFGS-B', jac=True, options={'ftol': 0.0, 'gtol': 1e-15, 'maxiter': 5000, 'maxls': 50}, args=(h, I, f, normal_unit, w_intensity))
                    res = minimize(cost_and_gradient, z_search, method='L-BFGS-B', bounds=bounds, jac=True, options={'ftol': 0.0, 'gtol': 1e-15, 'maxiter': 5000, 'maxls': 50}, args=(normal_unit, w_intensity))
                    
                    I_final = self.F(res.x)**2
                    intensity_check = np.abs(I_final - self.intensity)
                    
                    mpcollection.append(res.x)
                    
                    if res.success and intensity_check < 0.01:
                        final_analytical_grad = self.gradFs(res.x) # gradF_analytical(h, f, res.x)
                        norm_val = np.linalg.norm(final_analytical_grad)
                        final_grad_normed = final_analytical_grad / norm_val if norm_val != 0 else final_analytical_grad
                        
                        # Collinear verification mapped directly to degrees
                        raw_angle = self.angle_between(final_grad_normed, normal_unit)
                        collinear_angle_deg = np.rad2deg(np.minimum(raw_angle, np.pi - raw_angle))
                        
                        results[mode][f'mps{inx}'] = {'zcoordinate': res.x, 
                                                    'distance': np.abs(np.dot(normal_unit, res.x)),
                                                    'angle_deg': collinear_angle_deg, 
                                                    'I_final': I_final, 
                                                    'I_diff': intensity_check, 
                                                    'F_gradiant': final_grad_normed
                                                    }
                        
                        self.logger.info(f'[{mode.upper()}] Guess={z_2D_initial.round(4)} -> Found={res.x.round(4)} | I={I_final:.6f} | D = {np.abs(np.dot(normal_unit, res.x)):.8f} | Angle={collinear_angle_deg:.2f} deg.')
                        
                        success = True
                        break
                    else:
                        trycount += 1
                        
                if not success:
                    logging.warning(f"[{mode.upper()}] Optimization failed for: {z_2D_initial}\n")
            
            tangentcollection[mode][f'tangentpoint'] = np.array(mpcollection)
                
        return results, tangentcollection
    
    def linearize(self,): 
        
        axispoint = self.get_axispoints()
        
        count_Nan = np.count_nonzero(np.isnan(axispoint))
        nanmask = np.isnan(axispoint)
                        
        if count_Nan == 0:
            self.logger.info(f'[NEPALinearizer] found non-intersection: {count_Nan}')
            # print(f'\x1b[1;33m ---> [NEPALinearizer] From if count_Nan == {count_Nan}: \x1b[0m')
            
            points = np.vstack([np.array(axispoint), self.xp])
            
            ni, di   = self.findnormal(axispoint)
            n_select = ni
                        
            finalres, mp = self.tangent_extreme_points(tangent_guess=points, normal=n_select)
            bestmax_key, best_zmax, best_maxdist = max([(k, v['zcoordinate'], v['distance']) for k, v in finalres['maximize'].items()], key=lambda x: x[2])
            bestmin_key, best_zmin, best_mindist = min([(k, v['zcoordinate'], v['distance']) for k, v in finalres['minimize'].items()],  key=lambda x: x[2])
                        
            mp_select = np.array([best_zmin, best_zmax])
            d_min_max = np.array([best_mindist, best_maxdist])
            best_keys  = np.array([bestmin_key, bestmax_key])

            
        elif (count_Nan > 0) and (count_Nan < self.dim-1):
            self.logger.info(f'[NEPALinearizer] found non-intersection: {count_Nan}')
            # print(f'\x1b[1;33m ---> [NEPALinearizer] From elif count_Nan == {count_Nan}: \x1b[0m')
                                
            innerpnts = self.get_edgepoints()
            points = np.vstack([np.array(innerpnts), self.xp])
            
            dsi = np.linalg.norm(innerpnts[:, :self.dim-1], axis=1)
            p2  = innerpnts[dsi <np.max(dsi)]
            
            n_select, di   = self.findnormal(innerpnts)
            
            finalres, mp = self.tangent_extreme_points(tangent_guess=points, normal=n_select)
            bestmax_key, best_zmax, best_maxdist = max([(k, v['zcoordinate'], v['distance']) for k, v in finalres['maximize'].items()], key=lambda x: x[2])
            bestmin_key, best_zmin, best_mindist = min([(k, v['zcoordinate'], v['distance']) for k, v in finalres['minimize'].items()],  key=lambda x: x[2])
                        
            mp_select = np.array([best_zmin, best_zmax])
            d_min_max = np.array([best_mindist, best_maxdist])
            best_keys  = np.array([bestmin_key, bestmax_key])
            
        elif count_Nan >= self.dim-1:
            self.logger.info(f'[NEPALinearizer] found non-intersection: {count_Nan}')
            # print(f'\x1b[1;33m ---> [NEPALinearizer] Fromcount_Nan == self.dim-1 {count_Nan}: \x1b[0m')
            
            innerpnts   = self.get_edgepoints()
            pplane      = self.ponitonplane()
            points      = np.vstack([np.array(innerpnts), self.xp, pplane])
            
            dsi = np.linalg.norm(points[:], axis=1)
            
            n_select, di  = self.findnormal(points)
            di = np.min( points @ n_select)
            
            finalres, mp = self.tangent_extreme_points(tangent_guess=points, normal=n_select)
            bestmax_key, best_zmax, best_maxdist = max([(k, v['zcoordinate'], v['distance']) for k, v in finalres['maximize'].items()], key=lambda x: x[2])
            bestmin_key, best_zmin, best_mindist = min([(k, v['zcoordinate'], v['distance']) for k, v in finalres['minimize'].items()],  key=lambda x: x[2])
            
            mp_select = np.array([best_zmin, best_zmax])
            d_min_max = np.array([best_mindist, best_maxdist])
            best_keys  = np.array([bestmin_key, bestmax_key])
            
        else:
            
            print(f'\x1b[1;33m ---> else. Nothing is possible: \x1b[0m')
            print(f"--> count_Nan {count_Nan}")
            print(f"--> Wired isosurface check linearization routine ")
            print(f"--> predicted points are : \n {axispoint}")
        
        # print(out)  <- you get {"normal": normal, "boundary": boundary, "polytope points": completepoint} # print(f"linear: {linear}")
        
        results = {"normal": n_select, 
                   "boundary": {"innerdistance": d_min_max[0],"outerdistance": d_min_max[1]},
                   "polytope points": np.array(points)
                #    "tangelt_points": mp,
                #    "tangelt_points_selected": mp_select,
                #    "final_result": finalres,
                #    "best_keys": best_keys
                   }
        return results
        # return n_select, d_min_max, np.array(points), mp, mp_select, finalres, best_keys 
        

class NEPALinearizerLOCAL:
    
    def __init__( self, reflection, structurefactor, intensity=None, atoms=None, imax=0.5, logger=None):
        
        self.logger = logger or logging.getLogger("psc.linearizer")
        
        self.reflection = reflection
        self.structurefactor = np.asarray(structurefactor, dtype=float)
        self.dim = len(self.structurefactor)
        self.k = 2 * np.pi * reflection
        self.indices = np.arange(self.dim)
        self.imax = imax
        self.period  = [0, 1/(2*reflection)]
        
        # ------------------------------------------------------
        # Intensity
        # ------------------------------------------------------
        if intensity is None:
            if atoms is None:
                raise ValueError("Either intensity or atoms must be provided.")
            self.intensity = self.F(atoms)**2
        else:
            self.intensity = float(np.abs(intensity))
        
        arg = np.sqrt(self.intensity) / np.sum(self.structurefactor)
        arg = np.clip(arg, -1.0, 1.0)
        
        self.xp_scalar = np.arccos(arg) / self.k
        self.xp = np.full(self.dim, self.xp_scalar)
        
    def gradFs(self, coordinates):
        return -self.k * self.structurefactor * np.sin(self.k * coordinates)
    
    def F(self, zcoordinate):
        zcoordinate = np.asarray(zcoordinate, dtype=float)
        return np.sum(self.structurefactor * np.cos(self.k * zcoordinate))
    
    @staticmethod
    def findnormal(ps):

        ps = np.asarray(ps)

        centroid = ps.mean(axis=0)

        _, _, vh = np.linalg.svd(ps - centroid)

        normal = vh[-1]

        if normal[-1] < 0:
            normal *= -1

        ds = np.min(np.abs(ps @ normal))

        return normal, ds
    
    @staticmethod
    def angle_between(v1, v2):
        dot_pr = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(np.clip(dot_pr, -1.0, 1.0))
        
    def get_edgepoints(self): # getxyz
        
        edgepoints = []
        
        for j in range(self.dim):
            
            others = np.delete(self.indices, j)
            
            for comb in product(self.period, repeat=len(others)):

                tmpcoordinate = np.zeros(self.dim)
                
                for idx, value in zip(others, comb):
                    tmpcoordinate[idx] = value
                    
                z = hsurf_F(self.reflection, tmpcoordinate, self.structurefactor, self.intensity, j=j, s=1, s2=1)
                
                if not np.isnan(z):
                    tmpcoordinate[j] = z
                    edgepoints.append(tmpcoordinate.copy())
                    
        return np.array(np.unique(edgepoints, axis=0))
    
    def get_axispoints(self):  # get_pnts
        tot_coor = np.zeros((self.dim, self.dim))
        
        for i in range(self.dim):
            tot_coor[i, i] = hsurf_F2(self.reflection, tot_coor[i], self.structurefactor, self.intensity, j=i, s=1, s2=1)
        return np.array(tot_coor)
        
    def ponitonplane(self):
        
        points = []
        
        for j in range(0, self.dim):
            inx = np.delete(np.arange(self.dim), j)
            denominator = np.sum(self.structurefactor[inx])
                
            for zj in self.period:
                temp = np.array([zj] * self.dim)
                
                # dr = (1/self.k) * np.arccos((np.sqrt(self.intensity) - self.structurefactor[j] * np.cos(self.k * zj)) / denominator)
                
                arg = ( np.sqrt(self.intensity) - self.structurefactor[j]*np.cos(self.k*zj) ) / denominator
                arg = np.clip(arg, -1.0, 1.0)
                
                dr = np.arccos(arg)/self.k
                
                temp = np.where(np.isin(np.arange(self.dim), inx), dr, temp)
                
                points.append(temp) if not np.any(np.isnan(temp)) else None
        
        return np.array(points)
    
    def tangentpoints(self, tangent_guess, normal):
        
        # Define global lists to store values at each iteration
        z_final, grad_norm, anglelist = [], [], []
        
        z_final_all, grad_norm_all, anglelist_all, tangentpoint = [], [], [], []
        
        # Defining the error function
        def errfun(z0, normal):
            
            my_scale  = 1E10
            # z0 = z0 if z0.ndim == 1 else z0[0]
            z0 = np.asarray(z0, dtype=float)
            
            z = z0.copy()
            z[self.dim-1] = hsurf_F2( self.reflection, z0, self.structurefactor, self.intensity, j=self.dim-1, s=1, s2=1, nan=False )
            
            I_new = self.F(z)**2
            
            grad_iso_norm = self.gradFs(z)
            
            dotproduct = np.dot(grad_iso_norm, normal)
            angle = self.angle_between(grad_iso_norm, normal)
            
            # Append values to the global lists
            z_final.append(z.copy())
            grad_norm.append(grad_iso_norm.copy())
            anglelist.append(angle.copy())
            
            return ( I_new - self.intensity)**2 - my_scale * (dotproduct + 1)**2
        
        # Define searching space by bound
        # bound=(np.array(np.min(tangent_guess, axis=0)), np.array(np.max(tangent_guess, axis=0)))
        bound=(np.array(np.min(tangent_guess, axis=0)), np.array(np.max(tangent_guess, axis=0)))
        ds=(bound[1]-bound[0])/100
        
        diagonalpnt=tangent_guess[-1]
        
        # Iterate over different z_2D points
        for z2D in tangent_guess:
            
            z_final, grad_norm, anglelist = [], [], [] 
            
            z2D = np.array(z2D)
            z_direction = np.sign(np.array(diagonalpnt - z2D))
            
            ds=(bound[1]-bound[0])/100
            
            trycount = 0
            
            while trycount <=10:
                try:
                    res = least_squares(errfun, z2D, jac='2-point', bounds=bound, method='trf', ftol=1e-30, xtol=1e-10,
                                        gtol=1e-16, x_scale=1, loss='linear', f_scale=1.0,diff_step=None, tr_solver=None,
                                        tr_options={}, jac_sparsity=None, max_nfev=1000,verbose=0,
                                        args=(normal, ))
                    
                    # Append the result to the list
                    tangentpoint.append(res.x)
                    
                    break
                except (ValueError, RuntimeError) as e:
                    logging.warning(f"Optimization failed for initial guess = {z2D, z_direction} with error: {e} ")
                    
                    z_2Dp = (z2D + ds * z_direction)
                    z_2Dn = (z2D - ds * z_direction)
                    z_2Dm = z_2Dn if np.all(z_2Dn <= bound[1]) or np.all(z_2Dn >= bound[0]) else z_2Dp
                    z2D = np.where(z_direction<=0, z_2Dm, z2D)
                    
                    trycount += 1
            else:
                logging.warning(f"Optimization failed for initial guess = {z2D} from else")
                
            z_final_all.append(np.array(z_final))
            grad_norm_all.append(np.array(grad_norm))
            anglelist_all.append(np.array(anglelist))
            
        return np.array(tangentpoint), z_final_all, grad_norm_all, anglelist_all
    
    def tangent_extreme_points(self, tangent_guess, normal):
        
        # Dictionaries to split your final outputs cleanly
        results = {'maximize': {}, 'minimize': {}}
        tangentcollection = {'maximize': {}, 'minimize': {}}
        
        PHYSICAL_MAX = 1.0 / (2.0 * self.reflection)   # replace with self.period[0]
        normal_unit = np.array(normal) / np.linalg.norm(normal)
        
        # --- SWEEP BOTH EXTREMES IN ONE GO ---
        for mode in ['maximize', 'minimize']:
            # Set the directional sign toggler
            direction_sign = -1.0 if mode == 'maximize' else 1.0
            
            mpcollection = []
             
            for inx, z_2D in enumerate(tangent_guess):
                z_2D = np.array(z_2D, dtype=float)
                z_2D_initial = z_2D.copy()
                
                def cost_and_gradient(z_eval, normal_vector, w_intensity):
                    if z_eval.ndim > 1:
                        z_eval = z_eval[0]
                        
                    # sqrt_I = np.sqrt(self.intensity)
                    sqrt_I = self.intensity
                    
                    # 1. --- DYNAMIC SCALAR COST ---
                    F_val = self.F( z_eval)
                    # g_z   = np.sqrt(F_val**2) - sqrt_I
                    g_z   = F_val**2 - self.intensity
                    err_intensity = g_z**2
                    
                    # Applies -1 for max tracking, +1 for min tracking
                    cost_distance = direction_sign * np.dot(normal_vector, z_eval) 
                    total_cost = cost_distance + (w_intensity * err_intensity)
                    
                    # 2. --- DYNAMIC ANALYTICAL GRADIENT ---
                    grad_distance = direction_sign * normal_vector
                    analytical_gradF = self.gradFs(z_eval) # gradF_analytical(h, f, z_eval) 
                    grad_intensity = 2.0 * w_intensity * g_z * analytical_gradF
                    
                    total_gradient = grad_distance + grad_intensity
                    
                    return total_cost, total_gradient
                
                bounds = [(0.0, PHYSICAL_MAX)] * self.dim
                step_chunks = [0.0, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.05, 0.065, 0.07, 0.075] 
                starting_points = [z_2D_initial + (chunk * PHYSICAL_MAX) for chunk in step_chunks]
                
                trycount = 0
                success = False
                
                while trycount < len(starting_points):
                    z_search = np.clip(starting_points[trycount], 0.0, PHYSICAL_MAX)
                    w_intensity = 1E5
                    
                    # res = minimize(cost_and_gradient, z_search, method='L-BFGS-B', jac=True, options={'ftol': 0.0, 'gtol': 1e-15, 'maxiter': 5000, 'maxls': 50}, args=(h, I, f, normal_unit, w_intensity))
                    res = minimize(cost_and_gradient, z_search, method='L-BFGS-B', bounds=bounds, jac=True, options={'ftol': 0.0, 'gtol': 1e-15, 'maxiter': 5000, 'maxls': 50}, args=(normal_unit, w_intensity))
                    
                    I_final = self.F(res.x)**2
                    intensity_check = np.abs(I_final - self.intensity)
                    
                    mpcollection.append(res.x)
                    
                    if res.success and intensity_check < 0.01:
                        final_analytical_grad = self.gradFs(res.x) # gradF_analytical(h, f, res.x)
                        norm_val = np.linalg.norm(final_analytical_grad)
                        final_grad_normed = final_analytical_grad / norm_val if norm_val != 0 else final_analytical_grad
                        
                        # Collinear verification mapped directly to degrees
                        raw_angle = self.angle_between(final_grad_normed, normal_unit)
                        collinear_angle_deg = np.rad2deg(np.minimum(raw_angle, np.pi - raw_angle))
                        
                        results[mode][f'mps{inx}'] = {'zcoordinate': res.x, 
                                                    'distance': np.abs(np.dot(normal_unit, res.x)),
                                                    'angle_deg': collinear_angle_deg, 
                                                    'I_final': I_final, 
                                                    'I_diff': intensity_check, 
                                                    'F_gradiant': final_grad_normed
                                                    }
                        
                        self.logger.info(f"[{mode.upper()}] Guess={z_2D_initial.round(4)} -> Found={res.x.round(4)} | I={I_final:.6f} | D = {np.abs(np.dot(normal_unit, res.x)):.8f} | Angle={collinear_angle_deg:.2f}° \x1b[0m")
                        
                        success = True
                        break
                    else:
                        trycount += 1
                        
                if not success:
                    logging.warning(f"[{mode.upper()}] Optimization failed for: {z_2D_initial}\n")
            
            tangentcollection[mode][f'tangentpoint'] = np.array(mpcollection)
                
        return results, tangentcollection
    
    def linearize(self,): 
        
        axispoint = self.get_axispoints()
        
        count_Nan = np.count_nonzero(np.isnan(axispoint))
        nanmask = np.isnan(axispoint)
                        
        if count_Nan == 0:
            self.logger.info(f'[NEPALinearizer] found non-intersection: {count_Nan}')
            
            points = np.vstack([np.array(axispoint), self.xp])
            
            ni, di   = self.findnormal(axispoint)
            n_select = ni
                        
            finalres, mp = self.tangent_extreme_points(tangent_guess=points, normal=n_select)
            bestmax_key, best_zmax, best_maxdist = max([(k, v['zcoordinate'], v['distance']) for k, v in finalres['maximize'].items()], key=lambda x: x[2])
            bestmin_key, best_zmin, best_mindist = min([(k, v['zcoordinate'], v['distance']) for k, v in finalres['minimize'].items()],  key=lambda x: x[2])
                        
            mp_select = np.array([best_zmin, best_zmax])
            d_min_max = np.array([best_mindist, best_maxdist])
            best_keys  = np.array([bestmin_key, bestmax_key])

            
        elif (count_Nan > 0) and (count_Nan < self.dim-1):
            self.logger.info(f'[NEPALinearizer] found non-intersection: {count_Nan}')
            
            innerpnts = self.get_edgepoints()
            points = np.vstack([np.array(innerpnts), self.xp])
            
            dsi = np.linalg.norm(innerpnts[:, :self.dim-1], axis=1)
            p2  = innerpnts[dsi <np.max(dsi)]
            
            n_select, di   = self.findnormal(innerpnts)
            
            finalres, mp = self.tangent_extreme_points(tangent_guess=points, normal=n_select)
            bestmax_key, best_zmax, best_maxdist = max([(k, v['zcoordinate'], v['distance']) for k, v in finalres['maximize'].items()], key=lambda x: x[2])
            bestmin_key, best_zmin, best_mindist = min([(k, v['zcoordinate'], v['distance']) for k, v in finalres['minimize'].items()],  key=lambda x: x[2])
                        
            mp_select = np.array([best_zmin, best_zmax])
            d_min_max = np.array([best_mindist, best_maxdist])
            best_keys  = np.array([bestmin_key, bestmax_key])
            
        elif count_Nan >= self.dim-1:
            self.logger.info(f'[NEPALinearizer] found non-intersection: {count_Nan}')
            
            innerpnts   = self.get_edgepoints()
            pplane      = self.ponitonplane()
            points      = np.vstack([np.array(innerpnts), self.xp, pplane])
            
            dsi = np.linalg.norm(points[:], axis=1)
            
            n_select, di  = self.findnormal(points)
            di = np.min( points @ n_select)
            
            finalres, mp = self.tangent_extreme_points(tangent_guess=points, normal=n_select)
            bestmax_key, best_zmax, best_maxdist = max([(k, v['zcoordinate'], v['distance']) for k, v in finalres['maximize'].items()], key=lambda x: x[2])
            bestmin_key, best_zmin, best_mindist = min([(k, v['zcoordinate'], v['distance']) for k, v in finalres['minimize'].items()],  key=lambda x: x[2])
            
            mp_select = np.array([best_zmin, best_zmax])
            d_min_max = np.array([best_mindist, best_maxdist])
            best_keys  = np.array([bestmin_key, bestmax_key])
            
        else:
            
            print(f'\x1b[1;33m ---> else. Nothing is possible: \x1b[0m')
            print(f"--> count_Nan {count_Nan}")
            print(f"--> Wired isosurface check linearization routine ")
            print(f"--> predicted points are : \n {axispoint}")
        
        # print(out)  <- you get {"normal": normal, "boundary": boundary, "polytope points": completepoint} # print(f"linear: {linear}")
        
        results = {"normal": n_select, 
                   "boundary": {"innerdistance": d_min_max[0],"outerdistance": d_min_max[1]},
                   "polytope points": np.array(points),
                   "tangelt_points": mp,
                   "tangelt_points_selected": mp_select,
                   "final_result": finalres,
                   "best_keys": best_keys
                   }
        return results
        # return n_select, d_min_max, np.array(points), mp, mp_select, finalres, best_keys 
        


class NEPALinearizerV1:
    
    def __init__( self, reflection, structurefactor, intensity=None, atoms=None, imax=0.5):
                
        self.reflection = reflection
        self.structurefactor = np.asarray(structurefactor, dtype=float)
        self.dim = len(self.structurefactor)
        self.k = 2 * np.pi * reflection
        self.indices = np.arange(self.dim)
        self.imax = imax
        self.period  = [0, 1/(2*reflection)]
        
        # ------------------------------------------------------
        # Intensity
        # ------------------------------------------------------
        if intensity is None:
            if atoms is None:
                raise ValueError("Either intensity or atoms must be provided.")
            self.intensity = self.F(atoms)**2
        else:
            self.intensity = float(np.abs(intensity))

        arg = np.sqrt(self.intensity) / np.sum(self.structurefactor)
        arg = np.clip(arg, -1.0, 1.0)

        self.xp_scalar = np.arccos(arg) / self.k
        self.xp = np.full(self.dim, self.xp_scalar)
        
    def gradFs(self, coordinates):
        return -self.k * self.structurefactor * np.sin(self.k * coordinates)
    
    def F(self, zcoordinate):
        zcoordinate = np.asarray(zcoordinate, dtype=float)
        return np.sum(self.structurefactor * np.cos(self.k * zcoordinate))
    
    @staticmethod
    def findnormal(ps):

        ps = np.asarray(ps)

        centroid = ps.mean(axis=0)

        _, _, vh = np.linalg.svd(ps - centroid)

        normal = vh[-1]

        if normal[-1] < 0:
            normal *= -1

        ds = np.min(np.abs(ps @ normal))

        return normal, ds
    
    @staticmethod
    def angle_between(v1, v2):
        dot_pr = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(np.clip(dot_pr, -1.0, 1.0))
        
    def get_edgepoints(self): # getxyz
        
        edgepoints = []
        
        for j in range(self.dim):
            
            others = np.delete(self.indices, j)
            
            for comb in product(self.period, repeat=len(others)):

                tmpcoordinate = np.zeros(self.dim)
                
                for idx, value in zip(others, comb):
                    tmpcoordinate[idx] = value
                    
                z = hsurf_F(self.reflection, tmpcoordinate, self.structurefactor, self.intensity, j=j, s=1, s2=1)
                
                if not np.isnan(z):
                    tmpcoordinate[j] = z
                    edgepoints.append(tmpcoordinate.copy())
                    
        return np.array(np.unique(edgepoints, axis=0))
    
    def get_axispoints(self):  # get_pnts
        tot_coor = np.zeros((self.dim, self.dim))
        
        for i in range(self.dim):
            tot_coor[i, i] = hsurf_F2(self.reflection, tot_coor[i], self.structurefactor, self.intensity, j=i, s=1, s2=1)
        return np.array(tot_coor)
        
    def ponitonplane(self):
        
        points = []
        
        for j in range(0, self.dim):
            inx = np.delete(np.arange(self.dim), j)
            denominator = np.sum(self.structurefactor[inx])
                
            for zj in self.period:
                temp = np.array([zj] * self.dim)
                
                # dr = (1/self.k) * np.arccos((np.sqrt(self.intensity) - self.structurefactor[j] * np.cos(self.k * zj)) / denominator)
                
                arg = ( np.sqrt(self.intensity) - self.structurefactor[j]*np.cos(self.k*zj) ) / denominator
                arg = np.clip(arg, -1.0, 1.0)
                
                dr = np.arccos(arg)/self.k
                
                temp = np.where(np.isin(np.arange(self.dim), inx), dr, temp)
                
                points.append(temp) if not np.any(np.isnan(temp)) else None
        
        return np.array(points)
    
    def tangentpoints(self, tangent_guess, normal):
        
        # Define global lists to store values at each iteration
        z_final, grad_norm, anglelist = [], [], []
        
        z_final_all, grad_norm_all, anglelist_all, tangentpoint = [], [], [], []
        
        # Defining the error function
        def errfun(z0, normal):
            
            my_scale  = 1E10
            # z0 = z0 if z0.ndim == 1 else z0[0]
            z0 = np.asarray(z0, dtype=float)
            
            z = z0.copy()
            z[self.dim-1] = hsurf_F2( self.reflection, z0, self.structurefactor, self.intensity, j=self.dim-1, s=1, s2=1, nan=False )
            
            I_new = self.F(z)**2
            
            grad_iso_norm = self.gradFs(z)
            
            dotproduct = np.dot(grad_iso_norm, normal)
            angle = self.angle_between(grad_iso_norm, normal)
            
            # Append values to the global lists
            z_final.append(z.copy())
            grad_norm.append(grad_iso_norm.copy())
            anglelist.append(angle.copy())
            
            return ( I_new - self.intensity)**2 - my_scale * (dotproduct + 1)**2
        
        # Define searching space by bound
        # bound=(np.array(np.min(tangent_guess, axis=0)), np.array(np.max(tangent_guess, axis=0)))
        bound=(np.array(np.min(tangent_guess, axis=0)), np.array(np.max(tangent_guess, axis=0)))
        ds=(bound[1]-bound[0])/100
        
        diagonalpnt=tangent_guess[-1]
        
        # Iterate over different z_2D points
        for z2D in tangent_guess:
            
            z_final, grad_norm, anglelist = [], [], [] 
            
            z2D = np.array(z2D)
            z_direction = np.sign(np.array(diagonalpnt - z2D))
            
            ds=(bound[1]-bound[0])/100
            
            trycount = 0
            
            while trycount <=10:
                try:
                    res = least_squares(errfun, z2D, jac='2-point', bounds=bound, method='trf', ftol=1e-30, xtol=1e-10,
                                        gtol=1e-16, x_scale=1, loss='linear', f_scale=1.0,diff_step=None, tr_solver=None,
                                        tr_options={}, jac_sparsity=None, max_nfev=1000,verbose=0,
                                        args=(normal, ))
                    
                    # Append the result to the list
                    tangentpoint.append(res.x)
                    
                    break
                except (ValueError, RuntimeError) as e:
                    logging.warning(f"Optimization failed for initial guess = {z2D, z_direction} with error: {e} ")
                    
                    z_2Dp = (z2D + ds * z_direction)
                    z_2Dn = (z2D - ds * z_direction)
                    z_2Dm = z_2Dn if np.all(z_2Dn <= bound[1]) or np.all(z_2Dn >= bound[0]) else z_2Dp
                    z2D = np.where(z_direction<=0, z_2Dm, z2D)
                    
                    trycount += 1
            else:
                logging.warning(f"Optimization failed for initial guess = {z2D} from else")
                
            z_final_all.append(np.array(z_final))
            grad_norm_all.append(np.array(grad_norm))
            anglelist_all.append(np.array(anglelist))
            
        return np.array(tangentpoint), z_final_all, grad_norm_all, anglelist_all
    
    def linearize(self,): 
        
        axispoint = self.get_axispoints()
        
        count_Nan = np.count_nonzero(np.isnan(axispoint))
        nanmask = np.isnan(axispoint)
                        
        if count_Nan == 0:
            print(f'\x1b[1;33m ---> [NEPALinearizer] From if count_Nan == {count_Nan}: \x1b[0m')
            
            points = np.vstack([np.array(axispoint), self.xp])
            
            ni, di   = self.findnormal(axispoint)
            n_select = ni
            
            #distance_all, mp, pss = remaining(l, I, f, n_select, di, xp)
            mp, _, _, _ = self.tangentpoints(tangent_guess=points, normal=n_select)
                    
            dis_matrix=np.vstack([axispoint, mp])        
            
            distance_all = np.array([np.dot(n_select, i) for i in dis_matrix])
            
            d_min_max = [np.min(distance_all), np.max(distance_all)]
                        
            mp_select = [dis_matrix[np.where(distance_all == np.min(distance_all))][0],
                         dis_matrix[np.where(distance_all == np.max(distance_all))][0]
                        ]
        
        elif count_Nan == 1:
            print(f'\x1b[1;33m ---> [NEPALinearizer] From elif count_Nan == {count_Nan}: \x1b[0m')
                                
            innerpnts = self.get_edgepoints()
            points = np.vstack([np.array(innerpnts), self.xp])
            
            dsi = np.linalg.norm(innerpnts[:, :self.dim-1], axis=1)
            p2  = innerpnts[dsi <np.max(dsi)]
            
            n_select, di   = self.findnormal(p2)
            
            mp, _, _, _ = self.tangentpoints(tangent_guess=points, normal=n_select)
                    
            distance_all = points @ n_select # np.array([np.dot(n_select, i) for i in points])
            
            d_min_max = [np.min(distance_all), np.max(distance_all)]   
            
            mp_select = [points[np.where(distance_all == np.min(distance_all))][0],
                         points[np.where(distance_all == np.max(distance_all))][0]
                        ]
            
        elif count_Nan == self.dim-1:
            print(f'\x1b[1;33m ---> [NEPALinearizer] Fromcount_Nan == self.dim-1 {count_Nan}: \x1b[0m')
            
            innerpnts   = self.get_edgepoints()
            points      = np.vstack([np.array(innerpnts), self.xp])
            pplane      = self.ponitonplane()
            
            dsi = np.linalg.norm(points[:], axis=1)
            
            n_select, di  = self.findnormal(points)
            di = np.min( points @ n_select)
            
            mp, _, _, _ = self.tangentpoints(points, normal=n_select)
            
            distance_all = points @ n_select
            
            d_min_max = [np.min(distance_all), np.max(distance_all)]   
            
            mp_select = [points[np.where(distance_all == np.min(distance_all))][0], 
                         points[np.where(distance_all == np.max(distance_all))][0]
                        ]
        else:
            
            print(f'\x1b[1;33m ---> else. Nothing is possible: \x1b[0m')
            print(f"--> count_Nan {count_Nan}")
            print(f"--> Wired isosurface check linearization routine ")
            print(f"--> predicted points are : \n {axispoint}")
        
        return n_select, d_min_max, np.array(points), mp, mp_select

class NEPALinearizerV2:
    
    def __init__( self, reflection, structurefactor, intensity=None, atoms=None, imax=0.5):
                
        self.reflection = reflection
        self.structurefactor = np.asarray(structurefactor, dtype=float)
        self.dim = len(self.structurefactor)
        self.k = 2 * np.pi * reflection
        self.indices = np.arange(self.dim)
        self.imax = imax
        self.period  = [0, 1/(2*reflection)]
        
        # ------------------------------------------------------
        # Intensity
        # ------------------------------------------------------
        if intensity is None:
            if atoms is None:
                raise ValueError("Either intensity or atoms must be provided.")
            self.intensity = self.F(atoms)**2
        else:
            self.intensity = float(np.abs(intensity))

        arg = np.sqrt(self.intensity) / np.sum(self.structurefactor)
        arg = np.clip(arg, -1.0, 1.0)

        self.xp_scalar = np.arccos(arg) / self.k
        self.xp = np.full(self.dim, self.xp_scalar)
        
    def gradFs(self, coordinates):
        return -self.k * self.structurefactor * np.sin(self.k * coordinates)
    
    def F(self, zcoordinate):
        zcoordinate = np.asarray(zcoordinate, dtype=float)
        return np.sum(self.structurefactor * np.cos(self.k * zcoordinate))
    
    @staticmethod
    def findnormal(ps):

        ps = np.asarray(ps)

        centroid = ps.mean(axis=0)

        _, _, vh = np.linalg.svd(ps - centroid)

        normal = vh[-1]

        if normal[-1] < 0:
            normal *= -1

        ds = np.min(np.abs(ps @ normal))

        return normal, ds
    
    @staticmethod
    def angle_between(v1, v2):
        dot_pr = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(np.clip(dot_pr, -1.0, 1.0))
        
    def get_edgepoints(self): # getxyz
        
        edgepoints = []
        
        for j in range(self.dim):
            
            others = np.delete(self.indices, j)
            
            for comb in product(self.period, repeat=len(others)):

                tmpcoordinate = np.zeros(self.dim)
                
                for idx, value in zip(others, comb):
                    tmpcoordinate[idx] = value
                    
                z = hsurf_F(self.reflection, tmpcoordinate, self.structurefactor, self.intensity, j=j, s=1, s2=1)
                
                if not np.isnan(z):
                    tmpcoordinate[j] = z
                    edgepoints.append(tmpcoordinate.copy())
                    
        return np.array(np.unique(edgepoints, axis=0))
    
    def get_axispoints(self):  # get_pnts
        tot_coor = np.zeros((self.dim, self.dim))
        
        for i in range(self.dim):
            tot_coor[i, i] = hsurf_F2(self.reflection, tot_coor[i], self.structurefactor, self.intensity, j=i, s=1, s2=1)
        return np.array(tot_coor)
        
    def ponitonplane(self):
        
        points = []
        
        for j in range(0, self.dim):
            inx = np.delete(np.arange(self.dim), j)
            denominator = np.sum(self.structurefactor[inx])
                
            for zj in self.period:
                temp = np.array([zj] * self.dim)
                
                # dr = (1/self.k) * np.arccos((np.sqrt(self.intensity) - self.structurefactor[j] * np.cos(self.k * zj)) / denominator)
                
                arg = ( np.sqrt(self.intensity) - self.structurefactor[j]*np.cos(self.k*zj) ) / denominator
                arg = np.clip(arg, -1.0, 1.0)
                
                dr = np.arccos(arg)/self.k
                
                temp = np.where(np.isin(np.arange(self.dim), inx), dr, temp)
                
                points.append(temp) if not np.any(np.isnan(temp)) else None
        
        return np.array(points)

    def tangentpoints(self, tangent_guess, normal):

        tangentpoints, z_final_all, grad_norm_all, anglelist_all = [], [], [], []

        lower = np.zeros(self.dim)
        upper = np.full(self.dim, self.period[1])

        bounds = list(zip(lower, upper))

        def objective(z0, normal, z_hist, grad_hist, angle_hist):

            z = np.asarray(z0, dtype=float).copy()

            try:

                z[self.dim-1] = hsurf_F2( self.reflection, z, self.structurefactor, self.intensity, j=self.dim-1, s=1, s2=1, nan=False)

                if np.isnan(z[self.dim-1]):
                    return 1e10

                grad = self.gradFs(z)

                grad_norm   = np.linalg.norm(grad)
                normal_norm = np.linalg.norm(normal)

                if grad_norm < 1e-15:
                    return 1e10

                cosang = np.dot(grad, normal) / (grad_norm * normal_norm)
                # cosang = np.clip(cosang, -1.0, 1.0)

                angle = np.degrees(np.arccos(cosang))

                z_hist.append(z.copy())
                grad_hist.append(grad.copy())
                angle_hist.append(angle)

                return 1.0 - cosang*cosang

            except Exception:
                return 1e10

        for guess in tangent_guess:

            best_res = None

            z_hist, grad_hist, angle_hist = [], [], []

            starts = [np.asarray(guess, dtype=float)]

            n_random = max(10, 3*self.dim)

            random_starts = np.random.uniform( lower, upper, size=(n_random, self.dim))

            starts.extend(random_starts)

            for start in starts:

                try:

                    res = minimize( objective, start, args=(normal, z_hist, grad_hist, angle_hist), method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000} )

                    if not res.success:
                        continue

                    if best_res is None or res.fun < best_res.fun:
                        best_res = res

                except Exception as e:

                    logging.warning(f"Optimization failed for start={start} error={e}")

            if best_res is None:

                logging.warning(f"No tangent point found for guess={guess}")

                continue

            z_best = best_res.x.copy()

            z_best[self.dim-1] = hsurf_F2( self.reflection, z_best, self.structurefactor, self.intensity, j=self.dim-1, s=1, s2=1, nan=False)

            tangentpoints.append(z_best)

            z_final_all.append(np.asarray(z_hist))
            grad_norm_all.append(np.asarray(grad_hist))
            anglelist_all.append(np.asarray(angle_hist))

            logging.info(f"Tangent point found. objective={best_res.fun}")

        return np.asarray(tangentpoints), z_final_all, grad_norm_all, anglelist_all
    
    def linearize(self,): 
        
        axispoint = self.get_axispoints()
        
        count_Nan = np.count_nonzero(np.isnan(axispoint))
        nanmask = np.isnan(axispoint)
                        
        if count_Nan == 0:
            print(f'\x1b[1;33m ---> [NEPALinearizer] From if count_Nan == {count_Nan}: \x1b[0m')
            
            points = np.vstack([np.array(axispoint), self.xp])
            
            ni, di   = self.findnormal(axispoint)
            n_select = ni
            
            #distance_all, mp, pss = remaining(l, I, f, n_select, di, xp)
            mp, _, _, _ = self.tangentpoints(tangent_guess=points, normal=n_select)
                    
            dis_matrix=np.vstack([axispoint, mp])        
            
            distance_all = np.array([np.dot(n_select, i) for i in dis_matrix])
            
            d_min_max = [np.min(distance_all), np.max(distance_all)]
                        
            mp_select = [dis_matrix[np.where(distance_all == np.min(distance_all))][0],
                         dis_matrix[np.where(distance_all == np.max(distance_all))][0]
                        ]
        
        elif count_Nan == 1:
            print(f'\x1b[1;33m ---> [NEPALinearizer] From elif count_Nan == {count_Nan}: \x1b[0m')
                                
            innerpnts = self.get_edgepoints()
            points = np.vstack([np.array(innerpnts), self.xp])
            
            dsi = np.linalg.norm(innerpnts[:, :self.dim-1], axis=1)
            p2  = innerpnts[dsi <np.max(dsi)]
            
            n_select, di   = self.findnormal(p2)
            
            mp, _, _, _ = self.tangentpoints(tangent_guess=points, normal=n_select)
                    
            distance_all = points @ n_select # np.array([np.dot(n_select, i) for i in points])
            
            d_min_max = [np.min(distance_all), np.max(distance_all)]   
            
            mp_select = [points[np.where(distance_all == np.min(distance_all))][0],
                         points[np.where(distance_all == np.max(distance_all))][0]
                        ]
            
        elif count_Nan == self.dim-1:
            print(f'\x1b[1;33m ---> [NEPALinearizer] Fromcount_Nan == self.dim-1 {count_Nan}: \x1b[0m')
            
            innerpnts   = self.get_edgepoints()
            points      = np.vstack([np.array(innerpnts), self.xp])
            pplane      = self.ponitonplane()
            
            dsi = np.linalg.norm(points[:], axis=1)
            
            n_select, di  = self.findnormal(points)
            di = np.min( points @ n_select)
            
            mp, _, _, _ = self.tangentpoints(points, normal=n_select)
            
            distance_all = points @ n_select
            
            d_min_max = [np.min(distance_all), np.max(distance_all)]   
            
            mp_select = [points[np.where(distance_all == np.min(distance_all))][0], 
                         points[np.where(distance_all == np.max(distance_all))][0]
                        ]
        else:
            
            print(f'\x1b[1;33m ---> else. Nothing is possible: \x1b[0m')
            print(f"--> count_Nan {count_Nan}")
            print(f"--> Wired isosurface check linearization routine ")
            print(f"--> predicted points are : \n {axispoint}")
        
        return n_select, d_min_max, np.array(points), mp, mp_select

class NEPALinearizerV3:
    
    def __init__( self, reflection, structurefactor, intensity=None, atoms=None, imax=0.5):
                
        self.reflection = reflection
        self.structurefactor = np.asarray(structurefactor, dtype=float)
        self.dim = len(self.structurefactor)
        self.k = 2 * np.pi * reflection
        self.indices = np.arange(self.dim)
        self.imax = imax
        self.period  = [0, 1/(2*reflection)]
        
        # ------------------------------------------------------
        # Intensity
        # ------------------------------------------------------
        if intensity is None:
            if atoms is None:
                raise ValueError("Either intensity or atoms must be provided.")
            self.intensity = self.F(atoms)**2
        else:
            self.intensity = float(np.abs(intensity))

        arg = np.sqrt(self.intensity) / np.sum(self.structurefactor)
        arg = np.clip(arg, -1.0, 1.0)

        self.xp_scalar = np.arccos(arg) / self.k
        self.xp = np.full(self.dim, self.xp_scalar)
        
    def gradFs(self, coordinates):
        return -self.k * self.structurefactor * np.sin(self.k * coordinates)
    
    def F(self, zcoordinate):
        zcoordinate = np.asarray(zcoordinate, dtype=float)
        return np.sum(self.structurefactor * np.cos(self.k * zcoordinate))
    
    @staticmethod
    def findnormal(ps):

        ps = np.asarray(ps)

        centroid = ps.mean(axis=0)

        _, _, vh = np.linalg.svd(ps - centroid)

        normal = vh[-1]

        if normal[-1] < 0:
            normal *= -1

        ds = np.min(np.abs(ps @ normal))

        return normal, ds
    
    @staticmethod
    def angle_between(v1, v2):
        dot_pr = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(np.clip(dot_pr, -1.0, 1.0))
        
    def get_edgepoints(self): # getxyz
        
        edgepoints = []
        
        for j in range(self.dim):
            
            others = np.delete(self.indices, j)
            
            for comb in product(self.period, repeat=len(others)):

                tmpcoordinate = np.zeros(self.dim)
                
                for idx, value in zip(others, comb):
                    tmpcoordinate[idx] = value
                    
                z = hsurf_F(self.reflection, tmpcoordinate, self.structurefactor, self.intensity, j=j, s=1, s2=1)
                
                if not np.isnan(z):
                    tmpcoordinate[j] = z
                    edgepoints.append(tmpcoordinate.copy())
                    
        return np.array(np.unique(edgepoints, axis=0))
    
    def get_axispoints(self):  # get_pnts
        tot_coor = np.zeros((self.dim, self.dim))
        
        for i in range(self.dim):
            tot_coor[i, i] = hsurf_F2(self.reflection, tot_coor[i], self.structurefactor, self.intensity, j=i, s=1, s2=1)
        return np.array(tot_coor)
        
    def ponitonplane(self):
        
        points = []
        
        for j in range(0, self.dim):
            inx = np.delete(np.arange(self.dim), j)
            denominator = np.sum(self.structurefactor[inx])
                
            for zj in self.period:
                temp = np.array([zj] * self.dim)
                
                # dr = (1/self.k) * np.arccos((np.sqrt(self.intensity) - self.structurefactor[j] * np.cos(self.k * zj)) / denominator)
                
                arg = ( np.sqrt(self.intensity) - self.structurefactor[j]*np.cos(self.k*zj) ) / denominator
                arg = np.clip(arg, -1.0, 1.0)
                
                dr = np.arccos(arg)/self.k
                
                temp = np.where(np.isin(np.arange(self.dim), inx), dr, temp)
                
                points.append(temp) if not np.any(np.isnan(temp)) else None
        
        return np.array(points)
    
    def tangentpoints(self, tangent_guess, normal):

        tangentpoints, z_final_all, grad_norm_all, anglelist_all = [], [], [], []

        normal = np.asarray(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)

        lower = np.zeros(self.dim)
        upper = np.full(self.dim, self.period[1])
        bounds = (lower, upper)

        def residual(z0):

            z = np.asarray(z0, dtype=float).copy()

            # enforce isosurface constraint on last coordinate
            z[self.dim - 1] = hsurf_F2( self.reflection, z, self.structurefactor, self.intensity, j=self.dim - 1, s=1, s2=1, nan=False )

            grad = self.gradFs(z)

            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-14:
                return np.ones(self.dim) * 1e6

            grad_dir = grad / grad_norm

            # direction mismatch (main condition)
            r = grad_dir - normal

            return r

        for guess in tangent_guess:

            guess = np.asarray(guess, dtype=float)

            z_hist = []
            grad_hist = []
            angle_hist = []

            best_res = None

            starts = [guess]

            # multi-start (important for nonlinear surfaces)
            starts += list(np.random.uniform(lower, upper, size=(10 * self.dim, self.dim)))

            for start in starts:

                try:
                    res = least_squares(residual,  start,  bounds=bounds,  method="trf",  ftol=1e-12,  xtol=1e-12,  gtol=1e-12,  max_nfev=200)

                    if not res.success:
                        continue

                    if best_res is None or np.linalg.norm(res.fun) < np.linalg.norm(best_res.fun):
                        best_res = res

                except Exception as e:
                    logging.warning(f"Failed start={start} error={e}")

            if best_res is None:
                logging.warning(f"No tangent solution found for guess={guess}")
                continue

            z_best = best_res.x.copy()

            z_best[self.dim - 1] = hsurf_F2(self.reflection, z_best, self.structurefactor, self.intensity, j=self.dim - 1, s=1, s2=1, nan=False )

            tangentpoints.append(z_best)

            # diagnostics
            grad = self.gradFs(z_best)
            grad_norm_all.append(grad)

            cosang = np.dot(grad, normal) / (np.linalg.norm(grad) * np.linalg.norm(normal))
            cosang = np.clip(cosang, -1, 1)

            anglelist_all.append(np.degrees(np.arccos(cosang)))

            z_final_all.append(z_hist)

        return np.asarray(tangentpoints), z_final_all, grad_norm_all, anglelist_all
    
    
    
    def linearize(self,): 
        
        axispoint = self.get_axispoints()
        
        count_Nan = np.count_nonzero(np.isnan(axispoint))
        nanmask = np.isnan(axispoint)
                        
        if count_Nan == 0:
            print(f'\x1b[1;33m ---> [NEPALinearizer] From if count_Nan == {count_Nan}: \x1b[0m')
            
            points = np.vstack([np.array(axispoint), self.xp])
            
            ni, di   = self.findnormal(axispoint)
            n_select = ni
            
            #distance_all, mp, pss = remaining(l, I, f, n_select, di, xp)
            mp, _, _, _ = self.tangentpoints(tangent_guess=points, normal=n_select)
                    
            dis_matrix=np.vstack([axispoint, mp])        
            
            distance_all = np.array([np.dot(n_select, i) for i in dis_matrix])
            
            d_min_max = [np.min(distance_all), np.max(distance_all)]
                        
            mp_select = [dis_matrix[np.where(distance_all == np.min(distance_all))][0],
                         dis_matrix[np.where(distance_all == np.max(distance_all))][0]
                        ]
        
        elif count_Nan == 1:
            print(f'\x1b[1;33m ---> [NEPALinearizer] From elif count_Nan == {count_Nan}: \x1b[0m')
                                
            innerpnts = self.get_edgepoints()
            points = np.vstack([np.array(innerpnts), self.xp])
            
            dsi = np.linalg.norm(innerpnts[:, :self.dim-1], axis=1)
            p2  = innerpnts[dsi <np.max(dsi)]
            
            n_select, di   = self.findnormal(p2)
            
            mp, _, _, _ = self.tangentpoints(tangent_guess=points, normal=n_select)
                    
            distance_all = points @ n_select # np.array([np.dot(n_select, i) for i in points])
            
            d_min_max = [np.min(distance_all), np.max(distance_all)]   
            
            mp_select = [points[np.where(distance_all == np.min(distance_all))][0],
                         points[np.where(distance_all == np.max(distance_all))][0]
                        ]
            
        elif count_Nan == self.dim-1:
            print(f'\x1b[1;33m ---> [NEPALinearizer] Fromcount_Nan == self.dim-1 {count_Nan}: \x1b[0m')
            
            innerpnts   = self.get_edgepoints()
            points      = np.vstack([np.array(innerpnts), self.xp])
            pplane      = self.ponitonplane()
            
            dsi = np.linalg.norm(points[:], axis=1)
            
            n_select, di  = self.findnormal(points)
            di = np.min( points @ n_select)
            
            mp, _, _, _ = self.tangentpoints(points, normal=n_select)
            
            distance_all = points @ n_select
            
            d_min_max = [np.min(distance_all), np.max(distance_all)]   
            
            mp_select = [points[np.where(distance_all == np.min(distance_all))][0], 
                         points[np.where(distance_all == np.max(distance_all))][0]
                        ]
        else:
            
            print(f'\x1b[1;33m ---> else. Nothing is possible: \x1b[0m')
            print(f"--> count_Nan {count_Nan}")
            print(f"--> Wired isosurface check linearization routine ")
            print(f"--> predicted points are : \n {axispoint}")
        
        return n_select, d_min_max, np.array(points), mp, mp_select




# -------------------------------------------------------
# Status on 27.01.2026
# Following will be removed later
# -------------------------------------------------------


# -------------------------------------------------------
# Modules for EPA
# -------------------------------------------------------

def get_pnts_new(l, f, gi, imax=0.5):
    tot_coor=[]
    for i in range(len(f)):
        tem_coor    = np.zeros(len(f)) #tem_coor[i] = x[i]
        tem_coor[i] = hsurf_g(l, tem_coor, f, gi, i, s=1)
        
        # if (~np.isnan(tem_coor[i])):
        #     tem_coor[i] = tem_coor[i]
        # else:
        #     tem_coor[i] = imax/l
        #     #tem_coor[i] = hsurf_g(l, tem_coor, f, gi, i, s=1)
        tot_coor.append(tem_coor)
    
    return tot_coor

def linearizenD_EPA_old_deltelater (l:int, f: list, gi: int) ->list:
    
    k = 2*np.pi*l
    #======= 1. finding first three points
    pnt = get_pnts_new(l, f, gi)
    
    #======= 2. finding fourth point : the point on the surface :: At particular point x=y=z or x1*=x2*=x3*. so
    xp  = (1/k)*np.arccos(gi/np.sum(f))
    p4  =[xp]*len(f)
    
    if np.all(~np.isnan(pnt)): #~np.isnan(a):
        centroid = np.mean(pnt, axis=0)
        u, s, v  = np.linalg.svd(pnt-centroid)
        #n = v[-1] if np.all((pnt-centroid)[0]>=0) else -1*v[-1]
        n = np.abs(v[-1])
        
        d_int = np.double(np.sum(np.multiply(n,centroid)))
        d_out = np.double(np.sum(np.multiply(n,p4)))
        
        aps = [d_out/i for i in n]
                        
        for iv in range(len(f)):
            vv = np.zeros(len(f))
            vv [iv] = aps[iv]
            pnt = np.vstack([pnt, vv])
        
        normal = n
        d_all  = [d_int,d_out]
    else:
        
        pnt=[]
        
        pin = get_pnts_new4(l, f, gi, xinit=0)
        pnt = np.vstack([pin])
               
        centroidi   = np.mean(pin, axis=0)
        ui, si, vi  = np.linalg.svd(pin-centroidi)
        
        d_int    = np.abs(np.sum(np.multiply(-1*vi[-1],centroidi)))
                
        pon = get_pnts_new4(l, f, gi, xinit=0.5)
        pnt = np.vstack([pnt, pon])
       
        centroido   = np.mean(pon, axis=0)
        uo, so, vo  = np.linalg.svd(pon-centroido)
        
        d_out1      = np.abs(np.sum(np.multiply(vo[-1],centroido)))
        d_out2      = np.abs(np.dot([xp]*len(f),vo[-1]))
        
        d_all  = [d_int, d_out1] if d_out1 > d_out2 else  [d_int, d_out2]
        
        normal = np.abs(vo[-1])
        
    return normal, d_all


# -------------------------------------------------------
#  Modules for EPA 
# -------------------------------------------------------

def get_pnts_new4(l, f, gi, xinit=0, imax=0.5):
    
    k = 2*np.pi*l  ; tot_coor = []
    
    # point on axis and face diagonal
    for i in range(len(f)):        
        tem_coor    = np.zeros(len(f))
        tem_coor[i] = hsurf_g(l, tem_coor, f, gi, i, s=1)
        
        tot_coor.append(tem_coor)        
    
    if np.all(~np.isnan(np.array(tot_coor))):
        pass
    else:
        
        inx = np.argwhere(np.isnan(tot_coor).any(axis=1)).flatten()
        
        for iw in inx:
            tem_coor    = np.zeros(len(f))
            tem_coor[iw] = xinit/l
            
            mlist = np.delete(np.arange(len(f)), iw )
            
            temp_coor = (1/k)*np.arccos( (gi-f[iw]*np.cos(k*tem_coor[iw])) / 
                                         (np.sum([f[ii]*np.cos(k*tem_coor[ii]) for ii in mlist]))  )
            
            tem_coor[iw+1:] = temp_coor
            tem_coor[:iw]   = temp_coor
            
            tot_coor[iw]=tem_coor
    
    # point along body diagonal
    xp=(1/k)*np.arccos(gi/np.sum(f)) #; print(f'xp : {xp}')
    tot_coor = np.vstack([np.array(tot_coor), [xp]*len(f)])
    
    return np.array(tot_coor)

def pntonplane(l, f, gi):
    pr = 1 / (2 * l)
    k = 2 * np.pi * l
    pp = []
    
    for jinx in range(0, len(f)):
        inx = np.delete(np.arange(len(f)), jinx)  #; print("inx ", inx)
        denominator = np.sum([f[ji] for ji in inx])
            
        for zj in [0,]:
            temp = np.array([zj] * len(f))
            dr = (1/k) * np.arccos((gi - f[jinx] * np.cos(k * zj)) / denominator)
            temp = np.where(np.isin(np.arange(len(temp)), inx), dr, temp)
            
            pp.append(temp) if not np.any(np.isnan(temp)) else None
    
    return np.array(pp)

def findnormal(ps: list) -> list:
    centroid = np.mean(ps, axis=0)
    u, s, v  = np.linalg.svd(ps-centroid)
    #nor = np.abs(v[-1]) if np.all((ps-centroid)[0] >0)  else -1*v[-1] # if ((ps-centroid)[0][0]) >0 else -1*v[-1]
    if np.all(v[-1]>=0):
        nor=v[-1]
    else:
        nor = np.abs(v[-1]) if np.all((ps-centroid)[0] >0)  else -1*v[-1] # if ((ps-centroid)[0][0]) >0 else -1*v[-1]
    #nor = v[-1] if np.dot((ps[0] - centroid), v[-1]) > 0 else -1 * v[-1] 

    ds = np.min([np.abs(np.dot(i, nor)) for i in ps])  #  we can use np.sum(ps * nor, axis=1)
    
    #print(f"u : \n{u} \n v\n{v}\ns{s}")
    #print(f"\n ps-centroid :\n {ps-centroid} and nor is {nor}")
    #print(f"ds's {[np.abs(np.dot(i, nor)) for i in ps]}")
    
    return nor

def pointonnDface(l, f, gi):
    import itertools
    
    kp=[]
    f=np.array(f)
    
    for i in range(0, len(f)-1):
        k  = 2*np.pi*l
        z  = np.zeros(len(f))
        
        up   = np.delete(np.arange(len(f)), np.arange(i,len(f)))
        down = np.arange(i,len(f))
        
        gg = (1/k)*np.arccos( ( gi-np.sum([f[j]*np.cos(k*z[j]) for j in up ])) / np.sum(f[down]) )
        
        gg = gg if not np.isnan(gg) else 1/(2*l)
        z[down]=gg
        zpermutation = list(itertools.permutations(z))
        
        z_array = np.array(list(set(zpermutation)))
        kp = z_array if len(kp) == 0 else np.vstack([kp, z_array])
        
    return kp

def linearizenD_EPA(l:int, f: list, gi: int) ->list:
    
    k = 2*np.pi*l
    #======= 1. finding first three points
    pnt = get_pnts_new(l, f, gi)
    
    #======= 2. finding fourth point : the point on the surface :: At particular point x=y=z or x1*=x2*=x3*. so
    xp  = [ (1/k)*np.arccos(gi/np.sum(f)) ]*len(f)
    
    print(f'pnt \n {np.array(pnt)}')
    
    if np.all(~np.isnan(pnt)):
                
        centroid = np.mean(pnt, axis=0)
        u, s, v  = np.linalg.svd(pnt-centroid)
        normal = np.abs(v[-1])              #n = v[-1] if np.all((pnt-centroid)[0]>=0) else -1*v[-1]
        
        #print(f'centroid {centroid}')
        
        d_int = np.dot(normal, centroid)    #np.double(np.sum(np.multiply(normal,centroid)))
        d_out = np.dot(normal, xp)          #np.double(np.sum(np.multiply(normal,p4)))
        
        aps = [d_out/i for i in normal]
                        
        for iv in range(len(f)):
            vv = np.zeros(len(f))
            vv [iv] = aps[iv]
            #print(f'vv : {vv}')
            pnt = np.vstack([pnt, vv])
        
        d_all  = [d_int,d_out]
        pntx   = pnt
    else:
        #print(f'\n-------------------- isotype {2}')
                
        pnt=[]
        pFP = pointonnDface(l, f, gi)
        
        pin = get_pnts_new4(l, f, gi, xinit=0)
        pnt = np.vstack([pin])
            
        pon = get_pnts_new4(l, f, gi, xinit=0.5)
        pnt = np.vstack([pnt, pon])
       
        centroidi   = np.mean(pin, axis=0)
        ui, si, vi  = np.linalg.svd(pin-centroidi)
        #d_int = np.abs(np.dot(vi[-1],centroidi))    #np.abs(np.sum(np.multiply(-1*vi[-1],centroidi)))
                
        centroido   = np.mean(pon, axis=0)
        uo, so, vo  = np.linalg.svd(pon-centroido)

        normal = np.abs(vo[-1])
        
        #d_out1      = np.dot(normal, centroido)       #np.abs(np.sum(np.multiply(vo[-1],centroido)))
        #d_out2      = np.abs(np.dot(normal, xp))
        
        pntx = np.vstack([pnt, pFP])
        #print(f'pFP {pFP}')
        #pFPnormal=findnormal(pFP)
        #print(f"from FP: {pFPnormal, findnormal(pntx)} dist: {np.array([np.dot(ii, pFPnormal) for ii in pFP])}")
                
        dall = np.dot(pntx, normal)  #[d_int, d_out1] if d_out1 > d_out2 else  [d_int, d_out2]
        d_all = [np.min(dall), np.max(dall) ]
        
        checkstatus, checked_d = checklinear(l, f, gi, normal, d_all, j=len(f)-1)
        #print(f'dis old  {d_all} new {checked_d}')
        if checkstatus == True:
            pass
        else:
            d_all = checked_d
            #print(f'----------- dis new {checked_d}')
            checkstatus, checked_d = checklinear(l, f, gi, normal, d_all, j=len(f)-1)
            print(f'------------ dis new2 {checked_d}')
        
        d_all = [(np.floor(d_all[0]*10**6) - 1)/10**6,  (np.floor(d_all[1]*10**6) + 1)/10**6]
    #print(f'from linearizenD_EPA :: pnt \n {np.array(pnt)} \n\ndist: {np.dot(pnt, normal)}')
        
    return normal, d_all, pntx


# -------------------------------------------------------
# Modules for non EPA 
# -------------------------------------------------------

def linearizenD_nEPA(l, f, gi):
    
    from scipy.optimize import minimize, minimize_scalar, root, basinhopping
    
    def extremetangent(h, gi, f, normal, percentage=1):
        
        def func_vec(x0, h, gi, f, normal):
            k  = 2*np.pi*h
            gg = np.sum([ f[i]* np.sqrt(1- ((x0[i]*normal[i])/(k*f[i]))**2 ) for i in range(len(f))])
            return [gi-gg]*len(x0)
        
        def funcjac_vec(x0, h, f, n):
            k     = 2*np.pi*h
            gradg = [ f[i]*x0[i]*((n[i]/(k*f[i]))**2)*(1-((x0[i]*n[i])/(k*f[i]))**2)**(-1/2) for i in range(len(f)) ]
            return gradg
        
        k  = 2*np.pi*h  ; k1 = 1/k
        
        lam_max  = [ k*(f[i]/normal[i]) if (normal[i] >1E-5) else k*f[i]*normal[i] for i in range(len(f))]
        lam_mask = np.ma.masked_equal(lam_max, 0.0, copy=False)
        
        x0 = [(ik - ik%percentage)-1  if ik !=0 else normal[ci]*ik for ci, ik in enumerate(lam_max) ]
        
        res    = root(func_vec, x0, args=(h, gi, f, normal), options={'maxiter':100,'gtol': 1e-16, 'disp': True}).x
        
        midpnt = [ k1*np.arcsin((res[i]*normal[i])/(k*f[i])) for i in range(len(f))]
        
        return [res, midpnt]
    
    def remaining(l, gi, f, n_select, di):
        
        mp = np.abs(extremetangent(l, gi, f, n_select, percentage=5.)[1])
        do = np.abs(np.dot(n_select, mp))
        
        distance = [di, do]
        
        dok=checkisoa(l, gi, f, n_select, distance, n=100, s=1)
        
        if not dok[0]: # dok[1] = [di_n, do_n]
            do = dok[1][1] if ~dok[0] and do <dok[1][1] else do
            di = dok[1][0] if ~dok[0] and di >dok[1][0] else di
        
        distance  = [di, do]
        
        return distance
    
    def checkisoa(l, gi, f, normal, distance, n=100, s=1):
        j = len(f) - 1
        lspace = np.linspace(0, 1 / (2 * l), n)
        kj = [lspace] * (len(f) - 1)
        kz = np.meshgrid(*kj)
        
        gzp = hsurf_F2(l, [*kz], f, gi, j, s=1, s2=1)
        #print(f"gzp shape: {np.shape(gzp)}")
        
        o = getpoly_mitd(l, normal, distance, scom=np.array([[1] * len(f)]), dlist=np.array([[0] * len(f)]), imax=lspace.max())
        
        kz.append(np.array(gzp))
        
        t = [kzi.flatten() for kzi in kz]
        t = list(zip(*t))
        t2 = [list(ti) for ti in t]
        t2 = np.array(t2)
        tz = t2[~np.isnan(t2).any(axis=1)]
        
        #print(f"t2: {np.shape(t2)} tz: {np.shape(tz)}")

        check = [ti in o for ti in tz]

        if np.any(np.logical_not(np.array(check))):
            index = np.where(~np.array(check))[0]
            dr = [np.dot(normal, tz[inx]) for inx in index]
            return False, [np.min(dr), np.max(dr)]
        else:
            return [True]

    def checkiso(l, gi, f, normal, distance, n=100, s=1):
        
        j = len(f)-1
        lspace  = np.linspace(0, 1/(2*l), n)
        kj = [lspace]*(len(f)-1)
        kz = np.meshgrid(*kj)
        gz = np.zeros_like(kz[0])
        
        gzp = hsurf_F2(l, [*kz], f, gi, j, s=1, s2=1)
        o = getpoly_mitd(l, normal, distance, scom=np.array([[1]*len(f)]), dlist=np.array([[0]*len(f)]), imax=lspace.max())
        
        kz.extend([np.array(gzp)])
        tz = np.vstack(np.dstack([*kz]))
        x=tz[~np.isnan(tz).any(axis=1)]
        
        check=[i in o for i in x]
        #check=[ ti in o    for ti in tz   if ~np.all(np.isnan(ti)) ]
        
        if ~np.all(check):
                index = np.where(~np.array(check))[0]
                dr  = [np.dot(normal,x[inx]) for inx in index]
                return False, [np.min(dr), np.max(dr)]
        else:
            return [True]        
        return  
    
    def plane_eq2n(pnt):
        
        centroid = np.mean(pnt, axis=0)
        u, s, v  = np.linalg.svd(pnt-centroid)
        
        d_int = np.double(np.sum(np.multiply(np.abs(v[-1]),centroid)))
                
        return np.abs(v[-1]), d_int #np.abs(np.array([a, b, c])), d

    def getxyz_opt3(l, f, I, minmax=0):
        
        xyz = []
        
        for jj in [0, 0.5/l]:
            for i in range(len(f)-1):
                t = [0] * len(f)
                t[len(f)-1] = jj
                
                z = hsurf_F2(l, t, f, I, j=i, s=1, s2=1) # hsurf_g(l, t, f, I, j=i, s=1)
                
                if not np.isnan(z):
                    t[i] = z
                else:
                    t[i] = 0.5/l
                    z = hsurf_F2(l, t, f,  I, j=i, s=1, s2=1) # hsurf_g(l, t, f, I, j=i-1, s=1)
                    if not np.isnan(z):
                        t[i-1] = z
                    else:
                        t[i-1] = 0.5/l
                xyz.append(t)
        
        a = np.array(xyz)
        
        if minmax==0:
            #a[:,0:1] = np.min(a[:,0:1])
            a[:,0] = np.min(a[:,0], axis=0)
        else:
            #a[:,0:1] = np.max(a[:,0:1])
            a[:,0] = np.unique(a)[-2] if np.unique(a)[-1] == 0.5/l else  np.unique(a)[-1]
            
        return a, np.array(xyz)
    
    def getxyz(l, f, gi):
        xyz = []
        
        for i in range(len(f)-1):
            for jj in [0, 1/(2*l)]:
                t = [0]*len(f)
                t[len(f)-1] = jj
                z = hsurf_F(l, t, f, gi, j=i, s=1)
                #z = hsurf_F2a(gi, l, t, f, j=i, s=1, s2=1)
                
                if ~np.isnan(z):
                    t[i] = z
                else:
                    t[i] = 0.5/l
                    #z = hsurf_g(l, t, f, gi, j=i, s=1)
                    z = hsurf_F2(l, t, f, gi, j=i, s=1, s2=1)
                    
                    if not np.isnan(z):
                        t[i-1] = z
                    else:
                        t[i-1] = 0.5/l
                                
                xyz.append(t)
        
        a = np.copy(xyz)
        a = a[a[:, 1].argsort()][::-1]
        
        #print("\n a - before :: \n", a)
        a[0][:2] = a[1][:2] if np.all(a[1][:2] <= a[0][:2]) else a[0][:2]
        a[1][:2] = a[0][:2] if np.all(a[1][:2] >= a[0][:2]) else a[1][:2]
        
        a[2][:2] = a[3][:2] if np.all(a[3][:2] <= a[2][:2]) else a[2][:2]
        a[3][:2] = a[2][:2] if np.all(a[3][:2] >= a[2][:2]) else a[3][:2]
        
        #print("\n a - after :: \n", a)
        return np.array(a), np.array(xyz)    
    
    def get_pnt_nEPA(l, f, I):
        tot_coor=[]
        for i in range(len(f)):
            tem_coor    = np.zeros(len(f))
            tem_coor[i] = hsurf_F2(l, tem_coor, f, I, j=i, s=1, s2=1, nan=False) #hsurf_g(l, tem_coor, f, gi, i, s=1)
            tot_coor.append(tem_coor)
            
        return tot_coor
    
    def gt3(l, f, I):
        xyz = []
        
        t = [0] * len(f)
        z = hsurf_F2( l, t, f,  I, j=0, s=1, s2=1, nan=False)
        
        if not np.isnan(z):
            t[0] = z
        else:
            t[0] = 0.5/l
        ###
        xyz.append(t)
        
        for i in range(1,len(f)):
            t = [0] * len(f)
            t[i] = 0.5/l
            z = hsurf_F2( l, t, f,  I, j=0, s=1, s2=1, nan=False)
            
            if not np.isnan(z):
                t[0] = z
            else:
                t[0] = 0.5/l
            
            xyz.append(t)
        
        t = [0.5/l] * len(f)
        z = hsurf_F2( l, t, f,  I, j=0, s=1, s2=1, nan=False)
        
        if not np.isnan(z):
            t[0] = z
        else:
            t[0] = 0.5/l
        ###
        xyz.append(t)
        
        a = np.array(xyz) ; a[:,0] = np.unique(a)[-1] if np.unique(a)[-1] == 0.5/l else  np.unique(a)[-1]
        return xyz, a
    
    
    # ---> Main func starts
    
    pnt = get_pnt_nEPA(l, f, gi)
    count_Nan = np.count_nonzero(np.isnan(pnt))
    
    if   count_Nan == 0:
        f = np.array(f)
        k   = 2*np.pi*l ; xp = (1/k)*np.arccos(gi/np.sum(f))
        p4  = [xp]*len(f)
        
        pnt = np.vstack([np.array(pnt),p4])
        
        ni, di   = plane_eq2n(pnt)
        
        n_select     = ni
        distance_all = remaining(l, gi, f, n_select, di)
        
        points = np.array(pnt)
        
    elif count_Nan == 1:
        pi, pext    = getxyz(l, f, gi) # getxyz_opt3(l, f, gi, minmax=min) # 
        
        n_ext, dext = plane_eq2n(pext)
        ni, di      = plane_eq2n(pi)
        n_select    = ni #n_ext
        distance_all = remaining(l, gi, f, n_select, di)
        
        points = np.vstack([pi, pext])
        
        
    elif count_Nan > 1:        
        #pi, pext  = getxyz_opt3(l, f, gi, minmax=1)
        pi, pext    = gt3(l, f, gi)
        
        n_ext, dext = plane_eq2n(pext)
        ni, di      = plane_eq2n(pi)
        n_select    = ni # n_ext
        
        di = di if di !=0 and di<dext else dext       
        distance_all = remaining(l, gi, f, n_select, di)
        points = np.vstack([pi, pext])
    else:
        print("--> count_Nan ", count_Nan)
        print("--> Wired isosurface check linearization routine ")
        print("--> predicted points are : \n", pnt)
        
    return n_select, distance_all, points

