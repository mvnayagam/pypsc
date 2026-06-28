import numpy as np
import polytope as pc
from itertools import product

import logging
logger = logging.getLogger(__name__)

class Tessellator:

    def __init__(self, reflection, normal, distance, IorG='amplitude', limitingmat= None, imax=0.5, dtype=np.float64, logger=None):

        self.logger = logger or logging.getLogger("psc.tessellator")
        
        self.reflection = reflection
        self.normal     = np.asarray(normal, dtype=dtype)
        self.distance   = np.asarray(distance, dtype=dtype)
        self.IorG   = IorG
        self.imax   = dtype(imax)
        self.dtype  = dtype
        self.dim    = len(normal)
        self.k      = dtype(2 * np.pi * reflection)
        
        self.Identitymat = np.eye(self.dim, dtype=dtype)
        
        self.scom = np.array(list(product([-1,1], repeat=self.dim)), dtype=np.int8)
        
        if limitingmat is not None:
            self.limitingmat = np.asarray(limitingmat, dtype=dtype)
            self.minlimit = np.min(self.limitingmat, axis=0)
            self.maxlimit = np.max(self.limitingmat, axis=0)
        else:
            self.limitingmat = None
            self.minlimit = None
            self.maxlimit = None
        
        
    # -------------------------------------------------------
    # Utilities
    # -------------------------------------------------------

    def allowed_scom_mask(self, d, scom, eps=1e-12):

        mask = np.ones(len(scom), dtype=bool)

        lower = np.isclose(d, 0.0, atol=eps)
        upper = np.isclose(d, self.imax, atol=eps)

        if np.any(lower):
            mask &= np.all(scom[:, lower] == 1, axis=1)

        if np.any(upper):
            mask &= np.all(scom[:, upper] == -1, axis=1)

        return mask
    
    def getmesh(self, eps=1e-12):
        
        c = np.linspace(0, self.imax, int(2 * self.reflection * self.imax + 1), dtype=self.dtype )
        
        
        # --- > These two lines are replace with following one line to save memory
        # j = np.meshgrid(*([c] * self.dim), indexing='ij')
        # meshlist = np.stack(j, axis=-1).reshape(-1, self.dim)
        meshlist = np.stack(np.meshgrid(*([c]*self.dim), indexing='ij'), axis=-1).reshape(-1,self.dim)
        
        oo = np.cos(self.k * meshlist)
        
        mask = ( np.all(oo > eps, axis=1) | np.all(oo < -eps, axis=1) )
        
        return meshlist[mask]

    def getsigncombination(self):
        # np.array(list(product([-1, 1], repeat=self.dim)), dtype=np.int8 )
        return self.scom

    # -------------------------------------------------------
    # Fast asymmetric-unit filtering
    # -------------------------------------------------------

    def asym_filter(self, dlist):

        temp = np.tril(np.ones((self.dim, self.dim), dtype=self.dtype), 0)
        temp = 0.5 * np.vstack([ np.zeros((1, self.dim), dtype=self.dtype), temp ])
        asym = pc.qhull(temp)

        # vectorized inequality test
        mask = np.all( asym.A @ dlist.T <= asym.b[:, None] + 1e-12, axis=0 )
        return dlist[mask]

    # -------------------------------------------------------
    # Core engine - Base and complete polytope builder
    # -------------------------------------------------------
    
    def build_base_polys(self):
        
        #I = np.eye(self.dim, dtype=self.dtype)
        
        scom = self.getsigncombination()
        
        A_all, b0_all = [], []
        
        for i in scom:
            
            # A = np.vstack([ -i * self.normal, i * self.normal, -self.Identitymat, self.Identitymat ])
            if (self.limitingmat is not None):# and (self.reflection != 1):
                A = np.vstack([ -i * self.normal, i * self.normal, -self.Identitymat, self.Identitymat, -self.Identitymat, self.Identitymat])
            else:
                A = np.vstack([ -i * self.normal, i * self.normal, -self.Identitymat, self.Identitymat ])
            
            sign_last = i[-1]  # this direction controls the primary orientation of polytope 
            if sign_last > 0:
                b12 = np.array([-sign_last, sign_last]) * self.distance
            else:
                b12 = np.array([sign_last, -sign_last]) * self.distance
            
            # fixed offsets (independent of d)
            de0_inner = (i - 1) / (4 * self.reflection)
            de0_outer = (i + 1) / (4 * self.reflection)
            
            # b0 = np.concatenate([ b12, -de0_inner, de0_outer ])
            if (self.limitingmat is not None):# and (self.reflection != 1):
                b0 = np.concatenate([ b12, -de0_inner, de0_outer, self.maxlimit, self.maxlimit ])
            else:
                b0 = np.concatenate([ b12, -de0_inner, de0_outer ])
            
            A_all.append(A)
            b0_all.append(b0)
            
        return scom, np.array(A_all), np.array(b0_all)
    
    def _build_polytope(self, EPA=False, fixdlist=None):
        
        if fixdlist is not None:
            dlist = np.asarray(fixdlist, dtype=self.dtype)
        
        else:
            dlist = self.getmesh()
            
            dlist = dlist if self.reflection > 1 else np.delete(dlist, 1, 0)

            # -------------------------------------------------
            # EPA asymmetric reduction
            # -------------------------------------------------
            if EPA:
                dlist = self.asym_filter(dlist)

            # -------------------------------------------------
            # amplitude reduction
            # -------------------------------------------------
            
            if self.IorG == 'amplitude':
                
                cosvals = np.cos(self.k * dlist)
                
                mask = np.all(cosvals > 1e-12, axis=1)
                
                dlist = dlist[mask]
                
        # -------------------------------------------------
        # build all base polytopes once
        # -------------------------------------------------

        scom, A_all, b0_all = self.build_base_polys()

        
        # -------------------------------------------------
        # collect inequalities only
        # -------------------------------------------------
        
        poly_data = []

        for d in dlist:

            valid_mask = self.allowed_scom_mask(d, scom)

            if not np.any(valid_mask):
                continue

            A_valid  = A_all[valid_mask]
            b0_valid = b0_all[valid_mask]

            # vectorized shifts
            # shape:  (Npoly, rows, cols) @ (cols)
            shifts = np.einsum('ijk,k->ij', A_valid, d)
            
            b_all = b0_valid + shifts
            
            # -------------------------------------------------
            # feasibility pruning
            # -------------------------------------------------
            
            lower = b_all[:, -2*self.dim:-self.dim]
            upper = b_all[:, -self.dim:]
            
            feasible = (
                np.all(np.isfinite(b_all), axis=1)
                &
                np.all(lower < self.imax, axis=1)
                &
                np.all(upper > 0.0, axis=1)
            )

            if not np.any(feasible):
                continue
            
            A_keep = A_valid[feasible]
            b_keep = b_all[feasible]
            
            # delay Polytope creation
            for A, b in zip(A_keep, b_keep):
                poly_data.append((A.copy(), b.copy()))

        # -------------------------------------------------
        # region creation
        # -------------------------------------------------

        if len(poly_data) == 0:
            return pc.Region([])

        region = pc.Region([ pc.reduce( pc.Polytope(A, b) ) for A, b in poly_data ])
        
        return region
    
    # -------------------------------------------------------
    # Return polytope for EPA and NEPA
    # -------------------------------------------------------
    
    # for EPA
    def getpolytope_EPA( self ):
        return self._build_polytope( EPA=True)

    # fro NEPA
    def getpolytope_NEPA( self):
        return self._build_polytope(EPA=False)

    # At particular d
    def getpolytope_dfixed(self, dfixed):
        return self._build_polytope(EPA=False, fixdlist=np.asarray(dfixed, dtype=self.dtype))


# How to use this class:
# Input:
# rep = Tessellator(reflection=1, normal=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], distance=[0.2, 0.3, 0.4], IorG='amplitude', limitingmat=SOMEARRAY, imax=0.5)
# region_EPA = rep.getpolytope_EPA()
# region_nEPA = rep.getpolytope_NEPA()
# Output:
# pc.Region([...]) will be returned for given reflection within EPA or nEPA

# To get polytopes at particular d then. NOTE on how dfixed is fed.
# o = rep.getpolytope_dfixed(dfixed=np.array([[0.3333333333,0,0]]))




# -------------------------------------------------------
#  Common utilities for both EPA and nEPA (can be moved outside the class if needed)
# -------------------------------------------------------

def allowed_scom_indices(d, scom, imax=0.5, eps=1e-12):

    mask = np.ones(len(scom), dtype=bool)

    for k in range(len(d)):

        # lower boundary
        if np.isclose(d[k], 0.0, atol=eps):
            mask &= (scom[:, k] == +1)

        # upper boundary
        elif np.isclose(d[k], imax, atol=eps):
            mask &= (scom[:, k] == -1)

        # interior:
        # no restriction

    return np.where(mask)[0]

def getmesh(l: int, dim: int, imax: float = 0.5) -> np.ndarray:
    c = np.linspace(0, imax, int(2 * l * imax + 1))
    
    # create grid directly
    j = np.meshgrid(*([c] * dim), indexing="ij")
    meshlist = np.stack([g.ravel() for g in j], axis=-1)
    
    # compute cos for all at once
    oo = np.cos(2 * np.pi * l * meshlist)
    
    # build mask
    mask = (np.all(oo > 0, axis=1)) | (np.all(oo < 0, axis=1))
    
    meshlist = meshlist[mask]
    
    # sort priority: last column -> first column
    idx = np.lexsort(tuple(meshlist[:, i] for i in range(dim)))
    
    # return meshlist[mask]
    return meshlist[idx]

def getsigncombination(r: int) -> np.ndarray:
    # return np.array(list(product([-1, 1], repeat=r)))
    return np.array(list(product([-1, 1], repeat=r)))

# -------------------------------------------------------
#  Modules for EPA 
# -------------------------------------------------------

def getpolytope_EPA( l: int, normal: list, distance: list, IorG: str = 'amplitude', imax: float = 0.5 ):
    ''' returns a collection of polytope for given reflection l.
        The polytope parameters boundary distance and normal are
        the required inputs. This module is for both EPA and nEPA
        Also this do not assume I or G. So it will return 2*(l**m)
        polytope for given l. m is dimension of PS
    Args:
        l (int): reflection
        normal (list): direction of polytope
        distance (list): boundary distance
        imax (int, optional): Limit of PS. Defaults to 0.5.

    Returns:
        _type_: Region of polytope.
    '''    
    
    normal   = np.asarray(normal)
    distance = np.asarray(distance)
    n        = len(normal)

    # ----------------------------
    # Identity and constraint polytope (fixed)
    # ----------------------------
    I    = np.eye(n)
    
    # ----------------------------
    # Define the PSC box constraints once (fixed) - not used anymore in the loop, but kept here for reference
    # ----------------------------
    # Apsc = np.vstack([-I, I])
    # bpsc = np.concatenate([np.zeros(n), 0.5 * np.ones(n)])
    # psc = pc.Polytope(Apsc, bpsc)
    temp = np.tril( np.ones(shape=(n,n)) , 0 )
    temp = 0.5*np.vstack([[0]*n, temp])
    asym = pc.qhull(np.array(temp))
    
    
    # ----------------------------
    # 1. mesh (dlist)
    # ----------------------------
    dlist = getmesh(l, dim=n, imax=imax)
    
    # Apply choice of origin constraint for l=1 (valid only for l=1; for l>1 origin is not a free choice)
    dlist = dlist if l > 1 else np.delete(dlist, 1, 0)
    
    # Reduce translation vectors into assymetric unit (for EPA only)
    dlist = np.array([dlx for cc, dlx in enumerate(dlist) if dlx in asym])
        
    if IorG == 'amplitude':
        cosvals = np.cos(2 * np.pi * l * dlist)
        mask = np.all(cosvals > 0, axis=1)
        dlist = dlist[mask]
    
    # ----------------------------
    # 2. sign combinations (precompute once)
    # ----------------------------
    scom = getsigncombination(len(normal))
    scom = scom[scom[:, -1].argsort()][::-1]
    
    # ----------------------------
    # 3. BUILD BASE POLYTOPES at d = 0 ONLY
    # ----------------------------
    base_polys = []
    
    for i in scom:
        
        # A matrix (independent of d)
        A = np.vstack([ -i * normal, i * normal, -I, I ])
        
        sign_last = i[-1]  # this direction controls the primary orientation of polytope 
        if sign_last > 0:
            b12 = np.array([-sign_last, sign_last]) * distance
        else:
            b12 = np.array([sign_last, -sign_last]) * distance
                    
        # fixed offsets (independent of d)
        de0_inner = (i - 1) / (4 * l)
        de0_outer = (i + 1) / (4 * l)
        
        b0 = np.concatenate([ b12, -de0_inner, de0_outer ])
        
        base_polys.append((A, b0, i))
    
    # -------------------------------------------------
    # 4. SHIFT ONLY (FAST PART)
    # -------------------------------------------------
    polylist = []
    
    for d in dlist:
        
        # ---------------------------------------------
        # inward-pointing sign combinations ONLY
        # ---------------------------------------------
        valid_indices = allowed_scom_indices(d, scom, imax)
        
        for idx in valid_indices:
            
            A, b0, i = base_polys[idx]
            
            # shift
            b = b0 + A[:, :n] @ d
            
            # invalid numerical cases
            if not np.all(np.isfinite(b)):
                continue
            
            # exact box feasibility
            lower = b[-2*n:-n]
            upper = b[-n:]
            
            if np.any(lower >= 0.5) or np.any(upper <= 0.0):
                continue
            
            # keep candidate
            polylist.append((A, b, d, i))
            
    # -------------------------------------------------
    # 5. build Region once
    # -------------------------------------------------
    return pc.Region([ pc.Polytope(A, b) for A, b, _, _ in polylist ])


def getpolytope_mitdv1OLD(l: int, normal: list, distance: list, dlist: list, imax: float = 0.5 ):
    ''' returns a collection of polytope for given reflection l and mesh gird point dlist.
    '''    
    
    normal   = np.asarray(normal)
    distance = np.asarray(distance)
    n        = len(normal)
    
    # ----------------------------
    # Identity and constraint polytope (fixed)
    # ----------------------------
    I    = np.eye(n)
    
    # ----------------------------
    # 2. sign combinations (precompute once)
    # ----------------------------
    scom = getsigncombination(len(normal))
    scom = scom[scom[:, -1].argsort()][::-1]
    
    # ----------------------------
    # 3. BUILD BASE POLYTOPES at d = 0 ONLY
    # ----------------------------
    base_polys = []
    
    for i in scom:
        
        # A matrix (independent of d)
        A = np.vstack([ -i * normal, i * normal, -I, I ])
        
        sign_last = i[-1]  # this direction controls the primary orientation of polytope 
        if sign_last > 0:
            b12 = np.array([-sign_last, sign_last]) * distance
        else:
            b12 = np.array([sign_last, -sign_last]) * distance
                    
        # fixed offsets (independent of d)
        de0_inner = (i - 1) / (4 * l)
        de0_outer = (i + 1) / (4 * l)
        
        b0 = np.concatenate([ b12, -de0_inner, de0_outer ])
        
        base_polys.append((A, b0, i))
    
    # -------------------------------------------------
    # 4. SHIFT ONLY (FAST PART)
    # -------------------------------------------------
    polylist = []
    
    for d in dlist:
        
        # ---------------------------------------------
        # inward-pointing sign combinations ONLY
        # ---------------------------------------------
        valid_indices = allowed_scom_indices(d, scom, imax)
        
        for idx in valid_indices:
            
            A, b0, i = base_polys[idx]
            
            # shift
            b = b0 + A[:, :n] @ d
            
            # invalid numerical cases
            if not np.all(np.isfinite(b)):
                continue
            
            # exact box feasibility
            lower = b[-2*n:-n]
            upper = b[-n:]
            
            if np.any(lower >= 0.5) or np.any(upper <= 0.0):
                continue
            
            # keep candidate
            polylist.append((A, b, d, i))
            
    # -------------------------------------------------
    # 5. build Region once
    # -------------------------------------------------
    return pc.Region([ pc.Polytope(A, b) for A, b, _, _ in polylist ])


def getpolytope_mitd(l: int, normal: list, distance: list, dlist: list,  limitmat: list = None, imax: float = 0.5 ):
    ''' returns a collection of polytope for given reflection l and mesh gird point dlist.
    '''    
    
    normal   = np.asarray(normal)
    distance = np.asarray(distance)
    n        = len(normal)
    minlimit, maxlimit = np.min(np.array(limitmat), axis=0), np.max(np.array(limitmat), axis=0)
    
    # ----------------------------
    # Identity and constraint polytope (fixed)
    # ----------------------------
    I    = np.eye(n)
    
    # ----------------------------
    # 2. sign combinations (precompute once)
    # ----------------------------
    scom = getsigncombination(len(normal))
    scom = scom[scom[:, -1].argsort()][::-1]
    
    # ----------------------------
    # 3. BUILD BASE POLYTOPES at d = 0 ONLY
    # ----------------------------
    base_polys = []
    
    for i in scom:
        
        # A matrix (independent of d)
        # A = np.vstack([ -i * normal, i * normal, -I, I ])
        
        if (limitmat is not None) and (l != 1):
            A = np.vstack([ -i * normal, i * normal, -I, I, -I, I])
        else:
            A = np.vstack([ -i * normal, i * normal, -I, I ])
        
        
        sign_last = i[-1]  # this direction controls the primary orientation of polytope 
        if sign_last > 0:
            b12 = np.array([-sign_last, sign_last]) * distance
        else:
            b12 = np.array([sign_last, -sign_last]) * distance
        
        # fixed offsets (independent of d)
        de0_inner = (i - 1) / (4 * l)
        de0_outer = (i + 1) / (4 * l)
        
        # b0 = np.concatenate([ b12, -de0_inner, de0_outer ])
        if (limitmat is not None) and (l != 1):
            b0 = np.concatenate([ b12, -de0_inner, de0_outer, maxlimit, maxlimit ])
        else:
            b0 = np.concatenate([ b12, -de0_inner, de0_outer ])
        
        base_polys.append((A, b0, i))
    
    # -------------------------------------------------
    # 4. SHIFT ONLY (FAST PART)
    # -------------------------------------------------
    polylist = []
    
    for d in dlist:
        
        # ---------------------------------------------
        # inward-pointing sign combinations ONLY
        # ---------------------------------------------
        valid_indices = allowed_scom_indices(d, scom, imax) # np.arange(8) # -> if you want fill polytope at given d
        
        for idx in valid_indices:
            
            A, b0, i = base_polys[idx]
            
            # shift
            b = b0 + A[:, :n] @ d
            
            # invalid numerical cases
            if not np.all(np.isfinite(b)):
                continue
            
            # exact box feasibility
            lower = b[-2*n:-n]
            upper = b[-n:]
            
            if np.any(lower >= 0.5) or np.any(upper <= 0.0):
                continue
            
            # keep candidate
            polylist.append((A, b, d, i))
            
    # -------------------------------------------------
    # 5. build Region once
    # -------------------------------------------------
    return pc.Region([ pc.reduce(pc.Polytope(A, b)) for A, b, _, _ in polylist ])

# -------------------------------------------------------
# Modules for non EPA 
# -------------------------------------------------------

def getpolytope_nEPA( l: int, normal: list, distance: list, IorG: str = 'amplitude', limitmat: list = None, imax: float = 0.5 ):
    
    normal   = np.asarray(normal)
    distance = np.asarray(distance)
    n        = len(normal)
    minlimit, maxlimit = np.min(np.array(limitmat), axis=0), np.max(np.array(limitmat), axis=0)
    
    # ----------------------------
    # Identity and constraint polytope (fixed)
    # ----------------------------
    I    = np.eye(n)
    
    # ----------------------------
    # 1. mesh (dlist)
    # ----------------------------
    dlist = getmesh(l, dim=n, imax=imax)
    dlist = dlist if l > 1 else np.delete(dlist, 1, 0)
    
    if IorG == 'amplitude':
        cosvals = np.cos(2 * np.pi * l * dlist)
        mask = np.all(cosvals > 0, axis=1)
        dlist = dlist[mask]
    
    # ----------------------------
    # 2. sign combinations (precompute once)
    # ----------------------------
    scom = getsigncombination(len(normal))
    scom = scom[scom[:, -1].argsort()][::-1]
    
    # ----------------------------
    # 3. BUILD BASE POLYTOPES at d = 0 ONLY
    # ----------------------------
    base_polys = []
    
    for i in scom:
        
        # A matrix (independent of d)
        # A = np.vstack([ -i * normal, i * normal, -I, I ])
        if (limitmat is not None) and (l != 1):
            A = np.vstack([ -i * normal, i * normal, -I, I, -I, I])
        else:
            A = np.vstack([ -i * normal, i * normal, -I, I ])
        
        sign_last = i[-1]  # this direction controls the primary orientation of polytope 
        if sign_last > 0:
            b12 = np.array([-sign_last, sign_last]) * distance
        else:
            b12 = np.array([sign_last, -sign_last]) * distance
                    
        # fixed offsets (independent of d)
        de0_inner = (i - 1) / (4 * l)
        de0_outer = (i + 1) / (4 * l)
        
        # b0 = np.concatenate([ b12, -de0_inner, de0_outer ])
        if (limitmat is not None) and (l != 1):
            b0 = np.concatenate([ b12, -de0_inner, de0_outer, maxlimit, maxlimit ])
        else:
            b0 = np.concatenate([ b12, -de0_inner, de0_outer ])
        
        base_polys.append((A, b0, i))
    
    # -------------------------------------------------
    # 4. SHIFT ONLY (FAST PART)
    # -------------------------------------------------
    polylist = []
    
    for d in dlist:
        
        # ---------------------------------------------
        # inward-pointing sign combinations ONLY
        # ---------------------------------------------
        valid_indices = allowed_scom_indices(d, scom, imax)
        
        for idx in valid_indices:
            
            A, b0, i = base_polys[idx]
            
            # shift
            b = b0 + A[:, :n] @ d
            
            # invalid numerical cases
            if not np.all(np.isfinite(b)):
                continue
            
            # exact box feasibility
            lower = b[-2*n:-n]
            upper = b[-n:]
            
            if np.any(lower >= 0.5) or np.any(upper <= 0.0):
                continue
            
            # keep candidate
            polylist.append((A, b, d, i))
            
    # -------------------------------------------------
    # 5. build Region once
    # -------------------------------------------------
    return pc.Region([ pc.reduce(pc.Polytope(A, b)) for A, b, _, _ in polylist ])


def getpolytope_nEPAv1OLD( l: int, normal: list, distance: list, IorG: str = 'amplitude', imax: float = 0.5 ):
    ''' returns a collection of polytope for given reflection l.
        The polytope parameters boundary distance and normal are
        the required inputs. This module is for both EPA and nEPA
        Also this do not assume I or G. So it will return 2*(l**m)
        polytope for given l. m is dimension of PS
    Args:
        l (int): reflection
        normal (list): direction of polytope
        distance (list): boundary distance
        imax (int, optional): Limit of PS. Defaults to 0.5.

    Returns:
        _type_: Region of polytope.
    '''    
    
    normal   = np.asarray(normal)
    distance = np.asarray(distance)
    n        = len(normal)

    # ----------------------------
    # Identity and constraint polytope (fixed)
    # ----------------------------
    I    = np.eye(n)
    
    # ----------------------------
    # Define the PSC box constraints once (fixed) - not used anymore in the loop, but kept here for reference
    # ----------------------------
    # Apsc = np.vstack([-I, I])
    # bpsc = np.concatenate([np.zeros(n), 0.5 * np.ones(n)])
    # psc = pc.Polytope(Apsc, bpsc)
            
    # ----------------------------
    # 1. mesh (dlist)
    # ----------------------------
    dlist = getmesh(l, dim=n, imax=imax)
    dlist = dlist if l > 1 else np.delete(dlist, 1, 0)
    
    if IorG == 'amplitude':
        cosvals = np.cos(2 * np.pi * l * dlist)
        mask = np.all(cosvals > 0, axis=1)
        dlist = dlist[mask]
    
    # ----------------------------
    # 2. sign combinations (precompute once)
    # ----------------------------
    scom = getsigncombination(len(normal))
    scom = scom[scom[:, -1].argsort()][::-1]
    
    # ----------------------------
    # 3. BUILD BASE POLYTOPES at d = 0 ONLY
    # ----------------------------
    base_polys = []
    
    for i in scom:
        
        # A matrix (independent of d)
        A = np.vstack([ -i * normal, i * normal, -I, I ])
        
        sign_last = i[-1]  # this direction controls the primary orientation of polytope 
        if sign_last > 0:
            b12 = np.array([-sign_last, sign_last]) * distance
        else:
            b12 = np.array([sign_last, -sign_last]) * distance
                    
        # fixed offsets (independent of d)
        de0_inner = (i - 1) / (4 * l)
        de0_outer = (i + 1) / (4 * l)
        
        b0 = np.concatenate([ b12, -de0_inner, de0_outer ])
        
        base_polys.append((A, b0, i))
    
    # -------------------------------------------------
    # 4. SHIFT ONLY (FAST PART)
    # -------------------------------------------------
    polylist = []
    
    for d in dlist:
        
        # ---------------------------------------------
        # inward-pointing sign combinations ONLY
        # ---------------------------------------------
        valid_indices = allowed_scom_indices(d, scom, imax)
        
        for idx in valid_indices:
            
            A, b0, i = base_polys[idx]
            
            # shift
            b = b0 + A[:, :n] @ d
            
            # invalid numerical cases
            if not np.all(np.isfinite(b)):
                continue
            
            # exact box feasibility
            lower = b[-2*n:-n]
            upper = b[-n:]
            
            if np.any(lower >= 0.5) or np.any(upper <= 0.0):
                continue
            
            # keep candidate
            polylist.append((A, b, d, i))
            
    # -------------------------------------------------
    # 5. build Region once
    # -------------------------------------------------
    return pc.Region([ pc.Polytope(A, b) for A, b, _, _ in polylist ])


def  getpolytope(l: int, normal: list, distance: list, imax: int =0.5) -> list:
    """ returns a collection of polytope for given reflection l.
        The polytope parameters boundary distance and normal are
        the required inputs. This module is for both EPA and nEPA
        Also this do not assume I or G. So it will return 2*(l**m)
        polytope for given l. m is dimension of PS
    Args:
        l (int): reflection
        normal (list): direction of polytope
        distance (list): boundary distance
        imax (int, optional): Limit of PS. Defaults to 0.5.

    Returns:
        _type_: Region of polytope.
    """    
    
    polylist = []
    
    dlist = getmesh(l, normal, imax=0.5)
    
    scom  = getsigncombination(len(normal))
    scom  = scom[scom[:,len(normal)-1].argsort()][::-1]
    
    gpsc  = np.identity(len(normal))
    Apsc  = np.array(np.vstack([-gpsc, gpsc]))
    bpsc  = np.array([0]*len(normal) + [0.5]*len(normal))
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


# ===> I do not know why i wrote this module. but thinking that if coordinates of linearization point
# #      is known then this module can be used
# def repeat(p, d, f, imin, imax):
    
#     pts =[]
#     inx =np.argwhere(d != 0)
#     nz  =np.count_nonzero(d)
    
#     if nz == 0:
#         e1=np.copy(p)
#         pts.append(e1)
        
#     if nz != 0:
#         r,c = np.shape(p)
        
#         if (nz != len(d)):
            
#             if (np.all((d[inx[:,0]]+p[:,inx[:,0]])>=imin) and np.all((d[inx[:,0]]+p[:,inx[:,0]])<=imax)):
                              
#                 if (nz == 1):
#                     e2=np.copy(p)
#                     e2[:,inx[:,0]]=e2[:,inx[:,0]]+d[inx[:,0]]
#                     pts.append(e2)
            
#         if (np.all((d[inx[:,0]]-p[:,inx[:,0]])>=imin) and np.all((d[inx[:,0]]-p[:,inx[:,0]])<=imax)):
            
#             e4=np.copy(p)
#             e4[:,inx[:,0]]=d[inx[:,0]]-p[:,inx[:,0]]
#             pts.append(e4)
            
#             if (nz >1):
#                 for j in f:
#                     e4a=np.copy(p)
#                     e4a=e4a*j
                    
#                     e4a[:,inx[:,0]]=d[inx[:,0]]-e4a[:,inx[:,0]]
                    
#                     if (np.all(e4a>=imin) and np.all(e4a<=imax)):
#                         pts.append(e4a)
        
#     return pts



# def getmesh(l: int, coordinates: list, imax:int =0.5) -> list:
    
#     c = np.linspace(0,imax,int(2*l*imax+1) )
    
#     k = [c, c]*len(coordinates)
#     k = k[0:len(coordinates)]
    
#     j = np.meshgrid(*k)
    
#     [*dim] = np.shape(j)
    
#     f1=(np.array([j[i].reshape(-1,1) for i in range([*dim][0])]))
#     f2=np.hstack([f1[i] for i in range([*dim][0])])
    
#     meshlist=np.array(f2)
    
#     # plist = []
#     # for meshid in meshlist:
#     #     oo=np.cos(2*np.pi*l*meshid)
#     #     if (np.all(np.sign(oo) == 1) or np.all(np.sign(oo) == -1)):
#     #         plist.append(meshid)
    
#     oo    = np.cos(2*np.pi*l*meshlist)
#     mask  = (np.all(np.sign(oo) == 1, axis=1)) | (np.all(np.sign(oo) == -1, axis=1))
#     plist = meshlist[mask]
         
#     return np.array(plist)

# def getsigncombination(r:int) -> list:
#     scom=[]

#     for i in range(0, r+1):
#         t = [-1]*i+[1]*(r-i)
#         w = set(permutations(t))
#         for u in w:
#             scom.append(u)
#     return np.array(scom)

# def getsigncombination4(r: int) -> np.ndarray:
#     # 2^r combinations
#     n = 1 << r  # same as 2**r
#     # integers 0..2^r-1 in binary
#     bits = np.arange(n, dtype=np.uint32)[:, None] >> np.arange(r) & 1
#     # map {0,1} → {-1,1}
#     return 2*bits - 1

# def getpolytope_EPA(l: int, normal: list, distance: list, amplitudesign: int, IorG: str='amplitude', imax: int=0.5):
    
#     # temp=[]
#     # for i in range(len(normal)+1):
#     #     zero=np.zeros(len(normal)) 
#     #     zero[0:i]=0.5
#     #     temp.append(zero)
    
#     temp = np.tril( np.ones(shape=(len(normal), len(normal))) , 0 )
#     temp = imax*np.vstack([[0]*len(normal), temp])
#     asym = pc.qhull(np.array(temp))
       
#     polylist = []
    
#     dlist1 = getmesh(l, normal, imax=0.5)
    
#     if l==1:
#         dlist = np.delete(dlist1, 1, 0)
#     else:
#         #dinx = [ dlx for cc, dlx in enumerate(dlist1) if dlx in asym]
#         #dlist=np.array(dinx)
#         dlist=dlist1
    
#     scom  = getsigncombination(len(normal))
#     scom  = scom[scom[:,len(normal)-1].argsort()][::-1]
    
#     gpsc  = np.identity(len(normal))
#     Apsc  = np.array(np.vstack([-gpsc, gpsc]))
#     bpsc  = np.array([0]*len(normal) + [0.5]*len(normal))
#     psc   = pc.Polytope(Apsc, bpsc)
    
#     aa    = np.array(normal)
#     bb    = np.array(distance)
    
#     for d in dlist:
#         d  = np.array(d)
#         oo = np.cos(2*np.pi*l*d)
#         if IorG == 'amplitude':
#             if (np.all(np.sign(oo) == amplitudesign)):
#                 for i in scom:
                    
#                     A = []
#                     A.append(-i*aa)
#                     A.append( i*aa)
                    
#                     if i[len(normal)-1]>0:
#                         b=np.array(np.array([-i[len(normal)-1], i[len(normal)-1]])*(bb + np.sum([i[kk]*aa[kk]*d[kk] for kk in range(len(d))])))
                    
#                     else:
#                         b=np.array(np.array([i[len(normal)-1], -i[len(normal)-1]])*(bb + np.sum([i[kk]*aa[kk]*d[kk] for kk in range(len(d))])))
                    
#                     # ---> inner
#                     iden = np.identity(len(normal))
#                     for k in range(len(normal)):
#                         A=np.vstack([A,-1*iden[k]])
                    
#                     de = d + (i-1)*(1/(4*l))
#                     b=np.append(b, -de)
                    
#                     # ---> outter
#                     for k in range(len(normal)):
#                         A=np.vstack([A,iden[k]])
                        
#                     de = d + 1*(i+1)*(1/(4*l))
#                     b=np.append(b, de)
                    
#                     w=pc.Polytope(np.array(A),np.array(b))
                    
#                     if w.chebXc is not None:
#                         if (w <= psc):
#                             polylist.append(w)
                    
#         elif IorG == 'intensity':
            
#             if( np.all(np.sign(oo) == 1) or np.all(np.sign(oo) == -1) ):
#                 for i in scom:
                    
#                     A = []
#                     A.append(-i*aa)
#                     A.append( i*aa)
                    
#                     if i[len(normal)-1]>0:
#                         b=np.array(np.array([-i[len(normal)-1], i[len(normal)-1]])*(bb + np.sum([i[kk]*aa[kk]*d[kk] for kk in range(len(d))])))
                    
#                     else:
#                         b=np.array(np.array([i[len(normal)-1], -i[len(normal)-1]])*(bb + np.sum([i[kk]*aa[kk]*d[kk] for kk in range(len(d))])))
                    
#                     # ---> inner
#                     iden = np.identity(len(normal))
#                     for k in range(len(normal)):
#                         A=np.vstack([A,-1*iden[k]])
                    
#                     de = d + (i-1)*(1/(4*l))
#                     b=np.append(b, -de)
                    
#                     # ---> outter
#                     for k in range(len(normal)):
#                         A=np.vstack([A,iden[k]])
                        
#                     de = d + 1*(i+1)*(1/(4*l))
#                     b=np.append(b, de)
                    
#                     w=pc.Polytope(np.array(A),np.array(b))
                    
#                     if w.chebXc is not None:
#                         if (w <= psc):
#                             polylist.append(w)
#         else:
#             print("===> select correct option for IorG")
                        
#     return pc.Region(polylist)
