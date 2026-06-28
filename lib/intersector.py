import polytope as pc
import numpy as np

import logging
logger = logging.getLogger(__name__)



def find_intersection(s, r):
    u=[]

    for count, i in enumerate(r):
        v = s & i
        
        if type(v) is pc.Polytope:
            if not pc.is_empty(v):
                u.append(v)
        elif type(v) is pc.Region:
            for k in v:
                if not pc.is_empty(k):
                    u.append(k)
    return pc.Region(u)

def find_intersection_updated(s, r):
    u=[]
    
    if type(s) is pc.Polytope:
        #print("from if: ", np.shape(s.A)[1])
        dim=np.shape(s.A)[1]
    elif type(s) is pc.Region:
        #print("from elif: ", np.shape(s[0].A)[1])
        dim=np.shape(s[0].A)[1]
    else:
        pass
        #print("type is not found")
    
    gp  = np.identity(dim)
    A  = np.array(np.vstack([-gp, gp]))
    b  = np.array([0]*dim + [0.5]*dim)
    PS = pc.Polytope(A, b)
    
    for count, i in enumerate(r):
        v = s & i
        
        if type(v) is pc.Polytope:
            if not pc.is_empty(v):
                u.append(PS & v)
        elif type(v) is pc.Region:
            for k in v:
                if not pc.is_empty(k):
                    u.append( PS & k)
    return pc.Region(u)

def find_intersection_updated_v0(s, r):
    u=[]
    
    if type(s) is pc.Polytope:
        #print("from if: ", np.shape(s.A)[1])
        dim=np.shape(s.A)[1]
    elif type(s) is pc.Region:
        #print("from elif: ", np.shape(s[0].A)[1])
        dim=np.shape(s[0].A)[1]
    else:
        print("type is not found")
    
    gp  = np.identity(dim)
    A  = np.array(np.vstack([-gp, gp]))
    b  = np.array([0]*dim + [0.5]*dim)
    PSc = pc.Polytope(A, b)
    
    for count, i in enumerate(r):
        v = s & i
        
        if type(v) is pc.Polytope:
            if not pc.is_empty(v):
                #u.append(PSc.intersect(v))
                u.append(v)
        elif type(v) is pc.Region:
            for k in v:
                if not pc.is_empty(k):
                    #u.append(PSc.intersect(v))
                    u.append(v)
    return pc.Region(u)


class Intersector:
    '''
    This func is defined to find the intersection of successive / any pair of polytope regions
    It is written in serial method. Parallelization could save time. However it is yet to be tested
    
    Future goal : make it in parallel code when the number of polytopes in the region is large.
    
    '''
    
    def __init__(self, imax=0.5, logger=None):
        
        self.logger = logger or logging.getLogger("psc.intersector")
        
        self.imax = imax
        self.is_empty = pc.is_empty

    # -------------------------------------------------
    # Utilities
    # -------------------------------------------------

    @staticmethod
    def _flatten(x):
        if isinstance(x, pc.Polytope):
            return [x]
        return list(x)

    @staticmethod
    def _bbox(P):
        lb, ub = pc.bounding_box(P)
        return np.asarray(lb), np.asarray(ub)

    @staticmethod
    def _bbox_overlap(b1, b2):
        return np.all(b1[0] <= b2[1]) and np.all(b2[0] <= b1[1])

    def _prepare(self, polytopes):
        poly_list = self._flatten(polytopes)
        bbox = [self._bbox(p) for p in poly_list]
        return poly_list, bbox

    def _clip_box(self, dim):
        A = np.vstack([-np.eye(dim), np.eye(dim)])
        b = np.concatenate([np.zeros(dim), self.imax * np.ones(dim)])
        return pc.Polytope(A, b)
    
    # -------------------------------------------------
    # Internal clipping box
    # -------------------------------------------------
    
    def _post_box_clip(self, poly):

        if isinstance(poly, pc.Polytope):
            dimension = poly.dim
            
        elif isinstance(poly, pc.Region):
            dimension = poly[0].dim if len(poly) > 0 else 0
            
        else:
            raise TypeError( f"Unknown type: {type(poly).__name__}. Expected: pc.Polytope or pc.Region." )

        A = np.vstack([-np.eye(dimension), np.eye(dimension)])
        b = np.concatenate([ np.zeros(dimension), self.imax * np.ones(dimension) ])

        return pc.Polytope(A, b)

    # -------------------------------------------------
    # Main function
    # -------------------------------------------------
    def find_intersection(self, s, r, clipbox=False):

        s_list, s_bbox = self._prepare(s)
        r_list, r_bbox = self._prepare(r)

        out = []

        dim = s_list[0].dim if len(s_list) > 0 else r_list[0].dim
        clip_poly = self._clip_box(dim) if clipbox else None

        is_empty = self.is_empty

        for i, r_i in enumerate(r_list):
            rb = r_bbox[i]

            for j, s_i in enumerate(s_list):
                if not self._bbox_overlap(s_bbox[j], rb):
                    continue

                v = s_i.intersect(r_i)
                if is_empty(v):
                    continue

                items = v if isinstance(v, pc.Region) else [v]

                for k in items:

                    if clipbox:
                        k = clip_poly.intersect(k)

                    if not is_empty(k):
                        out.append(k)

        return pc.Region(out)

# Example usage:
# from x3Dintersection import Intersector
# process = Intersector(imax=0.5)
# s12 = process.find_intersection(l1, l2)
# s23 = process.find_intersection(s12, l3)
# s34 = process.find_intersection(s23, l4)