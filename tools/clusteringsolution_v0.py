import scipy
import numpy as np
import polytope as pc

def clustersolution(poly):
    
    # ---> Get distance matrix 
    col = []
    for count1, i in enumerate(poly):
        xg = np.mean(pc.extreme(i), axis=0)
        col.append(xg)
    
    cx = np.array(col)[:]
    dd = scipy.spatial.distance_matrix(cx,cx, p=2)
    
    # ---> Clustering based on distance: compare distance and error
    
    newsol, idxcollection = [], []
    
    for count2, i in enumerate(poly):
        xx   = pc.extreme(i) ; dmax = np.max(xx, axis=0) ; dmin = np.min(xx, axis=0)
        err  = np.abs(dmax-dmin)/1
                
        limit = (np.max(err)+np.min(err))/2 if np.max(err)<=0.1 else np.max(err)/2
        einx = np.where(dd[count2]<=limit)[0]
        
        if (len(einx) >1) and (np.all([xy not in idxcollection for xy in einx])):
            p  = np.zeros(len(err))
            for jj in einx:
                p  = np.vstack([p, pc.extreme(poly[jj])])
                idxcollection.append(jj)
                
            p  = np.delete(p,0,axis=0)
            #eq = pc.qhull(np.array(p))
            
            hull = scipy.spatial.ConvexHull(p)
            Abeq = hull.equations
            A =  Abeq[:,0:-1]
            b = -Abeq[:,-1]
            vertices_out = p[hull.vertices,:]
            resultantpoly = pc.Polytope(A=A, b=b, vertices = vertices_out)
            
            newsol.append(resultantpoly)
            
        else:
            if (count2 not in idxcollection):
                idxcollection.append(count2)
                newsol.append(poly[count2])
    
    inx = [[True if i.intersect(j) else False for i in newsol] for j in newsol]
    
    # ---> Clustering based on overlap: Remove any overlapping
    
    rr, ny = [], []
    for count3, ij in enumerate(inx):
        idx = np.where(ij)[0]
        if(np.all([ c3x not in rr for c3x in idx])):
            p = np.zeros(3)
            for ik in  idx:
                p = np.vstack([p, pc.extreme(newsol[ik])])
                rr.append(ik)
            
            p=np.delete(p,0,axis=0)
            #ny.append(pc.qhull(p))
            
            hull = scipy.spatial.ConvexHull(p)
            Abeq = hull.equations
            A =  Abeq[:,0:-1]
            b = -Abeq[:,-1]
            vert_out = p[hull.vertices,:]
            ny.append(pc.Polytope(A=A, b=b, vertices = vert_out))
    
    return pc.Region(ny)


