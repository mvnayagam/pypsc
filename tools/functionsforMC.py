import sys
import h5py
import re
import random

import polytope as pc
import intvalpy as ip
import numpy as np
from itertools import permutations
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from psc.lib.gspacer import g, F, hsurf_g
from psc.lib.getscatteringfactor import getf, getfs

from psc.lib.intersector  import find_intersection


np.set_printoptions(threshold=sys.maxsize)



def isosurfsold(h,xexp,f,j,fname):
    n, dis = [], []
    for l in range(1,h+1):
        gi = np.abs(g(l, xexp, f))
        
        normal, distance = linear_jonas(l, xexp, f, gi)
        
        n.append([l, normal, distance])
        fname.write("%3g\t%2.12f\t%2.12f\t%2.12f\t%2.12f\t%2.12f\n"%(l, normal[0],normal[1],normal[2], distance[0],distance[1]))
    return (n)

def get_pnts(l, x, f, gi):
    tot_coor=[]
    for i in range(len(x)):
        tem_coor    = np.zeros(len(x))
        tem_coor[i] = x[i]
        tem_coor[i] = hsurf_g(l, tem_coor, f, gi, i, s=1)
        tot_coor.append(tem_coor)
    
    return tot_coor

def plane_eq(pnt):
    
    p1 = np.array(pnt[0])
    p2 = np.array(pnt[1])
    p3 = np.array(pnt[2])
    p4 = np.array(pnt[3])
        
    # These two vectors are in the plane
    v1 = p3 - p2
    v2 = p1 - p2
    
    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp
    
    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d  = np.dot(cp, p3)
    
    n  = np.array([a, b, c])
    dp = np.sum(np.multiply(n,p4))
    ap, bp, cp = dp/a, dp/b, dp/c
    
    return a, b, c, d

def getfs_new(l, d, E, symbs = 'H', realfs=False):
    
    import numpy as np
    h, c, elec_charge = 6.626069e-34, 299792458, 1.60217646e-19
    angle = np.arcsin(l/d*h*c/(E*elec_charge))*180/np.pi
    fs = []
    
    for s in symbs:
        if realfs:
            fs.append( np.real(getf(angle, s, energy = E)[3]) )
        else:
            fs.append(getf(angle, s, energy = E)[3])
    return fs

def plot_isosurface_3d(l, h, gs, gx, gy, gzp, gzm, axs, cc, al=1, imax=0.5):
    for hi in range(l+1):
        if ( (hi/l <= (imax) and h%2 !=0) or (hi/l <= (imax) and h%2 ==0) ):
            if hi == 0:
                
                axs.plot_surface(gx, gy, gzp, color=cc, alpha=al) #,label='$h(%g, %2.2f)$'%(l,gi))
                
                surf = axs.plot_surface(gx, gy, gzp, color=cc, alpha=al, antialiased=True,
                                        facecolor=cc, linewidth=0, label=r'$l\mathrm{( %g,%1.2f)}$'%(l, gs))
                surf._facecolors2d = surf._facecolor3d
                surf._edgecolors2d = surf._edgecolor3d
                axs.plot_wireframe(gx, gy, gzm, color=cc, alpha=al, rstride=25, cstride=25,antialiased=True)
                
            else:
                if (hi/l < imax() and l%(2*hi) !=0):
                    
                    axs.plot_surface  (gx, gy, gzp + hi/l, color=cc, alpha=al)
                    axs.plot_wireframe(gx, gy, gzm + hi/l, color=cc, alpha=al, rstride=25, cstride=25)
                    
                axs.plot_surface  (gx, gy, -1*gzp + hi/l, color=cc, alpha=al)
                axs.plot_wireframe(gx, gy, -1*gzm + hi/l, color=cc, alpha=al, rstride=25, cstride=25)
                
                if (hi/l < imax() and l%(2*hi) ==0):
                    axs.plot_surface  (gx, gy, gzp + hi/l, color=cc, alpha=al)
                    axs.plot_wireframe(gx, gy, gzm + hi/l, color=cc, alpha=al, rstride=25, cstride=25)
                    
            if (l%(2*l) == 0 and l/h <= imax()):
                axs.plot_surface  (gx, gy, -1*gzp + (hi+2)/l, color=cc, alpha=al)
                axs.plot_wireframe(gx, gy, -1*gzp + (hi+2)/l, color=cc, alpha=al, rstride=25, cstride=25)
    return

def plot_poly3dnew(poly, ax, alpha=0.1, color='C0'):
    dim=poly.dim
    
    if dim == 2:
        v = ip.IntLinIncR2(-poly.A, -poly.b, show=False)
        for i in v:
            x, y = v[:,0], v[:,1]
            ax.fill(x, y, linestyle='-', color=color, linewidth=1, alpha=alpha)
            ax.scatter(x, y, s=0, color='black', alpha=0.5)
            
    if dim == 3:
        v = ip.lineqs3D(-poly.A, -poly.b, size=(3,3), show=False)
        for i in v:
            x, y, z = i[:,0], i[:,1], i[:,2]
            
            poly3d = [list(zip(x, y, z))]
            PC = Poly3DCollection(poly3d, lw=0.5)
            PC.set_alpha(alpha)
            PC.set_facecolor(color)
            ax.add_collection3d(PC)
            ax.plot(x, y, z, color='black', lw=0.1, alpha=1)
            ax.scatter(x, y, z, s=0.2, color='black')
        
    return

def fn_signcom(r):
    f=[]
    
    for i in range(0, r+1):
        t = [-1]*i+[1]*(r-i)
        w = set(permutations(t))
        for u in w:
            f.append(u)
    return np.array(f)

def get_pnts_new4(l, f, g, xinit=0, imax=0.5):
    k = 2*np.pi*l
    tot_coor=[]
    for i in range(len(f)):
        
        tem_coor    = np.zeros(len(f)) #np.array([1]*len(f))*xinit #np.zeros(len(f))
        tem_coor[i] = hsurf_g(l, tem_coor, f, g, i, s=1)
        tot_coor.append(tem_coor)
        
    
    if np.all(~np.isnan(np.array(tot_coor))):
        return np.array(tot_coor)
    else:
        #print("tem_coor : ", tot_coor)
        inx = np.argwhere(np.isnan(tot_coor).any(axis=1)).flatten()
        
        for iw in inx:
            tem_coor    = np.zeros(len(f)) #np.array([1]*len(f))*xinit #
            tem_coor[iw] = xinit/l
            #print("temp_coor : ",tem_coor)
            mlist       = list([ii for ii in range(len(f)) if ii != iw])
            #print("mlist : ", mlist, " iw ", iw)
            temp_coor = (1/k)*np.arccos( (g-f[iw]*np.cos(k*tem_coor[iw])) / 
                                         (np.sum([f[ii]*np.cos(k*tem_coor[ii]) for ii in mlist])))
            #print("temp_coor : ",temp_coor)
#                   ,(g-f[iw]*np.cos(k*tem_coor[iw])),
#                                             (np.sum([f[ii]*np.cos(k*tem_coor[ii]) for ii in mlist])) )
            tem_coor[iw+1:] = temp_coor
            tem_coor[:iw]   = temp_coor
            
            
            #for ji in mlist:
            #    tem_coor[ji] = (1/k)*np.arccos( (g-f[iw]*np.cos(k*tem_coor[iw])) / 
            #                                 (np.sum([f[ii]*np.cos(k*tem_coor[ii]) for ii in mlist])))
            #    print(mlist,"ji : ", ji, "tem_coor : ", tem_coor)
            
            tot_coor[iw]=tem_coor
            
    return np.array(tot_coor)

def get_pnts_new(l, x, f, gi, imax=0.5):
    tot_coor=[]
    for i in range(len(x)):
        tem_coor    = np.zeros(len(x))
        tem_coor[i] = x[i]
        tem_coor[i] = hsurf_g(l, tem_coor, f, gi, i, s=1)
        
        
        if (~np.isnan(tem_coor[i])):
            tem_coor[i] = tem_coor[i]
        else:
            tem_coor[i] = imax/l
            tem_coor[i] = hsurf_g(l, tem_coor, f, gi, i, s=1)
            
        tot_coor.append(tem_coor)
    
    return tot_coor

def get_mitd(l, normal, distance, scom, dlis, x=[1, 1, 1], imax=1/6):
    
    polylist = []
    
    gpsc  = np.identity(len(x))
    Apsc  = np.array(np.vstack([-gpsc, gpsc]))
    bpsc  = np.array([0]*len(x) + [imax]*len(x))
    psc   = pc.Polytope(Apsc, bpsc)
    
    #v=plot_poly3dnew(psc, axs, alpha=0.15, color ='C0')
    
    aa    = np.array(normal)
    bb    = np.array(distance)
    
    for d in dlis:
        d  = np.array(d)
        oo = np.cos(2*np.pi*l*d)
        if (np.all(np.sign(oo) == 1) or np.all(np.sign(oo) == -1)):
            for i in scom:
                
                A = []
                A.append(-i*aa)
                A.append( i*aa)
                
                if i[len(x)-1]>0:
                    b=np.array(np.array([-i[len(x)-1], i[len(x)-1]])*(bb + np.sum([i[kk]*aa[kk]*d[kk] for kk in range(len(d))])))
                
                else:
                    b=np.array(np.array([i[len(x)-1], -i[len(x)-1]])*(bb + np.sum([i[kk]*aa[kk]*d[kk] for kk in range(len(d))])))
                
                # ---> inner
                iden = np.identity(len(x))
                for k in range(len(x)):
                    A=np.vstack([A,-1*iden[k]])
                
                de = d + (i-1)*(1/(4*l))
                b=np.append(b, -de)
                
                # ---> outter
                for k in range(len(x)):
                    A=np.vstack([A,iden[k]])
                    
                de = d + 1*(i+1)*(1/(4*l))
                b=np.append(b, de)
                
                w=pc.Polytope(np.array(A),np.array(b))
                
                if w.chebXc is not None:
                    if (w.chebXc in psc):
                        polylist.append(w)
                
    return pc.Region(polylist)

def repeat_linearization(fn, l, pnt, meshlist, signcom, axss, imin=0, imax=0.5, plot=False):
    dlist, pi, po, poi = [], [], [], []
    
    for meshid in meshlist:
        oo=np.cos(2*np.pi*l*meshid)
        if (np.all(np.sign(oo) == 1) or np.all(np.sign(oo) == -1)):
            d = np.array(meshid)
            dlist.append(d)
            
            plist=fn_repeat(pnt, d, signcom, imin, imax)
            
            if plist:
                fn_write(fn, plist)
                
                for pl in plist:
                    
                    pi1=[pl[0], pl[1], pl[2]]
                    a1, b1, c1, d1 = plane_eq(pi1)
                    
                    pi.append( [-1*a1, -1*b1, -1*c1, d1])
                    poi.append([-1*a1, -1*b1, -1*c1, d1])
                    
                    if plot:
                        axss.add_collection3d(Poly3DCollection([pi1], facecolors='#79afd9',ec='k', lw=0.1, alpha=0.5))
                    
                    if ( len(pl) == 7 ):
                        po1=[pl[4], pl[5], pl[6]]
                    else:
                        po1=[pl[3], pl[4], pl[5]]
                    
                    a2, b2, c2, d2 = plane_eq(po1)
                    
                    po.append( [a2, b2, c2, -d2])
                    poi.append([a2, b2, c2, -d2])
                    if plot:
                        axss.add_collection3d(Poly3DCollection([po1], facecolors='#e3abff',ec='k', lw=0.1, alpha=0.5))
    return dlist, poi           

def isosurfs_EPA(h, xexp, f, j, fname):
    n  = [] 
    if h <= 2:
        for l in range(1,h+1):
            gi = np.abs(g(l, xexp, f))
            normal, distance = linear_jonas(l, xexp, f, gi)
            
            n.append([l, normal, distance, np.sign(g(l, xexp, f))])
            fname.write("%3g\t%2.12f\t%2.12f\t%2.12f\t%2.12f\t%2.12f\n"%(l, normal[0],normal[1],normal[2],
                                                             distance[0],distance[1]))
    else:
        gi = np.abs(g(h,xexp,f))
        normal, distance = linear_jonas(h, xexp, f, gi)
        
        n.append([h, normal, distance, np.sign(l, xexp, f)])
        fname.write("%3g\t%2.12f\t%2.12f\t%2.12f\t%2.12f\t%2.12f\n"%(h, normal[0],normal[1],normal[2],
                                                             distance[0],distance[1]))
    
    
    return (n)

def linear_jonas(l, xexp, f, gi):
    planedetails = []
    #======= 1. finding first three points
    pnt = get_pnts_new(l, xexp, f, gi)
    
    #======= 2. finding fourth point : the point on the surface :: At particular point x=y=z or x1*=x2*=x3*. so
    k   = 2*np.pi*l
    xp  = (1/k)*np.arccos(gi/np.sum(f))
    p4  = [xp]*len(f)
    
    pnt = np.vstack([np.array(pnt),p4])
    
    a, b, c, d = np.double(plane_eq(pnt))
    
    if ~np.isnan(a):
        
        n  = np.array([a, b, c], dtype='double')
        dp = np.double(np.sum(np.multiply(n,p4)))
        
        ap, bp, cp = dp/a, dp/b, dp/c
        
        #print("set1-->l = ", l, ";\t normal = [%2.6f, %2.6f, %2.6f]"%(a, b, c),"; ds=[%2.6f, %2.6f]\n"%(d,dp))
        planedetails.append([dp/a, dp/b, dp/c, dp])
        pp5=np.array([[ap, 0, 0], [0, bp, 0], [0, 0, cp]])
        
        pnt=np.vstack([np.array(pnt),pp5])
        
        normal = [a, b, c]
        d_all = [d, dp]
    else:
        px = []
        pnt=[]
        
        pin=get_pnts_new4(l, f, gi, xinit=0)
        pnt = np.vstack([pin])
        a, b, c, dis = plane_eq(pin)
                
        pon=get_pnts_new4(l, f, gi, xinit=0.5)
        pnt = np.vstack([pnt, pon])
        ap, bp, cp, dp = plane_eq(pon)
        
        #px = np.vstack([[xp, xp, xp],pon])
        #ax, bx, cx, dx, disx = plane_eq2(px)
        
        dx     = np.dot([xp, xp, xp],[a, b, c]) 
        
        normal = [a, b, c]
        d_all  = [dis, dx] if dx > dp else  [dis, dp]
        
    return normal, d_all

def isosurfs_nEPA(h, xexp, f, j, fname):
    n  = [] 
    if h <= 2:
        for l in range(1,h+1):
            gi = np.abs(g(l, xexp, f))
            normal, distance = jonas_3d_nEPA(l, xexp, f, gi)
            
            n.append([l, normal, distance])
            fname.write("%3g\t%2.12f\t%2.12f\t%2.12f\t%2.12f\t%2.12f\n"%(l, normal[0],normal[1],normal[2],
                                                             distance[0],distance[1]))
    else:
        gi = np.abs(g(h, xexp, f))
        normal, distance = jonas_3d_nEPA(h, xexp, f, gi)
        
        n.append([h, normal, distance])
        fname.write("%3g\t%2.12f\t%2.12f\t%2.12f\t%2.12f\t%2.12f\n"%(h, normal[0],normal[1],normal[2],
                                                             distance[0],distance[1]))
    
    return (n)

def jonas_3d_nEPA(l, xexp, f, gi):
        
    def getxyz_opt3(l, f, g, minmax=min):
        xyz = []
        for jj in [0, 0.5/l]:
            for i in range(len(f)-1):
                t = [0] * len(f)
                t[len(f)-1] = jj
                
                z = hsurf_g(l, t, f, gi, j=i, s=1)
                
                if not np.isnan(z):
                    t[i] = z
                else:
                    t[i] = 0.5/l
                    z = hsurf_g(l, t, f, gi, j=i-1, s=1)
                    t[i-1] = z
                
                xyz.append(t)
        
        a = np.array(xyz)
        
        if minmax == min:
            a[:,0:1] = np.min(a[:,0:1])
        else:
            a[:,0:1] = np.max(a[:,0:1])
        return a, np.array(xyz)
    
    
    
    def extremetangent(h, gcal, f, normal, percentage=1):
        
        from scipy.optimize import minimize, minimize_scalar, root, basinhopping
        
        def func_vec(x0, h, gcal, f, normal):
            k  = 2*np.pi*h
            gg = np.sum([ f[i]* np.sqrt(1- ((x0[i]*normal[i])/(k*f[i]))**2 ) for i in range(len(f))])
            return [gcal-gg]*len(x0)
        
        def funcjac_vec(x0, h, f, n):
            k     = 2*np.pi*h
            gradg = [ f[i]*x0[i]*((n[i]/(k*f[i]))**2)*(1-((x0[i]*n[i])/(k*f[i]))**2)**(-1/2) for i in range(len(f)) ]
            return gradg
        
        k  = 2*np.pi*h  ; k1 = 1/k
        
        lam_max  = [ k*(f[i]/normal[i]) if (normal[i] >1E-5) else k*f[i]*normal[i] for i in range(len(f))]
        lam_mask = np.ma.masked_equal(lam_max, 0.0, copy=False)
        
        x0 = [(ik - ik%percentage)-1  if ik !=0 else normal[ci]*ik for ci, ik in enumerate(lam_max) ]
        
        res    = root(func_vec, x0, args=(h, gcal, f, normal), options={'maxiter':100,'gtol': 1e-16, 'disp': True}).x
        
        midpnt = [ k1*np.arcsin((res[i]*normal[i])/(k*f[i])) for i in range(len(f))]
        
        return [res, midpnt]
    
    def plane_eq2n(pnt):
    
        p1 = np.array(pnt[0]) ;  p2 = np.array(pnt[1]) ; p3 = np.array(pnt[2])
        
        # These two vectors are in the plane
        v1 = p3 - p2
        v2 = p1 - p2
        
        # the cross product is a vector normal to the plane
        cp = np.cross(v1, v2)
        cp = np.array(cp, dtype=float)
        cp /= np.linalg.norm(cp)
        
        a, b, c = cp
        
        a = a if ~np.isnan(a) else 0.5/1
        b = b if ~np.isnan(b) else 0.5/1
        c = c if ~np.isnan(c) else 0.5/1
        
        n = np.array([a, b, c], dtype = float)
        d =  np.abs(np.dot(n, p1))
        
        return np.abs(np.array([a, b, c])), d
    
    def cheiso(l, xexp, f, normal, distance, j=2, n=500, s=1):
        
        lspace = np.linspace(0, 1/(2*l), n)
        gx, gy = np.meshgrid(lspace, lspace)
        g      = np.abs(g(l, xexp, f))
        gz     = np.zeros_like(gx)
        gzp    = hsurf_g(l, [gx, gy, gz], f, g, j, s=s)
        
        o      = get_mitd(l, normal, distance, scom=np.array([[1,1,1]]), dlis=np.array([[0,0,0]]), imax=np.max(lspace))
        points = np.column_stack((gx.ravel(), gy.ravel(), gzp.ravel()))
        x      = points[~np.isnan(points).any(axis=1)]
        check  = [i in o for i in x]
        
        if ~np.all(check):
            inx = np.where(~np.array(check))[0]
            dr  = [np.dot(normal,x[i]) for i in inx]
            return False, [np.min(dr), np.max(dr)]
        else:
            return [True]
        return
    
    def getxyz(l, f, g):
        xyz = []
        
        for i in range(len(f)-1):
            for jj in [0, 1/(2*l)]:
                t = [0]*len(f)
                t[len(f)-1] = jj
                z = hsurf_g(l, t, f, gi, j=i, s=1)
                
                if ~np.isnan(z):
                    t[i] = z
                else:
                    t[i] = 0.5/l
                    z = hsurf_g(l, t, f, gi, j=i-1, s=1)
                    t[i-1] = z
                
                xyz.append(t)
        a = np.copy(xyz)
        a = a[a[:, 1].argsort()][::-1]
        
        if  np.all( a[1][0:2] <= a[0][0:2]):
            a[0][0:2] = a[1][0:2]
        elif np.all( a[1][0:2] >= a[0][0:2]):
            a[1][0:2] = a[0][0:2]
        else:
            a[0][0:2] = a[0][0:2]
        
        if  np.all( a[2][0:2] <= a[3][0:2]):
            a[3][0:2] = a[2][0:2]
        elif np.all( a[2][0:2] >= a[3][0:2]):
            a[2][0:2] = a[3][0:2]
        else:
            a[2][0:2] = a[2][0:2]
        return np.array(a), np.array(xyz)
    
    def get_pnts(l, f, gi):
        tot_coor=[]
        for i in range(len(f)):
            tem_coor    = np.zeros(len(f))
            tem_coor[i] = hsurf_g(l, tem_coor, f, gi, i, s=1)
            tot_coor.append(tem_coor)
        return tot_coor

    def remaining(l, gi, f, n_select, di):
        
        mp =  np.abs(extremetangent(l, gi, f, n_select, percentage=5.)[1])
        do = np.abs(np.dot(n_select, mp))
        
        distance = [di, do]
        dok=cheiso(l, xexp, f, n_select, distance, j=2, n=500, s=1)
        
        if not dok[0]:
            do = dok[1][1] if ~dok[0] and do <dok[1][1] else do
            di = dok[1][0] if ~dok[0] and di >dok[1][0] else di
        
        distance  = [di, do]
        
        return distance
    
    pnt = get_pnts(l, f, gi)
    
    count_Nan = np.count_nonzero(np.isnan(pnt))
    
    if   count_Nan == 0:       ### case 1 isosurface
        
        k   = 2*np.pi*l   ;     xp  = (1/k)*np.arccos(gi/np.sum(f))
        p4  = [xp]*len(f)
    
        pnt = np.vstack([np.array(pnt),p4])
    
        ni, di = plane_eq2n(pnt)
        n_select = ni
        
        distance_all = remaining(l, gi, f, n_select, di)    
    
    elif count_Nan == 1:       ### case 2 isosurface
        
        pi, pext  = getxyz(l, f, gi)
        
        n_ext, dext = plane_eq2n(pext)
        ni, di      = plane_eq2n(pi)
        n_select    = n_ext
        
        distance_all = remaining(l, gi, f, n_select, di)
        
    elif count_Nan > 1:       ### case 3 isosurface
        
        pi, pext  = getxyz_opt3(l, f, g, minmax=min)
        
        n_ext, dext = plane_eq2n(pext)
        ni, di      = plane_eq2n(pi)
        n_select    = n_ext
        
        di = di if di !=0 and di<dext else dext
        
        distance_all = remaining(l, gi, f, n_select, di)
        
    else:       ### NO isosurface
        
        print("--> count_Nan is ", count_Nan,". Wired isosurface check linearization routine. \
                    predicted points are : \n", pnt)
    
    return [n_select, distance_all]

def fn_getintersections(h,a,xexp,fname,count):
    
    s  = []
    
    for j in range(h-1):
        #print("Doing for j's upto :: ", j+1," with j = ",j+2)
        try:
            if j == 0:
                ss = a[j].intersection(a[j+1])
            else:
                ss = s[-1].intersection(a[j+1])
        except:
            fname.write(f"Pair-{count}: TopologyException error for x = {xexp} for reflection {j+1}\n")
            continue
        
        if not ss:
            #print("===> ss is empty for j = ", j+2)
            ss=s[-1]
        
        s.append(ss)
        
    return (s, j)

def sample_floats(n, low, high, k=1):
    """ Return a k-length list of unique random floats
        in the range of low <= x <= high
    """
    result = []
    seen   = set()
    for j in range(n):
        temp   = []
        for i in range(k):
            #print(j, i)
            x = random.uniform(low, high)
            while x in seen:
                x = random.uniform(low, high)
            seen.add(x)
            temp.append(x)
            #print("New :",j, i, temp)
            
        result.append(temp)
    return result

def readoldsolution(pairID, fname):
    
    with h5py.File(fname, 'r') as file:
        
        p=file.get('allsolution/Pair'+str(pairID))
        polyold=[]
        
        for ic, ds in enumerate(p.keys()):
            if ic < int(len(p.keys())/2):
                db='b'+re.split('(\d+)',ds)[1]
                da=np.array(p[ds])
                polyold.append(pc.Polytope(np.array(p[ds]), np.array(p[db])))
    
    return polyold

def wrtdata(fname, rc, volume, err, final, extremepnts,volAsym):
    with h5py.File(fname, 'a') as f:
       
    ### Writing volume information in file 
        if 'vol' in f:
            v = str('/vol/')+str('v') +str(rc)
            f.create_dataset(v,  data=volume,  dtype='float64')
        else:
            gvol = f.create_group('vol')
            v = str('/vol/')+str('v') +str(rc)
            f.create_dataset(v,  data=volume,  dtype='float64')
    
    ### Writing error information in file 
        if 'error' in f:
            derr = str('/error/')+str('err') +str(rc)
            f.create_dataset(derr,  data=err,  dtype='float64')
        else:
            gerror = f.create_group('error')
            derr = str('/error/')+str('err') +str(rc)
            f.create_dataset(derr,  data=err,  dtype='float64')
        
    ### Writing polytope information in file 
        if 'polytope' in f:
            pa=str('/polytope/')+str('pa') +str(rc)
            pb=str('/polytope/')+str('pb') +str(rc)
            f.create_dataset(pa, data=final.A, dtype='float64')
            f.create_dataset(pb, data=final.b, dtype='float64')
        else:
            gpolytope = f.create_group('polytope')
            pa=str('/polytope/')+str('pa') +str(rc)
            pb=str('/polytope/')+str('pb') +str(rc)
            f.create_dataset(pa, data=final.A, dtype='float64')
            f.create_dataset(pb, data=final.b, dtype='float64')
        
    ### Writing extreme points of polytope information in file 
        if 'extreme' in f:
            ext=str('/extreme/')+str('pa') +str(rc)
            f.create_dataset(ext, data=extremepnts, dtype='float64')
        else:
            gpolytope = f.create_group('extreme')
            ext=str('/extreme/')+str('pa') +str(rc)
            f.create_dataset(ext, data=extremepnts, dtype='float64')
    
    ### Writing sum of all volums in Asym part 
        if 'total_volume_in_Asym' in f:
            v_asym = str('/total_volume_in_Asym/')+str('pair') +str(rc)
            f.create_dataset(v_asym,  data=volAsym,  dtype='float64')
        else:
            gerror = f.create_group('total_volume_in_Asym')
            v_asym = str('/total_volume_in_Asym/')+str('pair') +str(rc)
            f.create_dataset(v_asym,  data=volAsym,  dtype='float64')
    
    
    return

def wrtcoor(fname, pairs):
    
    ### Writing polygenerated coordinats to file 
    
    import h5py
    
    with h5py.File(fname, 'a') as f:
        for i, k in enumerate(pairs):
            ksort = np.sort(k)[::-1]
            
            if 'generatedcoordinate' in f:
                co = str('/generatedcoordinate/')+str(i)
                f.create_dataset(co, data=ksort,  dtype='float64')
            else:
                gco = f.create_group('generatedcoordinate')
                co  = str('/generatedcoordinate/')+str(i)
                f.create_dataset(co, data=ksort,  dtype='float64')
            
            if 'unsortedcoordinate' in f:
                co = str('/unsortedcoordinate/')+str(i)
                f.create_dataset(co, data=k,  dtype='float64')
            else:
                gco = f.create_group('unsortedcoordinate')
                co  = str('/unsortedcoordinate/')+str(i)
                f.create_dataset(co, data=k,  dtype='float64')
    return

def wrtvolume(fname, rc, volume, dx, dy, final):
    import h5py
    with h5py.File(fname, 'a') as f:
        v =str('v') +str(rc)
        dxx=str('dx')+str(rc)
        dyy=str('dy')+str(rc)
        
        f.create_dataset(v,  data=volume,  dtype='float64')
        f.create_dataset(dxx, data=dx, dtype='float64')
        f.create_dataset(dyy, data=dy, dtype='float64')
        
        pa=str('s')+str(rc)+str('a')
        pb=str('s')+str(rc)+str('b')
        
        f.create_dataset(pa, data=final.A, dtype='float64')
        f.create_dataset(pb, data=final.b, dtype='float64')
    return

def wrtallsolution(fname, rc, solutionall):
    
    import h5py
    
    def wrtfile(rc, file, sa):
        count=0
        for cou, ip in enumerate(sa):
            if type(ip) is pc.Polytope:
                
                allpa=str('/allsolution/Pair')+ str(rc)+ str('/a')+ str(count)
                allpb=str('/allsolution/Pair')+ str(rc)+ str('/b')+ str(count)
                
                file.create_dataset(allpa, data=ip.A, dtype='float64')
                file.create_dataset(allpb, data=ip.b, dtype='float64')
                count += 1
                
            elif type(ip) is pc.Region:
                for iq in ip:
                    count += 1
                    allpa=str('/allsolution/')+ str('/Pair/')+ str(rc)+ str('/a')+ str(rc)+ str(count)
                    allpb=str('/allsolution/')+ str('/Pair/')+ str(rc)+ str('/b')+ str(rc)+ str(count)
                    file.create_dataset(allpa, data=iq.A, dtype='float64')
                    file.create_dataset(allpb, data=iq.b, dtype='float64')
        
    with h5py.File(fname, 'a') as file:
        
        #---> Writing extreme points of polytope information in file 
        if 'allsolution' in file:
            grp=file['/allsolution/']
            sgp=str('Pair')+ str(rc)
            sg = grp.create_group(sgp)
            wrtfile(rc, file, solutionall)
            
        else:
            gpolytope = file.create_group('allsolution')
            spg=str('Pair')+ str(rc)
            sg = gpolytope.create_group(spg)
            
            wrtfile(rc, file, solutionall)
        
    return

def sort_mc(solulist):
    sl = []
    Asym = np.array([[-1,0,0],[0,-1,0],[0,0,-1],[1,0,0],[0,1,0],[0,0,1],[-1,1,0],[0,-1,1],[-1,-1,0] ]) # 
    bsym = np.array([0,0,0, 0.5,0.5,0.5,  0,0,0])
    psym=pc.Polytope(Asym, bsym)
        
    for i in solulist:
        for j in i:
            xg=np.mean(pc.extreme(j), axis=0)
            if xg in psym:
                sl.append(j)
    
    return (pc.Region(sl))

def intersection_mc(h, polylist):    
    s = []
    
    A = np.array([[-1,0,0],[0,-1,0],[0,0,-1],[1,0,0],[0,1,0],[0,0,1],[-1,1,0],[0,-1,1],[-1,-1,0] ]) # 
    b = np.array([0,0,0, 0.5,0.5,0.5,  0,0,0])
    p = pc.Polytope(A, b)
    
    for j in range(h-1): 
        try:
            if j == 0:
                s1 = find_intersection(polylist[j], polylist[j+1])
                #s.append(sort_mc(s1))
                s.append(p.intersect(s1))
            else:
                #s1=find_intersection(polylist[j+1],s[-1])
                s1=find_intersection( s[-1], polylist[j+1])
                s.append(s1)
        except:
            continue
    return(s)

def fn_write(fn, data):
    
    dimension, r, c = np.shape(data)
    
    for a in data:
        countr=0
        for i in a:
            countc=0
            for j in i:
                if countr <(r-1):
                    fn.write("%2.8f \t"%(j))
                else:
                    if countc <(c-1):
                        fn.write("%2.8f \t"%(j))
                    else:
                        fn.write("%2.8f\n"%(j))
                countc += 1
            countr += 1    
    return 

def fn_signcombination(r):
    '''
    Input
    
    r:
      this fn requires only the dimension of the required array
      r = 2 for 2D case , 3 for 3D case and so on 
    
    Output
    
    f:
      array f contains all possible combination of signs
    
    '''
    f=[]
    for i in range(1, r+1):
        t = [-1]*i+[1]*(r-i)
        w = set(permutations(t))
        for u in w:
            f.append(u)
    return np.array(f)

def fn_repeat(p, d, f, imin, imax):
    #print("fn_repeat(p, d, f, imin, imax) : ", f)
    pts =[]
    inx =np.argwhere(d != 0)
    nz  =np.count_nonzero(d)
    
    if nz == 0:
        e1=np.copy(p)
        pts.append(e1)
        
    if nz != 0:
        r,c = np.shape(p)
        
        if (nz != len(d)):
            
            if (np.all((d[inx[:,0]]+p[:,inx[:,0]])>=imin) and np.all((d[inx[:,0]]+p[:,inx[:,0]])<=imax)):
                              
                if (nz == 1):
                    e2=np.copy(p)
                    e2[:,inx[:,0]]=e2[:,inx[:,0]]+d[inx[:,0]]
                    pts.append(e2)
            
        if (np.all((d[inx[:,0]]-p[:,inx[:,0]])>=imin) and np.all((d[inx[:,0]]-p[:,inx[:,0]])<=imax)):
            
            e4=np.copy(p)
            e4[:,inx[:,0]]=d[inx[:,0]]-p[:,inx[:,0]]
            pts.append(e4)
            
            if (nz >1):
                for j in f:
                    e4a=np.copy(p)
                    e4a=e4a*j
                    
                    e4a[:,inx[:,0]]=d[inx[:,0]]-e4a[:,inx[:,0]]
                    
                    if (np.all(e4a>=imin) and np.all(e4a<=imax)):
                        pts.append(e4a)
        
    return pts

def fn_mesh(l, coordinates, imax):
    
    c = np.linspace(0,imax,int(2*l*imax+1) )
    
    k = [c, c]*len(coordinates)
    k = k[0:len(coordinates)]
    
    j = np.meshgrid(*k)
    
    [*dim] = np.shape(j)
    
    f1=(np.array([j[i].reshape(-1,1) for i in range([*dim][0])]))
    f2=np.hstack([f1[i] for i in range([*dim][0])])
    
    meshlist=np.array(f2)
    
    return meshlist

