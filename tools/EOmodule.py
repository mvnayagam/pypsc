## Import basic libs
import scipy, sys, os, re
import numpy as np
from pathlib import Path
from scipy.optimize import fsolve
from itertools import permutations, combinations

import shapely
from shapely.geometry import Polygon, mapping, Point
#from descartes import PolygonPatch
from shapely.validation import make_valid
from shapely.ops import unary_union

from psc.lib.gspacer import g, F, hsurf_F, hsurf_g, hsurf_F2

## My defs'

def fn_getEO(h, xcoor, f):
    eo   = []
    
    for l in range(1,h+1,1):
        gi = np.abs(g(l, xcoor, f))
        eo.append([gi, l])
    
    eo    = np.array(eo)
    eoh   = eo[eo[:, 0].argsort()][::-1]
    
    Is    = eoh[:,1]
    noofI = int(np.ceil(len(eoh[:,1])/2))
    
    Is_high = Is[0:noofI]
    Is_low  = Is[noofI:]
    
    return (Is_high, Is_low)

def multistrip(a,b,pnts):
    d=[]
    for i in range(a,b,1):
        d1=Polygon([(pnts[i][j], pnts[i][j+1]) for j in range(0,8,2)])
        d.append(d1)
    return shapely.geometry.MultiPolygon([poly for poly in d])

def fn_getpolygon(hsi, p, imax=0.5):
    
    r1, r2 = 0, 0
    a = []
    
    for i in hsi:
        r2=r2+i*i*(4*imax)**2
        aa = multistrip(int (r1), int (r2), p)
        
        try:
            aa = unary_union(aa)
        except:
            print("AssertionFailedException occured for RO h=", i, "trying with make_valid")
            aa = make_valid(aa)
        
        a.append(aa)
        r1=np.copy(r2)
    return a


def fn_intersection(a): # somehow same as getintersections(h,a,xexp,fname,count)
    s = []
    
    for j in range(len(a)-1):
        try:
            if j == 0:
                ss = a[j].intersection(a[j+1])
            else:
                ss = s[-1].intersection(a[j+1])
        except:
            print("===> \x1b[0;32mTopologyException error occured. Skipping and Continue at h=%i \x1b[0m"%(j+1))
            continue
            
#         if not ss:
#             print("===> ss is empty for j = ", j+2)
#             ss = s[-1]
        s.append(ss)
    
    return s

def getintersections(h,a,xexp,fname,count):
    
    s  = []
    
    for j in range(h-1):
        #print("Doing for j's upto :: ", j+1," with j = ",j+2)
        try:
            if j == 0:
                ss = a[j].intersection(a[j+1])
            else:
                ss = s[-1].intersection(a[j+1])
        except:
            fname.write('Pair-{} : TopologyException error for x1 = {:2.4} and x2 = {:2.4} at h = {}\n'.format(count,xexp[0], xexp[1], (j+1)))
            continue
        
        if not ss:
            #print("===> ss is empty for j = ", j+2)
            ss=s[-1]
        
        s.append(ss)
        
    return s, j


def fn_get_orderedI(h, xexp, f, skip12=True):
    eo   = []
    
    if skip12:
        
        for l in range(3,h+1,1):
            gi = np.abs(g(l, xexp, f))
            eo.append([gi, l])
            
        eo  = np.array(eo)
        #eoh = eo[eo[:, 0].argsort()][::-1]
        eoh = eo[eo[:, 0].argsort()]
        Is  = eoh[:,1]
    
    else:
        
        for l in range(1,h+1,1):
            gi = np.abs(g(l, xexp, f))
            eo.append([gi, l])
            
        eo  = np.array(eo)
        #eoh = eo[eo[:, 0].argsort()][::-1]
        eoh = eo[eo[:, 0].argsort()]
        Is  = eoh[:,1]
        
    return eoh, Is

def writesolution(fcoor, polys):
    
    if polys.geom_type == 'MultiPolygon':
        
        for i in polys:
            x, y = i.exterior.coords.xy
            pseudosol=list(zip(x,y))
            
            for ii in range(len(pseudosol)):
                if (isInside(pseudosol[ii])):
                    for xl in range(len(x)-1):
                        fcoor.write("%2.12f\t %2.12f\t"%(x[xl], y[xl]))
                    break
    else:
        x, y = i.exterior.coords.xy
        
        if (isInside([x[0], y[0]])):
            for xl in range(len(x)-1):
                xy=np.sort([x[xl], y[xl]])
                fcoor.write("%2.12f\t %2.12f\t"%(xy[0], xy[1]))
    
    return

def pseudosolution(x,y,fnpoly):  
    for xl in range(len(x)):        
        if xl == len(x)-1:
            fnpoly.write("%2.12f\t %2.12f\n" %(x[xl], y[xl]))
        else:
            fnpoly.write("%2.12f\t %2.12f\t" %(x[xl],y[xl]))
    return

def realsolution(x,y,fcoor):
    for xl in range(len(x)):
        if xl == len(x)-1:            
            fcoor.write("%2.12f\t %2.12f\n"%(x[xl], y[xl]))
        else:
            fcoor.write("%2.12f\t %2.12f\t"%(x[xl], y[xl]))
    return

def finesortsolu(x, icentroid):
    dx=np.abs(x[0] - icentroid.x) 
    dy=np.abs(x[1] - icentroid.y)
    
    if dx<1E-3 or dy <1E-3:
        return True
    else:
        return False

def sortsolu(finals, fcoor, xexp):
    
    if finals.geom_type == 'MultiPolygon':
        for i in finals:
            x, y = i.exterior.coords.xy
            
            for ii in range(len(x)):
                if (isInside([x[ii],y[ii]])):
                    finesortsolu(xexp, i.centroid)
                    fcoor.write("%2.12f\t %2.12f\t %2.12f\t "%(xexp[0], xexp[1], i.area))
                    realsolution(x, y, fcoor)
                    break
    else:
        x, y = finals.exterior.coords.xy
        
        for ii in range(len(x)):
            if (isInside([x[ii],y[ii]])):
                realsolution(x, y, fcoor)
                break
    return

def find_interception(x,y,m):
    return y-m*x

def findp3x(x, m, g=1.5, h=1):
    return m+np.sin(2*np.pi*h*x)/np.sqrt(1- (g - np.cos(2*np.pi*h*x))**2) 

def findp3y(x,h,g):
    return 1/(2*np.pi*h)*np.arccos(g-np.cos(2*np.pi*h*x))

def jonaspnts(gi,l):
    #### Jonas Area 
    k    = 2*np.pi*l
    gi   = np.abs(gi)
    #### Finding point p1 and p2  
    p1x  = (1/k)*np.arccos(gi/2) ;       #  x2* = x1*
    p1y  = p1x
    
    p2x  = (1/k)*np.arccos(gi-1)         #  x1* = pnt2 ; x2* = 0
    p2y  = 0
    
    m1   = (p2y-p1y)/(p2x-p1x)           # slope of First line
    n1   = find_interception(p2x,p2y,m1)
        
    #### Finding point p3, p4 and p5
    xini = 0.0
    p5x  = fsolve(findp3x,xini,args=(m1,gi,l), factor=1, epsfcn=1e-16, maxfev=10000000)[0]
    p5y  = findp3y(p5x,l,gi)
    
    n2   = find_interception(p5x,p5y,m1)
    
    p4x  = -n2 / (m1-1)
    p4y  = p4x
    
    p3x  = -n2/m1
    p3y  = 0
    
    
    pnt  = np.array([[p1x, p1y], [p2x, p2y], [p3x, p3y], [p4x, p4y],[ p5x, p5y]])
    
    return pnt

def jonaspnts_newa(gi,l,f):
    #### Jonas Area 
    k    = 2*np.pi*l
    gi   = np.abs(gi)
    #### Finding point p1 and p2  
    #p1x  = (1/k)*np.arccos(gi/2) ;       #  x2* = x1*
    #p1y  = p1x
    
    p1x  = (1/k)*np.arccos(gi-1)         #  x1* = pnt2 ; x2* = 0
    p1y  = 0
    
    p2y  = (1/k)*np.arccos(gi-1)         #  x1* = pnt2 ; x2* = 0
    p2x  = 0
    
    m1   = (p2y-p1y)/(p2x-p1x)           # slope of First line
    n1   = find_interception(p2x,p2y,m1)
        
    #### Finding point p3, p4 and p5
    xini = 0.0
    
    #p5x  = fsolve(findp3x,xini,args=(m1,gi,l), factor=1, epsfcn=1e-16, maxfev=10000000)[0]
    #p5y  = findp3y(p5x,l,gi)
    
    p5x = (1/k)*np.arccos(gi/np.sum(f))
    p5y = p5x #p4  = [xp]*len(f)
    
    n2   = find_interception(p5x,p5y,m1)
    
    p4y  = n2 #/ (m1-1)  # y = m1x+n2
    p4x  = 0 #p4x
    
    p3x  = -n2/m1
    p3y  = 0
    
    pnt  = np.array([[p2x, p2y], [p1x, p1y], [p3x, p3y], [p5x, p5y], [p4x, p4y]])
    
    return pnt


def writepolygons(fname, polys):
    
    for i in polys:
        x, y = i.exterior.coords.xy
        
        for xl in range(len(x)):
            #fname.write('{:10.10}\t\t{:10.10}\t\t'.format(x[xl],y[xl]))
            fname.write("%2.12f\t\t%2.12f\t\t"%(x[xl],y[xl]))
        fname.write("\n")
    
    return ()

def isInside(p, v1=np.array([0.0, 0.0]), v2=np.array([0.5,0.0]), v3=np.array([0.25, 0.25])):
    
    def get_area(vert1, vert2, vert3):
        veca = vert2-vert1
        vecb = vert3-vert1
        return 0.5*np.abs(np.cross(veca, vecb))
    
    A = get_area (v1, v2, v3)
    A1 = get_area (p, v2, v3)
    A2 = get_area (v1, p, v3)
    A3 = get_area (v1, v2, p)
    
    if(A >= A1 + A2 + A3):
        return True
    else:
        return False

def get_error_v2a(d):
    xlist=[d[i] for i in range(3,len(d), 2)]
    ylist=[d[i] for i in range(4,len(d), 2)]
    
    x_min=np.min(xlist)
    #x_min_inx=np.where(xlist == x_min)[0]
    #y_min=ylist[x_min_inx[0]]
    
    x_max=np.max(xlist)
    #x_max_inx=np.where(xlist == x_max)[0]
    #y_max=ylist[x_max_inx[0]]
    
    y_min=np.min(ylist)
    y_max=np.max(ylist)
    
    dx = np.abs(x_min-x_max)/2
    dy = np.abs(y_min-y_max)/2
    
    return (dx, dy)

def get_error_v2(d):
    xlist=d[0]
    ylist=d[1]
    
    x_min=np.min(xlist)
    #x_min_inx=np.where(xlist == x_min)[0]
    #y_min=ylist[x_min_inx[0]]
    
    x_max=np.max(xlist)
    #x_max_inx=np.where(xlist == x_max)[0]
    #y_max=ylist[x_max_inx[0]]
    
    
    y_min=np.min(ylist)
    y_max=np.max(ylist)
    
    dx = np.abs(x_min-x_max)/2
    dy = np.abs(y_min-y_max)/2
    
    return (dx, dy)

def pseudosolution(x,y,fnpoly):  
    for xl in range(len(x)):        
        if xl == len(x)-1:
            fnpoly.write("%2.12f\t %2.12f\n" %(x[xl], y[xl]))
        else:
            fnpoly.write("%2.12f\t %2.12f\t" %(x[xl],y[xl]))
    return

def realsolution(x,y,fcoor):
    for xl in range(len(x)):
        if xl == len(x)-1:            
            fcoor.write("%2.12f\t %2.12f\n"%(x[xl], y[xl]))
        else:
            fcoor.write("%2.12f\t %2.12f\t"%(x[xl], y[xl]))
    return

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
    return()

def fn_repeat(p, d, f, imin, imax):
    
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

