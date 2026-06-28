import logging
logger = logging.getLogger(__name__)

import os
import numpy as np

from datetime import datetime

from itertools import permutations

from shapely.validation import make_valid
from shapely.ops import unary_union
from shapely.geometry import Point, Polygon

from .gspacer import F, g
from .x2Dlinearizer2D import find_interception  

from .tessellator import getmesh, getsigncombination


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


def linear_JONOPT_nEPA(gi, l, f, j=1, error=0):
    
    pnt = []
    k   = 2*np.pi*l
    gi   = np.abs(gi) 
    
    p1x = (1/k)*np.arccos(gi*(1+error)/np.sum(f))
    p1y = p1x
    pnt.append([p1x, p1y])
    
    p2x = (1/k)*np.arccos((gi*(1+error)-f[1])/f[0])
    p2y = 0
    
    if np.isnan(p2x):
        p2x = 0.5/l
    
    pnt.append([p2x, p2y])
    
    m1  = (p2y-p1y)/(p2x-p1x)
    n1  = find_interception(p2x,p2y,m1)
    
    xini = 0.00
    p5   = fn_solveforx_v2(l, gi, f, m1, j, error)
    
    if ~np.all(np.isnan(p5)):
        
        if len(np.shape(p5)) > 1 :
            
            pnt = []
            
            if np.floor(p5[0,1]/p5[0,0]) <= 1 and np.floor(p5[1,1]/p5[1,0]) > 1:
                pLB=p5[0]
                pUB=p5[1]
            else:
                pLB=p5[1]
                pUB=p5[0]
            
            ### Lower Boundary (LB)
            nLB = pLB[1] - m1*pLB[0]
            
            pLB1y = 0.5/l
            pLB1x = (pLB1y - nLB)/m1
            
            pLB3x = np.abs(nLB/m1)
            pLB3y = 0
            #pLB3x = (1/k)*np.arccos((g*(1+error)-f[0])/f[1])
                        
            ### Upper Boundary (UB)
            
            nUB = pUB[1] - m1*pUB[0]
            pUB4x = np.abs(nUB/m1)
            pUB4y = 0
            
            pUB6y = 0.5/l
            pUB6x = (pUB6y - nUB)/m1
            
            ### Colloecting point in order
            pnt.append([pLB1x,  pLB1y])
            pnt.append([pLB[0], pLB[1]])
            pnt.append([pLB3x,  pLB3y])
            
            pnt.append([pUB4x,  pUB4y])
            pnt.append([pUB[0], pUB[1]])
            pnt.append([pUB6x,  pUB6y])
                        
            return np.array(pnt)
        
        elif ~np.all(np.isnan(p5)) and len(np.shape(p5)) == 1 :
            p5y = p5[1]
            p5x = p5[0]
        else:
            p5y = 0.5/l
            p5x  = (1/k)*np.arccos( gi/f[0] - (f[1]/f[0])*np.cos(k*p5y) )
    
    n2  = find_interception(p5x,p5y,m1)
    
    p4x = -n2 / (m1-1)
    p4y = p4x
    
    p3x = -n2 / m1
    p3y = 0
    
    pnt.append([p3x, p3y])
    pnt.append([p5x, p5y])
    
    
    #---> Part 2 linearization 
    
    p6x  = 0
    p6y  = (1/k)*np.arccos((gi*(1+error)-f[0])/f[1])
    
    if np.isnan(p6y):
        p6y = 0.5/l
        
    m3 = (-p6y+p1y)/(-p6x+p1x)
    
    p7 = fn_solveforx_v2(l, gi, f, m3, j, error)
    
    if ~np.all(np.isnan(p7)):
        p7x = p7[0]
        p7y = p7[1]
        
        n4   = find_interception(p7x,p7y,m3)
        p8x  = 0
        if ~np.isnan(n4) and n4<=0.5/l:
            p8y  = n4 
        elif ~np.isnan(n4) and n4>=0.5/l:
            p8y  = 0.5/l
        else:
            print("--> from def jonopt_error_v5: do not know what to do for p8y ")
        
    else:
        p7y = 0.5/l
        p7x  = (1/k)*np.arccos( gi/f[0] - (f[1]/f[0])*np.cos(k*p7y) )
        
        n4   = find_interception(p7x,p7y,m3)
        p8x  = 0
        p8y  = 0.5/l
        
        
    n4   = find_interception(p7x,p7y,m3)
    
    p9x = n4/(1-m3)
    p9y = p9x
    
    if p9y == p4y and p9x == p4x :
        pnt.append([p4x, p4y])
        pnt.append([p9x, p9y])
    else:
        if p9y > p4y and p9x > p4x :
                    
            p9x = (n2-n4) / (m3-m1)
            p9y = m3 * p9x + n4
            
            pnt.append([p4x, p4y])
            pnt.append([p9x, p9y])
            
            m49   = (p9y-p4y)/(p9x-p4x)
            n49   = p1y - m49*p1x
            
            pnewy = p9y
            pnewx = (pnewy - n49)/( m49 )
            
            
        else:
            
            p9x = (n2-n4) / (m3-m1)
            p9y = m3 * p9x + n4
            
            pnt.append([p9x, p9y])
        
            m49   = (p9y-p4y)/(p9x-p4x)
            n49   = p1y - m49*p1x
            pnewy = p9y
            pnewx = (pnewy - n49)/( m49 )
                
    pnt.append([p7x, p7y])
    pnt.append([p8x, p8y])
    pnt.append([p6x, p6y])
    pnt.append([p1x, p1y])
    
    
    return np.array(pnt)

def linear_JONAS_nEPA(gi, l, f, j=1, error=0):
    pnt  = []
    
    k    = 2*np.pi*l
    gi    = np.abs(gi)
      
    p1x  = 0
    p1y  = (1/k)*np.arccos((gi*(1+error)-f[0])/f[1])
    
    if np.isnan(p1y):
        p1y = 0.5/l
        p1x = findpx(p1y, l, g, f)
        pnt.append([p1x, p1y])
        
    else:
        pnt.append([p1x, p1y])        
    
    
    p2x  = (1/k)*np.arccos((gi*(1+error)-f[1])/f[0])
    p2y  = 0
    pnt.append([p2x, p2y]) 
    
    
    m1   = (p2y-p1y)/(p2x-p1x)
    n1   = find_interception(p2x,p2y,m1)
    
    
    p4   = fn_solveforx_v2(l, gi*(1+error), f, m1, j, error)    
    
    if ~np.all(np.isnan(p4)):
        
        if len(np.shape(p4)) > 1 :
            
            pnt = []
            
            if np.floor(p4[0,1]/p4[0,0]) <= 1 and np.floor(p4[1,1]/p4[1,0]) > 1:
                pLB=p4[0]
                pUB=p4[1]
            else:
                pLB=p4[1]
                pUB=p4[0]
            
            ### Lower Boundary (LB)
            nLB = pLB[1] - m1*pLB[0]
            
            pLB1y = 0.5/l
            pLB1x = (pLB1y - nLB)/m1
            
            pLB3x = np.abs(nLB/m1)
            pLB3y = 0
            
                        
            ### Upper Boundary (UB)
            
            nUB = pUB[1] - m1*pUB[0]
            pUB4x = np.abs(nUB/m1)
            pUB4y = 0
            
            pUB6y = 0.5/l
            pUB6x = (pUB6y - nUB)/m1
            
            ### Colloecting point in order
            pnt.append([pLB1x,  pLB1y])
            pnt.append([pLB[0], pLB[1]])
            pnt.append([pLB3x,  pLB3y])
            
            pnt.append([pUB4x,  pUB4y])
            pnt.append([pUB[0], pUB[1]])
            pnt.append([pUB6x,  pUB6y])
                        
            return np.array(pnt)
        
        elif ~np.all(np.isnan(p4)):
            p4y = p4[1]
            p4x = p4[0]
        else:
            
            p4y = 0.5/l
            p4x  = (1/k)*np.arccos( gi*(1+error)/f[0] - (f[1]/f[0])*np.cos(k*p4y) )
    
    n2   = find_interception(p4x,p4y,m1)
    
    p3x  = -n2/m1
    p3y  = 0
    
    pnt.append([p3x, p3y])
    pnt.append([p4x, p4y])
    
    p5x  = 0
    p5y  = n2
    
    if p5y > 0.5/l:
        p5y  = 0.5/l
        p5x = (p5y-n2)/m1
        pnt.append([p5x, p5y])
        
        pextra_x = 0
        pextra_y = 0.5/l
        pnt.append([pextra_x, pextra_y])
    else:
        pnt.append([p5x, p5y])
        
    return np.array(pnt)


def findpy(x, h, g, f):
    return (1/(2*np.pi*h))*np.arccos(g/f[1] - (f[0]/f[1])*np.cos(2*np.pi*h*x))

def findpx(y, h, g, f):
    return (1/(2*np.pi*h))*np.arccos(g/f[0] - (f[1]/f[0])*np.cos(2*np.pi*h*y))

def fn_solveforx_v2(l, g, f, m, j, error):
    
    k = 2 * np.pi * l
    
    i = list(range(j)) + list(range(j+1,len(f)))
    
    a = (1-m*m)/(f[j]*f[j])
    b = 2 * m*m * g /(f[j]*f[j])
    c = m*m*( 1 - (g*g) / (f[j]*f[j]) ) - np.array([ (f[ii]*f[ii]) / (f[j]*f[j]) for ii in i]).sum(axis = 0) 
    
    
    if a != 0:
        
        z1 = (-b + np.sqrt(b*b - 4*a*c))/(2*a)
        z2 = (-b - np.sqrt(b*b - 4*a*c))/(2*a)
        
        r1=(1/k)*np.arccos(z1/f[0])
        r2=(1/k)*np.arccos(z2/f[0])
        
        if ~np.isnan(r1) and np.isnan(r2):
            ry=findpy(r1,l,g,f)
            return np.array([r1, ry])
        
        elif np.isnan(r1) and ~np.isnan(r2):
            ry=findpy(r2,l,g,f)
            return np.array([r2, ry])
        
        elif np.isnan(r1) and np.isnan(r2) :
            
            return np.array([float("nan")])
        
        elif ~np.isnan(r1) and ~np.isnan(r2):
            
            r1y=findpy(r1,l,g,f)
            r2y=findpy(r2,l,g,f)
            
            #print("Multiple solution: r1, r1y= ", r1,r1y, " r2, r2y = ", r2, r2y)
            return np.array([ [r1, r1y], [r2, r2y] ])
        
        else:
            #print("I obtained weird solution. I do not know what to return r1= ", r1," r2= ", r2)
            return np.array([r1, r2])
    
    else:
        
        z1 = (-1*c/b)
        
        r1=(1/k)*np.arccos(z1/f[0])
        
        if ~np.isnan(r1):
            ry=findpy(r1,l,g,f)
            return np.array([r1, ry])
        elif np.isnan(r1):
            prx = (1/k)*np.arccos(g*(1+error)/np.sum(f))
            pry = prx
            return np.array([prx, pry])
        else:
            prx = (1/k)*np.arccos(g*(1+error)/np.sum(f))
            pry = prx
            return np.array([prx, pry])


##############################################################################################################

def linrep_nEPA(h, f, pnt, meshgrid, imin=0, imax=0.5):
    
    def getsigncomx(r):
        scom=[]
    
        for i in range(0, r+1):
            t = [-1]*i+[1]*(r-i)
            w = set(permutations(t))
            for u in w:
                scom.append(u)
        return np.array(scom)
    
    plist   = []
    signcom = getsigncomx(len(f))
    
    for i in signcom:
        pinner  = pnt*i
        plist.append(pinner)
            
    pfinal = []
    
    for jmesh in np.array(meshgrid):
        
        for ii in plist:
            ji = jmesh+ii
            if np.all(ji<=imax) and np.all(ji>=imin):
                pfinal.append(ji)
    
    return pfinal


def isosurface_nEPA(h, xexp, f, j, fname, SSorDS='DS', IorG='amplitude', imin=0, imax=0.5):
        
    polygon = []
    
    tlinear_t, tpoly_t = 0, 0
    
    for l in range(1,h+1):
        
        pl = [] ; dlist = []
        
        t1=datetime.now()
        
        signcom  = getsigncombination(len(xexp))
        meshlist = getmesh(l, xexp, imax)
        
        if IorG == 'amplitude':
            gi    = g(l, xexp, f)
            
            if SSorDS == 'double':
                pnts  = linear_JONOPT_nEPA(gi, l, f, j=1, error=0)
            if SSorDS == 'single':
                pnts  = linear_JONAS_nEPA (gi, l, f, j, error=0)
            
            tlinear=datetime.now()
            tlinear_t += (tlinear.timestamp() - t1.timestamp())
            
            for meshid in meshlist:
                oo=np.cos(2*np.pi*l*meshid)
                if (np.all( np.sign(oo) == np.sign(g(l, xexp, f)) )):
                    dlist.append(meshid)
            
            ph = linrep_nEPA(l, f, pnts, dlist, imin=0, imax=0.5)
            
            if l==1:
                #dlist0 = np.delete(dlist, 1, 0)
                ph = linrep_nEPA(l, f, pnts, dlist, imin=0, imax=0.5)
                pl.append(Polygon(ph[0]))
                polyFirstRO=Polygon(ph[0])
            else:
                ph = linrep_nEPA(l, f, pnts, dlist, imin=0, imax=0.5)
                for pp in ph:
                    if  Polygon(pp).intersection(polyFirstRO):
                        fn_write(fname, [pp])
                        pl.append(Polygon(pp))
            
            tpoly=datetime.now()
            tpoly_t += (tpoly.timestamp() - tlinear.timestamp())
            
        elif IorG == 'intensity':
                        
            I     = F(l, xexp, f)**2
            if SSorDS == 'double':
                pnts  = linear_JONOPT_nEPA(np.sqrt(I), l, f, j=1, error=0)
            if SSorDS == 'single':
                pnts  = linear_JONAS_nEPA (np.sqrt(I), l, f, j, error=0)
            
            tlinear=datetime.now()
            tlinear_t += (tlinear.timestamp() - t1.timestamp())
            
            for meshid in meshlist:
                oo=np.cos(2*np.pi*l*meshid)
                if (np.all(np.sign(oo) == 1) or np.all(np.sign(oo) == -1)):
                    dlist.append(meshid)
            
            if l==1:
                dlist0 = np.delete(dlist, 1, 0)                
                ph = linrep_nEPA(l, f, pnts, dlist0, imin=0, imax=0.5)
                pl.append(Polygon(ph[0]))
                polyFirstRO=Polygon(ph[0])
            else:
                ph = linrep_nEPA(l, f, pnts, dlist, imin=0, imax=0.5)
                for pp in ph:
                    if  Polygon(pp).intersection(polyFirstRO):
                        fn_write(fname, [pp])
                        pl.append(Polygon(pp))
            
            tpoly=datetime.now()
            tpoly_t += (tpoly.timestamp() - tlinear.timestamp())
        
        else:
             print("please select correct option for IorG. It should be either intensity of amplitude")
        
        try:
            aa = unary_union(pl)
            polygon.append([l, aa])
        except:
            #print("AssertionFailedException occured for RO h=", i, "trying with make_valid")
            aa = make_valid(pl)
            polygon.append([l, aa])
        
            
    return polygon, tlinear_t, tpoly_t

def intersect_new_nEPA(polycollection, xexp, fname, count):
    
    s  = []
    
    for j in range(len(polycollection)-1):
        
        try:
            if j == 0:
                #asym = Polygon([[0,0],[0.25,0.25],[0.5,0]])
                ss   = polycollection[j][1].intersection(polycollection[j+1][1])
            else:
                ss   = s[-1].intersection(polycollection[j+1][1])
        except:
            fname.write('Pair-{} : TopologyException error for x1 = {:2.4} and x2 = {:2.4} at h = {}\n'.format(count,xexp[0], xexp[1], polycollection[j][0]))
            continue
        
        if not ss:
            print("===> ss is empty for j = ", j+1)
            ss=s[-1]
        
        s.append(ss)
        
    return s

def writepolygon_amplitude(fname, pair, polys):
    
    sr   = np.argsort(pair)
    #pair = np.sort(pair)[::-1]
    
    if polys.geom_type == 'MultiPolygon':
        fname.write("%5g\t"%(len(polys)))
        for i in polys:
            x, y = i.exterior.coords.xy
            cen  = np.array([np.mean(x), np.mean(y)])
            pt=Point(pair)
            if (i.contains(pt) or pt.touches(i) or pt.distance(i) <1E-3):
                #np.all(np.isclose(cen, pair, rtol=5e-3, atol=1e-12))):
                for xl in range(len(x)):
                    xf = np.array([x[xl],y[xl]])
                    #xf = xf[sr.argsort()]
                    fname.write("%2.12f\t\t%2.12f\t\t"%(xf[0],xf[1]))
                #fname.write("\n")
    else:
        fname.write("1\t")
        
        x, y = polys.exterior.coords.xy
        
        for xl in range(len(x)):
            xf = np.array([x[xl],y[xl]])
            #xf = xf[sr.argsort()]
            fname.write("%2.12f\t\t%2.12f\t\t"%(xf[0],xf[1]))
            #fname.write("%2.12f\t\t%2.12f\t\t"%(x[xl],y[xl]))
        #fname.write("\n")
    return



def MC2DPS_nEPA(noofpair: int=1, noofRO: int=4, atomicscatteringfactors: list=[10, 4], SSorDS: str='double', IorG: str='amplitude', imax=0.5) -> None:
    
    """
    This module carry out MC simulation in 2D PS.
    Arg:
        noofpair : number of atomic pairs you want to generate
        noofRO   : total number of reflection orders to consider in the calculation
        atomicscatteringfactors: scattering factors of atoms. defaults to [10, 4]
        SSorDS   : either 'double' or 'single'. selects single or double segment linearization
        IorG     : selects either 'amplitude' or 'intensity' method
    """    
    
    TSI=datetime.now()
    fpath = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop', datetime.now().strftime('nEPA_2DMCresult-'+'%Y-%m-%d-%H%M%S'))
    
    if os.path.isdir(fpath):
        print("\x1b[0;34m===> Output files will be saved to \x1b[1;33m", fpath, "\x1b[0;34m location\n" )
    else:
        os.mkdir(fpath)
        print("\x1b[0;34m===> Dir \x1b[1;33mresults \x1b[0;34mis created. Output files will be saved to ", fpath," location\n\n" )
    
    #---> Generating required no of random positions in list
    rp = np.random.uniform(0.0, 0.5, size=(noofpair, 2))
    f = np.sort(atomicscatteringfactors)[::-1]
    for h in range(2, noofRO, 1):
        
        TS=datetime.now()
        
        print("\n\x1b[1;31m================== h = ",h,"=============")
        
        ftime       = open(os.path.join(fpath,'timeinfo_I_JONAS_%g.dat'%(h)), "at+")
        fcoor       = open(os.path.join(fpath,'allcoordinates_I_JONAS_%g.dat'%(h)), "at+")
        
        for coun, pairing in enumerate(rp): 
            
            count = 1+coun
            tinfo = [] ; tinfo.append(h) ; tinfo.append(count)
            
            j     = 1
            
            #---> checking the pai to be in asym
            #pairk = np.sort(pairing)[::-1]
            pairk = np.array(pairing)
            
            ll = 1 ; gi = g(ll, pairk, f)
            
            if gi>0:
                pair = pairk
            else:
                pair = 0.5-pairk
            
            print(f"\x1b[1;34m===> Pair-{count}. Current pair {pairing}\x1b[0m",end ="...\n")
            
            #### Step 1: Get isomatte. Call "isosurfs(h,pair,f,j,fname)" fn., 
            
            fstep1=open(os.path.join(fpath,'Pair-%g-jonaspnts_x-%1.2g-%1.2g.dat'%(count,pair[0],pair[1])), "wt+")  
            
            #polycollection = isosurface_nEPA(h, pair, f, j, fstep1, IorG='amplitude', imin=0, imax=0.5)
            polycollection, tlinear_t, tpoly_t = isosurface_nEPA(h, pair, f, j, fstep1, SSorDS, IorG='intensity', imin=0, imax=0.5)
            
            fstep1.close()
            
            tlinearize=datetime.now()
            tinfo.append(tlinear_t) ; tinfo.append(tpoly_t) 
            
            #### Step 3: Get intersections. Call "getintersections(h,a,pair,fname)"
            fstep3 = open(os.path.join(fpath,'zerrorinfo.txt'), "at+")
            
            solutions = intersect_new_nEPA(polycollection, pair, fstep3, count)
            
            tintesect=datetime.now()
            tinfo.append(tintesect.timestamp() - tlinearize.timestamp())
            
            #### Step 4: Get write final soultion. Call "writepolygons(fname, polys)"
            
            if (np.shape(solutions)[0]) != 0:
                
                fstep4=open(os.path.join(fpath,'Pair-%g-solutionfor_x-%1.2g-%1.2g.dat'%(count, pair[0],pair[1])), "wt+")
                
                finalsol=solutions[-1]
                
                fcoor.write("%2.12f\t%2.12f\t%2.12f\t%2.12f\t%2.12f\t\t"%( pairing[0],pairing[1],pair[0], pair[1], finalsol.area))
                
                writepolygon_amplitude(fcoor, pair, finalsol)
                
                fcoor.write("\n")
                fstep4.close()
                
                twrite=datetime.now()
                tinfo.append(twrite.timestamp()-tlinearize.timestamp())
                tinfo.append(twrite.timestamp()-TS.timestamp())
                
                ftime.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(tinfo[0],tinfo[1],tinfo[2],tinfo[3],tinfo[4],tinfo[5],tinfo[6]))
                print(tinfo,"\x1b[1;33m:)")
                
            os.remove(os.path.join(fpath,'Pair-%g-solutionfor_x-%1.2g-%1.2g.dat'%(count, pair[0],pair[1])))
            os.remove(os.path.join(fpath,'Pair-%g-jonaspnts_x-%1.2g-%1.2g.dat'%(count,pair[0],pair[1])))
        
        fstep3.close()
        fcoor.close()
        ftime.close()

    # ==========  End of MC Code  ==========
    TSF=datetime.now()
    print("\n\x1b[1;32m ---->  Total time taken in ", TSF.timestamp()-TSI.timestamp()," ( sec )")
    print("\n==========  End of MC run  ==========")
    
    return





