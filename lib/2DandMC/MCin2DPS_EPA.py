
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
from .x2Dlinearizer2D import double_segment_EPA, single_segment_EPA
from .tessellator import getmesh

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

def linrep(h, f, pnt, meshgrid, imin=0, imax=0.5):
    
    def getsigncom(r):
        scom=[]
    
        for i in range(0, r+1):
            t = [-1]*i+[1]*(r-i)
            w = set(permutations(t))
            for u in w:
                scom.append(u)
        return np.array(scom)
    
    plist   = []
    signcom = getsigncom(len(f))
    
    for i in signcom:
        pinner  = pnt*i
        plist.append(pinner)
        
        poutter = np.flip(pinner, axis=1)
        plist.append(poutter)
    
    pfinal = []
    
    for j in np.array(meshgrid):
        
        for ii in plist:
            ji = j+ii
            if np.all(ji<=imax) and np.all(ji>=imin):
                pfinal.append(ji)
    
    return pfinal


def isosurface_newEPA(h, xexp, f, j, fname, SSorDS='DS', IorG='amplitude', imin=0, imax=0.5):
    
    bp = Polygon([[0,0],[0.5,0 ],[0.25,0.25]])
    
    polygon = []
    
    tlinear_t, tpoly_t = 0, 0
    for l in range(1,h+1):
        
        pl = [] ; dlist = []
        
        t1=datetime.now()
        
        meshlist = getmesh(l, xexp, imax)
        
        if IorG == 'amplitude':
            gi    = np.abs(g(l, xexp, f))
                      
            if SSorDS=='double':
                #pnts  = linear_JONOPT_EPA(gi, l, xexp, f, error=0)
                pnts  = double_segment_EPA(gi, l, f, error=0)
            
            if SSorDS=='single':
                #pnts  = linear_JONAS_EPA(gi, l, f, error=0)
                pnts  = single_segment_EPA(gi, l, f, error=0)
                        
            tlinear=datetime.now()
            tlinear_t += (tlinear.timestamp() - t1.timestamp())
            
            for meshid in meshlist:
                oo=np.cos(2*np.pi*l*meshid)
                if (np.all( np.sign(oo) == np.sign(g(l, xexp, f)) )):
                    pt = Point(meshid)
                    if pt.touches(bp) or bp.contains(pt) or bp.covers(pt):
                        dlist.append(meshid)
            
            if l==1:
                #dlist0 = np.delete(dlist, 1, 0)
                ph = linrep(l, f, pnts, dlist, imin=0, imax=0.5)
                pl.append(Polygon(ph[0]))
                polyFirstRO=Polygon(ph[0])
            else:
                ph = linrep(l, f, pnts, dlist, imin=0, imax=0.5)
                for pp in ph:
                    if  Polygon(pp).intersection(polyFirstRO):
                        fn_write(fname, [pp])
                        pl.append(Polygon(pp))
            
            tpoly=datetime.now()
            tpoly_t += (tpoly.timestamp() - tlinear.timestamp())
            
        elif IorG == 'intensity':
                        
            I     = F(l, xexp, f)**2
            
            if SSorDS=='double':
                #pnts  = linear_JONOPT_EPA(np.sqrt(I), l, xexp, f, error=0)
                pnts  = double_segment_EPA(np.sqrt(I), l, f, error=0)
            if SSorDS=='single':
                #pnts  = linear_JONAS_EPA (np.sqrt(I), l, f, error=0)
                pnts  = single_segment_EPA(np.sqrt(I), l, f, error=0)
            
            tlinear=datetime.now()
            tlinear_t += (tlinear.timestamp() - t1.timestamp())
            
            for meshid in meshlist:
                oo=np.cos(2*np.pi*l*meshid)
                if (np.all(np.sign(oo) == 1) or np.all(np.sign(oo) == -1)):
                    pt = Point(meshid)
                    if pt.touches(bp) or bp.contains(pt) or bp.covers(pt):
                        dlist.append(meshid)
            
            if l==1:
                #dlist0 = np.delete(dlist, 1, 0)                
                ph = linrep(l, f, pnts, dlist, imin=0, imax=0.5)
                pl.append(Polygon(ph[0]))
                polyFirstRO=Polygon(ph[0])
            else:
                ph = linrep(l, f, pnts, dlist, imin=0, imax=0.5)
                for pp in ph:
                    if Polygon(pp).intersection(polyFirstRO):
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
   
def intersect_new(polycollection, xexp, fname, count):
    
    s  = []
    
    for j in range(len(polycollection)):
        
        try:
            if j == 0:
                asym = Polygon([[0,0],[0.25,0.25],[0.5,0]])
                ss   = polycollection[j][1].intersection(asym)
            else:
                ss   = s[-1].intersection(polycollection[j][1])
        except:
            fname.write('Pair-{} : TopologyException error for x1 = {:2.4} and x2 = {:2.4} at h = {}\n'.format(count,xexp[0], xexp[1], polycollection[j][0]))
            continue
        
        if not ss:
            print("===> ss is empty for j = ", j+1)
            ss=s[-1]
        
        s.append(ss)
        
    return s

def writepolygon_amplitude(fname, pair, polys):
    
    sr = np.argsort(pair)[::-1]
        
    if polys.geom_type == 'MultiPolygon':
        fname.write("%5g\t"%(len(polys)))
        for i in polys:
            x, y = i.exterior.coords.xy
            cen  = np.array([np.mean(x), np.mean(y)])
            if (i.contains(Point(pair)) or i.touches(Point(pair))):# or Point(pair).distance(i) <1E-3):
                for xl in range(len(x)):
                    xf = np.array([x[xl],y[xl]])
                    fname.write("%2.12f\t\t%2.12f\t\t"%(xf[0],xf[1]))
    else:
        fname.write("1\t")
        
        x, y = polys.exterior.coords.xy
        
        for xl in range(len(x)):
            xf = np.array([x[xl],y[xl]])
            #xf = xf[sr.argsort()]
            fname.write("%2.12f\t\t%2.12f\t\t"%(xf[0],xf[1]))
            #fname.write("%2.12f\t\t%2.12f\t\t"%(x[xl],y[xl]))
        #fname.write("\n")
    return ()

def writepolygon_EPA(fname, pair, polys):
    
    sr = np.argsort(pair)[::-1]
    
    if polys.geom_type == 'MultiPolygon':
        fname.write("%5g\t"%(len(polys)))
        for i in polys:
            x, y = i.exterior.coords.xy
            cen  = np.array([np.mean(x), np.mean(y)])
            if (i.contains(Point(pair)) or i.touches(Point(pair))):# or Point(pair).distance(i) <1E-3):
                
                for xl in range(len(x)):
                    xf = np.array([x[xl],y[xl]])
                    fname.write("%2.12f\t\t%2.12f\t\t"%(xf[0],xf[1]))
    else:
        fname.write("1\t")
        x, y = polys.exterior.coords.xy
        for xl in range(len(x)):
            xf = np.array([x[xl],y[xl]])
            fname.write("%2.12f\t\t%2.12f\t\t"%(xf[0],xf[1]))
    return ()




def MC2DPS_EPA(noofpair: int=1, noofRO: int=4, SSorDS: str='double', IorG: str='amplitude', imax=0.5) -> None:
    
    """
    This module carry out MC simulation in 2D PS.
    Arg:
        noofpair : number of atomic pairs you want to generate
        noofRO   : total number of reflection orders to consider in the calculation
        SSorDS   : either 'double' or 'single'. selects single or double segment linearization
        IorG     : selects either 'amplitude' or 'intensity' method
    """    
    
    
    TSI=datetime.now()
    fpath = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop', datetime.now().strftime('2DMCresult-'+'%Y-%m-%d-%H%M%S'))
    
    if os.path.isdir(fpath):
        print("\x1b[0;34m===> Output files will be saved to \x1b[1;33m", fpath, "\x1b[0;34m location\n" )
    else:
        os.mkdir(fpath)
        print("\x1b[0;34m===> Dir \x1b[1;33mresults \x1b[0;34mis created. Output files will be saved to ", fpath," location\n\n" )
    
    #---> Generating required no of random positions in list
    rp = np.random.uniform(0.0, 0.5, size=(noofpair, 2))
        
    for h in range(2,noofRO+1,1):
        
        print(f"\n\x1b[1;31m================== h={h} =============")
        
        TS=datetime.now()
        
        ftime    = open(os.path.join(fpath,'timeinfo_I_JONOPT_%g.dat'%(h)), "at+")
        fcoor    = open(os.path.join(fpath,'allcoordinates_I_JONOPT_%g.dat'%(h)), "at+")
                    
        bpoly=Polygon([[0,0],[0.5,0 ],[0.25,0.25]])
        
        for coun, pairing in enumerate(rp):
            
            count = 1+coun
            tinfo = [] ; tinfo.append(h) ; tinfo.append(count)
            
            f     = [1.0, 1.0]
            j     = 1
            
            #### checking the pai to be in asym
            pairk = np.sort(pairing)[::-1]
            
            ll = 1 ; gi = g(ll, pairk, f)
            
            if bpoly.contains(Point(pairk)):
                pair = pairk
            else:
                pair = np.sort(0.5-pairk)[::-1]
                    
            print(f"\x1b[1;34m===> Pair-{count} Current pair:{pairing} \x1b[0m",end ="...\n")
            
            #### Step 1: Get isomatte. Call "isosurfs(h,pair,f,j,fname)" fn
            
            fstep1=open(os.path.join(fpath,'Pair-%g-jonaspnts_x-%1.2g-%1.2g.dat'%(count,pair[0],pair[1])), "wt+")  
            
            polycollection, tlinear_t, tpoly_t = isosurface_newEPA(h, pair, f, j, fstep1, SSorDS, IorG, imin=0, imax=0.5)
            
            fstep1.close()
            
            tlinearize=datetime.now()
            tinfo.append(tlinear_t) ; tinfo.append(tpoly_t) 
            
            #### Step 3: Get intersections. Call "getintersections(h,a,pair,fname)"
            fstep3 = open(os.path.join(fpath,'zerrorinfo.txt'), "at+")
            
            solutions = intersect_new(polycollection, pair, fstep3, count)
            
            tintesect=datetime.now()
            tinfo.append(tintesect.timestamp() - tlinearize.timestamp())
            
            #### Step 4: Get write final soultion. Call "writepolygons(fname, polys)"
            
            if (np.shape(solutions)[0]) != 0:
                
                fstep4=open(os.path.join(fpath,'Pair-%g-solutionfor_x-%1.2g-%1.2g.dat'%(count, pair[0],pair[1])), "wt+")
                
                finalsol=solutions[-1]
                
                fcoor.write("%2.12f\t%2.12f\t%2.12f\t%2.12f\t%2.12f\t\t"%( pairing[0],pairing[1],pair[0], pair[1], finalsol.area))
                
                writepolygon_EPA(fcoor, pair, finalsol)
                
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