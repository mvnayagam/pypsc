import logging
logger = logging.getLogger(__name__)

import os
import numpy as np
import polytope as pc
from datetime import datetime

from .gspacer import g
from ..tools.createfolder import createmcdir
from .tessellator import getpolytope_EPA
from .intersector import find_intersection
from .linearizer import linearizenD_EPA
from .solutionIO import wrtcoor, wrtdata, wrtallsolution, wrttime_mc, readoldsolution

from .checklinearizer import checklinear
from .xlinearizationtools import radius_from_volume

def isosurfs_EPA(h, xexp, f, fname, verifylinearization=True):
    n  = [] 
    if h <= 2:
        for l in range(1,h+1):
            gi = g(l, xexp, f)
            normal, dist, ponts = linearizenD_EPA(l, f, np.abs(gi))
            
            n.append([l, normal, dist, np.sign(gi)])
            fname.write("%3g\t%2.6f\t%2.6f\t%2.6f\t%2.6f\t%2.6f\n"%(h, normal[0],normal[1],normal[2], dist[0],dist[1]))
            if verifylinearization:
                checklinear(l, f, gi, normal, dist, j=len(f)-1)
    else:
        gi = g(h,xexp,f)
        normal, dist, ponts = linearizenD_EPA(h, f, np.abs(gi))
        
        n.append([h, normal, dist, np.sign(gi)])
        fname.write("%3g\t%2.6f\t%2.6f\t%2.6f\t%2.6f\t%2.6f\n"%(h, normal[0],normal[1],normal[2], dist[0],dist[1]))
        if verifylinearization:
            checklinear(h, f, gi, normal, dist, j=len(f)-1)
    
    return n

def MCNDPS_EPA(dimension: int, noofpair: int, structure: list = [], noofRO: int=9, iorg: str='amplitude',
                     restart: bool=False, restarth: int=5, restartpath: str='./', imax=0.5, verifylinearization=False) -> None:
    
    TS0=datetime.now()
        
    # ---> define asymmetric part of PS
    temp = np.tril(np.ones(shape=(dimension, dimension)) , 0 )
    temp = imax*np.vstack([[0]*dimension, temp])
    asym = pc.qhull(np.array(temp))
    
    # ------------------------------------------------------------------------------------------
    # ---> Generating required no of random positions in list
    # ------------------------------------------------------------------------------------------
    if (not structure) and noofpair:
        rp = np.random.uniform(0.0, 0.5, size=(noofpair, dimension))
        print(f'---> I find noofpair variable to be true. I will create {noofpair} pairs')
        print(f'---> Generated pairs are: \n {rp}')
    elif (not noofpair) and structure:
        print(f'---> I find structure variable to be true. I use structures: \n {structure}')
        rp=structure
    else:
        print(f'---> Only structure or noofpair should be specified. I find both structure and noofpair')
        print(f'---> In this case I discard noofpair({noofpair}) variable and for further process select the structures variable: \n{structure}')
        rp=structure
            
    # ------------------------------------------------------------------------------------------
    #---> define result folder to save results
    # ------------------------------------------------------------------------------------------
    if not restart:
        
        hstart = 2
        
        #fpath = os.path.join(os.getcwd(), datetime.now().strftime('MCresult-'+'%Y-%m-%d-%H%M%S'))  # to creat MC folder within 2021_Freiberg
        fpath  = createmcdir()
        # fpath=os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop', datetime.now().strftime('MCresult-'+'%Y-%m-%d-%H%M%S')) 
        if os.path.isdir(fpath):
            print("\x1b[0;34m===> Output files will be saved to \x1b[1;33m", fpath, "\x1b[0;34m location\n" )
        else:
            os.mkdir(fpath)
            print("\x1b[0;34m===> Dir \x1b[1;33mresults \x1b[0;34mis created. Output files will be saved to ", fpath," location\n\n" )
    else:
        hstart = restarth
        fpath  = restartpath
        
    sn = []
    polyFirstRO = []
    for h in range(hstart,noofRO+1):
        
        sn2 = []
        TS=datetime.now()
               
        print("\n\x1b[1;31m=====> for h = \x1b[0m", h,"\x1b[1;31m==============================\n\x1b[0m")
                     
        fname = os.path.join(fpath,'pnew_%g.h5'%(h))
        
        if os.path.exists(fname):
            os.remove(fname)
                
        # ------------------------------------------------------------------------------------------
        # ---> write random pairs to file 
        # ------------------------------------------------------------------------------------------
        wrtcoor(fname, rp)
        
        for rc, pairs in enumerate(rp):
            tinfo = []
            f     = [1.0]*dimension
            j     = len(f)-1
            
            xexp  = np.sort(pairs)[::-1]
            ll    = 1
            sign  = np.sign(g(ll, xexp, f))
            if sign>0:
                xexp = xexp
            else:
                xexp = np.sort(0.5-xexp)[::-1]
            
            print("\x1b[1;34m===> Pair-%2g: \x1b[0m"%(rc+1),"\x1b[1;34mCurrent pair: \x1b[0m", pairs,"\x1b[1;34mSolving for: \x1b[0m", xexp)

            # ------------------------------------------------------------------------------------------
            # ---> Step 1: Do linearization 
            # ------------------------------------------------------------------------------------------
                        
            tinfo.append(h) ; tinfo.append(rc+1)
            t1=datetime.now()
            
            print("\x1b[1;32mstart:\x1b[0m",datetime.now().strftime("%H:%M:%S"),end=" ")
            
            fstep1=open(os.path.join(fpath,'Pair-%g.dat'%(rc+1)), "wt+")
            nar = isosurfs_EPA(h, xexp, f, fstep1, verifylinearization=False)
            fstep1.close()
            
            #print(f'\n nar [l, normal, dist, np.sign(gi)] : {nar}')
            tlinearize=datetime.now()
            tinfo.append(tlinearize.timestamp() - t1.timestamp())
            
            print("\x1b[1;32miso:\x1b[0m",datetime.now().strftime("%H:%M:%S"),end=" ")
                    
            # ------------------------------------------------------------------------------------------
            # ---> Step 2: Get polytope
            # ------------------------------------------------------------------------------------------
            polylist = []
            
            if not restart:
                if h <= 2:
                    for i in nar: # here i[0]->l,i[1]->normal,i[2]->distance,i[3]->amplitudesign
                        o = getpolytope_EPA(i[0], i[1], i[2], i[3], IorG=iorg, imax=0.5) # getpolytope_EPA(l, normal, d_all, amplitudesign, IorG='intensity')
                        if i[0]==1:
                            polyFirstRO.append(pc.Region([o.intersect(asym)]))
                            polylist.append(o)                        
                        else:
                            pplist=[]
                            for ipoly in o:
                                #if (asym.intersect(ipoly)):
                                if (ipoly.intersect(asym)):
                                    if (polyFirstRO[rc].intersect(ipoly)):  #if (ipoly.intersect(polyFirstRO[rc])):
                                        pplist.append(ipoly)
                            polylist.append(pc.Region(pplist))
                else:
                    pplist=[]
                    i = nar[0]
                    o = getpolytope_EPA(i[0], i[1], i[2], i[3], IorG=iorg, imax=0.5)
                    for ipoly in o:
                        if (asym.intersect(ipoly)):
                            if polyFirstRO[rc].intersect(ipoly):
                                pplist.append(ipoly)
                    polylist=pc.Region(pplist)
                    
            elif restart:
                # ---> Get polytope of first order
                hh=1
                ftemp=open(os.path.join(fpath,'temp-%g.dat'%(rc+1)), "wt+")
                nhh = isosurfs_EPA(hh, xexp, f, ftemp, verifylinearization=False)[0]
                ftemp.close()
                
                o = getpolytope_EPA(nhh[0], nhh[1], nhh[2], nhh[3], IorG=iorg, imax=0.5)
                polyFirstRO.append(pc.Region([o.intersect(asym)]))
                
                oldsolutionfile=os.path.join(fpath,'pnew_%g.h5'%(restarth-1))
                #oldsolution=readoldsolution(rc+1, oldsolutionfile)
                oldsolution=readoldsolution(rc+1, oldsolutionfile)
                
                #polylist.append(pc.Region(oldsolution))
                
                # ---> start remaining process 
                pplist=[]
                i = nar[0]
                o = getpolytope_EPA(i[0], i[1], i[2], i[3], IorG=iorg, imax=0.5)
                for ipoly in o:
                    if (asym.intersect(ipoly)):
                        if polyFirstRO[rc].intersect(ipoly):
                            pplist.append(ipoly)
                polylist=pc.Region(pplist)
            else:
                print(f'---> ERROR : start from scratch')       
            
            tgetpoly=datetime.now()
            tinfo.append(tgetpoly.timestamp() - tlinearize.timestamp())
            
            print("\x1b[1;32mpoly:\x1b[0m",datetime.now().strftime("%H:%M:%S"),end=" ")
                    
            # ------------------------------------------------------------------------------------------
            # ---> Step 3: Get intersection , write all solution
            # ------------------------------------------------------------------------------------------
            if h == 2:
                solun = find_intersection(polyFirstRO[rc], polylist[1])
                wrtallsolution(fname, rc+1, solun)
                sn2.append(solun)
            else:
                if not restart:
                    solun = find_intersection(sn[h-3][rc], polylist)
                    #solun = find_intersection_updated(sn[h-3][rc], polylist)
                else:
                    if len(sn)==0:
                        solun = find_intersection(pc.Region(oldsolution), polylist)
                        #solun = find_intersection_updated(pc.Region(oldsolution), polylist)                        
                    else:
                        solun = find_intersection(sn[-1][rc], polylist)
                        #solun = find_intersection_updated(sn[-1][rc], polylist)
                        
                wrtallsolution(fname, rc+1, solun)
                sn2.append(solun)
                        
            tintersect=datetime.now()
            tinfo.append(tintersect.timestamp() - tgetpoly.timestamp())
            
            print("\x1b[1;32mintersection:\x1b[0m",datetime.now().strftime("%H:%M:%S"),end=" ")
            
            # ------------------------------------------------------------------------------------------
            # ---> Final step: Sorting solution
            # ------------------------------------------------------------------------------------------            
                    
            print("\x1b[1;32msort sol:\x1b[0m",datetime.now().strftime("%H:%M:%S"))
                        
            def try_extreme(spolyi, retries=100):
                for _ in range(retries):
                    try:
                        #print(f'\n pc.extreme(polyi) {pc.extreme(spolyi)}')
                        return pc.extreme(spolyi)
                    except:
                        #print(f'\n pc.extreme(polyi) {pc.extreme(spolyi)}')
                        continue
                raise Exception(f"Failed to process spolyi after {retries} attempts")
            
            try:
                contains_coordinates = np.array([xexp in polyi for polyi in solun])
                print(f'contains_coordinates: {contains_coordinates}')
                if any(contains_coordinates):
                    inx = np.where(contains_coordinates == True)[0]
                    if len(inx)>1:
                        #ms  = [np.mean(pc.extreme(solun[si]), 0) for si in inx]
                        ms  = [np.mean(try_extreme(solun[si]), 0) for si in inx]
                        std = [np.linalg.norm(m - xexp) for m in ms]
                        best_inx = inx[np.argmin(std)]
                    else:
                        best_inx = inx[-1]
                    print(f'In try- inx: {inx} len(inx): {len(inx)} best_inx: {best_inx}')
                    solution_volume = pc.volume(solun[best_inx])
                else:
                    raise ValueError("No polytope contains coordinates.")
            
            except ValueError as e:
                
                ms = [np.mean(try_extreme(spolyi), 0) for spolyi in solun]
                
                std = np.array([np.linalg.norm(np.abs(msi - xexp)) for msi in ms])
                best_inx = np.argmin(std)
                print(f"in except: std {std} best_inx: {best_inx}  ms: {ms}")
                solution_volume = pc.volume(solun[best_inx])
                #print(f'The best polytope is at {best_inx}. Centroid is {ms[best_inx]} with std of {std[best_inx]}')
                
            print(f"L(s):{len(solun)} solution is at:{best_inx}", end=" ") 
            
            sortorder = np.argsort(pairs) if sign <0 else np.argsort(pairs)[::-1]
            
            allsolutions, vol_all = [], []
            #vol_all = [pc.volume(polyi) for polyi in solun]
            #allsolutions= np.array([np.mean(try_extreme(solus), 0) for solus in solun])
            for polyi in solun:
                if type(polyi) is pc.Polytope:
                    vol_all.append(pc.volume(polyi))
                    dm = np.array(np.mean(try_extreme(polyi), 0))[sortorder.argsort()]
                    allsolutions.append(dm)
                
                elif type(polyi) is pc.Region:
                    for iq in polyi:
                        vol_all.append(pc.volume(polyi))
                        dm = np.array(np.mean(try_extreme(polyi), 0))[sortorder.argsort()]
                        allsolutions.append(dm)

            grandradius = radius_from_volume(dim=dimension, volume=np.sum(vol_all))
            localmat  = pc.extreme(solun[best_inx])#, abs_tol=1E-7)
            
            # for zi, zj in enumerate(localmat):
            #     localmat[zi]=zj[sortorder.argsort()]
                    
            dmax = np.max(localmat, axis=0)
            dmin = np.min(localmat, axis=0)
            solution_error = np.abs(dmax-dmin)/2
            total_solNr = len(solun)
            
            mx=np.mean(localmat,0)
            
            m = np.mean(localmat,0)
            #m = m[sortorder.argsort()] 
            if sign <0:
                m= m[sortorder.argsort()]
                m = 0.5-m
            else:
                m= m[sortorder.argsort()]
            
            #wrtdata(fname, rc, solution_volume, solution_error, solun[best_inx], extremepnts=localmat, volAsym=np.sum(vol_all), grandradius=grandradius, Lsol=lsol)            
            twrite=datetime.now()
            tinfo.append(twrite.timestamp()-tintersect.timestamp())
            
            t2  = datetime.now()
            
            print("\x1b[1;32mwrt:\x1b[0m",datetime.now().strftime("%H:%M:%S"),"\x1b[1;32m Total: \x1b[0m{}".format(t2-t1))
            
            ttotal=datetime.now()
            tinfo.append(ttotal.timestamp()-t1.timestamp())
                        
            wrttime_mc(rc, fname, tinfo)  # tinfo = [h, PairID, t_linearize, t_polytope, t_intersect, t_write, t_total]
            wrtdata(rc, fname, m, solun[best_inx], solution_volume, solution_error, localmat, np.sum(vol_all), grandradius , total_solNr=total_solNr, allsolutions=np.array(allsolutions))
            
            print(f"---> predicted struc:{m}. Given struc: {pairs}")
            
            os.remove(os.path.join(fpath,'Pair-%g.dat'%(rc+1)))
        
        sn.append(sn2)
        
        TE=datetime.now()    
        print("\n intermediate total: \x1b[0m{}".format(TE-TS))
        
    TE=datetime.now()    
    print("Total: \x1b[0m{}".format(TE-TS0))
    
    return