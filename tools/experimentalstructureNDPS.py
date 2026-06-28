import os
import numpy as np
import polytope as pc

from datetime import datetime

from psc.lib.gspacer import g, F, hsurf_F2, hsurf_g
from psc.lib.linearizer import linearizenD_EPA_old_deltelater
from psc.lib.tessellator import getpolytope_EPA  #getpolytope
from psc.lib.intersector import find_intersection
from psc.lib.solutionIO import wrtcoor, wrtdata, wrtallsolution, wrttime_mc

def isosurfs_EPA(h, xexp, f, j, fname):
    n  = [] 
    if h <= 2:
        for l in range(1,h+1):
            gi = g(l, xexp, f)
            normal, dist = linearizenD_EPA_old_deltelater(l, f, np.abs(gi))
            
            n.append([l, normal, dist, np.sign(gi)])
            fname.write("%3g\t%2.6f\t%2.6f\t%2.6f\t%2.6f\t%2.6f\n"%(h, normal[0],normal[1],normal[2], dist[0],dist[1]))
    else:
        gi = g(h,xexp,f)
        normal, dist = linearizenD_EPA_old_deltelater(h, f, np.abs(gi))
        
        n.append([h, normal, dist, np.sign(gi)])
        fname.write("%3g\t%2.6f\t%2.6f\t%2.6f\t%2.6f\t%2.6f\n"%(h, normal[0],normal[1],normal[2], dist[0],dist[1]))
    
    return n

def experimentalstrucutre_EPA(ROlist:list, scatteringfactors:list = [1., 1., 1.],
                              atomsymbol: list = ['Mo', 'Se', 'Se'],
                              iorg: str='amplitude', imax=0.5) -> None:
    
    """This routine solve structure within EPA framework in nD-PS.

    Args:
        ROlist (int): number of reflection orders to consider in calculation. should be >=2 
        scatteringfactors (list): list of atomic structure factors of each atom. Always set to [1., 1., ....]  for EPA model
        atomsymbol (list): list of atomic symbols. Defaults to ['Mo', 'Se', 'Se'] for BaSrO3
        iorg (str, optional): which method to select from ['intensity', 'amplitude' ]. Defaults to 'amplitude'.
        imax (float, optional): The extend of PS. Defaults to 0.5.
    """
    
    TS0=datetime.now()
    
    #---> define asymmetric part of PS
    dimension = len(scatteringfactors)
    temp = np.tril(np.ones(shape=(dimension, dimension)) , 0 )
    temp = imax*np.vstack([[0]*dimension, temp])
    asym = pc.qhull(np.array(temp))
      
    
    #---> define result folder to save results
    #fpath = os.path.join(os.getcwd(), datetime.now().strftime('MCresult-'+'%Y-%m-%d-%H%M%S'))
    fpath  = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop', datetime.now().strftime('MCresult-'+'%Y-%m-%d-%H%M%S')) 
    if os.path.isdir(fpath):
        print("\x1b[0;34m===> Output files will be saved to \x1b[1;33m", fpath, "\x1b[0;34m location\n" )
    else:
        os.mkdir(fpath)
        print("\x1b[0;34m===> Dir \x1b[1;33mresults \x1b[0;34mis created. Output files will be saved to ", fpath," location\n\n" )
    
    sn = []
    polyFirstRO = []
    for h in ROlist:
        
        sn2 = []
        TS=datetime.now()
               
        print("\n\x1b[1;31m=====> for h = \x1b[0m", h,"\x1b[1;31m==============================\n\x1b[0m")
                     
        fname = os.path.join(fpath,'pnew_%g.h5'%(h))
        
        if os.path.exists(fname):
            os.remove(fname)
        
        #### write random pairs to file 
        #wrtcoor(fname, rp)
        
        for rc, pairs in enumerate(rp):
            tinfo = []
            f     = [1.0, 1.0, 1.0]
            j     = len(f)-1
            
            xexp  = np.sort(pairs)[::-1]
            ll    = 1
            if np.sign(g(ll, xexp, f))>0:
                xexp = xexp
            else:
                xexp = np.sort(0.5-xexp)[::-1]
            
            print("\x1b[1;34m===> Pair-%2g: "%(rc+1),"Current pair : \x1b[0m", pairs)
            
            #### Step 1: Get isomatten 
            
            tinfo.append(h) ; tinfo.append(rc+1)
            t1=datetime.now()
            
            print("\x1b[1;32mstar:\x1b[0m",datetime.now().strftime("%H:%M:%S"),end=" ")
            fstep1=open(os.path.join(fpath,'Pair-%g.dat'%(rc+1)), "wt+")
            nar = isosurfs_EPA(h,xexp,f,j,fstep1)
            fstep1.close()
            
            tlinearize=datetime.now()
            tinfo.append(tlinearize.timestamp() - t1.timestamp())
            
            print("\x1b[1;32miso:\x1b[0m",datetime.now().strftime("%H:%M:%S"),end=" ")
                    
            polylist = []
            
            iorgselect='amplitude'
            if h <= 2:
                for i in nar: # here i[0]->l,i[1]->normal,i[2]->distance,i[3]->amplitudesign
                    o = getpolytope_EPA(i[0], i[1], i[2], i[3], IorG=iorgselect, imax=0.5) # getpolytope_EPA(l, normal, d_all, amplitudesign, IorG='intensity')
                    
                    if i[0]==1:
                        polyFirstRO.append(pc.Region([o.intersect(asym)]))
                        polylist.append(o)                        
                    else:
                        pplist=[]
                        for ipoly in o:
                            if (asym.intersect(ipoly)):
                                if (polyFirstRO[rc].intersect(ipoly)):
                                    pplist.append(ipoly)
                        polylist.append(pc.Region(pplist))
            else:
                pplist=[]
                i = nar[0]
                o = getpolytope_EPA(i[0], i[1], i[2], i[3], IorG=iorgselect, imax=0.5)
                for ipoly in o:
                    if (asym.intersect(ipoly)):
                        if polyFirstRO[rc].intersect(ipoly):
                            pplist.append(ipoly)
                polylist=pc.Region(pplist)
            
            tgetpoly=datetime.now()
            tinfo.append(tgetpoly.timestamp() - tlinearize.timestamp())
            
            print("\x1b[1;32mpoly:\x1b[0m",datetime.now().strftime("%H:%M:%S"),end=" ")
                    
            #---> Get intersection , write all solution
            if h == 2:
                solun = find_intersection(polyFirstRO[rc], polylist[1])
                wrtallsolution(fname, rc+1, solun)
                sn2.append(solun)
            else:
                #pname   = os.path.join(fpath,'pnew_%g.h5'%(h-1))
                #polyold = readoldsolution(rc+1, pname)
                #solun   = find_intersection(pc.Region(polyold), polylist)
                
                solun   = find_intersection(sn[h-3][rc], polylist) #mcinter([pc.Region(polyold), polylist])
                
                wrtallsolution(fname, rc+1, solun)
                sn2.append(solun)
            
            tintersect=datetime.now()
            tinfo.append(tintersect.timestamp() - tgetpoly.timestamp())
            
            print("\x1b[1;32mintersection:\x1b[0m",datetime.now().strftime("%H:%M:%S"),end=" ")
            
            #### Sorting solution
                    
            print("\x1b[1;32msort sol:\x1b[0m",datetime.now().strftime("%H:%M:%S"),end=" ")
            
            #### Final step
            sol_all, vol_all  = [], []
            volAsym , sel1    = 0, -1
            
            for jc, poly in enumerate(solun):
                if xexp in poly:
                    
                    sol_all.append(poly)
                    if not vol_all:
                        sel1   = jc
                        vol_all.append(poly.volume)
                        volume = poly.volume
                        
                    else:
                        sel1   = sel1   if vol_all[-1] <= poly.volume else jc
                        volume = volume if vol_all[-1] <= poly.volume else poly.volume 
                        vol_all.append(poly.volume)
                
                volAsym += pc.volume(poly)
            
            print("L(sol)= ", len(solun)," s =",sel1) 
            
            rr=((3/4)*(1/np.pi)*(volAsym))**(1/3)
            
            sortorder = np.argsort(pairs)[::-1]
            localmat  = pc.extreme(solun[sel1])
            
            for zi, zj in enumerate(localmat):
                localmat[zi]=zj[sortorder.argsort()]
                    
            dmax = np.max(localmat, axis=0)
            dmin = np.min(localmat, axis=0)
            err  = np.abs(dmax-dmin)/2
            wrtdata(fname, rc, pc.volume(solun[sel1]), err, solun[sel1], localmat, [volAsym, rr], len(solun))
            
            twrite=datetime.now()
            tinfo.append(twrite.timestamp()-tintersect.timestamp())
            
            t2  = datetime.now()
            
            print("\x1b[1;32mwrt:\x1b[0m",datetime.now().strftime("%H:%M:%S"),"\x1b[1;32m Total: \x1b[0m{}".format(t2-t1))
            
            ttotal=datetime.now()
            tinfo.append(ttotal.timestamp()-t1.timestamp())
            
            wrttime_mc(rc, fname, tinfo)
            
            #### plot all
            
            m = np.mean(localmat,0)
            print("---> predicted value : ", m, 0.5-m,"xexp : ", pairs)
            
            os.remove(os.path.join(fpath,'Pair-%g.dat'%(rc+1)))
        
        sn.append(sn2)
        
        TE=datetime.now()    
        print("\n intermediate total: \x1b[0m{}".format(TE-TS))
        
    TE=datetime.now()    
    print("Total: \x1b[0m{}".format(TE-TS0))
    
    return

def MCinND_EPA(dimension: int, noofpair: int, noofRO: int=9, iorg: str='amplitude', imax=0.5) -> None:
    
    TS0=datetime.now()
    
    #---> define asymmetric part of PS
    temp = np.tril(np.ones(shape=(dimension, dimension)) , 0 )
    temp = imax*np.vstack([[0]*dimension, temp])
    asym = pc.qhull(np.array(temp))
    
    #---> Generating required no of random positions in list
    rp = np.random.uniform(0.0, 0.5, size=(noofpair, dimension))
    
    #---> define result folder to save results
    #fpath = os.path.join(os.getcwd(), datetime.now().strftime('MCresult-'+'%Y-%m-%d-%H%M%S'))
    fpath  = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop', datetime.now().strftime('MCresult-'+'%Y-%m-%d-%H%M%S')) 
    if os.path.isdir(fpath):
        print("\x1b[0;34m===> Output files will be saved to \x1b[1;33m", fpath, "\x1b[0;34m location\n" )
    else:
        os.mkdir(fpath)
        print("\x1b[0;34m===> Dir \x1b[1;33mresults \x1b[0;34mis created. Output files will be saved to ", fpath," location\n\n" )
    
    sn = []
    polyFirstRO = []
    for h in range(2,noofRO+1):
        
        sn2 = []
        TS=datetime.now()
               
        print("\n\x1b[1;31m=====> for h = \x1b[0m", h,"\x1b[1;31m==============================\n\x1b[0m")
                     
        fname = os.path.join(fpath,'pnew_%g.h5'%(h))
        
        if os.path.exists(fname):
            os.remove(fname)
        
        #### write random pairs to file 
        wrtcoor(fname, rp)
        
        for rc, pairs in enumerate(rp):
            tinfo = []
            f     = [1.0, 1.0, 1.0]
            j     = len(f)-1
            
            xexp  = np.sort(pairs)[::-1]
            ll    = 1
            if np.sign(g(ll, xexp, f))>0:
                xexp = xexp
            else:
                xexp = np.sort(0.5-xexp)[::-1]
            
            print("\x1b[1;34m===> Pair-%2g: "%(rc+1),"Current pair : \x1b[0m", pairs)
            
            #### Step 1: Get isomatten 
            
            tinfo.append(h) ; tinfo.append(rc+1)
            t1=datetime.now()
            
            print("\x1b[1;32mstar:\x1b[0m",datetime.now().strftime("%H:%M:%S"),end=" ")
            fstep1=open(os.path.join(fpath,'Pair-%g.dat'%(rc+1)), "wt+")
            nar = isosurfs_EPA(h,xexp,f,j,fstep1)
            fstep1.close()
            
            tlinearize=datetime.now()
            tinfo.append(tlinearize.timestamp() - t1.timestamp())
            
            print("\x1b[1;32miso:\x1b[0m",datetime.now().strftime("%H:%M:%S"),end=" ")
                    
            polylist = []
            
            iorgselect='amplitude'
            if h <= 2:
                for i in nar: # here i[0]->l,i[1]->normal,i[2]->distance,i[3]->amplitudesign
                    o = getpolytope_EPA(i[0], i[1], i[2], i[3], IorG=iorgselect, imax=0.5) # getpolytope_EPA(l, normal, d_all, amplitudesign, IorG='intensity')
                    
                    if i[0]==1:
                        polyFirstRO.append(pc.Region([o.intersect(asym)]))
                        polylist.append(o)                        
                    else:
                        pplist=[]
                        for ipoly in o:
                            if (asym.intersect(ipoly)):
                                if (polyFirstRO[rc].intersect(ipoly)):
                                    pplist.append(ipoly)
                        polylist.append(pc.Region(pplist))
            else:
                pplist=[]
                i = nar[0]
                o = getpolytope_EPA(i[0], i[1], i[2], i[3], IorG=iorgselect, imax=0.5)
                for ipoly in o:
                    if (asym.intersect(ipoly)):
                        if polyFirstRO[rc].intersect(ipoly):
                            pplist.append(ipoly)
                polylist=pc.Region(pplist)
            
            tgetpoly=datetime.now()
            tinfo.append(tgetpoly.timestamp() - tlinearize.timestamp())
            
            print("\x1b[1;32mpoly:\x1b[0m",datetime.now().strftime("%H:%M:%S"),end=" ")
                    
            #---> Get intersection , write all solution
            if h == 2:
                solun = find_intersection(polyFirstRO[rc], polylist[1])
                wrtallsolution(fname, rc+1, solun)
                sn2.append(solun)
            else:
                #pname   = os.path.join(fpath,'pnew_%g.h5'%(h-1))
                #polyold = readoldsolution(rc+1, pname)
                #solun   = find_intersection(pc.Region(polyold), polylist)
                
                solun   = find_intersection(sn[h-3][rc], polylist) #mcinter([pc.Region(polyold), polylist])
                
                wrtallsolution(fname, rc+1, solun)
                sn2.append(solun)
            
            tintersect=datetime.now()
            tinfo.append(tintersect.timestamp() - tgetpoly.timestamp())
            
            print("\x1b[1;32mintersection:\x1b[0m",datetime.now().strftime("%H:%M:%S"),end=" ")
            
            #### Sorting solution
                    
            print("\x1b[1;32msort sol:\x1b[0m",datetime.now().strftime("%H:%M:%S"),end=" ")
            
            #### Final step
            sol_all, vol_all  = [], []
            volAsym , sel1    = 0, -1
            
            for jc, poly in enumerate(solun):
                if xexp in poly:
                    
                    sol_all.append(poly)
                    if not vol_all:
                        sel1   = jc
                        vol_all.append(poly.volume)
                        volume = poly.volume
                        
                    else:
                        sel1   = sel1   if vol_all[-1] <= poly.volume else jc
                        volume = volume if vol_all[-1] <= poly.volume else poly.volume 
                        vol_all.append(poly.volume)
                
                volAsym += pc.volume(poly)
            
            print("L(sol)= ", len(solun)," s =",sel1) 
            
            rr=((3/4)*(1/np.pi)*(volAsym))**(1/3)
            
            sortorder = np.argsort(pairs)[::-1]
            localmat  = pc.extreme(solun[sel1])
            
            for zi, zj in enumerate(localmat):
                localmat[zi]=zj[sortorder.argsort()]
                    
            dmax = np.max(localmat, axis=0)
            dmin = np.min(localmat, axis=0)
            err  = np.abs(dmax-dmin)/2
            wrtdata(fname, rc, pc.volume(solun[sel1]), err, solun[sel1], localmat, [volAsym, rr], len(solun))
            
            twrite=datetime.now()
            tinfo.append(twrite.timestamp()-tintersect.timestamp())
            
            t2  = datetime.now()
            
            print("\x1b[1;32mwrt:\x1b[0m",datetime.now().strftime("%H:%M:%S"),"\x1b[1;32m Total: \x1b[0m{}".format(t2-t1))
            
            ttotal=datetime.now()
            tinfo.append(ttotal.timestamp()-t1.timestamp())
            
            wrttime_mc(rc, fname, tinfo)
            
            #### plot all
            
            m = np.mean(localmat,0)
            print("---> predicted value : ", m, 0.5-m,"xexp : ", pairs)
            
            os.remove(os.path.join(fpath,'Pair-%g.dat'%(rc+1)))
        
        sn.append(sn2)
        
        TE=datetime.now()    
        print("\n intermediate total: \x1b[0m{}".format(TE-TS))
        
    TE=datetime.now()    
    print("Total: \x1b[0m{}".format(TE-TS0))
    
    return