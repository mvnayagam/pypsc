import logging
logger = logging.getLogger(__name__)

import numpy as np
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, mark_inset)


def writepolygons(fname, polys):
    
    for i in polys:
        x, y = i.exterior.coords.xy
        
        for xl in range(len(x)):
            #fname.write('{:10.10}\t\t{:10.10}\t\t'.format(x[xl],y[xl]))
            fname.write("%2.12f\t\t%2.12f\t\t"%(x[xl],y[xl]))
        fname.write("\n")
    
    return ()

def isInside(p: np.array, 
             v1=np.array([0.0, 0.0]), 
             v2=np.array([0.5,0.0]), 
             v3=np.array([0.25, 0.25])
             ) -> bool:
    
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

def get_error_v3a(d):
    xlist=[d[i] for i in range(3,len(d), 2)]
    ylist=[d[i] for i in range(4,len(d), 2)]
    
    x_min=np.min(xlist)
    x_max=np.max(xlist)
    y_min=np.min(ylist)
    y_max=np.max(ylist)
    dx = (x_min-x_max)/2
    dy = (y_min-y_max)/2
    
    return (dx, dy)

def get_error(d):  # before: get_error_v3(d)
    xlist=d[0]
    ylist=d[1]
    
    x_min=np.min(xlist)
    x_max=np.max(xlist)
    y_min=np.min(ylist)
    y_max=np.max(ylist)
    
    dx = (x_min-x_max)/2
    dy = (y_min-y_max)/2
    
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

def analyzesolution(solution, xcoor, plotting=True, imax=0.5):
   
    xlim, ylim = [], []
    
    fnpoly=open("pseudosol.dat", "wt+")
    fcoor =open('realsol.dat', "at+")
        
    if solution == []:
        print("\n---> Solution list is empty. I can not do further steps. I am exiting now")
        sys.exit
    else:
        if plotting:
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize = (12, 4), subplot_kw = dict(aspect = 1.0))
            axins = inset_axes(axs, width="75%", height="75%",bbox_to_anchor=(.2, .4, .6, .5),bbox_transform=axs.transAxes, loc=2)
            mark_inset(axs, axins, loc1=1, loc2=3, fc="none", ec='0.25')
            
            axs.plot(xcoor[0], xcoor[1], 'o',color='k',ms=8,mew=1, mfc='none', alpha=1)
            axs.plot(imax/2,imax/2, 'o',c='crimson', mfc='white', ms=8, alpha=0.5)  # orangered
            axs.plot(imax/2,imax/2, 'o',c='crimson', alpha=0.5)
            axs.plot([0, 0.25, 0.5], [0, 0.25, 0.], '-.',c='crimson', alpha=0.5)
            axs.set_xlabel(r'$z_\mathrm{1}$' ,fontsize=16,labelpad=12)
            axs.set_ylabel(r'$z_\mathrm{2}$' ,fontsize=16,labelpad=12)
            axs.set_xlim(0,0.5)
            axs.set_ylim(0.,0.5)
            
        cc=(0.914716238473734, 0.1228224724422626, 0.3040725144466219, 0.5)
        
        if solution[-1].geom_type == 'MultiPolygon':
            for i in solution[-1].geoms: # for shpaely v<2 remove .geoms
                
                x, y = i.exterior.coords.xy
                pseudosol=list(zip(x,y))
                pseudosolution(x, y, fnpoly)
                
                for ii in range(len(pseudosol)): #for ii in range(len(x)):
                    if (isInside(pseudosol[ii])): #if (isInside([x[ii],y[ii]])):
                                            
                        fcoor.write("%2.12f\t %2.12f\t %2.12f\t "%(xcoor[0], xcoor[1], i.area))
                        realsolution(x, y, fcoor)
                        xe , ye = get_error([x, y])
                        
                        if plotting:
                            axs.fill_between  (x,y,linewidth=0, facecolor=cc, alpha=1)
                            axins.fill_between(x,y,linewidth=0, facecolor=cc, alpha=0.5)
                            
                            xlim.append([np.min(x), np.max(x)])
                            ylim.append([np.min(y), np.max(y)])
                            
                            axs.scatter (xcoor[0], xcoor[1], ec='none', alpha=0.4, c='darkblue', s=10E1*i.area )
                            axs.errorbar(xcoor[0], xcoor[1],xerr=1*np.abs(xe), alpha=0.4, ecolor = 'g', elinewidth = 2)
                            axs.errorbar(xcoor[0], xcoor[1],yerr=1*np.abs(ye), alpha=0.4, ecolor = 'b', elinewidth = 2)
                            
                            axins.plot(np.mean(x), np.mean(y), 'o',color='k',ms=8,mew=1, mfc='none', alpha=1)
                            axins.errorbar(np.mean(x), np.mean(y), xerr=1*np.abs(xe), alpha=0.5, ecolor = 'g', elinewidth = 2)
                            axins.errorbar(np.mean(x), np.mean(y), yerr=1*np.abs(ye), alpha=0.5, ecolor = 'b', elinewidth = 2)
                            axins.set_xlim(np.min(xlim), np.max(xlim))
                            axins.set_ylim(np.min(ylim), np.max(ylim))
                            fig.tight_layout()  
                            plt.show()
                        
                        # for shapely v<2 use this:
                        #print(f"===> Possible solution (centorid of polygon) :: {np.array(i.centroid)} or {0.5-np.array(i.centroid)}")
                        
                        print(f"===> Possible solution (centorid of polygon) :: {np.array(i.centroid.xy).reshape(1,-1)[0]} or {0.5-np.array(i.centroid.xy).reshape(1,-1)[0]}")
                        print(f"===> Assumed coordinate: {xcoor}")
                        print(f"===> Possible uncertainty in solution :: {xe, ye}")
                        print(f"===> Finals area                      :: {i.area}\n")
                        
                        break
        elif solution[-1].geom_type == 'LineString':
            fl = solution[-1].boundary
            xall, yall = [], []
            
            for i in fl:
                x, y = i.xy
                xall.append(x[0])
                yall.append(y[0])
                xe, ye = get_error([xall, yall])            
            print(f"===> Possible solution (centorid of polygon) :: [{np.mean(xall), np.mean(yall)} or [{0.5-np.mean(xall), 0.5-np.mean(yall)}]")
            print(f"===> Assumed coordinate: {xcoor}")
            print(f"===> Possible uncertainty in solution :: {xe, ye}")
            print(f"===> Finals area                      :: {i.area}\n")
            
            
            if plotting:
                axs.plot(np.mean(xall), np.mean(yall), 'or', ms=10, alpha=0.5)
                axs.plot(xall, yall, '-r', lw=2 )
                
                axs.errorbar(xcoor[0], xcoor[1], xerr=1*np.abs(xe), alpha=0.4, ecolor = 'g', elinewidth = 2)
                axs.errorbar(xcoor[0], xcoor[1], yerr=1*np.abs(ye), alpha=0.4, ecolor = 'b', elinewidth = 2)
                
                axins.plot(np.mean(xall), np.mean(yall), 'or', ms=10, alpha=0.5)
                axins.plot(xall, yall, '-r', lw=2 )
                axins.errorbar(np.mean(xall), np.mean(yall), xerr=1*np.abs(xe), alpha=0.5, ecolor = 'g', elinewidth = 2)
                axins.errorbar(np.mean(xall), np.mean(yall), yerr=1*np.abs(ye), alpha=0.5, ecolor = 'b', elinewidth = 2)
                
                fig.tight_layout()  
                plt.show()
            
            fcoor.write("%2.12f\t %2.12f\t %2.12f\t "%(xcoor[0], xcoor[1], i.area))
            realsolution(xall, yall, fcoor)        
        else:
            x, y = solution[-1].exterior.coords.xy
            pseudosol=list(zip(x,y))
            pseudosolution(x, y, fnpoly)
            
            for ii in range(len(pseudosol)):
                if (isInside(pseudosol[ii])):
                    realsolution(x, y, fcoor)
                    xlim.append([np.min(x), np.max(x)])
                    ylim.append([np.min(y), np.max(y)])
                    
                    if plotting:
                        axs.fill_between(x, y, alpha=1, linewidth=0, facecolor=cc)
                        axins.fill_between(x,y, alpha=1, linewidth=0, facecolor=cc)
                        axins.set_xlim(np.min(x), np.max(x))
                        axins.set_ylim(np.min(y), np.max(y))
                        fig.tight_layout()  
                        plt.show()
                    
                    d=np.array([x, y])
                    xe , ye = get_error(d)
                    
                    print(f"===> Possible solution (centorid of polygon) :: {np.array(solution[-1].centroid)} or {0.5-np.array(solution[-1].centroid)}")
                    print(f"===> Assumed coordinate :: {xcoor}")
                    print("===> Possible uncertanity in solution :: ", xe, ye)
                    print("===> Finals area                      :: ", solution[-1].area,"\n")
                    
                    break
                    
        fnpoly.close()
        fcoor.close()
        
    return