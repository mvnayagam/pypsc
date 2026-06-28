import numpy as np
import intvalpy as ip
import polytope as pc

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from psc.lib.gspacer import F, g, hsurf_F, hsurf_g, hsurf_F2


def plot_polytope(poly, ax, alpha=0.1, color='C0'):
    
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


def plot_isosurf(l, h, gs, gx, gy, gzp, gzm, axs, cc, al=1, imax=0.5):
    for hi in range(l+1):
        if ( (hi/l <= (imax) and h%2 !=0) or (hi/l <= (imax) and h%2 ==0) ):
            if hi == 0:
                surf = axs.plot_surface(gx, gy, gzp, color=cc, alpha=al, antialiased=True, ec=cc, capstyle='round',
                                        facecolor=cc, linewidth=0, label=r'$\mathcal{G}\mathrm{( %g,%1.2f)}$'%(l, gs))
                surf._facecolors2d = surf._facecolor3d
                surf._edgecolors2d = surf._edgecolor3d
                axs.plot_wireframe(gx, gy, gzm, color=cc, alpha=al, rstride=25, cstride=25,antialiased=True)
            
            else:
                if (hi/l < imax and l%(2*hi) !=0):
                    
                    axs.plot_surface  (gx, gy, gzp + hi/l, color=cc, facecolor=cc, linewidth=0,
                                 ec=cc,capstyle='round',linestyles='solid', alpha=al)
                    axs.plot_wireframe(gx, gy, gzm + hi/l, color=cc, alpha=al, rstride=25, cstride=25)
                    
                axs.plot_surface  (gx, gy, -1*gzp + hi/l, color=cc, facecolor=cc, linewidth=0,
                                 ec=cc,capstyle='round',linestyles='solid', alpha=al)
                axs.plot_wireframe(gx, gy, -1*gzm + hi/l, color=cc, alpha=al, rstride=25, cstride=25)
                
                if (hi/l < imax and l%(2*hi) ==0):
                    axs.plot_surface  (gx, gy, gzp + hi/l, color=cc, facecolor=cc, linewidth=0,
                                 ec=cc,capstyle='round',linestyles='solid', alpha=al)
                    axs.plot_wireframe(gx, gy, gzm + hi/l, color=cc, alpha=al, rstride=25, cstride=25)
                    
            if (l%(2*l) == 0 and l/h <= imax):
                axs.plot_surface  (gx, gy, -1*gzp + (hi+2)/l, color=cc, facecolor=cc, linewidth=0,
                                 ec=cc,capstyle='round',linestyles='solid', alpha=al)
                axs.plot_wireframe(gx, gy, -1*gzp + (hi+2)/l, color=cc, alpha=al, rstride=25, cstride=25)
    return



def plot_isosurfG(h, f, gi, noofpnts=500, imax=0.5, al=0.5, hstart=1):
       
    j = len(f)-1
    
    isos  = np.linspace(0, 0.5, noofpnts)
    kj = [isos]*(len(f)-1)
    [*dim] = np.shape(kj)
    kz = np.meshgrid(*kj)

    gz  = np.zeros_like(kz[0])
    
    kz.extend([np.array(gz)])
    #tz = np.vstack( np.dstack([*kz]))
    
    fig, axs = plt.subplots(1, 1, figsize=(12,5), subplot_kw={'projection': '3d','aspect':'auto'})
    
    plt.rc('xtick', labelsize=16); plt.rc('ytick', labelsize=16) 
    
    axs.set_xlim(0., imax); axs.set_ylim(0., imax); axs.set_zlim(0., imax);  axs.grid(False)
    
    axs.tick_params('z', labelsize=14); axs.tick_params('y', labelsize=14);  axs.tick_params('x', labelsize=14)
    
    axs.set_xlabel(r'$z_\mathrm{1}$', fontsize=16, labelpad=16)
    axs.set_ylabel(r'$z_\mathrm{2}$', fontsize=16, labelpad=16)
    axs.set_zlabel(r'$z_\mathrm{3}$', fontsize=16, labelpad=10)
    
    axs.view_init(elev=15, azim=-50, vertical_axis='z')
    
    fig.tight_layout()
    gx = kz[0]  ; gy = kz[1]
    
    for l in range(hstart, h+1):
        gzp = hsurf_g(l, [*kz], f, gi, j, s=1)
        gzm = hsurf_g(l, [*kz], f, gi, j, s=-1)
        
        ra = np.random.uniform(0, 1, 3) ; cc = (ra[0],ra[1],ra[2])
        
        for hi in range(l+1):
            if ( (hi/l <= imax and h%2 !=0) or (hi/l <= imax and h%2==0) ):
                if hi == 0:
                    surf = axs.plot_surface(gx, gy, gzp, color=cc, alpha=al, antialiased=True, ec=cc, capstyle='round',
                                            facecolor=cc, linewidth=0, label=r'$\mathcal{G}\mathrm{( %g,%1.2f)}$'%(l, gi))
                    surf._facecolors2d = surf._facecolor3d
                    surf._edgecolors2d = surf._edgecolor3d
                    axs.plot_wireframe(gx, gy, gzm, color=cc, alpha=al, rstride=25, cstride=25,antialiased=True)                
                else:
                    if (hi/l < imax and l%(2*hi) !=0):                    
                        axs.plot_surface  (gx, gy, gzp + hi/l, color=cc, facecolor=cc, linewidth=0,
                                     ec=cc,capstyle='round',linestyles='solid', alpha=al)
                        axs.plot_wireframe(gx, gy, gzm + hi/l, color=cc, alpha=al, rstride=25, cstride=25)
                    
                    axs.plot_surface  (gx, gy, -1*gzp + hi/l, color=cc, facecolor=cc, linewidth=0,
                                     ec=cc,capstyle='round',linestyles='solid', alpha=al)
                    axs.plot_wireframe(gx, gy, -1*gzm + hi/l, color=cc, alpha=al, rstride=25, cstride=25)
                
                    if (hi/l < imax and l%(2*hi) ==0):
                        axs.plot_surface  (gx, gy, gzp + hi/l, color=cc, facecolor=cc, linewidth=0,
                                     ec=cc,capstyle='round',linestyles='solid', alpha=al)
                        axs.plot_wireframe(gx, gy, gzm + hi/l, color=cc, alpha=al, rstride=25, cstride=25)                
                if (l%(2*l) == 0 and l/h <= imax):
                    axs.plot_surface  (gx, gy, -1*gzp + (hi+2)/l, color=cc, facecolor=cc, linewidth=0,
                                     ec=cc,capstyle='round',linestyles='solid', alpha=al)
                    axs.plot_wireframe(gx, gy, -1*gzp + (hi+2)/l, color=cc, alpha=al, rstride=25, cstride=25)
        
    axs.legend(prop = {'size' : 14}, loc=2, shadow=False, bbox_to_anchor=(0.05,0.95))
    return