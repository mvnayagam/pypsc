import os
import numpy as np
import intvalpy as ip
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from psc.lib.solutionIO import  readh5file_v2_updated
from psc.lib.gspacer import hsurf_g

def radarplotformc (reflections: list, fpath: str, ltoplot=[2, 9], figname: str='ex', figtype: str='png', savefig: bool=True):
    
    # ---> Get back the volume and mean volume
    reflections = reflections if type(reflections) == list else np.arange(2, reflections+1)
    print(f"--> processing the reflections: {reflections}")
    
    solution, grandradius, solution_error, meanvolume, total_solutionNr, total_volume_in_Asym = [], [] , [], [], [], []

    for ai, ls in enumerate(reflections):
        fname = os.path.join(fpath,'pnew_%g.h5'%(ls))
        print(f'---> Reading {fname}')    
        
        res = readh5file_v2_updated(fname)
        if ai == 0:
            pairs = res[0]
            pairs_unsorted = res[1]
            #print(f'pairs: {pairs} pairs_unsorted {pairs_unsorted}')
        
        solution.append(res[2])
        grandradius.append(res[3])
        solution_error.append(res[4])
        meanvolume.append(res[5])
        total_solutionNr.append(res[6])
        total_volume_in_Asym.append(res[7])

    solution, grandradius = np.array(solution), np.array(grandradius), 
    solution_error, meanvolume = np.array(solution_error), np.array(meanvolume), 
    total_solutionNr, total_volume_in_Asym = np.array(total_solutionNr), np.array(total_volume_in_Asym)
        
    cmap = plt.get_cmap("cool")
    norm = cm.colors.LogNorm(vmax=1*np.max(meanvolume), vmin=1*np.min(meanvolume))
        
    l=ltoplot

    print(f'\x1b[1;36m---> Given structure:\t\x1b[1;32m{pairs[0]}')
    #print(f'\x1b[1;36m---> found structure:\t\x1b[1;32m {solution[-1][0]}')
    
    # ---> set no of rows nr automatically. each row will have 4 columns
    nc = 4 ; nr=(nc-1)//4+1
    fig, axs = plt.subplots(nrows=nr, ncols=nc, figsize =(20,7), sharex=True, sharey=False, subplot_kw={'aspect':'auto', 'projection':'polar'},
                            constrained_layout=True, gridspec_kw={'wspace':0.0, 'hspace':-0.05}) #, 'height_ratios':[1, 1.15]
    
    IDs=[0]
    # Number of variables
    n = len(pairs[-1])
    angles = np.linspace(0, 360, n, endpoint=False)
    angles_rad = np.deg2rad(np.append(angles, angles[0]))
    
    for linx, lvalue in enumerate(l):
        for pinx, pair in enumerate(pairs):
            ax = axs[linx]
            
            vol_c=cmap(norm(meanvolume[lvalue-2][IDs[pinx]]),alpha=0.15)
                        
            # ---> Set the theta grid to match the number of variables
            ax.set_thetagrids(angles)
            
            # ---> Plot given actual coordinates
            pair = np.concatenate([pair,pair[:1]])
            centroid = solution[lvalue-2][IDs[pinx]]
            centroid = np.concatenate([centroid, centroid[0:1]])
            
            print(f'\x1b[1;36m---> found structure:\t\x1b[1;32m{centroid}\x1b[1;36m for l = {lvalue}')
            
            ax.plot(angles_rad, pair, 'o', ms=12, color='k', mew=2, mfc='w', lw=1.5, alpha=1.0)
            
            ax.plot(angles_rad, centroid, 'o--', ms=8, color=vol_c, mew=2, mfc='w', lw=1.5, alpha=1, label=r'$l\leq %g$'%(lvalue))
            #ax.fill(angles_rad, centroid, color=vol_c,alpha=0.15)
            
            # ---> Calculate limits or error and plot
            err = solution_error[lvalue-2][IDs[pinx]]
            err = np.concatenate([err,err[:1]])
                
            upper_bound = pair + 1* (err/2)
            lower_bound = np.abs(pair - 1*(err/2))
            #print(f'centroid: {centroid}. {pair}')
            
            ax.fill_between(angles_rad, pair, lower_bound, color=vol_c, alpha=0.3)
            ax.fill_between(angles_rad, pair, upper_bound, color=vol_c, alpha=0.3)
    
    # ---> Remove unused subplots (but keep the colorbar intact)
    last_used_ax = None
    for ax in axs.ravel():
        if not ax.has_data():  # Check if the subplot has data
            fig.delaxes(ax)  # Remove the subplot
        else:
            last_used_ax = ax
    # Adjust the layout
    plt.subplots_adjust()  # Add this line here
                  
    # ---> Add color map
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([np.min(meanvolume),np.max(meanvolume),100])
    cbar = fig.colorbar(sm, ax=last_used_ax, orientation='vertical', pad=0.1, aspect=30, shrink=0.6) # ax=axs[:] #ax=last_used_ax
    cbar.set_label(r'$\log~V_{\mathrm{average}}$', size=16, weight='bold', loc='center', labelpad=2.5)
    cbar.ax.tick_params(labelsize=14, width=1.2, length=8, which='major')
    cbar.ax.tick_params(labelsize=14, width=1.0, length=5, which='minor')
    
    # Customize each subplot
    for ax in axs.ravel():
        # Set the theta ticks and labels
        zz=[str('$z_')+str(i)+'$' for i in range(1,n+1,1)]
        ax.set_xticklabels(zz, fontsize=18)
        ax.set_rlabel_position(90)
        
        ax.tick_params(axis='x', colors='k', labelsize=14, pad=10)
        ax.tick_params(axis='y', colors='k', labelsize=14, pad=30)
        ax.set_ylim(0., 0.55)
        
        ax.yaxis.grid(True, ls='-', lw=1.0, which='major', c='k', alpha=0.35)
        ax.yaxis.grid(True, ls='--', lw=0.5, which='minor', c='k', alpha=0.20)
        ax.legend(loc=2, fontsize=14, bbox_to_anchor=(-0.15, 1.07))
            
    if savefig:
        fig.savefig(figname+"."+figtype, dpi=300, bbox_inches='tight')
        
    plt.show()

    return


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

def plot_segment(ax, p, cc, lww=1.5, al=0.5):
    for i in p:
        ax.plot(i[:,0], i[:,1],'-',lw=lww, c=cc, alpha=al)
    return

def plot_isosurface(h, l, I, gx, gy, gz1, gz2, gz3, gz4, axs, al=0.3, imax=0.5):
    
    for hi in range(l+1):
        if ( (hi/l <= imax and h%2 !=0) or (hi/l <= imax and h%2 ==0) ):
            if hi == 0:
                
                surf = axs.plot_surface(gx, gy, gz1, color='k', antialiased=True, facecolor='k', linewidth=1, alpha=al, 
                                        label=r'$\mathcal{F}\mathrm{( %g,%1.2f)}$'%(l, I))
                surf._facecolors2d = surf._facecolor3d
                surf._edgecolors2d = surf._edgecolor3d
                
                # axs.plot_surface   (gx, gy, gz1, color=cc,antialiased=True, edgecolor='none', alpha=al)
                axs.plot_wireframe (gx, gy, gz3, color='b', alpha=al, rstride=15, cstride=15,antialiased=True)
                
            else:
                if (hi/l < imax and l%(2*hi) !=0):
                    
                    axs.plot_surface  (gx, gy, gz1 + hi/l, color='k', alpha=al)
                    axs.plot_wireframe(gx, gy, gz3 + hi/l, color='b', alpha=al, rstride=25, cstride=25)
                    
                axs.plot_surface  (gx, gy, 1*gz2 + hi/l, color='r', alpha=al)
                axs.plot_wireframe(gx, gy, 1*gz4 + hi/l, color='g', alpha=al, rstride=25, cstride=25)
                
                if (hi/l < imax and l%(2*hi) ==0):
                    axs.plot_surface  (gx, gy, gz1 + hi/l, color='k', alpha=al)
                    axs.plot_wireframe(gx, gy, gz3 + hi/l, color='b', alpha=al, rstride=25, cstride=25)
                    
            if (l%(2*l) == 0 and l/h <= imax):
                axs.plot_surface  (gx, gy, 1*gz1 + (hi+2)/l, color='k', alpha=al)
                
                # axs.plot_wireframe(gx, gy, 1*gz2 + (hi+2)/l, color='r', alpha=al, rstride=25, cstride=25)
                axs.plot_surface  (gx, gy, 1*gz2 + (hi+2)/l, color='r', alpha=al)
    return



def plotisosurf_EPA(l, h, gi, ax, isos, giso1, giso2, cc, lw=0.12, imax=0.5):
    
    for hi in range(l+1):
        if ( (hi/l <= imax and h%2 !=0) or (hi/l <= imax and h%2 ==0) ):
            if hi == 0:
                ax.plot(isos, giso1 + hi/l, '-' ,lw=lw, c=cc,label=r'$\mathcal{G}\mathrm{( %g,%1.2f)}$'%(l, gi))
                ax.plot(isos, giso2 + hi/l, '--',lw=lw, c=cc)
            else:
                if (hi/l < imax and l%(2*hi) !=0):
                    ax.plot(isos, giso1 + hi/l, '-', lw=lw, c=cc)
                    ax.plot(isos, giso2 + hi/l, '--',lw=lw, c=cc)
                ax.plot(isos, -1*giso1  + hi/l, '-', lw=lw, c=cc)
                ax.plot(isos, -1*giso2  + hi/l, '--',lw=lw, c=cc)
                if (hi/l < imax and l%(2*hi) ==0):
                    ax.plot(isos, giso1    + hi/l, '-', lw=lw, c=cc)
                    ax.plot(isos, giso2    + hi/l, '--',lw=lw, c=cc)
                
            if (l%(2*l) == 0 and l/h <= imax):
                ax.plot(isos, -1*giso1 + (hi+2)/l, '-', lw=lw,c=cc)
                ax.plot(isos, -1*giso2 + (hi+2)/l, '--',lw=lw,c=cc)
    return

def plotisosurf_nEPA(l, h, gi, ax, isos, y1, y2, y3, y4, cc, lw=2, imax=0.5, alp=0.5):
    for hi in range(l+1):
        if ( (hi/l <= imax and h%2 !=0) or (hi/l <= imax and h%2 ==0) ):
            if hi == 0:
                ax.plot(isos, y1 + hi/(l), '-',  c='k', alpha=alp, label='h=%g'%(l))
                ax.plot(isos, y3 + hi/(l), '--', c='b', alpha=alp)
            else:
                if (hi/l <imax and l%(2*hi) !=0):
                    ax.plot(isos, y1 + hi/(l), '-',  c='k', alpha=alp)
                    ax.plot(isos, y3 + hi/(l), '--', c='b', alpha=alp)
                    
                ax.plot(isos, y2 + hi/(l), '-',  c='r', alpha=alp)
                ax.plot(isos, y4 + hi/(l), '--', c='g', alpha=alp)
                
                if (hi/l < imax and l%(2*hi) ==0):
                    ax.plot(isos, y1 + hi/(l), '-',  c='k', alpha=alp)
                    ax.plot(isos, y3 + hi/(l), '--', c='b', alpha=alp)
                
                if (l%(2*l) == 0 and l/h <= imax):
                    ax.plot(isos, y1 + (hi+2)/l, '-',  c='k', alpha=alp)
                    ax.plot(isos, y2 + (hi+2)/l, '-',  c='r', alpha=alp)
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


