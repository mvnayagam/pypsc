# ---> import modules
import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

from matplotlib.cm import ScalarMappable

from psc.lib.gspacer import g
from psc.lib.solutionIO import readh5file_v2


def plot3dmcresults(h:int, fpath: str):
    
    
    # ---> get back all generated coordinate NOTE: change 'h' and 'path' to your setting
    #      the path is set to Desktop\\MCres1. the script search the result in this location
    # fn = os.path.join(fpath,'pnew_2.h5')


    # ---> get back the voume and mean volume

    l = np.arange(2,h+1, 1)
    mmean, error, allvol   = [], [], []

    for ai, ls in enumerate(l):
        fn = os.path.join(fpath,'pnew_%g.h5'%(ls))
        if ai==0:
            pair = readh5file_v2(fn)[4]   ## to get unsorted pair
        mmean.append(readh5file_v2(fn)[1])

    cmap = plt.get_cmap("cool")
    norm_vol = cm.colors.LogNorm(vmax=1*np.max(mmean),  vmin=1*np.min(mmean))

    # ---> start plot
    nc = 4 ; nr = int(np.ceil(len(l)/nc))
    fig, axs = plt.subplots(nr, nc, figsize=(16,8),frameon=False,constrained_layout=True, subplot_kw={'projection': '3d','aspect':'auto'})

    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    for ai, ls in enumerate(l):
        
        fn = os.path.join(fpath,'pnew_%g.h5'%(ls))
        allvol.append(readh5file_v2(fn)[0])
        error.append(np.array(readh5file_v2(fn)[2]))
        centero=np.array(readh5file_v2(fn)[5])
        radi=np.array(readh5file_v2(fn)[6])
        
        noofsolus=np.array(readh5file_v2(fn)[7])
            
        #axs.flat[ai].plot(0,0,0, ms= 0, marker='o', c='m',label=r'$\mathcal{l}~\leq~%g~~V~=~%1.7f$'%(ls,mmean[ai]), alpha=0)
        axs.flat[ai].plot(0,0,0, ms= 0, marker='o', c='m',label=r'$\mathcal{l}~\leq~%g$'%(ls), alpha=0)
        axs.flat[ai].legend(loc='upper right', frameon='False',fancybox='False', framealpha=0.0,mode='expand',
                            prop={'size':14}, bbox_to_anchor=(-0.,1.05))
        
        
        for en, ei in enumerate(np.array(error[ai])):
            
            vol_c=cmap(norm_vol( mmean[ai]),alpha=0.15)
                
            ROO= 1; f = [1.0]*len((pair[en])[::-1])
            if np.sign(g(ROO, np.sort(pair[en])[::-1], f))>0:
                centero[en] = centero[en]
            else:
                sr=np.argsort(pair[en])[::-1] ; XP = np.sort(0.5-centero[en])[::-1]
                centero[en]=XP[sr.argsort()]
                ei  = ei[sr.argsort()]
                    
            if int(noofsolus[en]/2) >1:
                NOO=int(noofsolus[en])
                axs.flat[ai].plot(pair[en][0], pair[en][1], pair[en][2], 'oy', mew=0.5*NOO, ms=20,mfc='none', alpha=0.5)
            
            axs.flat[ai].plot(centero[en][0], centero[en][1], centero[en][2], 'og', ms=8, mew=1.5, mfc='none', alpha=0.5)
            axs.flat[ai].plot(pair[en][0], pair[en][1], pair[en][2], 'ok',          ms=5, mew=0.0, mfc='k', alpha=0.5)
            
            axs.flat[ai].errorbar(pair[en][0],pair[en][1],pair[en][2], xerr=ei[0], alpha=0.5, ecolor='g', elinewidth=2, fmt='none')
            axs.flat[ai].errorbar(pair[en][0],pair[en][1],pair[en][2], yerr=ei[1], alpha=0.5, ecolor='b', elinewidth=2, fmt='none')
            axs.flat[ai].errorbar(pair[en][0],pair[en][1],pair[en][2], zerr=ei[2], alpha=0.5, ecolor='k', elinewidth=2, fmt='none')
            
            axs.flat[ai].scatter(pair[en][0],pair[en][1],pair[en][2], s=3000*radi[en], linewidths=2, ec='C1', fc='none', alpha=0.75)
                    
            if ai<3:
                axs.flat[ai].zaxis.set_ticklabels([])
                axs.flat[ai].set_xlabel(r'$z_\mathrm{1}$', fontsize=16, labelpad=10)
                axs.flat[ai].set_ylabel(r'$z_\mathrm{2}$', fontsize=16, labelpad=10)
            if ai == 3:
                axs.flat[ai].set_xlabel(r'$z_\mathrm{1}$', fontsize=16, labelpad=10)
                axs.flat[ai].set_ylabel(r'$z_\mathrm{2}$', fontsize=16, labelpad=10)
                axs.flat[ai].set_zlabel(r'$z_\mathrm{3}$', fontsize=16, labelpad=10)
            axs.flat[ai].grid(False)
            
            axs.flat[ai].set_xlim(0.,0.5)
            axs.flat[ai].set_ylim(0.,0.5)
            axs.flat[ai].set_zlim(0.,0.5)
        
            axs.flat[ai].xaxis.set_pane_color(vol_c)
            axs.flat[ai].yaxis.set_pane_color(vol_c)
            axs.flat[ai].zaxis.set_pane_color(vol_c)
            
    fig.tight_layout()
    sm_vol =  ScalarMappable(norm=norm_vol, cmap=cmap)
    sm_vol.set_array([np.min(mmean),np.max(mmean),100])

    cbar_vol = fig.colorbar(sm_vol, ax=axs[:], orientation='vertical', pad=0.075, aspect=30, shrink=1.0)
    #cbar_vol = fig.colorbar(sm_vol, ax=axs[:], aspect=15, shrink=0.8,panchor=(0,0.5))
    cbar_vol.set_label(r'$\log~V_{\mathrm{average}}$', size=12, weight='bold', loc='center', labelpad=1.5)
    #cbar_vol.ax.tick_params(labelsize=12)
    cbar_vol.ax.tick_params(width=1.5, length=8, which='major')
    cbar_vol.ax.tick_params(width=1.0, length=5, which='minor')
    cbar_vol.ax.tick_params(labelsize=16)

    # ---> This will remove any unused axs in the fig. 
    for i in axs.flat:
        if not i.lines:
            i.remove()

    plt.subplots_adjust(left=0.01, right=0.785, bottom=0.10, top=1, wspace=0.05, hspace=0.075)

    #plt.savefig(os.path.join(os.getcwd(), 'MCres/folder1/folder2/EPA_I.pdf'), dpi=200, bbox_inches='tight')  # when want to save in some other location
    plt.savefig(os.path.join(fpath, 'MCresult.pdf'), dpi=250, bbox_inches='tight')   # when want to save in the some location of results directory

    plt.show()

    return 