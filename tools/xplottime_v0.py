
import warnings
warnings.filterwarnings('ignore')

import os
import h5py
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.cm import ScalarMappable

from psc.lib.solutionIO import readh5file_v2


def plottotaltime(reflections: list, fpath: str, figname: str='ex', figtype: str='pdf', savefig: bool=True):
    """_This fn plots the time information (total time) for each processed reflection.
    all data are read from the *.h5 file. ALong with total time, this will plot no of solutions and error
    1. Declare path of folder which contains the .h5 files for each reflections.
    2. Have a list of reflection
    3. select a name for file to save the plot of total time 
    4. select a extension of the file like pdf or pnf or jpeg
    5. savefig decides to save or not fig.
    
    Following are required_
    
    Args:
        reflections (list): List of processed reflections
        fpath (str): Path of folder which contains the h5 files
        figname (str, optional): Name to save plotted picture. Defaults to 'ex'.
        figtype (str, optional): type of extension pdf or png or jpeg. Defaults to 'pdf'.
        savefig (bool, optional): choice to switch off save fig. Defaults to True.
    """    
    print(f'---> I am reading files from the path :: {fpath}')
        
    mmean, meanl = [], []

    for ci, ls in enumerate(reflections):
        fn = os.path.join(fpath,'pnew_%g.h5'%(ls))
        print(f'---> fn : {fn}')    
        mmean.append(readh5file_v2(fn)[1])
        meanl.append(readh5file_v2(fn)[1])

    cmap  = plt.get_cmap("cool")
    norml = cm.colors.LogNorm(vmax=1*np.max(meanl), vmin=1*np.min(meanl))    
    norm  = cm.colors.LogNorm(vmax=1*np.max(mmean), vmin=1*np.min(mmean))
    
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize =(7,5), sharex=True, sharey=False, subplot_kw={'aspect':'auto'},
                            constrained_layout=True, gridspec_kw={'wspace': 0.00,'hspace': 0.00, 'height_ratios':[1, 1.15]})
    
    # plt.rc('xtick', labelsize=14)
    # plt.rc('ytick', labelsize=14)
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    
    emax, t,  nsol   = [], [], []
    tave, eave, vave = [], [], []

    err_c, sol_c = '#006400', 'C3'

    # ----------------------------------------------------
    # -------------- Axis setting
    # ----------------------------------------------------

    # -------------- solution axis
    ax_sol=axs[0].twinx()

    ax_sol.spines.bottom.set_visible(True)
    ax_sol.set_zorder(ax_sol.get_zorder() + 1)
    ax_sol.yaxis.label.set_color(sol_c)
    ax_sol.spines.right.set_color(sol_c)
    ax_sol.spines.right.set_linewidth(2)
    ax_sol.spines.right.set_position(("axes", 1))
    ax_sol.spines.right.set_visible(True)
    ax_sol.spines.left.set_visible(False)

    ax_sol.set_xticklabels([])
    ax_sol.set_ylabel("avg. no. of solutions", fontsize=14, labelpad=10)     

    ax_sol.tick_params(axis='y', direction='in', length=8.5, width=1.5, which='major', colors=sol_c, top=True, bottom=True, left=False, right=True)
    ax_sol.tick_params(axis='y', direction='in', length=5.5, width=1.5, which='minor', colors=sol_c, top=True, bottom=True, left=False, right=True)
    ax_sol.tick_params(axis='x', direction='in', length=8.5, width=1.5, which='major', colors='k',   top=True, bottom=True, left=False, right=True)
    ax_sol.tick_params(axis='x', direction='in', length=5.5, width=1.5, which='minor', colors='k',   top=True, bottom=True, left=False, right=True)

    # -------------- error axis
    ax = axs[0]
    ax.spines.bottom.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.yaxis.label.set_color(err_c)
    ax.spines.left.set_color(err_c)
    ax.spines.left.set_linewidth(2)

    ax.tick_params(axis='y', direction='in', length=8.5, width=1.3, which='major', colors=err_c, top=True, bottom=True, right=False)
    ax.tick_params(axis='y', direction='in', length=5.5, width=1.3, which='minor', colors=err_c, top=True, bottom=True, right=False)
    ax.tick_params(axis='x', direction='in', length=8.5, width=1.3, which='major', colors='k',   top=True, bottom=True, right=False)
    ax.tick_params(axis='x', direction='in', length=5.5, width=1.3, which='minor', colors='k',   top=True, bottom=True, right=False)

    ax.set_xticklabels([])
    ax.set_ylabel("avg. max. error", fontsize=14, labelpad=14)

    # -------------- time axis
    axs[1].tick_params(axis='both', direction='in', colors='k', length=8.5, width=1.3, which='major', top=True, bottom=True, left=True, right=True)
    axs[1].tick_params(axis='both', direction='in', colors='k', length=5.5, width=1.3, which='minor', top=True, bottom=True, left=True, right=True)

    axs[1].set_ylabel("$t_\mathrm{total}$ (s)",  fontsize=14, labelpad=14)
    axs[1].set_xlabel("Reflections",  fontsize=14, labelpad=12)

    # ----------------------------------------------------
    # -------------- plotting results --------------------
    # ----------------------------------------------------


    # --------------- Plot  time
    # ----------------------------------------------------
    boxpro     = dict(linestyle='--', linewidth=0.1, alpha=1)
    medianpro  = dict(color='k', linestyle='-', linewidth=1.5, alpha=1)
    meanpro    = dict(linestyle='-', linewidth=0.5, color='k')
    whiskerpro = dict(color="red", linewidth = 1.2, alpha=0.7)
    capprop    = dict(color="red", linewidth = 1.2, alpha=0.7) 
    
    for ci, i in enumerate(reflections):
        fn  = os.path.join(fpath,'pnew_%g.h5'%(i))
        
        with h5py.File(fn, 'r') as f:
            
            tinf = f.get('time_total')
            tt   = [np.array(tinf.get(ii)) for ii in np.array(tinf)] ; tt = np.array(tt)
            tti  = [np.max(np.array(tinf.get(ii)))for ii in np.array(tinf)]
            t.append(tt[:,6]); tave.append(tti)
            
            #er    = f.get('solution_error')
            #err=[np.array(er.get(i)) for i in np.array(er)]
            err=np.array(readh5file_v2(fn)[2]) 
            emaxi=[np.max(i) for i in err]
            emax.append(emaxi)
            
            #ns  = f.get('allsolution')
            #nss = [(np.shape(np.array(f.get('allsolution/Pair'+str(ii))))[0])/2 for ii in range(1, len(ns)+1)]
            nss=np.array(readh5file_v2(fn)[7]) 
            nsol.append(nss)
            
            vave.append(readh5file_v2(fn)[1])

    bp = axs[1].boxplot(t, labels=reflections, notch=False, patch_artist=True, boxprops=boxpro, meanprops=meanpro, meanline=False, showmeans=False, showfliers=False, 
                            medianprops=medianpro, whiskerprops=whiskerpro, capprops=capprop)

    for bi, box in enumerate(bp['boxes']):
        box.set(facecolor=cmap(norm(vave[bi])), linewidth=1, alpha=0.65) 
    for bi, box in enumerate(bp['whiskers']):
        if bi%2==0:
            binx = int(bi/2)
        box.set(color=cmap(norm(vave[binx])), linewidth=2, alpha=1)
    for bi, box in enumerate(bp['caps']):
        if bi%2==0:
            binx = int(bi/2)
        box.set(color=cmap(norm(vave[binx])), linewidth=2, alpha=1)

    # --------------- Plot Error
    # ---------------------------------------------------- 
    bp_err = ax.boxplot(emax, labels=reflections, notch=False, showfliers=False, whis=False, patch_artist=False, showcaps=False, showbox=False, showmeans=True, meanline=False,
                            medianprops=dict(linestyle='-', linewidth=0, alpha=1, color='k'),
                            meanprops=dict(linestyle='-', linewidth=0, color=err_c, marker="o",markerfacecolor=err_c, markeredgecolor="k", alpha=0.5))
                
    X, Y = [], []
    for m in bp_err['means'][:]:
        lx, ly = m.get_data() ; X.append(lx[0]); Y.append(ly[0])
    ax.plot(X,Y,'-',c=err_c, lw=2, alpha=0.75, zorder=-1)

    # --------------- Plot no. of solutions
    # ----------------------------------------------------
    bp_sol = ax_sol.boxplot(nsol, labels=reflections, notch=False, showfliers=False, whis=False, patch_artist=False, showcaps=False, showbox=False, showmeans=True, meanline=False,
                            medianprops=dict(linestyle='-', linewidth=0, alpha=1, color='k'),
                            meanprops=dict(linestyle='-', linewidth=0, color=sol_c, marker="o",markerfacecolor=sol_c, markeredgecolor="k", alpha=0.5))

    X, Y = [], []
    for m in bp_sol['means'][:]:
        lx, ly = m.get_data();  X.append(lx[0]);  Y.append(ly[0])
    ax_sol.plot(X, Y, '-', c=sol_c, lw=2, alpha=0.75, zorder=-1)
        
    # --------------- Plot colour bar
    # ----------------------------------------------------

    sm   =  ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axs[:], orientation='vertical', pad=0.075, aspect=30, shrink=1.0)
    cbar.set_label(r'$\mathrm{log~ Volume_{~average}}$', fontsize=16, loc='center', labelpad=3.0)
    cbar.ax.tick_params(width=1.5, length=8, which='major')
    cbar.ax.tick_params(width=1.0, length=5, which='minor')
    cbar.ax.tick_params(labelsize=16)
    #print(f'ax.yaxis.get_label().get_position()  : {axs[0].yaxis.get_label().get_position(), axs[1].yaxis.get_label().get_position() }')

    axs[1].yaxis.set_label_coords(-0.18, 0.5)
    axs[0].yaxis.set_label_coords(-0.18, 0.5)
    
    if savefig:
        fig.savefig(figname+"."+figtype, dpi=300, bbox_inches='tight')
        #plt.savefig(os.path.join(fpath, 'Timeinfo.pdf'), dpi=250, bbox_inches='tight')# when want to save in the some location of results directory
        
    
    plt.show()
    
    return

# def plottotaltime(reflections, fpath, figname='ex', figtype='pdf', savefig=True):
#     """_This fn plots the time information (total time) for each processed reflection.
#     all data are read from the *.h5 file. ALong with total time, this will plot no of solutions and error
#     1. Declare path of folder which contains the .h5 files for each reflections.
#     2. Have a list of reflection
#     3. select a name for file to save the plot of total time 
#     4. select a extension of the file like pdf or pnf or jpeg
#     5. savefig decides to save or not fig.
    
#     Following are required_
    
#     Args:
#         reflections (list): List of processed reflections
#         fpath (str): Path of folder which contains the h5 files
#         figname (str, optional): Name to save plotted picture. Defaults to 'ex'.
#         figtype (str, optional): type of extension pdf or png or jpeg. Defaults to 'pdf'.
#         savefig (bool, optional): choice to switch off save fig. Defaults to True.
#     """    
#     print(f'---> I am reading files from the path :: {fpath}')
        
#     mmean, meanl = [], []

#     for ci, ls in enumerate(reflections):
#         fn = os.path.join(fpath,'pnew_%g.h5'%(ls))
#         #print(f'---> fn : {fn}')    
#         mmean.append(readh5file_v2(fn)[1])
#         meanl.append(readh5file_v2(fn)[1])

#     cmap  = plt.get_cmap("cool")
#     norml = cm.colors.LogNorm(vmax=1*np.max(meanl), vmin=1*np.min(meanl))    
#     norm  = cm.colors.LogNorm(vmax=1*np.max(mmean), vmin=1*np.min(mmean))
    
#     fig, axs = plt.subplots(nrows=2, ncols=1, figsize =(7,5), sharex=True, sharey=False, subplot_kw={'aspect':'auto'},
#                             constrained_layout=True, gridspec_kw={'wspace': 0.00,'hspace': 0.00, 'height_ratios':[1, 1.15]})

#     plt.rc('xtick', labelsize=14)
#     plt.rc('ytick', labelsize=14)

#     emax, t,  nsol   = [], [], []
#     tave, eave, vave = [], [], []

#     err_c, sol_c = '#006400', 'C3'

#     # ----------------------------------------------------
#     # -------------- Axis setting
#     # ----------------------------------------------------

#     # -------------- solution axis
#     ax_sol=axs[0].twinx()

#     ax_sol.spines.bottom.set_visible(True)
#     ax_sol.set_zorder(ax_sol.get_zorder() + 1)
#     ax_sol.yaxis.label.set_color(sol_c)
#     ax_sol.spines.right.set_color(sol_c)
#     ax_sol.spines.right.set_linewidth(2)
#     ax_sol.spines.right.set_position(("axes", 1))
#     ax_sol.spines.right.set_visible(True)
#     ax_sol.spines.left.set_visible(False)

#     ax_sol.set_xticklabels([])
#     ax_sol.set_ylabel("avg. no. of solutions", fontsize=14, labelpad=10)     

#     ax_sol.tick_params(axis='y', direction='in', length=8.5, width=1.5, which='major', colors=sol_c, top=True, bottom=True, left=False, right=True)
#     ax_sol.tick_params(axis='y', direction='in', length=5.5, width=1.5, which='minor', colors=sol_c, top=True, bottom=True, left=False, right=True)
#     ax_sol.tick_params(axis='x', direction='in', length=8.5, width=1.5, which='major', colors='k',   top=True, bottom=True, left=False, right=True)
#     ax_sol.tick_params(axis='x', direction='in', length=5.5, width=1.5, which='minor', colors='k',   top=True, bottom=True, left=False, right=True)

#     # -------------- error axis
#     ax = axs[0]
#     ax.spines.bottom.set_visible(False)
#     ax.spines.right.set_visible(False)
#     ax.yaxis.label.set_color(err_c)
#     ax.spines.left.set_color(err_c)
#     ax.spines.left.set_linewidth(2)

#     ax.tick_params(axis='y', direction='in', length=8.5, width=1.3, which='major', colors=err_c, top=True, bottom=True, right=False)
#     ax.tick_params(axis='y', direction='in', length=5.5, width=1.3, which='minor', colors=err_c, top=True, bottom=True, right=False)
#     ax.tick_params(axis='x', direction='in', length=8.5, width=1.3, which='major', colors='k',   top=True, bottom=True, right=False)
#     ax.tick_params(axis='x', direction='in', length=5.5, width=1.3, which='minor', colors='k',   top=True, bottom=True, right=False)

#     ax.set_xticklabels([])
#     ax.set_ylabel("avg. max. error", fontsize=14, labelpad=14)

#     # -------------- time axis
#     axs[1].tick_params(axis='both', direction='in', colors='k', length=8.5, width=1.3, which='major', top=True, bottom=True, left=True, right=True)
#     axs[1].tick_params(axis='both', direction='in', colors='k', length=5.5, width=1.3, which='minor', top=True, bottom=True, left=True, right=True)

#     axs[1].set_ylabel("$t_\mathrm{total}$ (s)",  fontsize=14, labelpad=14)
#     axs[1].set_xlabel("Reflections",  fontsize=14, labelpad=12)

#     # ----------------------------------------------------
#     # -------------- plotting results --------------------
#     # ----------------------------------------------------


#     # --------------- Plot  time
#     # ----------------------------------------------------
#     boxpro     = dict(linestyle='--', linewidth=0.1, alpha=1)
#     medianpro  = dict(color='k', linestyle='-', linewidth=1.5, alpha=1)
#     meanpro    = dict(linestyle='-', linewidth=0.5, color='k')
#     whiskerpro = dict(color="red", linewidth = 1.2, alpha=0.7)
#     capprop    = dict(color="red", linewidth = 1.2, alpha=0.7) 
    
#     for ci, i in enumerate(reflections):
#         fn  = os.path.join(fpath,'pnew_%g.h5'%(i))
        
#         with h5py.File(fn, 'r') as f:
            
#             tinf = f.get('timeexe')
#             tt   = [np.array(tinf.get(ii)) for ii in np.array(tinf)] ; tt = np.array(tt)
#             tti  = [np.max(np.array(tinf.get(ii)))for ii in np.array(tinf)]
#             t.append(tt[:,6]); tave.append(tti)
            
#             er    = f.get('error')
#             err   = [np.array(er.get(i)) for i in np.array(er)]
#             emaxi = [np.max(i) for i in err]
#             emax.append(emaxi)
            
#             ns  = f.get('allsolution')
#             nss = [(np.shape(np.array(f.get('allsolution/Pair'+str(ii))))[0])/2 for ii in range(1, len(ns)+1)]
#             nsol.append(nss)
            
#             vave.append(readh5file_v2(fn)[1])

#     bp = axs[1].boxplot(t, labels=reflections, notch=False, patch_artist=True, boxprops=boxpro, meanprops=meanpro, meanline=False, showmeans=False, showfliers=False, 
#                             medianprops=medianpro, whiskerprops=whiskerpro, capprops=capprop)

#     for bi, box in enumerate(bp['boxes']):
#         box.set(facecolor=cmap(norm(vave[bi])), linewidth=1, alpha=0.65) 
#     for bi, box in enumerate(bp['whiskers']):
#         if bi%2==0:
#             binx = int(bi/2)
#         box.set(color=cmap(norm(vave[binx])), linewidth=2, alpha=1)
#     for bi, box in enumerate(bp['caps']):
#         if bi%2==0:
#             binx = int(bi/2)
#         box.set(color=cmap(norm(vave[binx])), linewidth=2, alpha=1)

#     # --------------- Plot Error
#     # ---------------------------------------------------- 
#     bp_err = ax.boxplot(emax, labels=reflections, notch=False, showfliers=False, whis=False, patch_artist=False, showcaps=False, showbox=False, showmeans=True, meanline=False,
#                             medianprops=dict(linestyle='-', linewidth=0, alpha=1, color='k'),
#                             meanprops=dict(linestyle='-', linewidth=0, color=err_c, marker="o",markerfacecolor=err_c, markeredgecolor="k", alpha=0.5))
                
#     X, Y = [], []
#     for m in bp_err['means'][:]:
#         lx, ly = m.get_data() ; X.append(lx[0]); Y.append(ly[0])
#     ax.plot(X,Y,'-',c=err_c, lw=2, alpha=0.75, zorder=-1)

#     # --------------- Plot no. of solutions
#     # ----------------------------------------------------
#     bp_sol = ax_sol.boxplot(nsol, labels=reflections, notch=False, showfliers=False, whis=False, patch_artist=False, showcaps=False, showbox=False, showmeans=True, meanline=False,
#                             medianprops=dict(linestyle='-', linewidth=0, alpha=1, color='k'),
#                             meanprops=dict(linestyle='-', linewidth=0, color=sol_c, marker="o",markerfacecolor=sol_c, markeredgecolor="k", alpha=0.5))

#     X, Y = [], []
#     for m in bp_sol['means'][:]:
#         lx, ly = m.get_data();  X.append(lx[0]);  Y.append(ly[0])
#     ax_sol.plot(X, Y, '-', c=sol_c, lw=2, alpha=0.75, zorder=-1)
        
#     # --------------- Plot colour bar
#     # ----------------------------------------------------

#     sm   =  ScalarMappable(norm=norm, cmap=cmap)
#     cbar = fig.colorbar(sm, ax=axs[:], orientation='vertical', pad=0.075, aspect=30, shrink=1.0)
#     cbar.set_label(r'$\mathrm{log~ Volume_{~average}}$', fontsize=16, loc='center', labelpad=3.0)
#     cbar.ax.tick_params(width=1.5, length=8, which='major')
#     cbar.ax.tick_params(width=1.0, length=5, which='minor')
#     cbar.ax.tick_params(labelsize=16)
#     #print(f'ax.yaxis.get_label().get_position()  : {axs[0].yaxis.get_label().get_position(), axs[1].yaxis.get_label().get_position() }')

#     axs[1].yaxis.set_label_coords(-0.18, 0.5)
#     axs[0].yaxis.set_label_coords(-0.18, 0.5)
    
#     if savefig:
#         fig.savefig(figname+"."+figtype, dpi=300, bbox_inches='tight')
    
#     plt.show()
    
#     return



