from psc.lib.solutionreader import SolutionReader

import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

params = {'axes.labelsize': 14, 'axes.titlesize': 14, 
          'xtick.labelsize':14, 'ytick.labelsize':14,
          'legend.fontsize':14}
plt.rcParams.update(params)


class Plotstyle:

    @staticmethod
    def gridstyle(ax, ax2=None):
        ax.grid(which='major', linestyle='-', linewidth=0.8)
        ax.grid(which='minor', linestyle=':', linewidth=0.5)
        ax.set_axisbelow(True)

        if ax2 is not None:
            ax2.grid(which='major', zorder=-1000)
            ax2.set_axisbelow(True)

    @staticmethod
    def switchonticks(ax, ax2=None, axcolor=None, minor=True, fs=14):
        ax.tick_params(direction='in', which='both', top=True, right=True)

        if ax2 is not None:
            ax2.tick_params(direction='in', which='both', colors='r')
            ax.tick_params(axis='x', direction='in', which='major', length=8, top=True, bottom=True)
            ax.tick_params(axis='x', direction='in', which='minor', length=4, top=True, bottom=True)

            ax.tick_params(axis='y', direction='in', which='major', length=8, left=True, right=False)
            ax.tick_params(axis='y', direction='in', which='minor', length=4, left=True, right=False)

            ax.spines['right'].set_visible(False)
            # ax.spines['top'].set_visible(False)

            if minor:
                ax2.minorticks_on()
            else:
                ax2.minorticks_off()
            
            ax2.tick_params(axis='y', direction='in', which='major', length=8, left=False, right=True)
            ax2.tick_params(axis='y', direction='in', which='minor', length=4, left=False, right=True)
            ax2.tick_params(axis='y', which='both', colors='r')
            
            ax2.yaxis.label.set_color('r')
            ax2.spines['right'].set_color('r')
            ax2.spines['left'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.set_axisbelow(True)
            
        else:
            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_visible(True)
            ax.tick_params(axis='both', direction='in', which='major', length=8, top=True, bottom=True, left=True, right=True)
            ax.tick_params(axis='both', direction='in', which='minor', length=4, top=True, bottom=True, left=True, right=True)
        
        if axcolor is not None:
            ax.tick_params(axis='y', which='both', colors=axcolor)
            ax.yaxis.label.set_color(axcolor)
            ax.spines['left'].set_color(axcolor)
        else:
            ax.tick_params(axis='y', which='both', colors='b')
            ax.yaxis.label.set_color('b')
            ax.spines['left'].set_color('b')
                        
        ax.set_axisbelow(True)


class Plotting(Plotstyle):

    def __init__(self, datafile, savefig=True):
        self.datafile = datafile
        self.savefig = savefig

    # =========================================================
    # -------------------- ERROR HANDLER -----------------------
    # =========================================================
    def _error_norm(self, error):

        error = np.asarray(error)

        if error.ndim == 1:
            return np.abs(error)

        elif error.ndim == 2:
            return np.linalg.norm(error, axis=1)

        else:
            raise ValueError(f"Unsupported error shape: {error.shape}")

    # =========================================================
    # -------------------- CONVERGENCE -------------------------
    # =========================================================
    def _is_converged(self, x, window=3, tol=1e-3):
        x = np.asarray(x)
        if len(x) < window + 1:
            return False, None
        eps = 1e-12
        rel = np.abs(x[1:] - x[:-1]) / (np.abs(x[:-1]) + eps)
        tail = rel[-window:]
        conv = np.all(tail < tol)
        idx = len(x) - window if conv else None
        return conv, idx
    
    # =========================================================
    # -------------------- VOLUME PLOT -------------------------
    # =========================================================
    def plot_volume(self, run=None, tol=1e-3, window=3):

        data = self.datafile.volume_convergence(run)

        v = np.asarray(data["volume"])
        x = np.asarray(data["key"])

        conv, idx = self._is_converged(v, window, tol)

        plt.figure(figsize=(8,6))

        plt.plot(x, v, "ob-")
        plt.scatter(x, v)

        if conv:
            plt.axvline(x[idx], linestyle="--")

        plt.title("Volume Convergence")
        plt.xlabel("step")
        plt.ylabel("volume")

        self.gridstyle(plt.gca())
        self.switchonticks(plt.gca(), minor=False)

        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

        return {"converged": conv, "index": idx}
    
    # =========================================================
    # -------------------- ERROR PLOT --------------------------
    # =========================================================
    def plot_error(self, run=None, tol=1e-3, window=3):

        data = self.datafile.error_convergence(run)
        
        e = self._error_norm(data["error"])
        x = np.asarray(data["key"])
        conv, idx = self._is_converged(e, window, tol)
        
        plt.figure(figsize=(8,6))
        plt.plot(x, e, "ob-")
        plt.scatter(x, e)
        
        if conv:
            plt.axvline(x[idx], linestyle="--")
            
        plt.title("Error Convergence (L2 norm)")
        plt.xlabel("step")
        plt.ylabel("error")
        
        self.gridstyle(plt.gca())
        self.switchonticks(plt.gca(), minor=False)
        
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

        return {"converged": conv, "index": idx}
    
    # =========================================================
    # ---------------- DUAL CONVERGENCE ------------------------
    # =========================================================
    def plot_dual_convergence(self, run=None, tol=1e-3, window=3):

        vol_data = self.datafile.volume_convergence(run)
        err_data = self.datafile.error_convergence(run)
        
        v = np.asarray(vol_data["volume"])
        e = self._error_norm(err_data["error"])
        
        keys_v = vol_data["key"]
        keys_e = err_data["key"]
        
        common_keys = [k for k in keys_v if k in keys_e]
        
        v_map = dict(zip(keys_v, v))
        e_map = dict(zip(keys_e, e))
        
        v = np.array([v_map[k] for k in common_keys])
        e = np.array([e_map[k] for k in common_keys])
        x = np.arange(len(common_keys))
        
        fig, ax1 = plt.subplots(figsize=(8,6))
        
        ax1.plot(x, v, "ob-")
        ax1.set_xlabel("Intersection step")
        ax1.set_ylabel("Volume")
        ax1.set_yscale("log")
        
        ax2 = ax1.twinx()
        ax2.plot(x, e, "or-")
        ax2.set_ylabel("Error (L2 norm)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(common_keys, rotation=45, ha="right")
        
        conv_v, idx_v = self._is_converged(v, window, tol)
        conv_e, idx_e = self._is_converged(e, window, tol)
        
        if conv_v:
            ax1.axvline(idx_v, linestyle="--")
        if conv_e:
            ax2.axvline(idx_e, linestyle="--")
            
        self.gridstyle(ax1)
        self.switchonticks(ax1, ax2=ax2, axcolor='b', minor=False)
        
        plt.title("Dual Convergence: Volume vs Error")
        plt.tight_layout()
        
        plt.savefig("plot_dual_convergence.pdf", dpi=300, bbox_inches='tight')
        plt.show()

        return {"volume_converged": conv_v, "error_converged": conv_e, "volume_index": idx_v, "error_index": idx_e}

    # =========================================================
    # ------------ UNIFIED CONVERGENCE METRIC ------------------
    # =========================================================
    def convergence_metric(self, run=None):

        vol_data = self.datafile.volume_convergence(run)
        err_data = self.datafile.error_convergence(run)
        
        vol = np.asarray(vol_data["volume"])
        err = np.asarray(err_data["error"])
        
        keys_v = vol_data["key"]
        keys_e = err_data["key"]
        err = self._error_norm(err)
        
        # align by common keys
        common_keys = [k for k in keys_v if k in keys_e]
        
        v_map = dict(zip(keys_v, vol))
        e_map = dict(zip(keys_e, err))
        
        vol = np.array([v_map[k] for k in common_keys])
        err = np.array([e_map[k] for k in common_keys])
        
        v0 = vol[0] if vol[0] != 0 else 1.0
        e0 = err[0] if err[0] != 0 else 1.0
        
        vn = vol / v0
        en = err / e0
        
        metric = vn * en
        
        return { "metric": metric, "volume_norm": vn, "error_norm": en, "keys": common_keys}
        
    def plot_convergence_metric(self, run=None):
        
        data = self.convergence_metric(run)
        
        c = data["metric"]
        x = data['keys'] #np.arange(len(c))
        
        fig, ax = plt.subplots(figsize=(8,6))
        
        ax.plot(x, c, "ob-")
        ax.set_yscale("log")
        ax.set_xlabel("Intersection step")
        ax.set_ylabel("Convergence metric")
        self.gridstyle(ax)
        self.switchonticks(ax, minor=False)
        
        plt.title("Unified Convergence Metric (Volume × Error)")
        plt.tight_layout()
        
        plt.savefig("plot_convergence_metric.pdf", dpi=300, bbox_inches='tight')
        plt.show()
        
    
    # =========================================================
    # ------------ TIME PLOTS ------------------
    # =========================================================

    # 1. Per-level timing plot
    def plot_per_level_timing(self, run=None):

        data = self.datafile.timing_summary(run)["per_level"]

        plt.figure(figsize=(8,5))

        for module, levels in data.items():

            x = list(levels.keys())
            y = list(levels.values())

            plt.plot(x, y, marker="o", label=module)
        
        plt.title(r"Per-reflection timing ($l_1$ – $l_{N}$)")
        plt.xlabel("Reflection")
        plt.ylabel("Time (s)")

        plt.legend()
        plt.xticks(rotation=0)
        
        self.gridstyle(plt.gca())
        self.switchonticks(plt.gca(), minor=False)

        plt.tight_layout()
        
        plt.savefig("plot_per_level_timing.pdf", dpi=300, bbox_inches='tight')
        plt.show()

    # 2. Scalar timing bar chart
    def plot_scalar_timing(self, run=None):

        data = self.datafile.timing_summary(run)["total"]

        plt.figure(figsize=(8,5))

        # plt.bar(list(data.keys()), list(data.values()))
        plt.bar( list(data.keys()), list(data.values()), color="skyblue",       # bar colour
                width=0.2,             # bar thickness (0-1)
                edgecolor="gray",     # border colour
                linewidth=1,           # border thickness
                alpha=0.7              # transparency
                )

        plt.title("Total time per moduler")
        plt.ylabel("time (s)")
        plt.xticks(rotation=45)

        self.gridstyle(plt.gca())
        self.switchonticks(plt.gca(), minor=False)

        plt.tight_layout()
        
        plt.savefig("plot_scalar_timing.pdf", dpi=300, bbox_inches='tight')
        plt.show()

    # 3. Intersector timing plot
    def plot_intersector_timing(self, run=None):

        data = self.timing_summary(run)["intersector"]

        # flatten (only one module expected)
        for module, values in data.items():

            x = list(values.keys())
            y = list(values.values())

            plt.figure(figsize=(8,5))
            plt.plot(x, y, marker="o")

            plt.title(f"Intersector Timing: {module}")
            plt.xlabel("state")
            plt.ylabel("time (s)")

            plt.xticks(rotation=45)

            self.gridstyle(plt.gca())
            self.switchonticks(plt.gca(), minor=False)

            plt.tight_layout()
            plt.savefig("plot_intersector_timing.pdf", dpi=300, bbox_inches='tight')
            plt.show()

    # =========================================================
    # -------------------- REPORT ------------------------------
    # =========================================================
    def report(self, run=None):
        v = self.plot_volume(run)
        e = self.plot_error(run)
        return {"volume_converged": v["converged"], "volume_index": v["index"], "error_converged": e["converged"], "error_index": e["index"]}

## ---- Usage
# datafile = SolutionReader(str(Path("./tmp/pairID0_pscsolver_20260628-074027-c2aa89.h5")))
# plot = Plotting(datafile)
# plot.report()
# plot.plot_dual_convergence()
# plot.plot_convergence_metric()
# plot.plot_per_level_timing()
# plot.plot_scalar_timing()
# plot.plot_intersector_timing()


# Usage
# reader = SolutionReader(str(Path("./tmp/pscsolver_20260623-010411-8f782e.h5")))
# plot = Plotting(reader)
# # plot.report()
# # plot.plot_dual_convergence()
# # plot.plot_convergence_metric()
# plot.plot_per_level_timing()
# plot.plot_scalar_timing()