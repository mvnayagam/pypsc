import h5py
import numpy as np
import polytope as pc
from pathlib import Path
import re

class SolutionReader:

    def __init__(self, filename):
        self.filename = str(filename)
    
    # =========================================================
    # -------------------- HELPERS ----------------------------
    # =========================================================
    
    def _sort_keys(self, keys):
        return sorted(keys, key=lambda x: self._num(x))
    
    def _num(self, key):
        
        m = re.findall(r"\d+", str(key))
        return int(m[0]) if len(m) > 0 else None
    
    def _safe_keys(self, h5, path):

        if path not in h5:
            return []
        return list(h5[path].keys())
    
    # =========================================================
    # -------------------- RUNS -------------------------------
    # =========================================================
    def get_runs(self):
        with h5py.File(self.filename, "r") as h5:
            return sorted([k for k in h5.keys() if k.startswith("run_")])
    
    # =========================================================
    # -------------------- VOLUME -----------------------------
    # =========================================================
    def volume_convergence(self, run=None):
        
        with h5py.File(self.filename, "r") as h5:
            run = run or self.get_runs()[-1]
            base = f"{run}/stages/intersector"
            volume, keys = [], []
            
            for k in self._safe_keys(h5, base):
                p = f"{base}/{k}/stats/total_volume"
                
                if p in h5:
                    keys.append(k)
                    volume.append(h5[p][()])
        
        return { "key": np.array(keys), "volume": np.array(volume) }
    
    # =========================================================
    # -------------------- ERROR ------------------------------
    # =========================================================
    def polytope_error(self, polys):
        
        ext = []
        
        for p in polys:
            try:
                ext.append(pc.extreme(p))
            except:
                continue
        if len(ext) == 0:
            return np.array([])
        ext = np.vstack(ext)
        
        return (np.max(ext, axis=0) - np.min(ext, axis=0)) / 2
    
    # =========================================================
    # -------------------- ERROR CONVERGENCE -----------------
    # =========================================================
    def error_norm(self, error):
        
        if len(error)==0:
            return np.nan
        return np.linalg.norm(error)
    
    def error_convergence(self, run=None):
        
        with h5py.File(self.filename, "r") as h5:
            run = run or self.get_runs()[-1]
            base = f"{run}/stages/intersector"
            
            error, keys = [], []
            
            # for k in self._safe_keys(h5, base):
            for k in self._sort_keys(self._safe_keys(h5, base)):
                
                polys = []
                
                for name in self._safe_keys(h5, f"{base}/{k}"):
                    if name.startswith("poly_"):
                        # g = h5[f"{base}/{k}/{name}"]
                        g = h5.get(f"{base}/{k}/{name}")
                        try:
                            A = g["A"][()]
                            b = g["b"][()]
                            polys.append(pc.Polytope(A, b))
                        except:
                            continue
                if len(polys) == 0:
                    continue
                keys.append(k)
                # error.append(self.polytope_error(polys))
                error.append(self.error_norm(self.polytope_error(polys)))
        res = { "key": np.array(keys), "error": np.array(error) }
        
        return res
     
    # =========================================================
    # -------------------- TIME CONVERGENCE -----------------
    # =========================================================
    
    def timing_summary(self, run=None):
        
        run = run or self.get_runs()[-1]
        base = f"{run}/timing"
        
        total, per_level, intersector = {}, {}, {}
        
        with h5py.File(self.filename, "r") as h5:
            if base not in h5:
                return {}
            
            for module in h5[base].keys():
                mpath = f"{base}/{module}"
                
                # -------------------------
                # CASE 1: TOTAL TIMING
                # -------------------------
                if module == "total":
                    total = {}
                    for name in self._safe_keys(h5, mpath):
                        p = f"{mpath}/{name}/value"
                        if p in h5:
                            total[name] = h5[p][()]
                
                # -------------------------
                # CASE 2: PER LEVEL (l1-l10)
                # -------------------------
                elif any(k.startswith("l") for k in self._safe_keys(h5, mpath)):
                    per_level[module] = {}
                    for k in self._sort_keys(self._safe_keys(h5, mpath)):
                        p = f"{mpath}/{k}"
                        try:
                            per_level[module][k] = h5[p][()]
                        except:
                            continue
                
                # -------------------------
                # CASE 3: INTERSECTOR s12,s23...
                # -------------------------
                elif module == "Intersector":
                    intersector = {}
                    for k in self._safe_keys(h5, mpath):
                        p = f"{mpath}/{k}"
                        try:
                            intersector[k] = h5[p][()]
                        except:
                            continue
        
        return { "per_level": per_level, "total": total, "intersector": intersector }
    
    # =========================================================
    # -------------------- SUMMARY ----------------------------
    # =========================================================
    def summary(self, run=None):
        
        v = self.volume_convergence(run)
        e = self.error_convergence(run)
        
        timing = self.timing_summary(run)
        keys = list(v["key"])
        e_map = dict(zip(e["key"], e["error"]))
        volume = np.array(v["volume"])
        error = np.array([e_map.get(k, np.nan) for k in keys])
        step = np.array([self._num(k) for k in keys])
        
        return { "key": keys,
                "step": step,
                
                # physical quantities
                "volume": volume,
                "error": error,
                
                # complete timing tree
                "timing": timing,
                
                # convenient shortcuts
                "level_time": timing["per_level"],
                "module_time": timing["total"],
                "intersector_time": timing["intersector"]
                }

# # # USAGE
# reader = SolutionReader(str(Path("./tmp/pscsolver_20260623-010411-8f782e.h5")))
# data = reader.summary()
# print(data["key"])
# print(data["volume"])
# print(data["error"])
# data