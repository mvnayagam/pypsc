import math
import h5py
import logging
import numpy as np
import polytope as pc
from pathlib import Path
from datetime import datetime
import re


import math
import h5py
import logging
import numpy as np
import polytope as pc
from pathlib import Path
from datetime import datetime

class SolutionIO:

    def __init__(self, filename, logger=None):
        self.logger = logger or logging.getLogger("psc.solutionIO")
        self.filename = filename
        self.run_name = None
        self.run_path = None

    # =========================================================
    # -------------------- HELPERS ----------------------------
    # =========================================================

    @staticmethod
    def _vol(poly): return pc.volume(poly)

    @staticmethod
    def _centroid(poly): return np.mean(pc.extreme(poly), axis=0)

    @staticmethod
    def _err(ext): return np.abs(np.max(ext, axis=0) - np.min(ext, axis=0)) / 2

    def _radius(self, dim, vol): return ((vol * math.gamma(dim/2 + 1)) / (math.pi ** (dim/2))) ** (1/dim)

    def _norm(self, d):
        if isinstance(d, np.ndarray): return d
        if isinstance(d, (np.floating, np.integer)): return d.item()
        if isinstance(d, np.bool_): return bool(d)
        return d

    def _grp(self, h5, p): return h5[p] if p in h5 else h5.create_group(p)

    def _runid(self, h5): return (max([int(k.split("_")[1]) for k in h5.keys() if k.startswith("run_")] + [0]) + 1)

    def _encode(self, v): return v.encode("utf-8") if isinstance(v, str) else v

    # =========================================================
    # -------------------- CORE IO ----------------------------
    # =========================================================

    def write_dataset(self, h5, path, data):
        if path in h5: del h5[path]
        data = self._norm(data)

        if isinstance(data, dict):
            g = h5.create_group(path)
            for k, v in data.items():
                if v is None: continue
                self.write_dataset(h5, f"{path}/{k}", v)

        elif isinstance(data, list):
            if len(data) == 0: h5.create_dataset(path, data=[])
            elif all(isinstance(x, (int, float, np.number)) for x in data):
                h5.create_dataset(path, data=np.asarray(data))
            else:
                g = h5.create_group(path)
                for i, v in enumerate(data):
                    self.write_dataset(h5, f"{path}/{i}", v)

        elif isinstance(data, np.ndarray):
            h5.create_dataset(path, data=data)

        elif isinstance(data, (int, float, bool, np.number)):
            h5.create_dataset(path, data=data)

        elif isinstance(data, str):
            h5.create_dataset(path, data=data.encode("utf-8"))

        elif isinstance(data, pc.Polytope):
            g = h5.create_group(path)
            g.create_dataset("A", data=np.asarray(data.A))
            g.create_dataset("b", data=np.asarray(data.b))

        elif isinstance(data, pc.Region):
            g = h5.create_group(path)
            for i, p in enumerate(data):
                sub = g.create_group(f"poly_{i}")
                sub.create_dataset("A", data=np.asarray(p.A))
                sub.create_dataset("b", data=np.asarray(p.b))

        else:
            raise TypeError(f"Unsupported type {type(data)} at {path}")

    # =========================================================
    # -------------------- RUN -------------------------------
    # =========================================================
    def init_run(self):
        with h5py.File(self.filename, "a") as h5:
            self.run_name = f"run_{self._runid(h5):04d}"
            self.run_path = f"/{self.run_name}"
            for p in ["stages", "logs", "state", "results", "metadata"]:
                self._grp(h5, f"{self.run_path}/{p}")
            self.write_metadata(start_time=str(datetime.now()), status="running")

    # =========================================================
    # -------------------- METADATA ---------------------------
    # =========================================================
    def write_metadata(self, **kw):
        with h5py.File(self.filename, "a") as h5:
            p = f"{self.run_path}/metadata"
            self._grp(h5, p)
            for k, v in kw.items():
                if v is None: continue
                self.write_dataset(h5, f"{p}/{k}", v)

    # =========================================================
    # ------------------- TIMING INFO -------------------------
    # =========================================================
    def write_timing(self, stage, name=None, value=None):

        with h5py.File(self.filename, "a") as h5:

            path = f"{self.run_path}/timing/{stage}"
            self._grp(h5, path)

            if value is None:
                value = float(name)   # allow shortcut mode

                key = "value"

            else:
                key = name
                value = float(value)

            if key in h5[path]:
                del h5[path][key]

            h5[path].create_dataset(key, data=value)
    
    # =========================================================
    # -------------------- STATE + CHECKPOINT -----------------
    # =========================================================
    def save_state(self, step, data):
        with h5py.File(self.filename, "a") as h5:
            p = f"{self.run_path}/state/{step}"
            self.write_dataset(h5, p, data)
            self.write_dataset(h5, f"{self.run_path}/state/last_step", step)
            self.write_dataset(h5, f"{self.run_path}/state/last_time", str(datetime.now()))

    def checkpoint(self, stage, substep="", index=-1):
        with h5py.File(self.filename, "a") as h5:
            p = f"{self.run_path}/state/checkpoint"
            self._grp(h5, p)
            self.write_dataset(h5, f"{p}/stage", stage)
            self.write_dataset(h5, f"{p}/substep", substep)
            if index is not None:
                self.write_dataset(h5, f"{p}/index", index)
            self.write_dataset(h5, f"{p}/time", str(datetime.now()))

    def load_checkpoint(self):
        with h5py.File(self.filename, "r") as h5:
            p = f"{self.run_path}/state/checkpoint"
            if p not in h5: return None
            def r(x): return x[()].decode() if isinstance(x[()], bytes) else x[()]
            return {"stage": r(h5[f"{p}/stage"]), "substep": r(h5[f"{p}/substep"]), "index": int(h5[f"{p}/index"][()]) if f"{p}/index" in h5 else None}

    # =========================================================
    # -------------------- STAGES -----------------------------
    # =========================================================
    def log_stage(self, h5, stage): self.write_dataset(h5, f"{self.run_path}/logs/{stage}", str(datetime.now()))

    def write_stage(self, stage, data):
        with h5py.File(self.filename, "a") as h5:
            p = f"{self.run_path}/stages/{stage}"
            self._grp(h5, p)
            for k, v in data.items():
                if v is None: continue
                self.write_dataset(h5, f"{p}/{k}", v)
            self.log_stage(h5, stage)

    
    # =========================================================
    # -------------------- WRITER -----------------------
    # =========================================================
    def write_polytope(self, L, polys):
        import h5py
        from pathlib import Path

        fn = Path(self.filename).with_name(f"{Path(self.filename).stem}_L{L}_polytope.h5")

        with h5py.File(fn, "w") as f:
            f.create_dataset("count", data=len(polys))
            self.write_dataset(f, "polytopes", polys)

        return str(fn)
        
    
    # =========================================================
    # -------------------- INTERSECTION -----------------------
    # =========================================================

    def write_intersection(self, key, polys):
        with h5py.File(self.filename, "a") as h5:
            p = f"{self.run_path}/stages/intersector/{key}"
            self.write_dataset(h5, p, polys)
            self.write_dataset(h5, f"{p}/count", len(polys))

    def write_intersection_stats(self, key, polys, dim):
        with h5py.File(self.filename, "a") as h5:
            base = f"{self.run_path}/stages/intersector/{key}/stats"
            geom = f"{self.run_path}/stages/intersector/{key}"
            self._grp(h5, base)

            vols = []
            for i, p in enumerate(polys):
                v = pc.volume(p)
                e = self._err(pc.extreme(p))
                c = np.mean(pc.extreme(p), axis=0)
                vols.append(v)
                self.write_dataset(h5, f"{geom}/volume/v{i}", v)
                self.write_dataset(h5, f"{geom}/centroid/p{i}", c)
                self.write_dataset(h5, f"{geom}/error/e{i}", e)

            total = np.sum(vols)
            self.write_dataset(h5, f"{base}/total_volume", total)
            self.write_dataset(h5, f"{base}/grand_radius", self._radius(dim, max(total, 0)))
            self.write_dataset(h5, f"{base}/number_of_solution", len(polys))

    # =========================================================
    # -------------------- CONVERGENCE ------------------------
    # =========================================================

    def write_convergence(self, data):
        with h5py.File(self.filename, "a") as h5:
            p = f"{self.run_path}/stages/intersector_convergence"
            self._grp(h5, p)
            for k, v in data.items():
                if v is None: continue
                if k == "key": v = np.array(v, dtype="S")
                else: v = np.asarray(v)
                self.write_dataset(h5, f"{p}/{k}", v)

    # =========================================================
    # -------------------- MERGER -----------------------------
    # =========================================================

    def write_solutionmerger(self, merged, comp):
        with h5py.File(self.filename, "a") as h5:
            p = f"{self.run_path}/stages/merger"
            self._grp(h5, p)
            self.log_stage(h5, "solutionmerger")

            self.write_dataset(h5, f"{p}/n_regions", len(merged))
            self.write_dataset(h5, f"{p}/success", comp == 1)
            self.write_dataset(h5, f"{p}/components", comp)

            for i, r in enumerate(merged):
                self.write_dataset(h5, f"{p}/merged_regions/region_{i}", r)

    # =========================================================
    # -------------------- SELECTOR ---------------------------
    # =========================================================

    def write_solutionselector(self, best_i, best_err, best_x, selectedpolytope, best_ext, loginfo):
        with h5py.File(self.filename, "a") as h5:
            p = f"{self.run_path}/stages/solutionselector"
            self._grp(h5, p)
            self.log_stage(h5, "solutionselector")

            self.write_dataset(h5, f"{p}/best_i", best_i)
            self.write_dataset(h5, f"{p}/best_err", best_err)
            self.write_dataset(h5, f"{p}/best_x", best_x)
            self.write_dataset(h5, f"{p}/best_ext", best_ext)
            self.write_dataset(h5, f"{p}/loginfo", loginfo)
            self.write_dataset(h5, f"{p}/selectedpolytope", selectedpolytope)

    # =========================================================
    # -------------------- FINAL ------------------------------
    # =========================================================

    def write_final_solution(self, sol, vol, coord, normal):
        with h5py.File(self.filename, "a") as h5:
            p = f"{self.run_path}/results"
            self._grp(h5, p)
            self.write_dataset(h5, f"{p}/solution", sol)
            self.write_dataset(h5, f"{p}/volume", vol)
            self.write_dataset(h5, f"{p}/coordinates", coord)
            self.write_dataset(h5, f"{p}/normal", normal)
            self.write_dataset(h5, f"{self.run_path}/metadata/status", "finished")


# class SolutionIO:

#     def __init__(self, filename, logger=None):
#         self.logger = logger or logging.getLogger("psc.solutionIO")
#         self.filename = filename
#         self.run_name = None
#         self.run_path = None

#     # =========================================================
#     # -------------------- HELPERS ----------------------------
#     # =========================================================

#     @staticmethod
#     def _vol(poly): return pc.volume(poly)

#     @staticmethod
#     def _centroid(poly): return np.mean(pc.extreme(poly), axis=0)

#     @staticmethod
#     def _err(ext): return np.abs(np.max(ext, axis=0) - np.min(ext, axis=0)) / 2

#     def _radius(self, dim, vol):
#         return ((vol * math.gamma(dim / 2 + 1)) / (math.pi ** (dim / 2))) ** (1 / dim)

#     def _norm(self, x):
#         if isinstance(x, np.ndarray): return x
#         if isinstance(x, (np.floating, np.integer)): return x.item()
#         if isinstance(x, np.bool_): return bool(x)
#         return x

#     def _grp(self, h5, path):
#         return h5[path] if path in h5 else h5.create_group(path)

#     def _run_id(self, h5):
#         runs = [k for k in h5.keys() if k.startswith("run_")]
#         ids = [int(k.split("_")[1]) for k in runs] if runs else [0]
#         return max(ids) + 1

#     def _encode(self, v):
#         return v.encode("utf-8") if isinstance(v, str) else v

#     # =========================================================
#     # -------------------- CORE IO ----------------------------
#     # =========================================================

#     def write_dataset(self, h5, path, data):

#         if path in h5:
#             del h5[path]

#         data = self._norm(data)

#         # dict
#         if isinstance(data, dict):
#             g = h5.create_group(path)
#             for k, v in data.items():
#                 self.write_dataset(h5, f"{path}/{k}", v)

#         # list
#         elif isinstance(data, list):
#             if len(data) == 0:
#                 h5.create_dataset(path, data=[])
#             elif all(isinstance(x, (int, float, np.number)) for x in data):
#                 h5.create_dataset(path, data=np.asarray(data))
#             else:
#                 g = h5.create_group(path)
#                 for i, v in enumerate(data):
#                     self.write_dataset(h5, f"{path}/{i}", v)

#         # numpy
#         elif isinstance(data, np.ndarray):
#             h5.create_dataset(path, data=data)

#         # scalar
#         elif isinstance(data, (int, float, bool, np.number)):
#             h5.create_dataset(path, data=data)

#         # string
#         elif isinstance(data, str):
#             h5.create_dataset(path, data=data.encode("utf-8"))

#         # polytope
#         elif isinstance(data, pc.Polytope):
#             g = h5.create_group(path)
#             g.create_dataset("A", data=np.asarray(data.A))
#             g.create_dataset("b", data=np.asarray(data.b))

#         # region
#         elif isinstance(data, pc.Region):
#             g = h5.create_group(path)
#             for i, p in enumerate(data):
#                 sub = g.create_group(f"poly_{i}")
#                 sub.create_dataset("A", data=np.asarray(p.A))
#                 sub.create_dataset("b", data=np.asarray(p.b))

#         else:
#             raise TypeError(f"Unsupported type {type(data)} at {path}")

#     # =========================================================
#     # -------------------- RUN INIT ---------------------------
#     # =========================================================

#     def init_run(self):

#         with h5py.File(self.filename, "a") as h5:
#             rid = self._run_id(h5)
#             self.run_name = f"run_{rid:04d}"
#             self.run_path = f"/{self.run_name}"

#             for p in ["stages", "logs", "state", "results", "metadata"]:
#                 self._grp(h5, f"{self.run_path}/{p}")

#             self.write_metadata(start_time=str(datetime.now()), status="running")

#     # =========================================================
#     # -------------------- METADATA ---------------------------
#     # =========================================================

#     def write_metadata(self, **kw):

#         with h5py.File(self.filename, "a") as h5:
#             p = f"{self.run_path}/metadata"
#             self._grp(h5, p)

#             for k, v in kw.items():
#                 if v is not None:
#                     self.write_dataset(h5, f"{p}/{k}", v)

#     # =========================================================
#     # -------------------- POLYTOPE ---------------------------
#     # =========================================================

#     def write_polytope(self, L, polys):

#         fname = Path(self.filename).with_name(
#             f"{Path(self.filename).stem}_L{L}_polytope.h5"
#         )

#         with h5py.File(fname, "w") as h5:
#             h5.create_dataset("n", data=len(polys))
#             self.write_dataset(h5, "polytopes", polys)

#         return fname

#     # =========================================================
#     # -------------------- STAGE ------------------------------
#     # =========================================================

#     def write_stage(self, stage, data):

#         with h5py.File(self.filename, "a") as h5:
#             p = f"{self.run_path}/stages/{stage}"
#             self._grp(h5, p)

#             for k, v in data.items():
#                 if v is not None:
#                     self.write_dataset(h5, f"{p}/{k}", v)

#     def log_stage(self, h5, stage):
#         self.write_dataset(h5, f"{self.run_path}/logs/{stage}", str(datetime.now()))

#     # =========================================================
#     # -------------------- INTERSECTION -----------------------
#     # =========================================================

#     def write_intersection(self, key, polys):

#         with h5py.File(self.filename, "a") as h5:
#             p = f"{self.run_path}/stages/intersector/{key}"
#             self._grp(h5, p)

#             self.write_dataset(h5, p, polys)
#             self.write_dataset(h5, f"{p}/count", len(polys))

#     # =========================================================
#     # -------------------- STATS ------------------------------
#     # =========================================================

#     def write_intersection_stats(self, key, polys, dim):

#         with h5py.File(self.filename, "a") as h5:

#             base = f"{self.run_path}/stages/intersector/{key}/stats"
#             geom = f"{self.run_path}/stages/intersector/{key}"

#             self._grp(h5, base)

#             vols = []

#             for i, p in enumerate(polys):

#                 v = pc.volume(p)
#                 ext = pc.extreme(p)

#                 vols.append(v)

#                 self.write_dataset(h5, f"{geom}/volume/v{i}", v)
#                 self.write_dataset(h5, f"{geom}/centroid/p{i}", np.mean(ext, axis=0))
#                 self.write_dataset(h5, f"{geom}/error/e{i}", self._err(ext))

#             total = np.sum(vols)

#             self.write_dataset(h5, f"{base}/total_volume", total)
#             self.write_dataset(h5, f"{base}/grand_radius", self._radius(dim, total))
#             self.write_dataset(h5, f"{base}/number_of_solution", len(polys))

#     # =========================================================
#     # -------------------- CONVERGENCE ------------------------
#     # =========================================================

#     def write_convergence(self, data):

#         with h5py.File(self.filename, "a") as h5:

#             p = f"{self.run_path}/stages/intersector_convergence"
#             self._grp(h5, p)

#             for k, v in data.items():
#                 if v is None:
#                     continue
#                 if k == "key":
#                     v = np.array(v, dtype="S")
#                 else:
#                     v = np.asarray(v)

#                 self.write_dataset(h5, f"{p}/{k}", v)

#     # =========================================================
#     # -------------------- MERGER -----------------------------
#     # =========================================================

#     def write_solutionmerger(self, merged_regions, components):

#         with h5py.File(self.filename, "a") as h5:

#             p = f"{self.run_path}/stages/merger"
#             self._grp(h5, p)

#             self.write_dataset(h5, f"{p}/n_regions", len(merged_regions))
#             self.write_dataset(h5, f"{p}/success", components == 1)
#             self.write_dataset(h5, f"{p}/components", components)

#             for i, r in enumerate(merged_regions):
#                 self.write_dataset(h5, f"{p}/merged_regions/region_{i}", r)

#     # =========================================================
#     # -------------------- SELECTOR ---------------------------
#     # =========================================================

#     def write_solutionselector(self, best_i, best_err, best_x, loginfo):

#         with h5py.File(self.filename, "a") as h5:

#             p = f"{self.run_path}/stages/solutionselector"
#             self._grp(h5, p)

#             self.write_dataset(h5, f"{p}/best_i", best_i)
#             self.write_dataset(h5, f"{p}/best_err", best_err)
#             self.write_dataset(h5, f"{p}/best_x", best_x)
#             self.write_dataset(h5, f"{p}/loginfo", loginfo)

#     # =========================================================
#     # -------------------- STATE ------------------------------
#     # =========================================================

#     def save_state(self, step, data):

#         with h5py.File(self.filename, "a") as h5:

#             p = f"{self.run_path}/state/{step}"

#             self.write_dataset(h5, p, data)
#             self.write_dataset(h5, f"{self.run_path}/state/last_step", step)
#             self.write_dataset(h5, f"{self.run_path}/state/last_time", str(datetime.now()))

#     # =========================================================
#     # -------------------- FINAL ------------------------------
#     # =========================================================

#     def write_final_solution(self, solution, volume, coordinates, normal):

#         with h5py.File(self.filename, "a") as h5:

#             p = f"{self.run_path}/results"
#             self._grp(h5, p)

#             self.write_dataset(h5, f"{p}/solution", solution)
#             self.write_dataset(h5, f"{p}/volume", volume)
#             self.write_dataset(h5, f"{p}/coordinates", coordinates)
#             self.write_dataset(h5, f"{p}/normal", normal)

#             self.write_dataset(h5, f"{self.run_path}/metadata/status", "finished")

#     # =========================================================
#     # ---------------- CHECKPOINT CONTROLLER ------------------
#     # =========================================================
    
#     # ---------------- WRITE CHECKPOINT ------------------
#     def write_checkpoint(self, stage, substep=None, index=None):
#         with h5py.File(self.filename, "a") as h5:
#             p = f"{self.run_path}/state/checkpoint"
#             self._grp(h5, p)
#             self.write_dataset(h5, f"{p}/stage", stage)
#             if substep is not None: self.write_dataset(h5, f"{p}/substep", substep)
#             if index is not None: self.write_dataset(h5, f"{p}/index", int(index))
#             self.write_dataset(h5, f"{p}/time", str(datetime.now()))
    
#     # ---------------- LOAD CHECKPOINT ------------------
#     def load_checkpoint(self):
#         with h5py.File(self.filename, "r") as h5:
#             p = f"{self.run_path}/state/checkpoint"
#             if p not in h5: return None
#             return {
#                 "stage": h5[f"{p}/stage"][()].decode() if isinstance(h5[f"{p}/stage"][()], bytes) else str(h5[f"{p}/stage"][()]),
#                 "substep": h5[f"{p}/substep"][()].decode() if f"{p}/substep" in h5 else None,
#                 "index": int(h5[f"{p}/index"][()]) if f"{p}/index" in h5 else None
#             }

#     def checkpoint(self, stage, substep=None, index=None):
#         self.save_state(stage, {"substep": substep, "index": index})
#         self.write_checkpoint(stage, substep=substep, index=index)


    
# import math
# import h5py
# import logging
# import numpy as np
# import polytope as pc
# from pathlib import Path
# from datetime import datetime

# class SolutionIO:

#     def __init__(self, filename, logger=None):
        
#         self.logger = logger or logging.getLogger("psc.solutionIO")
        
#         self.filename = filename
#         self.run_name = None
#         self.run_path = None
    
#     @staticmethod
#     def polytope_volume(poly):
#         return pc.volume(poly)
    
#     @staticmethod
#     def polytope_centroid(poly):
#         ext = pc.extreme(poly)
#         return np.mean(ext, axis=0)
    
#     @staticmethod
#     def solution_error(ext):
#         dmax = np.max(ext, axis=0)
#         dmin = np.min(ext, axis=0)
#         return np.abs(dmax - dmin) / 2

#     def radius_from_volume(self, dim: int, volume: float) -> float:
#         """returns radius of N-dimensional sphere

#         Args:
#             dim (int): dimension of PS
#             volume (float): volume of polytope in PS

#         Returns:
#             float: returns radius of N-dimensional sphere
#         """    
#         # Calculate radius of sphere in 'dimension'al space from its volume
#         R = ((volume * math.gamma((dim / 2) + 1)) / (math.pi ** (dim / 2))) ** (1 / dim)
#         return R

#     def write_metadata(self, **kwargs):
        
#         with h5py.File(self.filename, 'a') as h5file:    
#             metadata_path = f'{self.run_path}/metadata'
#             self.create_group( h5file, metadata_path)
            
#             for key, value in kwargs.items():
#                 if value is None:
#                     continue
                
#                 self.write_dataset( h5file, f'{metadata_path}/{key}', value)

#     def create_group(self, h5file, group_name: str):
#         return h5file[group_name] if group_name in h5file else h5file.create_group(group_name)
    
#     def get_polytope_filename(self, L):
#         path = Path(self.filename)
#         return path.with_name(f"{path.stem}_L{L}_polytope.h5")
        
#     def write_polytope(self, L, polys):
#         filename = self.get_polytope_filename(L)
        
#         with h5py.File(filename, "w") as h5file:
#             h5file.create_dataset("number_of_polytope", data=len(polys))
#             self.write_polytope_data( h5file, "polytopes", polys )
#         return filename

#     def _write_intersection_internal(self, h5file, key: str, polys):
#         path = f"intersector/{key}"
#         if isinstance(h5file, str):
#             with h5py.File(h5file, "a") as f:
#                 self._write_intersection_internal(f, key, polys)
#             return
#         self.write_polytope_data(h5file, path, polys)
#         h5file.create_dataset(f"{path}/count", data=len(polys))
    
#     def write_intersection(self, key: str, polys):
#         with h5py.File(self.filename, "a") as h5file:
#             self.log_stage(h5file, "write_intersection")
            
#             path = f"{self.run_path}/stages/intersector/{key}"
#             self.write_polytope_data(h5file, path, polys)
#             self.write_dataset(h5file, f"{path}/count", len(polys)) 
    
#     def _normalize(self, data):
        
#         if isinstance(data, np.ndarray):
#             return data
        
#         if isinstance(data, (np.floating, np.integer)):
#             return data.item()
        
#         if isinstance(data, np.bool_):
#             return bool(data)
        
#         return data
    
#     def write_dataset(self, h5file, path: str, data):
        
#         if path in h5file:
#             del h5file[path]
        
#         data = self._normalize(data)
        
#         # -------------------------
#         # dict -> create group
#         # -------------------------
#         if isinstance(data, dict):            
#             group = h5file.create_group(path)
            
#             for key, value in data.items():
#                 self.write_dataset( h5file, f"{path}/{key}",  value)
        
#         # -------------------------
#         # numpy scalar -> convert
#         # -------------------------
#         elif isinstance(data, (np.floating, np.integer)):
#             h5file.create_dataset(path, data=data.item())
        
#         # -------------------------
#         # list -> convert to array
#         # -------------------------
#         elif isinstance(data, list):
#             h5file.create_dataset(path, data=np.asarray(data))

#         # -------------------------
#         # numpy array -> store directly
#         # -------------------------
#         elif isinstance(data, np.ndarray):
#             h5file.create_dataset(path, data=data)
        
#         elif isinstance(data, bool):
#             h5file.create_dataset(path, data=data)

#         # -------------------------
#         # python scalar
#         # -------------------------
#         elif isinstance(data, (int, float)):
#             h5file.create_dataset(path, data=data)
        
#         # -------------------------
#         # string
#         # -------------------------
#         elif isinstance(data, str):
#             h5file.create_dataset(path, data=data.encode("utf-8"))
        
#         elif isinstance(data, pc.Polytope):
            
#             group = h5file.create_group(path)
#             group.create_dataset("A", data=np.asarray(data.A))
#             group.create_dataset("b", data=np.asarray(data.b))
        
#         elif isinstance(data, pc.Region):
            
#             group = h5file.create_group(path)

#             for i, poly in enumerate(data):

#                 sub = group.create_group(f"poly_{i}")
#                 sub.create_dataset("A", data=np.asarray(poly.A))
#                 sub.create_dataset("b", data=np.asarray(poly.b))
        
#         # -------------------------
#         # fallback (important for debugging)
#         # -------------------------
#         else:
#             raise TypeError(f"Unsupported type for HDF5: {type(data)} at {path}")

#     def write_polytope_data(self, h5file, path, polytope_data):
        
#         if polytope_data is None:
#             return
        
#         h5file.require_group(path)
#         poly_count = 0
        
#         for count, poly in enumerate(polytope_data):
            
#             # -------------------------
#             # pc.Polytope
#             # -------------------------
#             if isinstance(poly, pc.Polytope):
                
#                 self.write_dataset( h5file, f"{path}/polytope_{poly_count}/A", poly.A )
#                 self.write_dataset( h5file, f"{path}/polytope_{poly_count}/b", poly.b )
#                 poly_count += 1
            
#             # -------------------------
#             # pc.Region
#             # -------------------------
#             elif isinstance(poly, pc.Region):
                
#                 for subpoly in poly:
#                     self.write_dataset( h5file, f"{path}/polytope_{poly_count}/A", subpoly.A )
#                     self.write_dataset( h5file, f"{path}/polytope_{poly_count}/b", subpoly.b )
#                     poly_count += 1
            
#             else:
#                 raise TypeError( f"Unsupported polytope object: {type(poly)}" )

#     def init_run(self):
#         with h5py.File(self.filename, 'a') as h5file:
#             runs = [k for k in h5file.keys() if k.startswith('run_')]
#             run_id = len(runs) + 1
#             self.run_name = f'run_{run_id:04d}'
#             self.run_path = f'/{self.run_name}'
#             self.create_group(h5file, self.run_path)
#             self.create_group(h5file, f'{self.run_path}/stages')
#             self.create_group(h5file, f'{self.run_path}/logs')
#             self.create_group(h5file, f'{self.run_path}/state')
#             self.write_dataset(h5file, f'{self.run_path}/metadata/start_time', str(datetime.now()))
            
#     def log_stage(self, h5file, stage_name: str):
#         self.write_dataset(h5file, f'{self.run_path}/logs/{stage_name}', str(datetime.now()))
        
#     def write_stage(self, stage_name: str, data: dict):
        
#         with h5py.File(self.filename, 'a') as h5file:
            
#             stage_path = f'{self.run_path}/stages/{stage_name}'
#             self.create_group(h5file, stage_path)
            
#             for key, value in data.items():
#                 if value is None:
#                     continue
                
#                 self.write_dataset(h5file, f'{stage_path}/{key}', value)
                
#             self.log_stage(h5file, stage_name)
    
#     def write_intersection_stats(self, key: str, polys, dim: int):
        
#         with h5py.File(self.filename, "a") as h5file:
            
#             self.log_stage(h5file, "intersector_stats")
                        
#             base = f"{self.run_path}/stages/intersector/{key}/stats"
#             geom = f"{self.run_path}/stages/intersector/{key}"

#             self.create_group(h5file, base)

#             volumes, centroids, errors = [], [], []

#             for i, poly in enumerate(polys):

#                 v = pc.volume(poly)
#                 ext = pc.extreme(poly)
#                 c = np.mean(ext, axis=0)
#                 e = self.solution_error(ext)

#                 volumes.append(v)
#                 centroids.append(c)
#                 errors.append(e)

#                 self.write_dataset(h5file, f"{geom}/volume/v{i}", v)
#                 self.write_dataset(h5file, f"{geom}/centroid/p{i}", c)
#                 self.write_dataset(h5file, f"{geom}/error/e{i}", e)

#             volumes = np.array(volumes)
#             errors = np.array(errors)

#             total_volume = np.sum(volumes)
#             grand_radius = self.radius_from_volume(dim, total_volume)
#             total_error = np.mean(errors, axis=0)

#             self.write_dataset(h5file, f"{base}/total_volume", total_volume)
#             self.write_dataset(h5file, f"{base}/grand_radius", grand_radius)
#             self.write_dataset(h5file, f"{base}/number_of_solution", len(polys))
#             self.write_dataset(h5file, f"{base}/total_error", total_error)
            
#     def write_convergence(self, data: dict):

#         with h5py.File(self.filename, "a") as h5file:
            
#             self.log_stage(h5file, "intersector_convergence")
            
#             path = f"{self.run_path}/stages/intersector_convergence"
#             self.create_group(h5file, path)

#             for k, v in data.items():

#                 if v is None:
#                     continue

#                 # -------------------------
#                 # string list handling
#                 # -------------------------
#                 if k == "key":
#                     v = np.array(v, dtype="S")

#                 else:
#                     v = np.asarray(v)

#                 self.write_dataset(h5file, f"{path}/{k}", v)

#     def write_solutionmerger(self, merged_regions, components):
#         with h5py.File(self.filename, "a") as h5file:

#             path = f"{self.run_path}/stages/merger"
#             self.create_group(h5file, path)
            
#             self.log_stage(h5file, "solutionmerger")
                        
#             # -------------------------
#             # metadata (ADD HERE)
#             # -------------------------
#             self.write_dataset(h5file, f"{path}/n_regions", len(merged_regions))
#             self.write_dataset(h5file, f"{path}/success", components == 1)

#             # -------------------------
#             # components (scalar / int)
#             # -------------------------
#             self.write_dataset(h5file, f"{path}/components", np.asarray(components))
            
#             # -------------------------
#             # merged regions (geometry)
#             # -------------------------
#             for i, reg in enumerate(merged_regions):

#                 reg_path = f"{path}/merged_regions/region_{i}"

#                 # Polytope / Region handling
#                 if isinstance(reg, dict):
#                     self.write_dataset(h5file, reg_path, reg)

#                 else:
#                     self.write_dataset(h5file, reg_path, reg)

#     def write_solutionselector(self, best_i, best_err, best_x, loginfo):
#         with h5py.File(self.filename, "a") as h5file:
#             self.log_stage(h5file, "solutionselector")
            
#             path = f"{self.run_path}/stages/solutionselector"
            
            
#             self.create_group(h5file, path)
#             self.write_dataset(h5file, f"{path}/best_i", best_i)
#             self.write_dataset(h5file, f"{path}/best_err", best_err)
#             self.write_dataset(h5file, f"{path}/best_x", best_x)
            
#             for i, info in enumerate(loginfo):
#                 self.write_dataset(h5file, f"{path}/loginfo/polytope_{i}", info)

    
#     def save_state(self, step: str, data: dict):
        
#         with h5py.File(self.filename, "a") as h5file:
            
#             path = f"{self.run_path}/state/{step}"
            
#             self.write_dataset(h5file, path, data)
#             self.write_dataset(h5file, f"{self.run_path}/state/last_step", step)
#             self.write_dataset(h5file, f"{self.run_path}/state/last_time", str(datetime.now()))
    
#     def get_last_state(self):
        
#         with h5py.File(self.filename, 'r') as h5file:
            
#             path = f'{self.run_path}/state/last_step'
            
#             if path not in h5file:
#                 return None
            
#             return str(h5file[path][()])
        
#     def write_final_solution(self, solution, volume, coordinates, normal):
        
#         with h5py.File(self.filename, 'a') as h5file:
                
#             self.create_group(h5file, f'{self.run_path}/results')
#             self.write_dataset(h5file, f'{self.run_path}/results/final_polytope', solution)
#             self.write_dataset(h5file, f'{self.run_path}/results/volume', volume)
#             self.write_dataset(h5file, f'{self.run_path}/results/coordinates', coordinates)
#             self.write_dataset(h5file, f'{self.run_path}/results/normal', normal)
            
#     def readdata(self, path: str):
        
#         with h5py.File(self.filename, 'r') as h5file:
            
#             full_path = f'{self.run_path}{path}' if not path.startswith('/') else path
            
#             if path != "/" and full_path not in h5file:
                
#                 self.logger.warning(f'{full_path} not found')
                
#                 return None
            
#             if path != "/":
#                 return np.array(h5file[full_path])
            
#             def read_group(g):
                
#                 out = {}
                
#                 for k, v in g.items():
#                     if isinstance(v, h5py.Dataset):
#                         out[k] = np.array(v)
#                     else:
#                         out[k] = read_group(v)
#                 return out
            
#             return read_group(h5file[self.run_path])


# # How to use this class
# io = SolutionIO("psc_run.h5")  # create object
# io.write_metadata( version="PSC-v1.0.0", reflections=10, intensities={ 'l1': 1.2, 'l2': -0.3, 'l10': 5.6 } ) # at start of simulation

# io.write_intersection(reflection_id=5, polytope_data=poly_list ) # during calculation
# io.write_stage( "solution_space", { "polytope": solution_polytope, "volume": volume, "num_regions": n_regions} ) # pre-final before doing any mergeing 

# io.write_stage("merged_solution", { "polytope": merged_polytope, "volume": merged_volume } ) # after merging polytopes

# io.write_final_solution( solution=best_polytope, volume=best_volume, coordinates=coords, normal=normal_vec ) # after selecting particular solution space


# # # Reading data back

# data = io.readdata("results/volume") # to read specific path in h5file
# data = io.readdata("/") # use this to read full file



# ----------------------------------------------------------------------------------------------------------------

# I made this comment before making classes

# More standard method. This will be used in next versions both readdata and writedata are more flexible.
# Any number of parameters can be added or deleted. If either one of the function is edited, correspondingly 
# other function should also be edited
# _status on : 20.09.2024 18:18
# ----------------------------------------------------------------------------------------------------------------


def write_or_create_group(fname: str, group_name: str):
    """Helper function to create a group if it doesn't exist."""
    return fname[group_name] if group_name in fname else fname.create_group(group_name)

def write_dataset(fname: str, path: str, data: float) -> None:
    """Helper function to write data to the file."""
    #print(f"path : {path}")
    return fname.create_dataset(path, data=data, dtype='float64')

def write_polytope_data(fname: str, var_name: str, polytope_data: np.ndarray, pair_inx: int):
    """Helper function to write polytope data to the file."""
    if polytope_data is not None:
        tcount = 0
        write_or_create_group(fname, f'{var_name}/')
        for count, poly in enumerate(polytope_data):
            if isinstance(poly, pc.Polytope):  # Handling `Polytope`
                pa = f'/{var_name}/Pair{pair_inx}/pa{count+tcount}'
                pb = f'/{var_name}/Pair{pair_inx}/pb{count+tcount}'
                write_dataset(fname, pa, poly.A)
                write_dataset(fname, pb, poly.b)
            
            if isinstance(poly, pc.Region):  # Handling `Region`
                for counti, subpoly in enumerate(poly):
                    pa = f'/{var_name}/Pair{pair_inx}/pa{counti+count}'
                    pb = f'/{var_name}/Pair{pair_inx}/pb{counti+count}'
                    write_dataset(fname, pa, subpoly.A)
                    write_dataset(fname, pb, subpoly.b)
                tcount += 1
    return

def writedata(pair_inx: int, fname: str,
            allsolution: np.ndarray=None,
            allsolution_polytope:np.ndarray=None,
            coordinate_sorted: list=None, 
            distance: list=None,
            generatedcoordinate: list=None,
            gmagnitude: float=None, 
            grandradius: float=None, 
            normal: list=None, 
            solution: np.ndarray=None,
            solution_error: float=None,
            solution_extremepoint: np.ndarray=None,
            solution_polytope: np.ndarray=None,
            solution_totalNr: int=None,
            solution_volume: float=None, 
            time_total: list=None, 
            total_volume_in_Asym: float=None,
            ) -> None:
    
    """
     _ write the data at end of structure solving process or at any step
     _ This fn can be called to write one or more data sets. all data sets are None initially
       Requored data sets should be set to True by providing some data
     _ Ex: to write individual info, call this fn as 
        * writedata(pair_inx, fname, allsolution=allsolution)
        * writedata(pair_inx, fname, allsolution_polytope=allsolution_polytope)
        * writedata(pair_inx, fname, normal=normal)
        * writedata(pair_inx, fname, distance=distance)
        * writedata(pair_inx, fname, gmagnitude=gmagnitude)
        * writedata(pair_inx, fname, coordinate_sorted=coordinate_sorted)
        * writedata(pair_inx, fname, generatedcoordinate=generatedcoordinate)
        * writedata(pair_inx, fname, grandradius=grandradius)
        * writedata(pair_inx, fname, solution=solution)
        * writedata(pair_inx, fname, solution_error=solution_error)
        * writedata(pair_inx, fname, solution_extremepoint=exterem)
        * writedata(pair_inx, fname, solution_polytope=solution_poly)
        * writedata(pair_inx, fname, solution_totalNr=solution_totalNr)
        * writedata(pair_inx, fname, solution_volume=solution_volume)
        * writedata(pair_inx, fname, total_volume_in_Asym=volum_in_Asym)
        * writedata(pair_inx, fname, time_total=totaltime)
     _ Ex: to write more data at once: writedata(pair_inx, fname, allsolution=allsolution, time_total=totaltime)
     
     _ totaltime info contains [l, PairID, t_linearize, t_polytope, t_intersect, t_write, t_total]
     
     _ if you want to write more data include them in data_to_write tuple.
    """   
    
    # A list of tuples containing the group name, path, and the corresponding data.
    # In futue one can expand it to any order
    data_to_write = [
        ('allsolution'           , f'/allsolution/Pair{pair_inx}'         , allsolution           ),
        ('coordinate_sorted'     , f'/coordinate_sorted/pair{pair_inx}'   , coordinate_sorted     ),
        ('distance'              , f'/distance/distanceofPair{pair_inx}'  , distance              ),
        ('generatedcoordinate'   , f'/generatedcoordinate/pair{pair_inx}' , generatedcoordinate   ),
        ('gmagnitude'            , f'/gmagnitude/gofPair{pair_inx}'       , gmagnitude            ),       
        ('grandradius'           , f'/grandradius/r{pair_inx}'            , grandradius           ),
        ('normal'                , f'/normal/normalofPair{pair_inx}'      , normal                ),
        ('solution'              , f'/solution/solutionofPair{pair_inx}'  , solution              ),
        ('solution_error'        , f'/solution_error/eofPair{pair_inx}'   , solution_error        ),
        ('solution_extremepoint' , f'/solution_extremepoint/pa{pair_inx}' , solution_extremepoint ),
        ('solution_totalNr'      , f'/solution_totalNr/nsol{pair_inx}'    , solution_totalNr      ),
        ('solution_volume'       , f'/solution_volume/v{pair_inx}'        , solution_volume       ),
        ('time_total'             , f'/time_total/tofPair{pair_inx}'      , time_total            ),
        ('total_volume_in_Asym'  , f'/total_volume_in_Asym/v{pair_inx}'   , total_volume_in_Asym  ),
    ]
    #('distance'              , f'/distance/distanceofPair{pair_inx}'  , distance              ),               
    with h5py.File(fname, 'a') as fname:
        
        # Single for loop for many parameters other than `solution_polytope` and `allsolution_polytope`
        for group, path, data in data_to_write:
            if data is not None:  # Only write data if it's not None
                #print(f'---> Calling group: {group}')
                write_or_create_group(fname, group)
                write_dataset(fname, path, data)
        
        # Special handling for `solution_polytope` since it has multiple datasets (A and B)
        if solution_polytope is not None:
            write_polytope_data(fname, var_name='solution_polytope', polytope_data=[solution_polytope], pair_inx=pair_inx)

        # Special handling for `allsolution_polytope` since it has multiple datasets (A and B)
        if allsolution_polytope is not None:
            write_polytope_data(fname, var_name='allsolution_polytope', polytope_data=allsolution_polytope, pair_inx=pair_inx)
            
    return

def readdata(fname, datasets=None, pairID=None):
    
    with h5py.File(fname, 'r') as h5file:
        available_datasets = {
            'allsolution': 'allsolution',
            'generatedcoordinate': 'coordinate',
            'coordinate_sorted': 'coordinate_sorted',
            'grandradius': 'grandradius',
            'solution': 'solution',
            'solution_error': 'solution_error',
            'solution_volume': 'solution_volume',
            'solution_extremepoint': 'extremepoint',
            'solution_totalNr': 'total_solutionNr',
            'total_volume_in_Asym': 'total_volume_in_Asym',
            'distance': 'distance',
            'gmagnitude': 'gmagnitude',
            'normal': 'normal',
            'time_total': 'time_total',
            'allsolution_polytope': 'allsolution_polytope',
            'solution_polytope': 'solution_polytope'
        }
        
        data = {}
        
        # Ensure datasets is a list
        if isinstance(datasets, str):
            datasets = [datasets]
        
        if datasets is None:            
            datasets = list(available_datasets.keys())
        
        # Helper function to read general datasets
        def read_general_dataset(group, pairID=None):
            #print(f'group.get(gid) : {list(group.keys())}')
            
            if pairID is None:
                pairs = group.keys()
            else:
                pairID = pairID if isinstance(pairID, list) else [pairID]
                forpair = list(group.keys())
                pairs = [forpair[j] for j in pairID]
                                
            return {k: np.array(group.get(k)) for k in pairs}
        
        # Helper function to read polytope datasets
        def read_polytope(group, dataset_name, pairID=None):
            polydata = {}
            if pairID is None:
                pairs = group.keys()
            else:
                pairID = pairID if isinstance(pairID, list) else [pairID]
                pairs = [f'Pair{i}' for i in pairID]
                
            for gid in pairs:
                p = group.get(gid)
                
                if p is None:
                    continue
                
                polyold = []
                
                for ic, ds in enumerate(p.keys()):
                    if ic < len(p.keys()) // 2:
                        #db = f'pb{re.split('(\d+)',ds)[1]}'
                        db = f'pb{ic}'
                        da = f'pa{ic}'
                        polyold.append(pc.Polytope(np.array(p[da]), np.array(p[db])))
                polydata[gid] = polyold
            return polydata
        
        # Iterate through datasets and pairIDs
        for dataset in datasets:
            data[dataset] = {}
            if dataset in available_datasets:
                if dataset in ('solution_polytope', 'allsolution_polytope'):
                    group = h5file.get(dataset)
                    if group is not None:
                        data[dataset] = read_polytope(group, dataset, pairID)
                else:
                    group = h5file.get(dataset)
                    if group is not None:
                        data[dataset] = read_general_dataset(group, pairID)

    return data

def readdata_v0(fname, datasets=None, pairID=None):
    
    with h5py.File(fname, 'r') as h5file:
        
        available_datasets = {
            'allsolution': 'allsolution',
            'generatedcoordinate': 'coordinate',
            'coordinate_sorted': 'coordinate_sorted',
            'grandradius': 'grandradius',
            'solution': 'solution',
            'solution_error': 'solution_error',
            'solution_volume': 'solution_volume',
            'solution_extremepoint': 'extremepoint',
            'solution_totalNr': 'total_solutionNr',
            'total_volume_in_Asym': 'total_volume_in_Asym',
            'distance': 'distance',
            'gmagnitude': 'gmagnitude',
            'normal': 'normal',
            'time_total': 'time_total',
            'allsolution_polytope': 'allsolution_polytope',
            'solution_polytope': 'solution_polytope'
        }
        
        data = {}
        
        # Ensure datasets is a list
        if isinstance(datasets, str):
            datasets = [datasets]
        
        if datasets is None:            
            datasets = list(available_datasets.keys())
        
        # Helper function to read general datasets
        def read_general_dataset(group):
            #return np.array([np.array(group.get(i)) for i in group])
            return {k: np.array(group.get(k)) for k in group}
        
        # Helper function to read polytope datasets
        def read_polytope(group, dataset_name, pairID=None):
            polydata = {}
            if pairID is None:
                pairs = group.keys()
            else:
                pairID = pairID if isinstance(pairID, list) else [pairID]
                pairs = [f'Pair{i}' for i in pairID]
                
            for gid in pairs:
                p = group.get(gid)
                
                if p is None:
                    continue
                
                polyold = []
                
                for ic, ds in enumerate(p.keys()):
                    if ic < len(p.keys()) // 2:
                        #db = f'pb{re.split('(\d+)',ds)[1]}'
                        db = f'pb{ic}'
                        da = f'pa{ic}'
                        polyold.append(pc.Polytope(np.array(p[da]), np.array(p[db])))
                polydata[gid] = polyold
            return polydata
        
        # Iterate through datasets and pairIDs
        for dataset in datasets:
            data[dataset] = {}
            if dataset in available_datasets:
                if dataset in ('solution_polytope', 'allsolution_polytope'):
                    group = h5file.get(dataset)
                    if group is not None:
                        data[dataset] = read_polytope(group, dataset, pairID)
                else:
                    group = h5file.get(dataset)
                    if group is not None:
                        data[dataset] = read_general_dataset(group)

    return data



# ----------------------------------------------------------------------------------------------------------------

# I made this comment before making classes
# Below defs are older. in future they will be deleted 
# _status on : 19.06.2026 15:10
# ----------------------------------------------------------------------------------------------------------------


def wrtdata_v0(fname, rc, volume, err, final, extremepnts,volAsym, Lsol):
    
    with h5py.File(fname, 'a') as f:
    ### Writing volume information in file 
        if 'lenofsolution' in f:
            lsolu = str('/lenofsolution/')+str('lsol') +str(rc)
            f.create_dataset(lsolu,  data=Lsol,  dtype='float64')
        else:
            gvol = f.create_group('lenofsolution')
            lsolu = str('/lenofsolution/')+str('lsol') +str(rc)
            f.create_dataset(lsolu,  data=Lsol,  dtype='float64')
    
    ### Writing volume information in file 
        if 'vol' in f:
            v = str('/vol/')+str('v') +str(rc)
            f.create_dataset(v,  data=volume,  dtype='float64')
        else:
            gvol = f.create_group('vol')
            v = str('/vol/')+str('v') +str(rc)
            f.create_dataset(v,  data=volume,  dtype='float64')
    
    ### Writing error information in file 
        if 'error' in f:
            derr = str('/error/')+str('err') +str(rc)
            f.create_dataset(derr,  data=err,  dtype='float64')
        else:
            gerror = f.create_group('error')
            derr = str('/error/')+str('err') +str(rc)
            f.create_dataset(derr,  data=err,  dtype='float64')
    
    ### Writing polytope information in file 
        if 'polytope' in f:
            pa=str('/polytope/')+str('pa') +str(rc)
            pb=str('/polytope/')+str('pb') +str(rc)
            f.create_dataset(pa, data=final.A, dtype='float64')
            f.create_dataset(pb, data=final.b, dtype='float64')
        else:
            gpolytope = f.create_group('polytope')
            pa=str('/polytope/')+str('pa') +str(rc)
            pb=str('/polytope/')+str('pb') +str(rc)
            f.create_dataset(pa, data=final.A, dtype='float64')
            f.create_dataset(pb, data=final.b, dtype='float64')
    
    ### Writing extreme points of polytope information in file 
        if 'extreme' in f:
            ext=str('/extreme/')+str('pa') +str(rc)
            f.create_dataset(ext, data=extremepnts, dtype='float64')
        else:
            gpolytope = f.create_group('extreme')
            ext=str('/extreme/')+str('pa') +str(rc)
            f.create_dataset(ext, data=extremepnts, dtype='float64')
    
    ### Writing sum of all volums in Asym part 
        if 'total_volume_in_Asym' in f:
            v_asym = str('/total_volume_in_Asym/')+str('pair') +str(rc)
            f.create_dataset(v_asym,  data=volAsym,  dtype='float64')
        else:
            gerror = f.create_group('total_volume_in_Asym')
            v_asym = str('/total_volume_in_Asym/')+str('pair') +str(rc)
            f.create_dataset(v_asym,  data=volAsym,  dtype='float64')        
    return

def wrtcoor(fname, pairs):
    
    ### Writing polygenerated coordinats to file
        
    with h5py.File(fname, 'a') as f:
        for i, Pair in enumerate(pairs):
            Pair_sort = np.sort(Pair)[::-1]
            
            if 'generatedcoordinate' in f:
                co = str('/generatedcoordinate/')+str(i)
                f.create_dataset(co, data=Pair,  dtype='float64')
            else:
                gco = f.create_group('generatedcoordinate')
                co  = str('/generatedcoordinate/')+str(i)
                f.create_dataset(co, data=Pair,  dtype='float64')
            
            # if 'coordinate' in f:
            #     co = str('/coordinate/')+str(i)
            #     f.create_dataset(co, data=Pair,  dtype='float64')
            # else:
            #     gco = f.create_group('coordinate')
            #     co  = str('/coordinate/')+str(i)
            #     f.create_dataset(co, data=Pair,  dtype='float64')
            
            if 'coordinate_sorted' in f:
                co = str('/coordinate_sorted/')+str(i)
                f.create_dataset(co, data=Pair_sort, dtype='float64')
            else:
                gco = f.create_group('coordinate_sorted')
                co  = str('/coordinate_sorted/')+str(i)
                f.create_dataset(co, data=Pair_sort, dtype='float64')
    return

def wrtvolume(fname, rc, volume, dx, dy, final):
    
    with h5py.File(fname, 'a') as f:
        v =str('v') +str(rc)
        dxx=str('dx')+str(rc)
        dyy=str('dy')+str(rc)
        
        f.create_dataset(v,  data=volume,  dtype='float64')
        f.create_dataset(dxx, data=dx, dtype='float64')
        f.create_dataset(dyy, data=dy, dtype='float64')
        
        pa=str('s')+str(rc)+str('a')
        pb=str('s')+str(rc)+str('b')
        
        f.create_dataset(pa, data=final.A, dtype='float64')
        f.create_dataset(pb, data=final.b, dtype='float64')
    return

def wrtallsolution(fname, rc, solutionall):
        
    def wrtfile(rc, file, sa):
        count=0
        for cou, ip in enumerate(sa):
            if type(ip) is pc.Polytope:
                
                allpa=str('/allsolution_polytope/Pair')+ str(rc)+ str('/a')+ str(count)
                allpb=str('/allsolution_polytope/Pair')+ str(rc)+ str('/b')+ str(count)
                
                file.create_dataset(allpa, data=ip.A, dtype='float64')
                file.create_dataset(allpb, data=ip.b, dtype='float64')
                count += 1
                
            elif type(ip) is pc.Region:
                for iqinx, iq in enumerate(ip):
                    count += 1
                    #cou = iqinx + cou
                    allpa=str('/allsolution_polytope/')+ str('/Pair/')+ str(rc)+ str('/a')+ str(rc)+ str(count)
                    allpb=str('/allsolution_polytope/')+ str('/Pair/')+ str(rc)+ str('/b')+ str(rc)+ str(count)
                    file.create_dataset(allpa, data=iq.A, dtype='float64')
                    file.create_dataset(allpb, data=iq.b, dtype='float64')
        
    with h5py.File(fname, 'a') as file:
        
        #---> Writing extreme points of polytope information in file 
        if 'allsolution_polytope' in file:
            grp=file['/allsolution_polytope/']
            sgp=str('Pair')+ str(rc)
            sg = grp.create_group(sgp)
            wrtfile(rc, file, solutionall)
            
        else:
            gpolytope = file.create_group('allsolution_polytope')
            spg=str('Pair')+ str(rc)
            sg = gpolytope.create_group(spg)
            
            wrtfile(rc, file, solutionall)
        
    return

def wrttime_mc(rc, fname, timeinfo):
    with h5py.File(fname, 'a') as f:
        if 'time_total' in f:
            dtime = str('/time_total/')+str('time_total') +str(rc)
            f.create_dataset(dtime,  data=timeinfo,  dtype='float64')
        else:
            gtime = f.create_group('time_total')
            dtime = str('/time_total/')+str('tforPair') +str(rc)
            f.create_dataset(dtime,  data=timeinfo,  dtype='float64')
        
    return

def readoldsolution_v0(pairID, fname):
    
    with h5py.File(fname, 'r') as file:
        
        p=file.get('allsolution/Pair'+str(pairID))
        polyold=[]
        
        for ic, ds in enumerate(p.keys()):
            if ic < int(len(p.keys())/2):
                # db='b'+re.split('(\d+)',ds)[1]
                db = 'b' + re.split(r'(\d+)', ds)[1]
                da=np.array(p[ds])
                polyold.append(pc.Polytope(np.array(p[ds]), np.array(p[db])))
    
    return polyold

def readh5file(fn):
    
    with h5py.File(fn, 'r') as f:
        mean = []
        ls = list(f.items())
        
        vols = f.get('vol')
        vall = [np.array(vols.get(i)) for i in np.array(vols)]
        mean.append(np.mean(np.array(vall)))
        
        er=f.get('error')
        error=[np.array(er.get(i)) for i in np.array(er)]
                
        koorkey  = f.get('generatedcoordinate')
        koor = [np.array(koorkey.get(i)) for i in np.array(koorkey)]
        
        unsortkoorkey=f.get('coordinate')
        unsortkoor=[np.array(unsortkoorkey.get(i)) for i in np.array(unsortkoorkey)]
        
        ext=f.get('extreme')
        extre=[np.mean(ext.get(i),0) for i in np.array(ext)]
        
        radi=f.get('total_volume_in_Asym')
        radius=[np.array(radi.get(i)[1]) for i in np.array(radi)]
        
        
        noofsolu=f.get('allsolution')
        noofsolution = [ (np.shape( np.array(f.get('allsolution/Pair'+str(ii))) )[0])/2 for ii in range(1, len(noofsolu)+1) ]
        
    return np.array(vall), mean[0], error, np.array(koor),  np.array(unsortkoor), extre, radius, noofsolution


def readh5file_v2(fn):
    with h5py.File(fn, 'r') as f:
        tinfo, maxerror, mean = [], [], []
        
        ls = list(f.items())
        
        vol = f.get('solution_volume')
        volume = [np.array(vol.get(i)) for i in np.array(vol)]
        mean.append(np.mean(np.array(volume)))
        
        koorkey  = f.get('generatedcoordinate')
        koordinate = [np.array(koorkey.get(i)) for i in np.array(koorkey)]
        
        err  = f.get('solution_error')
        error = [np.array(err.get(i)) for i in np.array(err)]
                
        ext=f.get('solution_extremepoint')
        centroid=[np.mean(ext.get(i),0) for i in np.array(ext)]
        
        radkey=f.get('grandradius')
        radius=[np.array(radkey.get(i)) for i in np.array(radkey)]
        
        #noofsolu=f.get('allsolution')
        #noofsolution = np.shape(np.array([f.get('allsolution/Pair'+str(ii)) for ii in range(0, len(noofsolu)) ][0]))[0]
        noofsolu=f.get('solution_totalNr')
        noofsolution=[np.array(noofsolu.get(i)) for i in np.array(noofsolu)]

        tinf = f.get('time_total')
        tt   = [np.array(tinf.get(ii)) for ii in np.array(tinf)] ; tt = np.array(tt)[0]
        #tti  = np.max(tt[2:]) #[np.max(np.array(tinf.get(ii)))for ii in np.array(tinf)]
        tinfo.append(tt)
        
        er    = f.get('solution_error')
        err   = [np.array(er.get(i)) for i in np.array(er)]
        emaxi = [np.max(i) for i in err]
        maxerror.append(emaxi)
        
    return np.array(volume), mean[0], error, mean, np.array(koordinate), centroid, radius, noofsolution, tinfo, maxerror


def readh5file_v2_updated(fn):
    import h5py
    with h5py.File(fn, 'r') as f:
        mean_volume = []
        ls = list(f.items())
        
        allsolkey  = f.get('allsolution')
        allsolution = [np.array(allsolkey.get(i)) for i in np.array(allsolkey)]
                
        coorkey  = f.get('generatedcoordinate')
        coordinate = np.array([np.array(coorkey.get(i)) for i in np.array(coorkey)])
        
        sortkoorkey  = f.get('coordinate_sorted')
        coordinate_sorted = np.array([np.array(sortkoorkey.get(i)) for i in np.array(sortkoorkey)])
        
        radiuskey   = f.get('grandradius')
        grandradius = np.array([np.array(radiuskey.get(i)) for i in np.array(radiuskey)])
        
        solutkey  = f.get('solution')
        solution  = np.array([np.array(solutkey.get(i)) for i in np.array(solutkey)])
        
        errorkey  = f.get('solution_error')
        solution_error = np.array([np.array(errorkey.get(i)) for i in np.array(errorkey)])
        
        volkey = f.get('solution_volume')
        solution_volume = np.array([np.array(volkey.get(i)) for i in np.array(volkey)])
        #mean_volume.append(np.mean(np.array(solution_volume)))
        
        extkey=f.get('solution_extremepoint')
        extremepoint=[np.array(extkey.get(i)) for i in np.array(extkey)]
        
        soluNrkey=f.get('solution_totalNr')
        total_solutionNr = np.array([np.array(soluNrkey.get(i)) for i in np.array(soluNrkey)])
        
        vinAsymkey=f.get('total_volume_in_Asym')
        total_volume_in_Asym=np.array([np.array(vinAsymkey.get(i)) for i in np.array(vinAsymkey)])
        
        #print(f'grandradius {grandradius}, solution_error {solution_error}, np.array(mean_volume) {mean_volume}, extremepoint {extremepoint}, total_solutionNr {total_solutionNr}, total_volume_in_Asym {total_volume_in_Asym}')
                
    return coordinate, coordinate_sorted, solution, grandradius, solution_error, solution_volume, total_solutionNr, total_volume_in_Asym, extremepoint, allsolution

def readoldsolution(pairID, fname):
    
    with h5py.File(fname, 'r') as file:
        
        p=file.get('allsolution_polytope/Pair'+str(pairID))
        polyold=[]
        
        for ic, ds in enumerate(p.keys()):
            if ic < int(len(p.keys())/2):
                db = 'b' + re.split(r'(\d+)', ds)[1]
                # db='b'+re.split('(\d+)',ds)[1]
                da=np.array(p[ds])
                polyold.append(pc.Polytope(np.array(p[ds]), np.array(p[db])))
    
    return polyold


def wrtdata(pair_inx, fname, solution, solution_polytope, solution_volume, solution_error, solution_extremepnts, vol_in_Asym, grandradius , total_solNr, allsolutions):
    
    with h5py.File(fname, 'a') as f:
        
        # ---> Writing found solution for given structure in file
        if 'solution' in f:
            solkey = str('/solution/')+str('solutionforPair') +str(pair_inx)
            f.create_dataset(solkey, data=solution, dtype='float64')
        else:
            gvol   = f.create_group('solution')
            solkey = str('/solution/')+str('solutionforPair') +str(pair_inx)
            f.create_dataset(solkey, data=solution, dtype='float64')

        # ---> Writing all found solutions in file
        if 'allsolution' in f:
            allsolution = str('/allsolution/')+str('Pair') +str(pair_inx)
            f.create_dataset(allsolution,  data=allsolutions,  dtype='float64')
        else:
            gerror = f.create_group('allsolution')
            allsolution = str('/allsolution/')+str('Pair') +str(pair_inx)
            f.create_dataset(allsolution, data=allsolutions, dtype='float64')   
        
        # ---> Writing polytope contains given structure in file 
        if 'solution_polytope' in f:
            pa=str('/solution_polytope/')+str('pa') +str(pair_inx)
            pb=str('/solution_polytope/')+str('pb') +str(pair_inx)
            f.create_dataset(pa, data=solution_polytope.A, dtype='float64')
            f.create_dataset(pb, data=solution_polytope.b, dtype='float64')
        else:
            gpolytope = f.create_group('solution_polytope')
            pa=str('/solution_polytope/')+str('pa') +str(pair_inx)
            pb=str('/solution_polytope/')+str('pb') +str(pair_inx)
            f.create_dataset(pa, data=solution_polytope.A, dtype='float64')
            f.create_dataset(pb, data=solution_polytope.b, dtype='float64')

        # ---> Writing volume of polytope contains given structure in file
        if solution_volume:
            if 'solution_volume' in f:
                v = str('/solution_volume/')+str('v') +str(pair_inx)
                f.create_dataset(v,  data=solution_volume,  dtype='float64')
            else:
                gvol = f.create_group('solution_volume')
                v = str('/solution_volume/')+str('v') +str(pair_inx)
                f.create_dataset(v,  data=solution_volume,  dtype='float64')

        # ---> Writing error information in file
        if np.all(solution_error):
            if 'solution_error' in f:
                derr = str('/solution_error/')+str('err') +str(pair_inx)
                f.create_dataset(derr,  data=solution_error,  dtype='float64')
            else:
                gerror = f.create_group('solution_error')
                derr = str('/solution_error/')+str('err') +str(pair_inx)
                f.create_dataset(derr,  data=solution_error,  dtype='float64')
        
        # ---> Writing extreme points of polytope contains given structure in file 
        if 'solution_extremepoint' in f:
            ext=str('/solution_extremepoint/')+str('pa') +str(pair_inx)
            f.create_dataset(ext, data=solution_extremepnts, dtype='float64')
        else:
            gpolytope = f.create_group('solution_extremepoint')
            ext=str('/solution_extremepoint/')+str('pa') +str(pair_inx)
            f.create_dataset(ext, data=solution_extremepnts, dtype='float64')
                
        # ---> Writing sum of all volume in asymmetric PS 
        if vol_in_Asym:
            if 'total_volume_in_Asym' in f:
                v_asym = str('/total_volume_in_Asym/')+str('pair') +str(pair_inx)
                f.create_dataset(v_asym,  data=vol_in_Asym,  dtype='float64')
            else:
                gerror = f.create_group('total_volume_in_Asym')
                v_asym = str('/total_volume_in_Asym/')+str('pair') +str(pair_inx)
                f.create_dataset(v_asym,  data=vol_in_Asym,  dtype='float64')  
        
        # ---> Writing grand radius information in file
        if grandradius:
            if 'grandradius' in f:
                radius = str('/grandradius/')+str('r') +str(pair_inx)
                f.create_dataset(radius,  data=grandradius,  dtype='float64')
            else:
                gradius = f.create_group('grandradius')
                radius = str('/grandradius/')+str('r') +str(pair_inx)
                f.create_dataset(radius,  data=grandradius,  dtype='float64')
                
        # ---> Writing total no of solution information in file
        if  total_solNr:
            if 'solution_totalNr' in f:
                lsolu = str('/solution_totalNr/')+str('lsol') +str(pair_inx)
                f.create_dataset(lsolu,  data=total_solNr,  dtype='float64')
            else:
                gsol  = f.create_group('solution_totalNr')
                lsolu = str('/solution_totalNr/')+str('lsol') +str(pair_inx)
                f.create_dataset(lsolu,  data=total_solNr,  dtype='float64')
    
    return



