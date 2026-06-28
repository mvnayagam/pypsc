import numpy as np
import h5py
import time
import uuid
from datetime import datetime
from pathlib import Path
from psc.lib.pscsolver import PSCSolver

import logging

class MCSolver:

    # def __init__(self, reflections, structurefactor, dimension, noofpair=100, imax=0.5, output="./tmp"):
    def __init__(self, reflections, structurefactor, noofpair=100, intensity_type="amplitude", output="./tmp", verifylinearization=True, write_polytope=True, run_id=None, imax=0.5):
        
        self.reflections = np.asarray(reflections)
        self.sf = np.asarray(structurefactor)
        self.noofpair = noofpair
        self.imax = imax
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.intensity_type = intensity_type
        
        self.dimension = len(self.sf)
                
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        
        self.logger = logging.getLogger(f"psc.mc.{self.run_id}")
        
        self.filename = self.output / f"MC_results_{self.run_id}.h5"
        
        
    def generate_atoms(self):
        return np.sort(np.random.uniform(0.0, 0.5, size=3))[::-1] #np.random.uniform(0.0, 0.5, size=self.dimension)

    # ---------------- INIT ----------------
    def init_file(self):

        with h5py.File(self.filename, "w") as h5:

            mc = h5.create_group("MC")

            mc.create_dataset("structures", shape=(0, self.dimension), maxshape=(None, self.dimension), dtype="f8")
            mc.create_dataset("best_solution", shape=(0, self.dimension), maxshape=(None, self.dimension), dtype="f8")
            mc.create_dataset("error", shape=(0,), maxshape=(None,), dtype="f8")

            mc.create_group("timing")

            mc.create_dataset("log_poly_idx", shape=(0,), maxshape=(None,), dtype="i8")
            mc.create_dataset("log_error", shape=(0,), maxshape=(None,), dtype="f8")
            mc.create_dataset("log_success", shape=(0,), maxshape=(None,), dtype="i1")
            mc.create_dataset("log_init", shape=(0, self.dimension), maxshape=(None, self.dimension), dtype="f8")
            mc.create_dataset("log_opt", shape=(0, self.dimension), maxshape=(None, self.dimension), dtype="f8")
            mc.create_dataset("log_msg", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())

            mc.create_group("solutionpolytope")

    # ---------------- POLYTOPE ----------------
    def _poly_to_array(self, p):

        if isinstance(p, np.ndarray): return p
        if hasattr(p, "vertices"): return np.asarray(p.vertices, dtype=np.float64)
        if hasattr(p, "A") and hasattr(p, "b"): return np.hstack([np.asarray(p.A), np.asarray(p.b).reshape(-1, 1)])
        raise TypeError(f"Unsupported polytope type: {type(p)}")

    # ---------------- TIME WRITER ----------------
    def _write_timeinfo(self, grp, d):

        def rec(g, x):
            for k, v in x.items():
                if isinstance(v, dict):
                    sub = g.create_group(k)
                    rec(sub, v)
                else:
                    g.create_dataset(k, data=v)

        rec(grp, d)

    # ---------------- APPEND ----------------
    def append_result(self, atoms, solution, error, solutionpolytope, loginfo, idx, timeinfo):

        with h5py.File(self.filename, "a") as h5:

            mc = h5["MC"]

            for name, val in [("structures", atoms), ("best_solution", solution)]:
                d = mc[name]; n = d.shape[0]; d.resize(n + 1, axis=0); d[n] = val

            d = mc["error"]; n = d.shape[0]; d.resize(n + 1, axis=0); d[n] = error

            tgrp = mc["timing"]
            if f"structure_{idx}" in tgrp: del tgrp[f"structure_{idx}"]
            grp = tgrp.create_group(f"structure_{idx}")
            self._write_timeinfo(grp, timeinfo)

            idx_arr = np.array([x["polytope_index"] for x in loginfo], dtype=np.int64)
            err_arr = np.array([x["error"] for x in loginfo], dtype=np.float64)
            suc_arr = np.array([x["success"] for x in loginfo], dtype=np.int8)
            init_arr = np.array([x["initialsolution"] for x in loginfo], dtype=np.float64)
            opt_arr = np.array([x["optimizedsolution"] for x in loginfo], dtype=np.float64)
            msg_arr = np.array([x["message"] for x in loginfo], dtype=h5py.string_dtype())

            def add(name, arr):
                d = mc[name]; n = d.shape[0]; d.resize(n + len(arr), axis=0); d[n:] = arr

            add("log_poly_idx", idx_arr)
            add("log_error", err_arr)
            add("log_success", suc_arr)
            add("log_init", init_arr)
            add("log_opt", opt_arr)
            add("log_msg", msg_arr)
            
            pgrp = mc["solutionpolytope"]
            if f"structure_{idx}" in pgrp: del pgrp[f"structure_{idx}"]
            grp = pgrp.create_group(f"structure_{idx}")

            if isinstance(solutionpolytope, (list, tuple)): solutionpolytope = solutionpolytope[0]

            A = np.asarray(getattr(solutionpolytope, "A", None))
            b = np.asarray(getattr(solutionpolytope, "b", None))

            if A is None or b is None: raise TypeError(f"Polytope missing A/b: {type(solutionpolytope)}")

            grp.create_dataset("polytope_A", data=A, compression="gzip")
            grp.create_dataset("polytope_b", data=b, compression="gzip")

    # ---------------- RUN ----------------
    def run(self):

        self.init_file()

        for i in range(self.noofpair):
            
            atoms = self.generate_atoms() #  np.sort(pairs)[::-1]
            t0 = time.perf_counter()
            
            self.logger.info(f"MC cycle {i+1}/{self.noofpair} - structure : {atoms} ")
            
            solver = PSCSolver(reflections=self.reflections, structurefactor=self.sf, structure=atoms, intensity_type=self.intensity_type, imax=self.imax, verifylinearization=True, write_polytope=False, output=self.output, run_id=self.run_id, pair_id=i)
            
            result = solver.solve()
            elapsed = time.perf_counter() - t0
            
            # timeinfo = {"total": elapsed, **result.get("timeinfo", {})}
            timeinfo = dict(result.get("timeinfo", {}))
            timeinfo["total"] = elapsed
            
            self.append_result(
                atoms=atoms,
                solution=result["best_x"],
                error=result["best_error"],
                solutionpolytope=result["solutionpolytope"],
                loginfo=result["loginfo"],
                idx=i,
                timeinfo=timeinfo)

        self.logger.info(f"MC finished. Results saved: {self.filename}")

# USAGE
# mc = MCSolver( reflections=np.arange(1,5), structurefactor=[1,1,1], noofpair=3, intensity_type="intensity" )
# mc.run()


# class MCSolver:

#     # def __init__(self, reflections, structurefactor, dimension, noofpair=100, imax=0.5, output="./tmp"):
#     def __init__(self, reflections, structurefactor, noofpair=10, intensity_type="amplitude", output="./tmp", verifylinearization=True, write_polytope=True, run_id=None, imax=0.5):
        
#         self.reflections = np.asarray(reflections)
#         self.sf = np.asarray(structurefactor)
#         self.noofpair = noofpair
#         self.imax = imax
#         self.output = Path(output)
#         self.output.mkdir(parents=True, exist_ok=True)
#         self.intensity_type = intensity_type
        
#         self.dimension = len(self.sf)
                
#         self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        
#         self.filename = self.output / f"MC_results_{self.run_id}.h5"

#     def generate_atoms(self):
#         return np.sort(np.random.uniform(0.0, 0.5, size=3))[::-1] #np.random.uniform(0.0, 0.5, size=self.dimension)

#     # ---------------- INIT ----------------
#     def init_file(self):

#         with h5py.File(self.filename, "w") as h5:

#             mc = h5.create_group("MC")

#             mc.create_dataset("structures", shape=(0, self.dimension), maxshape=(None, self.dimension), dtype="f8")
#             mc.create_dataset("best_solution", shape=(0, self.dimension), maxshape=(None, self.dimension), dtype="f8")
#             mc.create_dataset("error", shape=(0,), maxshape=(None,), dtype="f8")

#             mc.create_group("timing")

#             mc.create_dataset("log_poly_idx", shape=(0,), maxshape=(None,), dtype="i8")
#             mc.create_dataset("log_error", shape=(0,), maxshape=(None,), dtype="f8")
#             mc.create_dataset("log_success", shape=(0,), maxshape=(None,), dtype="i1")
#             mc.create_dataset("log_init", shape=(0, self.dimension), maxshape=(None, self.dimension), dtype="f8")
#             mc.create_dataset("log_opt", shape=(0, self.dimension), maxshape=(None, self.dimension), dtype="f8")
#             mc.create_dataset("log_msg", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())

#             mc.create_group("solutionpolytope")

#     # ---------------- POLYTOPE ----------------
#     def _poly_to_array(self, p):

#         if isinstance(p, np.ndarray): return p
#         if hasattr(p, "vertices"): return np.asarray(p.vertices, dtype=np.float64)
#         if hasattr(p, "A") and hasattr(p, "b"): return np.hstack([np.asarray(p.A), np.asarray(p.b).reshape(-1, 1)])
#         raise TypeError(f"Unsupported polytope type: {type(p)}")

#     # ---------------- TIME WRITER ----------------
#     def _write_timeinfo(self, grp, d):

#         def rec(g, x):
#             for k, v in x.items():
#                 if isinstance(v, dict):
#                     sub = g.create_group(k)
#                     rec(sub, v)
#                 else:
#                     g.create_dataset(k, data=v)

#         rec(grp, d)

#     # ---------------- APPEND ----------------
#     def append_result(self, atoms, solution, error, solutionpolytope, loginfo, idx, timeinfo):

#         with h5py.File(self.filename, "a") as h5:

#             mc = h5["MC"]

#             for name, val in [("structures", atoms), ("best_solution", solution)]:
#                 d = mc[name]; n = d.shape[0]; d.resize(n + 1, axis=0); d[n] = val

#             d = mc["error"]; n = d.shape[0]; d.resize(n + 1, axis=0); d[n] = error

#             tgrp = mc["timing"]
#             if f"structure_{idx}" in tgrp: del tgrp[f"structure_{idx}"]
#             grp = tgrp.create_group(f"structure_{idx}")
#             self._write_timeinfo(grp, timeinfo)

#             idx_arr = np.array([x["polytope_index"] for x in loginfo], dtype=np.int64)
#             err_arr = np.array([x["error"] for x in loginfo], dtype=np.float64)
#             suc_arr = np.array([x["success"] for x in loginfo], dtype=np.int8)
#             init_arr = np.array([x["initialsolution"] for x in loginfo], dtype=np.float64)
#             opt_arr = np.array([x["optimizedsolution"] for x in loginfo], dtype=np.float64)
#             msg_arr = np.array([x["message"] for x in loginfo], dtype=h5py.string_dtype())

#             def add(name, arr):
#                 d = mc[name]; n = d.shape[0]; d.resize(n + len(arr), axis=0); d[n:] = arr

#             add("log_poly_idx", idx_arr)
#             add("log_error", err_arr)
#             add("log_success", suc_arr)
#             add("log_init", init_arr)
#             add("log_opt", opt_arr)
#             add("log_msg", msg_arr)
            
#             pgrp = mc["solutionpolytope"]
#             if f"structure_{idx}" in pgrp: del pgrp[f"structure_{idx}"]
#             grp = pgrp.create_group(f"structure_{idx}")

#             if isinstance(solutionpolytope, (list, tuple)): solutionpolytope = solutionpolytope[0]

#             A = np.asarray(getattr(solutionpolytope, "A", None))
#             b = np.asarray(getattr(solutionpolytope, "b", None))

#             if A is None or b is None: raise TypeError(f"Polytope missing A/b: {type(solutionpolytope)}")

#             grp.create_dataset("polytope_A", data=A, compression="gzip")
#             grp.create_dataset("polytope_b", data=b, compression="gzip")

#     # ---------------- RUN ----------------
#     def run(self):

#         self.init_file()

#         for i in range(self.noofpair):
            
#             atoms = self.generate_atoms() #  np.sort(pairs)[::-1]
#             t0 = time.perf_counter()
            
#             print(f"MC cycle {i+1}/{self.noofpair} - structure : {atoms} ")
            
#             solver = PSCSolver(reflections=self.reflections, structurefactor=self.sf, atoms=atoms, intensity_type=self.intensity_type, 
#                                 imax=self.imax, verifylinearization=True, write_polytope=False, output=self.output, run_id=self.run_id)
            
#             result = solver.solve()
#             elapsed = time.perf_counter() - t0
            
#             # timeinfo = {"total": elapsed, **result.get("timeinfo", {})}
#             timeinfo = dict(result.get("timeinfo", {}))
#             timeinfo["total"] = elapsed
            
#             self.append_result(
#                 atoms=atoms,
#                 solution=result["best_x"],
#                 error=result["best_error"],
#                 solutionpolytope=result["solutionpolytope"],
#                 loginfo=result["loginfo"],
#                 idx=i,
#                 timeinfo=timeinfo)

#         print(f"MC finished. Results saved: {self.filename}")

# # --- USAGE
# # mc = MCSolver( reflections=np.arange(1,5), structurefactor=[1,1,1], noofpair=3, intensity_type="intensity" )

# # mc.run()