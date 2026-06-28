
import logging
from psc.config.logconfig import setup_logging
import uuid
import numpy as np

import time
from pathlib import Path
from datetime import datetime

from psc.lib.gspacer import GSpacer
from psc.lib.choiceoforigin import ChoiceOfOrigin
from psc.lib.linearizer import EPALinearizer, NEPALinearizer
from psc.lib.checklinearizer import CheckLinearizer
from psc.lib.tessellator import Tessellator
from psc.lib.intersector import Intersector
from psc.lib.solutionmerger import SolutionMerger
from psc.lib.solutionselector import Solutionselector
from psc.lib.solutionIO import SolutionIO
import polytope as pc

# User controls this location
log_folder = Path("./tmp")
setup_logging(log_folder)


class PSCSolver:
    
    def __init__(self, reflections, structurefactor, structure, imax=0.5, intensity_type="amplitude", output="./tmp", verifylinearization=True, write_polytope=True, run_id=None, pair_id=None, author=None):

        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.run_id = run_id
        self.pair_id = pair_id
        
        self.logger = logging.getLogger(f"psc.{run_id}.solver")

        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        
        if pair_id is not None:
            self.io = SolutionIO(self.output / f"pairID{pair_id}_pscsolver_{run_id}.h5", logger=self.logger)
        else:
            self.io = SolutionIO(self.output / f"pscsolver_{run_id}.h5", logger=self.logger)
        
        # self.io = SolutionIO(self.output / f"pscsolver_{run_id}.h5", logger=self.logger)

        self.reflections = np.asarray(reflections)
        self.sf = np.asarray(structurefactor)
        self.structure = np.asarray(structure)
        self.dimension = len(structure)
        self.imax = imax
        self.intensity_type = intensity_type
        self.verifylinearization = verifylinearization
        self.write_polytope = write_polytope
        
        self.author = author

        self.solutions = {}
        self.timeinfo  = {}
        self.logger.info(f"Created HDF5 file: {self.io.filename}")


    def _format_amplitudes(self, amplitudes_reflection_map):
        lines = []
        for reflection, value in amplitudes_reflection_map.items():
            lines.append(f"{int(reflection):>5} : {float(value): .6f}")
        return "\n".join(lines)
                
    def solve(self):
        t0 = datetime.now()
        self.logger.info( f"Solver started at :  {datetime.now()-t0}" )
        self.logger.info(f"Solving {self.dimension}D structure")
        
        self.io.init_run()
                        
        print(f"HDF5 file: {self.io.filename}")
        
        self.logger.info("")
        self.logger.info(f"# {'-' * 39} Details of structure {'-' * 39}#")
        self.logger.info(f"Assumed or given reflection : {self.reflections}")
        self.logger.info(f"Structure factors: {self.sf}")
        self.logger.info(f"Structure: {self.structure}")
        self.logger.info(f"Art of framework method : {self.intensity_type}")
        self.logger.info(f"Output path: {self.output}")
        self.logger.info(f"# {'-' * 100} #")
        self.logger.info("")
        
        ck = self.io.load_checkpoint()
        if ck is not None:
            self.logger.info(f"Resuming from {ck}")
        
        # ----------------------------------------------------------------
        # Step 1: Get amplitudes/intensity and map with reflections
        # ----------------------------------------------------------------
        
        gs = GSpacer(self.reflections, self.structure, self.sf)
        amplitudes = gs.g_vectorized()
        amplitudes_reflection_map = dict(zip(self.reflections, amplitudes))
        
        # self.logger.info(f"Calculated intensity/amplitude \n\t {amplitudes_reflection_map}")
        self.logger.info( "Calculated intensity/amplitude:\n" + self._format_amplitudes(amplitudes_reflection_map))
        
        # asymmetric part        
        # asym = gs.Asym(self.dimension)
        
        coo = ChoiceOfOrigin()
        amplitudes_reflection_coo_map = coo.apply( amplitudes_reflection_map )
        
        # self.logger.info(f"Choice of origin applied intensity/amplitude\n\t {amplitudes_reflection_coo_map}")
        self.logger.info( "Calculated intensity/amplitude:\n" + self._format_amplitudes(amplitudes_reflection_coo_map))
        
        self.io.write_metadata(version="PSC-v0.2.0", reflections=len(self.reflections), author=self.author, structurefactor=self.sf, structure=self.structure, amplitudes=amplitudes_reflection_map)
        
        self.io.save_state( "input", { "reflections": self.reflections, "structure": self.structure, "structurefactor": self.sf, "amplitudes": amplitudes, "intensity_type": self.intensity_type})
        
        # ----------------------------------------------------------------
        # Step 2: Get linearized for each amplitudes/intensity.
        #         Get linearization parameters
        # ----------------------------------------------------------------
        
        self.logger.info("")
        self.logger.info(f"Starting linearizer at : {datetime.now()-t0}")
        
        linearize_obj, lineartime = {}, {}
        for l in self.reflections:
            t1 = time.perf_counter()
            self.logger.info(f"Linearizing reflection : {l}")
            
            linearo=EPALinearizer(reflection=l, structurefactor=self.sf, structure=self.structure, amplitude=amplitudes_reflection_map[l], imax=0.5, logger=self.logger.getChild("linearizer"))
            linear=linearo.linearize()
            # print(out)  <- you get {"normal": normal, "boundary": boundary, "polytope points": completepoint} # print(f"linear: {linear}")
            
            self.io.write_timing("Linearizer", f"l{l}", time.perf_counter()-t1)
            
            linearize_obj[f'reflection{l}'] = linear
            lineartime[f"l{l}"] = time.perf_counter()-t1
            
            if self.verifylinearization:
                t1 = time.perf_counter()
                
                oo = CheckLinearizer(reflection=l, structurefactor=self.sf, amplitude=amplitudes_reflection_map[l], normal=linear['normal'],distance=np.array([linear['boundary']['innerdistance'], linear['boundary']['outerdistance'] ]), logger=self.logger.getChild("checklinear"))
                o = oo.run(method='gridbased')
                
                if o['status']:
                    self.logger.info(f"! Status : {o['status']} Linearization for reflection {l} is {o['message']}")
                else:
                    self.logger.warning(f"! Status : {o['status']} Linearization for reflection {l} is {o['message']}")
                
                linearize_obj[f'reflection{l}'].update(o)
                
                self.io.write_timing("CheckLinearizer", f"l{l}", time.perf_counter()-t1)
            
            self.io.write_stage( "linearizer", {f"L{l}": linearize_obj[f'reflection{l}']} )
            
            self.io.save_state( "linearizer_done", { "last_reflection": int(l), "completed": True } )
            
            self.io.checkpoint("linearizer", substep=f"L{l}", index=int(l))
        
        self.timeinfo['linearizer'] = lineartime
        self.timeinfo['linearizertotal'] = sum(lineartime.values())
        
        self.io.write_timing("total/Linearizer", sum(lineartime.values()) )
                        
        # ----------------------------------------------------------------
        # Step 3: Get Tessellator and fill complet PS. Get intersector and Get solution space
        # ----------------------------------------------------------------
        
        self.logger.info("")
        self.logger.info(f"Starting Tessellator at : {datetime.now()-t0}")
        
        polytope_obj, tessellator_summary, tesstime = {}, [], {}
        for l in self.reflections:
            self.logger.info(f"Tessellating for reflection : {l}")
            
            t1 = time.perf_counter()
            linear=linearize_obj[f'reflection{l}']
            
            if linear['status']:
                distance = np.array([ linear['boundary']['innerdistance'], linear['boundary']['outerdistance'] ])
            else:
                distance =linear['dist_new']
                        
            rep = Tessellator(reflection=l, normal= linear['normal'], distance=distance, IorG='amplitude',  imax=0.5, logger=self.logger.getChild("tessellator")) # add later- limitingmat=linear['polytope points']
            
            polys=rep.getpolytope_EPA()
            
            self.logger.info(f"Found polytopes for reflection {l} is : {len(polys)}")
            
            self.io.write_timing("Tessellator", f"l{l}", time.perf_counter()-t1)
            tesstime[f"l{l}"] =  time.perf_counter()-t1
            
            poly_file = None
            if self.write_polytope:
                t1 = time.perf_counter()
                poly_file = self.io.write_polytope( l, polys )
                                
                self.io.write_timing("TessellatorWritting", f"l{l}", time.perf_counter()-t1)
                
            tessellator_summary.append( [l, len(polys), str(poly_file)] )
            polytope_obj[f'reflection{l}'] = polys
            
        self.io.write_stage( "tessellator_summary", { "reflection": np.array([x[0] for x in tessellator_summary]), 
                                                     "number_of_polytope": np.array([x[1] for x in tessellator_summary]),
                                                      "polytope_file": np.array([x[2] if x[2] else "" for x in tessellator_summary], dtype="S") })
        
        self.io.save_state( "tessellator_done", { "last_reflection": int(l), "number_of_polytopes": len(polys), "completed": True})
        
        self.timeinfo['tessellator'] = tesstime
        self.timeinfo['tessellatortotal'] = sum(tesstime.values())
        
        self.io.write_timing("total/Tessellator", sum(tesstime.values()) )
        
        
        
        # ----------------------------------------------------------------
        # Step 4: Get intersector and Get solution space
        # ----------------------------------------------------------------
        
        self.logger.info("")
        self.logger.info(f"Starting Intersector at : {datetime.now()-t0}")
        
        intersector = Intersector(imax=0.5, logger=self.logger.getChild("intersector"))
        
        convergence = {"key": [], "total_volume": [], "grand_radius": [], "shrink_ratio": [], "volume_drop": [], "solution_count": [], "volume_ratio": []}
        intersection_obj, intersector_summary, intertime = {}, [], {}
        
        def fold(state, new_polytope):
            return intersector.find_intersection(state, new_polytope)
        
        # -------------------------
        # INITIAL STATE
        # -------------------------
        
        t1 = time.perf_counter()
        
        state = fold(polytope_obj['reflection1'], polytope_obj['reflection2'] )
        key = "s12"
        V = np.sum([pc.volume(p) for p in state])
        R = self.io._radius(self.dimension, V)
        N = len(state)
        
        convergence["key"].append(key)
        convergence["total_volume"].append(V)
        convergence["grand_radius"].append(R)
        convergence["shrink_ratio"].append(1.0)
        convergence["volume_drop"].append(0.0)
        convergence["solution_count"].append(N)
        convergence["volume_ratio"].append(1.0)
        
        self.io.write_intersection(key, state)
        self.io.write_intersection_stats(key, state, self.dimension)
        self.io.checkpoint("intersector", substep=key, index=0)
        
        self.io.write_timing("Intersector", f"{key}", time.perf_counter()-t1)
        
        intersection_obj[key] = state
        
        intersector_summary.append([key, N])
        
        self.logger.info(f"Found {N} solution spaces at reflections 1 & 2 - {key}")
        
        # -------------------------
        # STREAM REDUCTION LOOP
        # -------------------------
        
        for i, l_curr in enumerate(self.reflections[2:], start=2):
            t1 = time.perf_counter()
            
            key = f's{i}{l_curr}'
            
            t0_iter = time.perf_counter()
            
            state = fold(state, polytope_obj[f'reflection{l_curr}'])
            intersection_obj[key] = state
            
            # t1 = time.perf_counter()
            
            # V = np.sum([pc.volume(p) for p in state])
            V = sum(pc.volume(p) for p in state)
            R = self.io._radius(self.dimension, V)
            N = len(state)
            
            prev_V = convergence["total_volume"][-1]
            shrink = V / prev_V
            drop = prev_V - V
            
            intertime[key] = t1 - t0_iter
            
            convergence["key"].append(key)
            convergence["total_volume"].append(V)
            convergence["grand_radius"].append(R)
            convergence["shrink_ratio"].append(shrink)
            convergence["volume_drop"].append(drop)
            convergence["solution_count"].append(N)
            convergence["volume_ratio"].append(V / convergence["total_volume"][0])
            
            self.io.write_intersection(key, state)
            self.io.write_intersection_stats(key, state, self.dimension)
            self.io.write_convergence(convergence)
            
            intersector_summary.append([key, N])
            
            self.io.checkpoint("intersector", substep=key, index=i)
            self.logger.info(f"Found {N} solution spaces at reflection {l_curr} - {key}")
            
            self.io.write_timing("Intersector", f"{key}", time.perf_counter()-t1)
            
        self.timeinfo['intersector'] = intertime
        self.timeinfo['intersectortotal'] = sum(intertime.values())                
        
        self.io.write_timing("total/Intersector", sum(intertime.values()))
        
        # # ----------------------------------------------------------------
        # # Step 4: Get intersector and Get solution space
        # # ----------------------------------------------------------------
        
        # self.logger.info("")
        # self.logger.info(f"Starting Intersector at : {datetime.now()-t0}")
        
        # intersection_obj, intersector_summary, intertime = {}, [], {}
        # convergence = { "key": [], "total_volume": [], "grand_radius": [], "shrink_ratio": [], "volume_drop": [], "solution_count": [], "volume_ratio":[]}
        # prev_volume = None
        
        # intersector = Intersector(imax=0.5, logger=self.logger.getChild("intersector"))
        # self.logger.info("Intersecting reflections : 1 & 2")
        
        # prev_key = 's12'
        # t1 = time.perf_counter()
        # intersection_obj[prev_key] = intersector.find_intersection(polytope_obj['reflection1'], polytope_obj['reflection2'] )
        
        # self.logger.info(f"Found {len(intersection_obj[prev_key])} solution spaces after reflections 1 and 2 - solution index {prev_key}")
        
        # self.io.write_timing("Intersector", f"l_{prev_key}", time.perf_counter()-t1)
        
        # t1 = time.perf_counter()
        # self.io.write_intersection(prev_key, intersection_obj[prev_key])
        # intersector_summary.append([prev_key, len(intersection_obj[prev_key])])
        # self.io.write_timing("IntersectorWritting", f"l_{prev_key}", time.perf_counter()-t1)
        
        # intertime[prev_key] = time.perf_counter()-t1
        
        # for inx, l in enumerate(self.reflections):
            
        #     if inx <= 1: continue
            
        #     l_prev = self.reflections[l - 2]
        #     l_curr = self.reflections[l - 1]
            
        #     t1 = time.perf_counter()
            
        #     new_key = f's{l_prev}{l_curr}'
        #     intersection_obj[new_key] = intersector.find_intersection( intersection_obj[prev_key], polytope_obj[f'reflection{l_curr}'])
            
        #     self.io.write_timing("Intersector", f"l_{new_key}", time.perf_counter()-t1)
            
        #     t1 = time.perf_counter()
        #     self.io.write_intersection(new_key, intersection_obj[new_key])
        #     intersector_summary.append([new_key, len(intersection_obj[new_key])])
        #     prev_key = new_key
            
        #     self.io.write_timing("IntersectorWritting", f"l_{new_key}", time.perf_counter()-t1)
        #     self.logger.info(f"Found {len(intersection_obj[new_key])} solution spaces after reflection {l} - solution index {new_key}")
            
        #     intertime[new_key] = time.perf_counter()-t1
            
        #     V = np.sum([pc.volume(p) for p in intersection_obj[new_key]])
        #     R = self.io._radius(self.dimension, V)
        #     N = len(intersection_obj[new_key])
            
        #     if prev_volume is None:
        #         shrink = 1.0
        #         drop = 0.0
        #     else:
        #         shrink = V / prev_volume
        #         drop = prev_volume - V
            
        #     prev_volume = V
            
        #     convergence["key"].append(new_key)
        #     convergence["total_volume"].append(V)
        #     convergence["grand_radius"].append(R)
        #     convergence["shrink_ratio"].append(shrink)
        #     convergence["volume_drop"].append(drop)
        #     convergence["solution_count"].append(N)
        #     convergence["volume_ratio"].append(V / convergence["total_volume"][0])
        
        #     self.io.checkpoint("intersector", substep=new_key, index=inx)
                
        #     self.io.write_stage( "intersector_summary", {"key": np.array([x[0] for x in intersector_summary], dtype="S"),
        #                                                 "count": np.array([x[1] for x in intersector_summary]) })
            
        #     self.io.write_intersection_stats(new_key, intersection_obj[new_key], self.dimension)
            
        #     self.io.write_convergence(convergence)
            
        #     self.io.save_state( "intersector_done", { "last_intersection": new_key, "number_of_regions": len(intersection_obj[new_key]), "completed": True})
        
        # self.timeinfo['intersector'] = intertime
        # self.timeinfo['intersectortotal'] = sum(intertime.values())
        
        # ----------------------------------------------------------------
        # Step 5: Merge solution space
        # ----------------------------------------------------------------        
        self.logger.info("")
        keys = list(intersection_obj.keys())
        
        self.logger.info(f"Starting Solution Merger at : {datetime.now()-t0}")
        self.logger.info(f"Available keys in intersection object {keys}")
        self.logger.info(f"Intersection at {keys[-1]} will be merged")
        
        t1 = time.perf_counter()
        # k = 's34' ; merger = SolutionMerger(intersection_obj[k])
        merger = SolutionMerger(intersection_obj[keys[-1]], logger=self.logger.getChild("merger"))
        merged_regions, components = merger.runmerger()
        
        self.logger.info(f"Components mergerged into single polytope: {components}")
        self.logger.info(f"Number of polytops in merged region: {len(merged_regions)}")
        self.logger.info(f"Number of polytops in unmerged region: {len(intersection_obj[keys[-1]])}")
        
        self.io.write_solutionmerger(merged_regions, components)
        
        self.io.save_state( "merger_done", { "components": components, "number_of_regions": len(merged_regions), "completed": True})
        
        self.io.checkpoint("solutionmerger", substep="done")
        
        self.io.write_timing("total/SolutionMerger", time.perf_counter()-t1)
        
        self.timeinfo['solutionmerger'] = time.perf_counter()-t1
        
        # ----------------------------------------------------------------
        # Step 5: Find solution space for given reflection-amplitudes dataset
        # ----------------------------------------------------------------
        t1 = time.perf_counter()
        
        self.logger.info("")
        self.logger.info(f"Starting SolutionSelector at : {datetime.now()-t0}")
        solutionspace =  Solutionselector(polytope=merged_regions, Amp_measured=amplitudes, reflections=self.reflections, structure_factor=self.sf, imax=0.5)
        
        best_idx, best_x, best_err, best_ext, loginfo  = solutionspace.run()
        
        # print(f"\n\nbest_idx {best_idx}, best_err {best_err}, best_x {best_x}  {np.array(best_x).tolist()}, loginfo {loginfo}")
        
        self.io.write_solutionselector(best_idx, best_x, best_err, merged_regions[best_idx], best_ext, loginfo)
        
        self.io.save_state( "solution_done", {"best_index": int(best_idx), "error": float(best_err), "completed": True } )
        
        self.io.write_timing("total/Solutionselector", time.perf_counter()-t1)
        
        self.timeinfo['solutionselector'] = time.perf_counter()-t1
        
        return { "best_index": best_idx,
                "best_error": best_err,
                "best_x": best_x,
                "loginfo": loginfo, 
                "solutionpolytope": merged_regions[best_idx],
                "h5": self.io.filename,
                "timeinfo": self.timeinfo
                }

# best_i, best_err, best_x, loginfo, merged_regions[best_i]

# ------ USAGE
# solver = PSCSolver(reflections=np.arange(1,10), structurefactor=[1, 1, 1], structure=[0.41985, 0.36804, 0.20758]) #[0.372,0.213,0.12, 0.05]
# best_i, best_err, best_x, loginfo, polytope = solver.solve()




class PSCSolverNEPA:

    def __init__(self, reflections, structurefactor, structure=None, intensity=None, variant='NEPA', intensity_type="amplitude", output="./tmp", verifylinearization=True, write_polytope=True, run_id=None, imax=0.5):
        
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            
        self.run_id = run_id
        self.logger = logging.getLogger(f"psc.{run_id}.solver")
        
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        
        self.io = SolutionIO(self.output / f"pscsolver_{run_id}.h5", logger=self.logger)
        
        self.reflections = np.asarray(reflections)
        self.sf = np.asarray(structurefactor)
        self.dimension = self.sf.size
        
        self.imax = imax
        self.variant = variant
        self.intensity_type = intensity_type
        self.verifylinearization = verifylinearization
        self.write_polytope = write_polytope
        
        self.structure = None
        self.intensity = None
        
        if structure is None and intensity is None:
            raise ValueError("Either intensity or structure must be provided.")

        if structure is not None:
            self.structure = np.asarray(structure)

        if intensity is not None:
            self.intensity = np.asarray(intensity)

        if self.variant not in ["EPA", "NEPA"]:
            raise ValueError(f"Unknown variant: {self.variant}")

        self.solutions = {}
        self.timeinfo = {}

        self.logger.info(f"Created HDF5 file: {self.io.filename}")


    def _format_amplitudes(self, amplitudes_reflection_map):
        return "\n".join([f"{int(r):>5} : {float(v): .6f}" for r,v in amplitudes_reflection_map.items()])
    
    def solve(self):
        
        t0 = datetime.now()
        self.logger.info(f"Solver started at: {datetime.now()-t0}")
        self.logger.info(f"Solving {self.dimension}D structure")
        
        self.io.init_run()
        
        
        if self.structure is not None:
            self.logger.info(f"Strucutre before COO: {self.structure}")
            self.structure = np.sort(0.5-self.structure) #[::-1]
            gs = GSpacer(self.reflections, self.structure, self.sf)
            values = gs.g_vectorized()
            
            self.logger.info(f"Strucutre after COO: {self.structure}")
            
            if self.variant == "NEPA":
                amplitudes = np.sign(values)*values**2
            else:
                amplitudes = values
            amplitudes_reflection_map = dict(zip(self.reflections, amplitudes))
        
        elif self.intensity is not None:
            amplitudes = self.intensity
            amplitudes_reflection_map = dict(zip(self.reflections, amplitudes))
        
        else:
            raise ValueError("No valid structure or intensity found.")
        
        self.logger.info("Calculated amplitudes values:\n" + self._format_amplitudes(amplitudes_reflection_map))
        
        coo = ChoiceOfOrigin()
        amplitudes_reflection_coo_map = coo.apply( amplitudes_reflection_map )
        
        self.logger.info(f"Reflection: {self.reflections}")
        self.logger.info(f"Structure factors: {self.sf}")
        self.logger.info(f"Structure: {self.structure}")
        self.logger.info(f"Variant: {self.variant}")
        self.logger.info(f"Intensity type: {self.intensity_type}")
        self.logger.info(f"Output path: {self.output}")
        self.logger.info( "Calculated intensity/amplitude:\n" + self._format_amplitudes(amplitudes_reflection_coo_map))
        
        if self.structure is not None:
            self.io.write_metadata(version="PSC-v1.0", reflections=len(self.reflections), author="Muthu", structurefactor=self.sf, atoms=self.structure, structuretype="Theoretical_data", amplitudes=amplitudes_reflection_map)
            
            self.io.save_state( "input", { "reflections": self.reflections, "atoms": self.structure, "structurefactor": self.sf, "amplitudes": amplitudes, "intensity_type": self.intensity_type})
        
        if self.intensity is not None:
            self.io.write_metadata(version="PSC-v1.0", reflections=len(self.reflections), author="Muthu", structurefactor=self.sf, structuretype="Experimental_data", amplitudes=amplitudes_reflection_map)
            
            self.io.save_state( "input", { "reflections": self.reflections, "structurefactor": self.sf, "amplitudes": amplitudes, "intensity_type": self.intensity_type})
                    
        # ----------------------------------------------------------------
        # Step 2: Get linearized for each amplitudes/intensity.
        #         Get linearization parameters
        # ----------------------------------------------------------------
        
        self.logger.info("")
        self.logger.info(f"Starting linearizer at : {datetime.now()-t0}")
        
        linearize_obj, lineartime = {}, {}
        for l in self.reflections:
            t1 = time.perf_counter()
            self.logger.info(f"Linearizing reflection : {l} Intensity: {amplitudes_reflection_map[l]} ")
            
            if self.variant == "EPA":
                linearo=EPALinearizer(reflection=l, structurefactor=self.sf, atoms=self.structure, amplitude=amplitudes_reflection_map[l], imax=0.5, logger=self.logger.getChild("linearizer"))
                linear=linearo.linearize()
                # print(out)  <- you get {"normal": normal, "boundary": boundary, "polytope points": completepoint} # print(f"linear: {linear}")                
            else:
                linearo=NEPALinearizer(reflection=l, structurefactor=self.sf, intensity=amplitudes_reflection_map[l], atoms=self.structure, imax=0.5, logger=self.logger.getChild("linearizer"))
                linear=linearo.linearize()
                # print(out)  <- you get {"normal": normal, "boundary": boundary, "polytope points": completepoint} # print(f"linear: {linear}")
                
            self.io.write_timing("Linearizer", f"l{l}", time.perf_counter()-t1)
            
            linearize_obj[f'reflection{l}'] = linear
            lineartime[f"l{l}"] = time.perf_counter()-t1
            
            if self.verifylinearization:
                t1 = time.perf_counter()
                
                oo = CheckLinearizer(reflection=l, structurefactor=self.sf, amplitude=np.abs(amplitudes_reflection_map[l]), normal=linear['normal'],distance=np.array([linear['boundary']['innerdistance'], linear['boundary']['outerdistance'] ]), logger=self.logger.getChild("checklinear"), variant=self.variant)
                
                o = oo.run(method='gridbased')
                if o['status']:
                    self.logger.info(f"! Status : {o['status']} Linearization for reflection {l} is {o['message']}")
                else:
                    self.logger.warning(f"! Status : {o['status']} Linearization for reflection {l} is {o['message']}")
                
                linearize_obj[f'reflection{l}'].update(o)
                                
                self.io.write_timing("CheckLinearizer", f"l{l}", time.perf_counter()-t1)
            
            self.io.write_stage( "linearizer", {f"L{l}": linearize_obj[f'reflection{l}']} )
            self.io.save_state( "linearizer_done", { "last_reflection": int(l), "completed": True } )
            self.io.checkpoint("linearizer", substep=f"L{l}", index=int(l))
        self.timeinfo['linearizer'] = lineartime
        self.timeinfo['linearizertotal'] = sum(lineartime.values())
        self.io.write_timing("total/Linearizer", sum(lineartime.values()) )
        
        # ----------------------------------------------------------------
        # Step 3: Get Tessellator and fill complet PS. Get intersector and Get solution space
        # ----------------------------------------------------------------

        self.logger.info("")
        self.logger.info(f"Starting Tessellator at : {datetime.now()-t0}")
        
        polytope_obj, tessellator_summary, tesstime = {}, [], {}
        for l in self.reflections:
            self.logger.info(f"Tessellating for reflection : {l}")
            
            t1 = time.perf_counter()
            linear=linearize_obj[f'reflection{l}']
            
            if linear['status']:
                distance = np.array([ linear['boundary']['innerdistance'], linear['boundary']['outerdistance'] ])
            else:
                distance =linear['dist_new']
            
            rep = Tessellator(reflection=l, normal= linear['normal'], distance=distance, IorG=self.intensity_type,  imax=0.5, logger=self.logger.getChild("tessellator")) # add later- limitingmat=linear['polytope points']
            
            if self.variant == "EPA":
                polys=rep.getpolytope_EPA()
            else:
                polys=rep.getpolytope_NEPA()
                                
            self.logger.info(f"Found polytopes for reflection {l} is : {len(polys)}")
            self.io.write_timing("Tessellator", f"l{l}", time.perf_counter()-t1)
            tesstime[f"l{l}"] =  time.perf_counter()-t1
            
            poly_file = None
            if self.write_polytope:
                t1 = time.perf_counter()
                poly_file = self.io.write_polytope( l, polys )
                                
                self.io.write_timing("TessellatorWritting", f"l{l}", time.perf_counter()-t1)
                
            tessellator_summary.append( [l, len(polys), str(poly_file)] )
            polytope_obj[f'reflection{l}'] = polys
            
        self.io.write_stage( "tessellator_summary", {"reflection": np.array([x[0] for x in tessellator_summary]), 
                                                     "number_of_polytope": np.array([x[1] for x in tessellator_summary]),
                                                     "polytope_file": np.array([x[2] if x[2] else "" for x in tessellator_summary], dtype="S") })
        
        self.io.save_state( "tessellator_done", { "last_reflection": int(l), "number_of_polytopes": len(polys), "completed": True})
        self.timeinfo['tessellator'] = tesstime
        self.timeinfo['tessellatortotal'] = sum(tesstime.values())
        self.io.write_timing("total/Tessellator", sum(tesstime.values()) )
        
        # ----------------------------------------------------------------
        # Step 4: Get intersector and Get solution space
        # ----------------------------------------------------------------
        
        self.logger.info("")
        self.logger.info(f"Starting Intersector at : {datetime.now()-t0}")
        
        intersector = Intersector(imax=0.5, logger=self.logger.getChild("intersector"))
        
        convergence = {"key": [], "total_volume": [], "grand_radius": [], "shrink_ratio": [], "volume_drop": [], "solution_count": [], "volume_ratio": []}
        intersection_obj, intersector_summary, intertime = {}, [], {}
        
        def fold(state, new_polytope):
            return intersector.find_intersection(state, new_polytope)
        
        # INITIAL STATE
        
        t1 = time.perf_counter()
        
        state = fold(polytope_obj['reflection1'], polytope_obj['reflection2'] )        
        key = "s12"
        
        V = np.sum([pc.volume(p) for p in state])
        R = self.io._radius(self.dimension, V)
        N = len(state)
        
        convergence["key"].append(key)
        convergence["total_volume"].append(V)
        convergence["grand_radius"].append(R)
        convergence["shrink_ratio"].append(1.0)
        convergence["volume_drop"].append(0.0)
        convergence["solution_count"].append(N)
        convergence["volume_ratio"].append(1.0)
        
        self.io.write_intersection(key, state)
        self.io.write_intersection_stats(key, state, self.dimension)
        self.io.checkpoint("intersector", substep=key, index=0)
        
        self.io.write_timing("Intersector", f"{key}", time.perf_counter()-t1)
        
        intersection_obj[key] = state
        
        intersector_summary.append([key, N])
        
        self.logger.info(f"Found {N} solution spaces at reflections 1 & 2 - {key}")

        # STREAM REDUCTION LOOP        
        for i, l_curr in enumerate(self.reflections[2:], start=2):
            t1 = time.perf_counter()
            
            key = f's{i}{l_curr}'
            
            t0_iter = time.perf_counter()
            
            state = fold(state, polytope_obj[f'reflection{l_curr}'])
            intersection_obj[key] = state
            
            # # TESTING SELF:STRUCTURE
            # for i in state:
            #     if self.structure in i:
            #         print(f"\x1b[1;32m l: {key} {True} \x1b[0m")
            #     else:
            #         print(f"\x1b[1;31m l: {key} {False} \x1b[0m")
                      
            # V = np.sum([pc.volume(p) for p in state])
            V = sum(pc.volume(p) for p in state)
            R = self.io._radius(self.dimension, V)
            N = len(state)
            
            prev_V = convergence["total_volume"][-1]
            shrink = V / prev_V
            drop = prev_V - V
            
            intertime[key] = t1 - t0_iter
            
            convergence["key"].append(key)
            convergence["total_volume"].append(V)
            convergence["grand_radius"].append(R)
            convergence["shrink_ratio"].append(shrink)
            convergence["volume_drop"].append(drop)
            convergence["solution_count"].append(N)
            convergence["volume_ratio"].append(V / convergence["total_volume"][0])
            
            self.io.write_intersection(key, state)
            self.io.write_intersection_stats(key, state, self.dimension)
            self.io.write_convergence(convergence)
            
            intersector_summary.append([key, N])
            
            self.io.checkpoint("intersector", substep=key, index=i)
            self.logger.info(f"Found {N} solution spaces at reflection {l_curr} - {key}")
            
            self.io.write_timing("Intersector", f"{key}", time.perf_counter()-t1)
            
        self.timeinfo['intersector'] = intertime
        self.timeinfo['intersectortotal'] = sum(intertime.values())                
        
        self.io.write_timing("total/Intersector", sum(intertime.values()))
        
        # ----------------------------------------------------------------
        # Step 5: Merge solution space
        # ----------------------------------------------------------------        
        self.logger.info("")
        keys = list(intersection_obj.keys())
        
        self.logger.info(f"Starting Solution Merger at : {datetime.now()-t0}")
        self.logger.info(f"Available keys in intersection object {keys}")
        self.logger.info(f"Intersection at {keys[-1]} will be merged")
        
        t1 = time.perf_counter()
        merger = SolutionMerger(intersection_obj[keys[-1]], logger=self.logger.getChild("merger"))
        merged_regions, components = merger.runmerger()
        
        self.logger.info(f"Components mergerged into single polytope: {components}")
        self.logger.info(f"Number of polytops in merged region: {len(merged_regions)}")
        self.logger.info(f"Number of polytops in unmerged region: {len(intersection_obj[keys[-1]])}")
        
        self.io.write_solutionmerger(merged_regions, components)
        
        self.io.save_state( "merger_done", { "components": components, "number_of_regions": len(merged_regions), "completed": True})
        
        self.io.checkpoint("solutionmerger", substep="done")
        
        self.io.write_timing("total/SolutionMerger", time.perf_counter()-t1)
        
        self.timeinfo['solutionmerger'] = time.perf_counter()-t1
        
        # ----------------------------------------------------------------
        # Step 5: Find solution space for given reflection-amplitudes dataset
        # ----------------------------------------------------------------
                
        t1 = time.perf_counter()
        
        self.logger.info("")
        self.logger.info(f"Starting SolutionSelector at : {datetime.now()-t0}")
        solutionspace =  Solutionselector(polytope=merged_regions, Amp_measured=np.abs(amplitudes), reflections=self.reflections, structure_factor=self.sf, structure=self.structure,variant='NEPA', imax=0.5)
        
        best_idx, best_x, best_err, best_ext, loginfo  = solutionspace.run()
        self.io.write_solutionselector(best_idx, best_x, best_err, merged_regions[best_idx], best_ext, loginfo)
        
        self.io.save_state( "solution_done", {"best_index": int(best_idx), "error": float(best_err), "completed": True } )
        
        self.io.write_timing("total/Solutionselector", time.perf_counter()-t1)
        
        self.timeinfo['solutionselector'] = time.perf_counter()-t1
        
        return { "best_index": best_idx,
                "best_error": best_err,
                "best_x": best_x,
                "loginfo": loginfo, 
                "solutionpolytope": merged_regions[best_idx],
                "h5": self.io.filename,
                "timeinfo": self.timeinfo
                }


# ------ USAGE
# solver = PSCSolverNEPA(reflections=np.arange(1,5), structurefactor=[10, 9, 1], structure=[0.328,0.218, 0.125], intensity_type='intensity', write_polytope=False)
# result = solver.solve()
# result




# class PSCSolver:
    
#     def __init__(self, reflections, structurefactor, structure, imax=0.5, intensity_type="amplitude", output="./tmp", verifylinearization=True, write_polytope=True, run_id=None):

#         # if run_id is None:
#         run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
#         self.run_id = run_id
#         self.logger = logging.getLogger(f"psc.{run_id}.solver")

#         self.output = Path(output)
#         self.output.mkdir(parents=True, exist_ok=True)

#         self.io = SolutionIO(self.output / f"pscsolver_{run_id}.h5", logger=self.logger)

#         self.reflections = np.asarray(reflections)
#         self.sf = np.asarray(structurefactor)
#         self.structure = np.asarray(structure)
#         self.dimension = len(structure)
#         self.imax = imax
#         self.intensity_type = intensity_type
#         self.verifylinearization = verifylinearization
#         self.write_polytope = write_polytope

#         self.solutions = {}
#         self.timeinfo  = {}
#         self.logger.info(f"Created HDF5 file: {self.io.filename}")


#     def _format_amplitudes(self, amplitudes_reflection_map):
#         lines = []
#         for reflection, value in amplitudes_reflection_map.items():
#             lines.append(f"{int(reflection):>5} : {float(value): .6f}")
#         return "\n".join(lines)
                
#     def solve(self):
#         t0 = datetime.now()
#         self.logger.info( f"Solver started at :  {datetime.now()-t0}" )
#         self.logger.info(f"Solving {self.dimension}D structure")
        
#         self.io.init_run()
                        
#         print(f"HDF5 file: {self.io.filename}")
        
#         self.logger.info("")
#         self.logger.info(f"# {'-' * 39} Details of structure {'-' * 39}#")
#         self.logger.info(f"Assumed or given reflection : {self.reflections}")
#         self.logger.info(f"Structure factors: {self.sf}")
#         self.logger.info(f"Structure: {self.structure}")
#         self.logger.info(f"Art of framework method : {self.intensity_type}")
#         self.logger.info(f"Output path: {self.output}")
#         self.logger.info(f"# {'-' * 100} #")
#         self.logger.info("")
        
#         ck = self.io.load_checkpoint()
#         if ck is not None:
#             self.logger.info(f"Resuming from {ck}")
        
#         # ----------------------------------------------------------------
#         # Step 1: Get amplitudes/intensity and map with reflections
#         # ----------------------------------------------------------------
        
#         gs = GSpacer(self.reflections, self.structure, self.sf)
#         amplitudes = gs.g_vectorized()
#         amplitudes_reflection_map = dict(zip(self.reflections, amplitudes))
        
#         # self.logger.info(f"Calculated intensity/amplitude \n\t {amplitudes_reflection_map}")
#         self.logger.info( "Calculated intensity/amplitude:\n" + self._format_amplitudes(amplitudes_reflection_map))
        
#         # asymmetric part        
#         # asym = gs.Asym(self.dimension)
        
#         coo = ChoiceOfOrigin()
#         amplitudes_reflection_coo_map = coo.apply( amplitudes_reflection_map )
        
#         # self.logger.info(f"Choice of origin applied intensity/amplitude\n\t {amplitudes_reflection_coo_map}")
#         self.logger.info( "Calculated intensity/amplitude:\n" + self._format_amplitudes(amplitudes_reflection_coo_map))
        
#         self.io.write_metadata(version="PSC-v1.0", reflections=len(self.reflections), author="Muthu", structurefactor=self.sf, structure=self.structure, amplitudes=amplitudes_reflection_map)
        
#         self.io.save_state( "input", { "reflections": self.reflections, "structure": self.structure, "structurefactor": self.sf, "amplitudes": amplitudes, "intensity_type": self.intensity_type})
        
#         # ----------------------------------------------------------------
#         # Step 2: Get linearized for each amplitudes/intensity.
#         #         Get linearization parameters
#         # ----------------------------------------------------------------
        
#         self.logger.info("")
#         self.logger.info(f"Starting linearizer at : {datetime.now()-t0}")
        
#         linearize_obj, lineartime = {}, {}
#         for l in self.reflections:
#             t1 = time.perf_counter()
#             self.logger.info(f"Linearizing reflection : {l}")
            
#             linearo=EPALinearizer(reflection=l, structurefactor=self.sf, structure=self.structure, amplitude=amplitudes_reflection_map[l], imax=0.5, logger=self.logger.getChild("linearizer"))
#             linear=linearo.linearize()
#             # print(out)  <- you get {"normal": normal, "boundary": boundary, "polytope points": completepoint} # print(f"linear: {linear}")
            
#             self.io.write_timing("Linearizer", f"l{l}", time.perf_counter()-t1)
            
#             linearize_obj[f'reflection{l}'] = linear
#             lineartime[f"l{l}"] = time.perf_counter()-t1
            
#             if self.verifylinearization:
#                 t1 = time.perf_counter()
                
#                 oo = CheckLinearizer(reflection=l, structurefactor=self.sf, amplitude=amplitudes_reflection_map[l], normal=linear['normal'],distance=np.array([linear['boundary']['innerdistance'], linear['boundary']['outerdistance'] ]), logger=self.logger.getChild("checklinear"))
#                 o = oo.run(method='gridbased')
                
#                 if o['status']:
#                     self.logger.info(f"! Status : {o['status']} Linearization for reflection {l} is {o['message']}")
#                 else:
#                     self.logger.warning(f"! Status : {o['status']} Linearization for reflection {l} is {o['message']}")
                
#                 linearize_obj[f'reflection{l}'].update(o)
                
#                 self.io.write_timing("CheckLinearizer", f"l{l}", time.perf_counter()-t1)
            
#             self.io.write_stage( "linearizer", {f"L{l}": linearize_obj[f'reflection{l}']} )
            
#             self.io.save_state( "linearizer_done", { "last_reflection": int(l), "completed": True } )
            
#             self.io.checkpoint("linearizer", substep=f"L{l}", index=int(l))
        
#         self.timeinfo['linearizer'] = lineartime
#         self.timeinfo['linearizertotal'] = sum(lineartime.values())
#         # ----------------------------------------------------------------
#         # Step 3: Get Tessellator and fill complet PS. Get intersector and Get solution space
#         # ----------------------------------------------------------------
        
#         self.logger.info("")
#         self.logger.info(f"Starting Tessellating at : {datetime.now()-t0}")
        
#         polytope_obj, tessellator_summary, tesstime = {}, [], {}
#         for l in self.reflections:
#             self.logger.info(f"Tessellating for reflection : {l}")
            
#             t1 = time.perf_counter()
#             linear=linearize_obj[f'reflection{l}']
            
#             if linear['status']:
#                 distance = np.array([ linear['boundary']['innerdistance'], linear['boundary']['outerdistance'] ])
#             else:
#                 distance =linear['dist_new']
                        
#             rep = Tessellator(reflection=l, normal= linear['normal'], distance=distance, IorG='amplitude',  imax=0.5, logger=self.logger.getChild("tessellator")) # add later- limitingmat=linear['polytope points']
            
#             polys=rep.getpolytope_EPA()
            
#             self.logger.info(f"Found polytopes for reflection {l} is : {len(polys)}")
            
#             self.io.write_timing("Tessellator", f"l{l}", time.perf_counter()-t1)
#             tesstime[f"l{l}"] =  time.perf_counter()-t1
            
#             poly_file = None
#             if self.write_polytope:
#                 t1 = time.perf_counter()
#                 poly_file = self.io.write_polytope( l, polys )
                                
#                 self.io.write_timing("TessellatorWritting", f"l{l}", time.perf_counter()-t1)
                
#             tessellator_summary.append( [l, len(polys), str(poly_file)] )
#             polytope_obj[f'reflection{l}'] = polys
            
#         self.io.write_stage( "tessellator_summary", { "reflection": np.array([x[0] for x in tessellator_summary]), 
#                                                      "number_of_polytope": np.array([x[1] for x in tessellator_summary]),
#                                                       "polytope_file": np.array([x[2] if x[2] else "" for x in tessellator_summary], dtype="S") })
        
#         self.io.save_state( "tessellator_done", { "last_reflection": int(l), "number_of_polytopes": len(polys), "completed": True})
        
#         self.timeinfo['tessellator'] = tesstime
#         self.timeinfo['tessellatortotal'] = sum(tesstime.values())
                
#         # ----------------------------------------------------------------
#         # Step 4: Get intersector and Get solution space
#         # ----------------------------------------------------------------
        
#         self.logger.info("")
#         self.logger.info(f"Starting Intersector at : {datetime.now()-t0}")
        
#         intersection_obj, intersector_summary, intertime = {}, [], {}
#         convergence = { "key": [], "total_volume": [], "grand_radius": [], "shrink_ratio": [], "volume_drop": [], "solution_count": [], "volume_ratio":[]}
#         prev_volume = None
        
#         intersector = Intersector(imax=0.5, logger=self.logger.getChild("intersector"))
#         self.logger.info("Intersecting reflections : 1 & 2")
        
#         prev_key = 's12'
#         t1 = time.perf_counter()
#         intersection_obj[prev_key] = intersector.find_intersection(polytope_obj['reflection1'], polytope_obj['reflection2'] )
        
#         self.logger.info(f"Found {len(intersection_obj[prev_key])} solution spaces after reflections 1 and 2 - solution index {prev_key}")
        
#         self.io.write_timing("Intersector", f"l_{prev_key}", time.perf_counter()-t1)
        
#         t1 = time.perf_counter()
#         self.io.write_intersection(prev_key, intersection_obj[prev_key])
#         intersector_summary.append([prev_key, len(intersection_obj[prev_key])])
#         self.io.write_timing("IntersectorWritting", f"l_{prev_key}", time.perf_counter()-t1)
        
#         intertime[prev_key] = time.perf_counter()-t1
        
#         for inx, l in enumerate(self.reflections):
            
#             if inx <= 1: continue
            
#             l_prev = self.reflections[l - 2]
#             l_curr = self.reflections[l - 1]
            
#             t1 = time.perf_counter()
            
#             new_key = f's{l_prev}{l_curr}'
#             intersection_obj[new_key] = intersector.find_intersection( intersection_obj[prev_key], polytope_obj[f'reflection{l_curr}'])
            
#             self.io.write_timing("Intersector", f"l_{new_key}", time.perf_counter()-t1)
            
#             t1 = time.perf_counter()
#             self.io.write_intersection(new_key, intersection_obj[new_key])
#             intersector_summary.append([new_key, len(intersection_obj[new_key])])
#             prev_key = new_key
            
#             self.io.write_timing("IntersectorWritting", f"l_{new_key}", time.perf_counter()-t1)
#             self.logger.info(f"Found {len(intersection_obj[new_key])} solution spaces after reflection {l} - solution index {new_key}")
            
#             intertime[new_key] = time.perf_counter()-t1
            
#             V = np.sum([pc.volume(p) for p in intersection_obj[new_key]])
#             R = self.io._radius(self.dimension, V)
#             N = len(intersection_obj[new_key])
            
#             if prev_volume is None:
#                 shrink = 1.0
#                 drop = 0.0
#             else:
#                 shrink = V / prev_volume
#                 drop = prev_volume - V
            
#             prev_volume = V
            
#             convergence["key"].append(new_key)
#             convergence["total_volume"].append(V)
#             convergence["grand_radius"].append(R)
#             convergence["shrink_ratio"].append(shrink)
#             convergence["volume_drop"].append(drop)
#             convergence["solution_count"].append(N)
#             convergence["volume_ratio"].append(V / convergence["total_volume"][0])
        
#             self.io.checkpoint("intersector", substep=new_key, index=inx)
                
#         self.io.write_stage( "intersector_summary", {"key": np.array([x[0] for x in intersector_summary], dtype="S"),
#                                                      "count": np.array([x[1] for x in intersector_summary]) })
        
#         self.io.write_intersection_stats(new_key, intersection_obj[new_key], self.dimension)
        
#         self.io.write_convergence(convergence)
        
#         self.io.save_state( "intersector_done", { "last_intersection": new_key, "number_of_regions": len(intersection_obj[new_key]), "completed": True})
        
#         self.timeinfo['intersector'] = intertime
#         self.timeinfo['intersectortotal'] = sum(intertime.values())
        
#         # ----------------------------------------------------------------
#         # Step 4: Merge solution space
#         # ----------------------------------------------------------------        
#         self.logger.info("")
#         self.logger.info(f"Starting SolutionMerger at : {datetime.now()-t0}")
#         keys = list(intersection_obj.keys())
        
#         self.logger.info(f"Starting Solution Merger at : {datetime.now()-t0}")
#         self.logger.info(f"Available keys in intersection object {keys}")
#         self.logger.info(f"Intersection at {keys[-1]} will be merged")
        
#         t1 = time.perf_counter()
#         # k = 's34' ; merger = SolutionMerger(intersection_obj[k])
#         merger = SolutionMerger(intersection_obj[keys[-1]], logger=self.logger.getChild("merger"))
#         merged_regions, components = merger.runmerger()
        
#         self.logger.info(f"Components mergerged into single polytope: {components}")
#         self.logger.info(f"Number of polytops in merged region: {len(merged_regions)}")
#         self.logger.info(f"Number of polytops in unmerged region: {len(intersection_obj[keys[-1]])}")
        
#         self.io.write_solutionmerger(merged_regions, components)
        
#         self.io.save_state( "merger_done", { "components": components, "number_of_regions": len(merged_regions), "completed": True})
        
#         self.io.checkpoint("solutionmerger", substep="done")
        
#         self.io.write_timing("SolutionMerger", time.perf_counter()-t1)
        
#         self.timeinfo['solutionmerger'] = time.perf_counter()-t1
        
#         # ----------------------------------------------------------------
#         # Step 5: Find solution space for given reflection-amplitudes dataset
#         # ----------------------------------------------------------------
#         t1 = time.perf_counter()
        
#         self.logger.info("")
#         self.logger.info(f"Starting SolutionSelector at : {datetime.now()-t0}")
#         solutionspace =  Solutionselector(polytope=merged_regions, Amp_measured=amplitudes, reflections=self.reflections, structure_factor=self.sf, imax=0.5)
        
#         best_i, best_err, best_x, loginfo  = solutionspace.run()
#         #print(f"\n\nbest_i {best_i}, best_err {best_err}, best_x {best_x}, loginfo {loginfo}")
        
#         self.io.write_solutionselector(best_i, best_err, best_x, merged_regions[best_i], loginfo)
        
#         self.io.save_state( "solution_done", {"best_index": int(best_i), "error": float(best_err), "completed": True } )
        
#         self.io.write_timing("Solutionselector", time.perf_counter()-t1)
        
#         self.timeinfo['solutionselector'] = time.perf_counter()-t1
        
#         return { "best_index": best_i,
#                 "best_error": best_err,
#                 "best_x": best_x,
#                 "loginfo": loginfo, 
#                 "solutionpolytope": merged_regions[best_i],
#                 "h5": self.io.filename,
#                 "timeinfo": self.timeinfo
#                 }
