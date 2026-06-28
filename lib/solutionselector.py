import numpy as np
from scipy.optimize import minimize
import polytope as pc

import logging

class Solutionselector:

    """
    Select the best polytope that reproduces measured amplitudes.
    """
    
    def __init__(self, polytope, Amp_measured, reflections, structure_factor, structure=None, variant='EPA', imax=0.5, forward_model=None, logger=None):
        
        self.logger = logger or logging.getLogger("psc.solutionselector")
        
        self.polytope         = polytope
        self.Amp_measured     = Amp_measured
        self.reflections      = reflections
        self.structure_factor = structure_factor
        self.variant = variant
        self.structure = structure
        self.imax             = imax
        if self.variant=='EPA':
            self.forward_model = (forward_model if forward_model is not None else self.default_forward_model)
        else:
            self.forward_model = (forward_model if forward_model is not None else self.NEPA_forward_model)
            
        self.trace            = []
        # print("Using forward model:", self.forward_model.__name__)
    
    @staticmethod
    def _err(ext):
        if ext is None or len(ext)==0:
            return np.nan
        return np.abs(np.max(ext,axis=0)-np.min(ext,axis=0))/2
    
    # ---------------------------------------------------
    # Logging helper
    # ---------------------------------------------------    
    
    def log_trace(self, idx, err, x0, x_opt, best_ext, success, message):
        
        self.trace.append({ 
                           "polytope_index": idx,
                           "error": float(err), 
                           "initialsolution": None if x0 is None else np.array(x0).tolist(), 
                           "optimizedsolution": None if x_opt is None else np.array(x_opt).tolist(), 
                           "best_ext": best_ext, 
                           "success": bool(success), 
                           "message": str(message)
                           })
        # self.logger.info(f"Initial solution (centroid of selected polytope): {x0}")
        # self.logger.info(f"Opti,ized solution (optimized centroid ): {x_opt}")
        
    
    # ---------------------------------------------------
    # Default forward model
    # ---------------------------------------------------
    
    def default_forward_model( self, reflections, atom_positions, structure_factor ):
        return np.array([np.sum(structure_factor * np.cos(2*np.pi*hidx*atom_positions)) for hidx in reflections ])
    
    def NEPA_forward_model(self, reflections, structure, structure_factor):
            return np.array([np.sum(structure_factor * np.cos(2*np.pi*hidx*structure)) for hidx in reflections ])**2

    # ---------------------------------------------------
    # Fit one polytope
    # ---------------------------------------------------

    def fit_polytope(self, polytope, Amp_measured, initial_solution, reflections, structure_factor):
        """
        fit_polytope finds the centriod of polytope and calculates strucutre factor amplitude or intensity. Following given experimental intensity, the error is calculated as the difference between the calculated and measured amplitude or intensity.

        Args:
            polytope (_type_): input polytope to be fitted
            Amp_measured (_type_): measured amplitudes over 1 to h reflections
            initial_solution (_type_): initial solution for the optimization. Can be the centroid of the polytope vertices or the Chebyshev center.
            reflections (_type_): reflection list (e.g. h = 1, 2, ..., 10)
            structure_factor (_type_): structure factors

        Returns:
            _type_: _description_
        """
        
        A, b = polytope.A, polytope.b
        
        def constraint(x):
            return b - A @ x
        
        cons = {'type': 'ineq', 'fun': constraint}
        
        def objective(x):
            return np.linalg.norm( self.forward_model(reflections, x, structure_factor) - Amp_measured)
        
        # bounds = [(0, self.imax)] * len(initial_solution)
        ext = pc.extreme(polytope)
        if ext is not None and len(ext)>0:
            bounds = list(zip(np.min(ext,axis=0), np.max(ext,axis=0)))
        else:
            bounds = [(0,self.imax)]*len(initial_solution)
        
        res = minimize( objective, initial_solution, method='SLSQP', constraints=cons, bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-9} )
        
        # if not res.success:
        if res.x is None:
            return np.inf, None, False, res.message
                
        return res.fun, res.x, res.success, res.message
    
    # ---------------------------------------------------
    # Select best polytope
    # ---------------------------------------------------
    
    def select_best_polytope(self):
        """
        select_best_polytope : Gives best polytope that can reproduce given reflective amplitude or intensity. The best polytope is selected based on the lowest error between the calculated and measured amplitudes.
        
        Returns:
            best_idx: polytope index with the lowest error
            best_err: lowest error value
            best_x: best parameter values. Note that the order of parameters in best_x may not correspond to the order of atoms in the original structure. To get the sorted parameter values, sort it as per order of atoms. This is because the structure factor is invariant to the permutation of atom positions, so we can only determine the sorted parameter values from the amplitudes, not their original order.
            
            Note: The best_x is the optimal solution (starting from the centroid or Chebyshev center of polytope as initial guess) found by the optimization process, which may not be unique due to the nature of the problem. The error is calculated as the difference between the calculated and measured amplitudes, and the best polytope is the one that minimizes this error.
            
            self.trace: A list of dictionaries containing detailed information about the fitting process for each polytope, including the index of the polytope, the error, the initial solution, the optimized solution, and whether the optimization was successful. This can be used for further analysis or debugging.
                        
        """
                
        best_idx, best_err, best_x , best_ext =  -1, np.inf, None, None
        
        for idx, P in enumerate(self.polytope):
            
            # Guess initial solution. Better options are centroid-of-polytope or Chebyshev center. Check if the centroid is a valid.
            # use it as initial solution if it is valid, otherwise fall back to the Chebyshev center
            
            xi = pc.extreme(P)
            x0 = xi.mean(axis=0) if ( xi is not None and len(xi) > 0 and np.isfinite(xi).all() and (xi >= 0).all() and (xi <= self.imax).all() ) else P.chebXc
            
            if x0 is None:
                continue
            
            err, x_opt, success, message = self.fit_polytope( P, self.Amp_measured, x0, self.reflections, self.structure_factor)
            ext = np.array(self._err(xi)).tolist()
                                                
            if success and err < best_err: #if err < best_err:
                best_idx, best_x, best_err, best_ext = idx, np.array(x_opt).tolist(), np.array(err).tolist(), ext
            
            
            # Logging every polytope fit result
            self.log_trace(idx, err, x0, x_opt, best_ext, success, message)
            # self.logger.info(f"Polytope index {idx} | solution {x_opt} | error {err}")
            
            if self.structure is None:
                self.logger.info(f"Polytope index: {idx} | Inital solution (centroid of polytope): {np.array(x0).tolist()} | Final optimized solution: {x_opt} | error on structure: {ext} | error on intensity: {err}")
            else:
                self.logger.info(f"Polytope index: {idx} | Inital solution (centroid of polytope): {np.array(x0).tolist()} | Final optimized solution: {x_opt} | error on structure: {ext} | error on intensity: {err}")
            
        return best_idx, best_x, best_err, best_ext, self.trace

    # ---------------------------------------------------
    # Run pipeline
    # ---------------------------------------------------

    def run(self):

        best_idx, best_x, best_err, best_ext, loginfo = self.select_best_polytope()
        self.logger.info(f"Best polytope index: {best_idx}")
        self.logger.info(f"Error: {best_err}")
        
        if best_x is not None:
            self.logger.info(f"Best parameters: {np.array(best_x).tolist()} | sorted parameter: {(np.sort(best_x)[::-1]).tolist()} | given structure: {self.structure} | Error on structure: {best_ext} | Error on Intensity: {best_err}\n\n ---- SOLVER ENDED ---- \n\n")
        else:
            self.logger.info("Best parameters: None")
        
        return best_idx, best_x, best_err, best_ext, loginfo


#--------------------------------
# Example usage
#-------------------------------

# f          = [1, 1, 1]
# atoms      = np.array( [0.27064, 0.21298, 0.11325])
# amplitudes = [g(i, atoms, f) for i in range(1, 11)]
# polys      = [i for i in q8]       # result of some interaction of polytopes, e.g. q8 or q9
# h          = np.arange(1, 11) 

# solution =  Solutionselector(polytope=polys, Amp_measured=amplitudes, reflections=h, structurefactor=f, imax=0.5)
# best_i, best_err, best_x, loginfo  = solution.run()

#--------------------------------
# --- Possible output ---

# print("Best polytope index:", best_i)
# print("Error:", best_err)
# print(f"Best parameters: {best_x} or {np.sort(best_x)[::-1]} given: {atoms}") 

# -------------------------------------------------
# full details about each polytope fit result
# for entry in loginfo:
#     print(f"Polytope index: {entry['polytope_index']}, Error: {entry['error']}, Success: {entry['success']}, Message: {entry['message']}")
#     print(f"  Initial solution: {entry['initialsolution']}")

# or

# import pandas as pd
# df = pd.DataFrame(solution.trace)  # or pd.DataFrame(loginfo)
# print(df.columns) # to check available columns
# print(df['initialsolution'][2], df['optimizedsolution'][2])