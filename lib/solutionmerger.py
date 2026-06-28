import numpy as np
import polytope as pc

from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from collections import defaultdict

import logging

class SolutionMerger:
    """
    Task: Detect connected overlapping polytope regions and merge each region into a single polytope.
    
    Input : list of polytopes (e.g. from EPA, nEPA, or any source)
    
    Output: list of merged polytopes, list of components (indices of original polytopes in each region)
    
    """
    
    # -------------------------------------------------
    # INIT
    # -------------------------------------------------
    def __init__(self, polytopes, logger=None):
        ''' Initialize with a list of polytopes '''
        
        self.logger = logger or logging.getLogger("psc.solutionmerger")
        
        self.polytopes = polytopes
        self.n = len(polytopes)
        
        self.boxes = None
        self.graph = None
        self.components = None
        self.merged_regions = None
        
    # -------------------------------------------------
    # Bounding box from vertices
    # -------------------------------------------------
    @staticmethod
    def bounding_box(P):
        ''' Compute the axis-aligned bounding box of a polytope P using its vertices '''
        
        V = np.asarray(pc.extreme(P))
        
        mins = V.min(axis=0)
        maxs = V.max(axis=0)
        
        return mins, maxs
    
    # -------------------------------------------------
    # Bounding box overlap test
    # -------------------------------------------------
    @staticmethod
    def bbox_overlap(box1, box2):
        ''' Check if two bounding boxes overlap '''
        
        min1, max1 = box1
        min2, max2 = box2
        
        return np.all(max1 >= min2) and np.all(max2 >= min1)
    
    # -------------------------------------------------
    # Exact LP feasibility intersection
    # -------------------------------------------------
    @staticmethod
    def polytope_intersect(P1, P2):
        ''' Check if two polytopes intersect by solving an LP feasibility problem '''
        
        A = np.vstack([P1.A, P2.A])
        b = np.hstack([P1.b, P2.b])
        
        c = np.zeros(A.shape[1])
        
        res = linprog( c, A_ub=A, b_ub=b, method="highs" )
        
        return res.success
    
    # -------------------------------------------------
    # Precompute bounding boxes
    # -------------------------------------------------
    def compute_boxes(self):
        ''' Precompute bounding boxes for all polytopes '''
        
        self.boxes = [ self.bounding_box(P) for P in self.polytopes ]
    
    # -------------------------------------------------
    # Build overlap graph
    # -------------------------------------------------
    def build_graph(self):
        ''' Build a graph where nodes are polytopes and edges indicate intersection '''
        
        if self.boxes is None:
            self.compute_boxes()
            
        graph = defaultdict(list)
        
        for i in range(self.n):
            
            for j in range(i + 1, self.n):
                
                # -----------------------------
                # FAST rejection
                # -----------------------------
                if not self.bbox_overlap( self.boxes[i], self.boxes[j] ):
                    continue
                
                # -----------------------------
                # Exact LP feasibility
                # -----------------------------
                if self.polytope_intersect( self.polytopes[i], self.polytopes[j] ):
                    graph[i].append(j)
                    graph[j].append(i)
        
        self.graph = graph
        
    # -------------------------------------------------
    # Connected components
    # -------------------------------------------------
    def find_components(self):
        ''' Find connected components in the overlap graph '''
        
        if self.graph is None:
            self.build_graph()
            
        visited = set()
        components = []
        
        for i in range(self.n):
            if i in visited:
                continue
            
            stack = [i]
            comp = []
            
            while stack:
                
                node = stack.pop()
                
                if node in visited:
                    continue
                
                visited.add(node)
                comp.append(node)
                
                for nb in self.graph[node]:
                    
                    if nb not in visited:
                        stack.append(nb)
            
            components.append(comp)
            
        self.components = components
        
    # -------------------------------------------------
    # Merge one component into one polytope
    # -------------------------------------------------
    def merge_component(self, component):
        ''' Merge a single component (list of indices) into one polytope using convex hull of vertices '''
        
        all_vertices = []
        
        for idx in component:
            
            V = np.asarray( pc.extreme(self.polytopes[idx]))
            
            all_vertices.append(V)
            
        V_all = np.vstack(all_vertices)
        
        hull = ConvexHull(V_all)
        
        A = hull.equations[:, :-1]
        b = -hull.equations[:, -1]
        
        return pc.Polytope(A, b)
    
    # -------------------------------------------------
    # Merge all regions
    # -------------------------------------------------
    def merge_regions(self):
        ''' Merge all components into merged polytopes '''
        
        if self.components is None:
            self.find_components()
        
        merged = []
        
        for comp in self.components:
            merged.append(self.merge_component(comp))
            
        self.merged_regions = merged
        
    # -------------------------------------------------
    # Full pipeline
    # -------------------------------------------------
    def runmerger(self):
        ''' Run the full pipeline: compute boxes, build graph, find components, merge regions '''
        
        self.compute_boxes()
        
        self.build_graph()
        
        self.find_components()
        
        self.merge_regions()
                
        return self.merged_regions, self.components


# -------------------------------------------------
# Method to use the class
# -------------------------------------------------
# 
# getmerged = SolutionMerger(p67)
# %time merged_regions, components = getmerged.runmerger()
#
# or simply
#
# %time cp67final = SolutionMerger(p67).runmerger()