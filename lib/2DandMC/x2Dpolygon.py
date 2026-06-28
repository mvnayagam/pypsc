import numpy as np

import shapely
from shapely.geometry import Polygon
from shapely.validation import make_valid
from shapely.ops import unary_union

import logging
logger = logging.getLogger(__name__)

def multistrip(r1:int, r2:int, pnts):
    d=[]
    for i in range(r1, r2,1):
        df=int(len(pnts[i]))
        d1=Polygon([(pnts[i][j], pnts[i][j+1]) for j in range(0,df,2)])
        d.append(d1)
    return shapely.geometry.MultiPolygon([poly for poly in d])



# -------------------- For EPA linearization

def getploygons_EPA_SS(h: int, points: np.array, IorG : str ='intensity', imax=0.5):
    
    r1, r2 = 0, 0
    a  = []
    
    for i in range(1,h+1):
        #r1 = r1+4*(i-1)**2
        #r2 = r2+4*i*i
    
        r2=r2+i*i*2  if IorG == 'intensity' else r2+i*i
        aa = multistrip(int(r1), int(r2),points)
        
        try:
            aa = unary_union(aa)
        except:
            print("AssertionFailedException occured for RO h=", i, "trying with make_valid")
            aa = make_valid(aa)
    
        a.append(aa)
        r1=np.copy(r2)
        
    return (a)


def getploygons_EPA_DS(h: int, points: np.array, IorG : str ='intensity', imax=0.5):
    
    r1, r2 = 0, 0
    a  = []
    
    for i in range(1,h+1):
        #r1 = r1+4*(i-1)**2
        #r2 = r2+4*i*i
    
        r2=r2+i*i*4  if IorG == 'intensity' else r2+2*i*i
        aa = multistrip(int(r1), int(r2),points)
        
        try:
            aa = unary_union(aa)
        except:
            print("AssertionFailedException occured for RO h=", i, "trying with make_valid")
            aa = make_valid(aa)
    
        a.append(aa)
        r1=np.copy(r2)
        
    return (a)


# -------------------- For nEPA linearization

def getploygons_nEPA(h: int, points: np.array, imax=0.5):
    
    r1, r2 = 0, 0
    a  = []
    
    for i in range(1,h+1):
        #r1 = r1+4*(i-1)**2
        #r2 = r2+4*i*i
    
        r2=r2+2*i*i
        aa = multistrip(int(r1), int(r2),points)
    
        try:
            aa = unary_union(aa)
        except:
            print("AssertionFailedException occured for RO h=", i, "trying with make_valid")
            aa = make_valid(aa)
    
        a.append(aa)
        r1=np.copy(r2)
        
    return (a)



# -------------------- Intersection between polygons

def polyintersect(h:int, polylist: list, fname: str):
    
    s  = []
    
    for j in range(h-1):
        #print("Doing for j's upto :: ", j+1," with j = ",j+2)
        try:
            if j == 0:
                ss = polylist[j].intersection(polylist[j+1])
            else:
                ss = s[-1].intersection(polylist[j+1])
        except:
            fname.write(f'---> TopologyException error for at h = {j+1}\n')
            continue
        
        if not ss:
            #print("===> ss is empty for j = ", j+2)
            ss=s[-1]
        
        s.append(ss)
        
    #return (s, j)
    return s


def polyintersect_MC(h:int, polylist: list, xcoor: list, fname: str, count: int=0):
    
    s  = []
    
    for j in range(h-1):
        #print("Doing for j's upto :: ", j+1," with j = ",j+2)
        try:
            if j == 0:
                ss = polylist[j].intersection(polylist[j+1])
            else:
                ss = s[-1].intersection(polylist[j+1])
        except:
            fname.write(f'Pair-{count} : TopologyException error for x1 = {xcoor[0]} and x2 = {xcoor[1]} at h = {j+1}\n')
            continue
        
        if not ss:
            #print("===> ss is empty for j = ", j+2)
            ss=s[-1]
        
        s.append(ss)
        
    #return (s, j)
    return s
