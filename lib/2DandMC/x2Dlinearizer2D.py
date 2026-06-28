import numpy as np

import logging
logger = logging.getLogger(__name__)

# ----------------------------------------------
# ====================> General fns
# ----------------------------------------------

def find_interception(x,y,m):
    return y-m*x

def findp3x(x, m, gi, h):
    k=2*np.pi*h
    return m+np.sin(k*x)/np.sqrt(1- (gi - np.cos(k*x))**2) 

def findp3y(x,h,gi):
    k=2*np.pi*h
    return (1/(k))*np.arccos(gi-np.cos(k*x))

def findpy(x,h,gi,f):
    k=2*np.pi*h
    return (1/k)*np.arccos(gi/f[1] - (f[0]/f[1])*np.cos(k*x))

def findpx(y,h,gi,f):
    k=2*np.pi*h
    return (1/k)*np.arccos(gi/f[0] - (f[1]/f[0])*np.cos(k*y))


# ----------------------------------------------
# ====================> EPA linearization
# ----------------------------------------------

def double_segment_EPA(gi, l, f, error=0):
    
    #### Jonas Area 
    k    = 2*np.pi*l ;  gi   = np.abs(gi)
    
    #### Finding point p1 and p2  
    p1x  = (1/k)*np.arccos(gi*(1+error)/np.sum(f)) ;       #  x2* = x1*
    p1y  = p1x
    
    p2x  = (1/k)*np.arccos(gi*(1+error)-1)         #  x1* = pnt2 ; x2* = 0
    p2y  = 0
    
    m1   = (p2y-p1y)/(p2x-p1x)           # slope of First line
    n1   = find_interception(p2x,p2y,m1)
        
    #### Finding point p3, p4 and p5
    
    j = 1
    p5 = fn_solveforx_v2(l, gi, f, m1, j, error)[0]
    p5x, p5y = p5[0], p5[1]
        
    n2   = find_interception(p5x,p5y,m1)
    
    p4x  = -n2 / (m1-1)
    p4y  = p4x
    
    p3x  = -n2/m1
    p3y  = 0
    
    #pnt  = np.array([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, p5x, p5y])
    pnt  = np.array([[p1x, p1y], [p2x, p2y], [p3x, p3y], [p5x, p5y], [p4x, p4y]])
    
    return pnt

def single_segment_EPA(gi, l, f, error=0):
    #### Jonas Area 
    k    = 2*np.pi*l
    gi   = np.abs(gi)
    
    #### Finding point p1 and p2  
    
    p1x  = (1/k)*np.arccos(gi*(1+error)-1)         #  x1* = pnt2 ; x2* = 0
    p1y  = 0
    
    p2y  = (1/k)*np.arccos(gi*(1+error)-1)         #  x1* = pnt2 ; x2* = 0
    p2x  = 0
    
    m1   = (p2y-p1y)/(p2x-p1x)           # slope of First line
    n1   = find_interception(p2x,p2y,m1)
        
    #### Finding point p3, p4 and p5
    
    p5x = (1/k)*np.arccos(gi*(1-error)/np.sum(f))
    p5y = p5x #p4  = [xp]*len(f)
    
    n2   = find_interception(p5x,p5y,m1)
    
    p4y  = n2 #/ (m1-1)  # y = m1x+n2
    p4x  = 0 #p4x
    
    p3x  = -n2/m1
    p3y  = 0
    
    pnt  = np.array([[p2x, p2y], [p1x, p1y], [p3x, p3y], [p5x, p5y], [p4x, p4y]])
    
    return pnt


# ----------------------------------------------
# ====================> nEPA linearization
# ----------------------------------------------


def single_segment_nEPA(gi, l, f, j=1, error=0):
    pnt = []
    
    k   = 2*np.pi*l
    gi   = np.abs(gi)
      
    p1x  = 0
    p1y  = (1/k)*np.arccos((gi*(1+error)-f[0])/f[1])
    
    if np.isnan(p1y):
        p1y = 0.5/l
        p1x = findpx(p1y, l, gi, f)
        pnt.append([p1x, p1y])
        
    else:
        pnt.append([p1x, p1y])        
     
    
    p2x  = (1/k)*np.arccos((gi*(1+error)-f[1])/f[0])
    p2y  = 0
    pnt.append([p2x, p2y]) 
    
    
    m1   = (p2y-p1y)/(p2x-p1x)
    n1   = find_interception(p2x,p2y,m1)
    
    p4   = fn_solveforx_v2(l, gi, f, m1, j, error)
            
    if ~np.all(np.isnan(p4)):
        
        if len(p4) > 1:
            pnt = []
            
            if (p4[0,1]/p4[0,0]) <= 1 and (p4[1,1]/p4[1,0]) > 1:
            #if np.floor(p4[0,1]/p4[0,0]) <= 1 and np.floor(p4[1,1]/p4[1,0]) > 1:
                pLB=p4[0]
                pUB=p4[1]
            else:
                pLB=p4[1]
                pUB=p4[0]
            
            ### Lower Boundary (LB)
            nLB = pLB[1] - m1*pLB[0]
            
            pLB1y = 0.5/l
            pLB1x = (pLB1y - nLB)/m1
            
            pLB3x = np.abs(nLB/m1)
            pLB3y = 0
            #pLB3x = (1/k)*np.arccos((g*(1+error)-f[0])/f[1])
                        
            ### Upper Boundary (UB)
            
            nUB = pUB[1] - m1*pUB[0]
            pUB4x = np.abs(nUB/m1)
            pUB4y = 0
            
            pUB6y = 0.5/l
            pUB6x = (pUB6y - nUB)/m1
            
            ### Colloecting point in order
            pnt.append([pLB1x,  pLB1y])
            pnt.append([pLB[0], pLB[1]])
            pnt.append([pLB3x,  pLB3y])
            
            pnt.append([pUB4x,  pUB4y])
            pnt.append([pUB[0], pUB[1]])
            pnt.append([pUB6x,  pUB6y])
                        
            return np.array(pnt)
        
        elif ~np.all(np.isnan(p4)):
            p4y = p4[0][1]
            p4x = p4[0][0]
        else:
            
            p4y = 0.5/l
            p4x  = (1/k)*np.arccos( gi/f[0] - (f[1]/f[0])*np.cos(k*p4y) )
    
    n2   = find_interception(p4x,p4y,m1)
    p3x  = -n2/m1
    p3y  = 0
    
    pnt.append([p3x, p3y])
    pnt.append([p4x, p4y])
    
    p5x  = 0
    p5y  = n2
    
    if p5y > 0.5/l:
        p5y  = 0.5/l
        p5x = (p5y-n2)/m1
        pnt.append([p5x, p5y])
        
        pextra_x = 0
        pextra_y = 0.5/l
        pnt.append([pextra_x, pextra_y])
    else:
        pnt.append([p5x, p5y])
        
    return np.array(pnt)

def double_segment_nEPA(gi, l, f, j=1, error=0):
    pnt, pntLB, pntUB = [], [], []
    
    k   = 2*np.pi*l
    gi   = np.abs(gi) 
    
    p1x = (1/k)*np.arccos(gi*(1+error)/np.sum(f))
    p1y = p1x
    
    p2y = 0
    p2x = (1/k)*np.arccos((gi*(1+error)-f[1]*np.cos(k*p2y))/f[0])
    if np.isnan(p2x):
        p2x = 0.5/l
        
    m1  = (p2y-p1y)/(p2x-p1x)
    n1  = find_interception(p2x,p2y,m1)
    
    p5   = fn_solveforx_v2(l, gi, f, m1, j, error)
    
    # ---> get lower and upper tangent points
    if ~np.all(np.isnan(p5)):
        if len(p5) > 1:
            if (p5[0,1]/p5[0,0]) <= 1 and  (p5[1,1]/p5[1,0]) <= 1:
                pLB=p5
            elif (p5[0,1]/p5[0,0]) <= 1 and (p5[1,1]/p5[1,0]) > 1:
                pLB=p5[0]
                pUB=p5[1]
            else:
                pLB=p5[1]
                pUB=p5[0]
        elif len(p5) == 1 :
            pLB = p5[0]
        else:
            p5y = 0.5/l
            p5x = (1/k)*np.arccos( gi/f[0] - (f[1]/f[0])*np.cos(k*p5y) )
            pLB = np.array([p5x, p5y])
    
    # ---> use only lower tangent points for lower part
    nLB = pLB[1] - m1*pLB[0]
    p3x = np.abs(nLB/m1)
    p3y = 0
    
    p4x = nLB / (1-m1)
    p4y = p4x
    
    ### Colloecting point in order
    pnt.append([p1x, p1y])
    pnt.append([p2x, p2y])
    pnt.append([p3x, p3y])
    pnt.append([pLB[0], pLB[1]])
    pnt.append([p4x, p4y])
    
    # ---> Part 2 linearize upper part of isosurface
    p8x = 0
    p8y = (1/k)*np.arccos((gi*(1+error)-f[0])/f[1])
    if np.isnan(p8y):
        p8y = 0.5/l
        p8x = (1/k)*np.arccos( (gi*(1+error)-f[1]*np.cos(k*p8y))/f[0] )
    
    m2  = (p8y-p1y)/(p8x-p1x)
    n2  = find_interception(p8x,p8y, m2)
    
    p11 = fn_solveforx_v2(l, gi, f, m2, j, error)
    
    # ---> get lower and upper tangent points
    if ~np.all(np.isnan(p11)):
        if len(p11) > 1:
            if (p11[0,1]/p11[0,0]) < 1 and (p11[1,1]/p11[1,0]) >= 1:
                pLB=p11[0]
                pUB=p11[1]
                p10=np.array([p11[1]])
            elif (p11[0,1]/p11[0,0]) >= 1 and (p11[1,1]/p11[1,0]) >= 1:
                p10=p11
                pLB=p11[0]
                pUB=p11[1]
            else:
                p10=p11[1]
                pLB=p11[1]
                pUB=p11[0]
        elif len(p11) == 1 :
            pUB = p11[0]
            p10 = p11
        else:
            p11y = 0.5/l
            p11x = (1/k)*np.arccos( gi/f[0] - (f[1]/f[0])*np.cos(k*p11y) )
            pUB = np.array([p11x, p11y])
            p10 = pUB
    
    if len(p10) >1:
        # ---> use only lower tangent points for lower part
        
        p14x=pLB[0] ; p14y=pLB[1]
        
        nUB1=pLB[1]-m2*pLB[0]
        p15y = 0.5/l
        p15x = (p15y - nUB1)/m2
        
        p16x = nUB1 / (1-m2)
        p16y = p16x
        
        pnt.append([p16x, p16y])
        pnt.append([p14x, p14y])
        pnt.append([p15x, p15y])  

        nUB = pUB[1] - m2*pUB[0]
        p12y = 0.5/l
        p12x = (p12y - nUB)/m2
        
        p13x = nUB / (1-m2)
        p13y = p13x       
        
        pnt.append([p12x, p12y])
        pnt.append([pUB[0], pUB[1]])
        pnt.append([p13x, p13y])   
    else:
        nUB = pUB[1] - m2*pUB[0]
        
        ptypex = 0
        ptypey = (1/k)*np.arccos( (gi*(1+error)-f[0]*np.cos(k*ptypex))/f[1] )
        
        if np.isnan(ptypey):
            #print("---> I found type II isosurface")
            p12y = 0.5/l
            p12x = (p12y - nUB)/m2
        else:
            #print("---> I found type I isosurface")
            p12x = 0 
            p12y = nUB
        
        p13x = nUB / (1-m2)
        p13y = p13x
        
        pnt.append([p1x, p1y])
        pnt.append([p8x, p8y])
        pnt.append([p12x, p12y])
        pnt.append([pUB[0], pUB[1]])
        pnt.append([p13x, p13y])   
        pnt.append([p1x, p1y])
                
    return np.array(pnt)

def fn_solveforx_v2(l, gi, f, m, j, error):
    
    k = 2 * np.pi * l
    
    i = list(range(j)) + list(range(j+1,len(f)))
    
    a = (1-m*m)/(f[j]*f[j])
    b = 2 * m*m * gi /(f[j]*f[j])
    c = m*m*( 1 - (gi*gi) / (f[j]*f[j]) ) - np.array([ (f[ii]*f[ii]) / (f[j]*f[j]) for ii in i]).sum(axis = 0) 
    
    
    if a != 0:
        
        z1 = (-b + np.sqrt(b*b - 4*a*c))/(2*a)
        z2 = (-b - np.sqrt(b*b - 4*a*c))/(2*a)
        
        r1=(1/k)*np.arccos(z1/f[0])
        r2=(1/k)*np.arccos(z2/f[0])
        
        if ~np.isnan(r1) and np.isnan(r2):
            ry=findpy(r1, l, gi, f)
            return np.array([[r1, ry]])
        
        elif np.isnan(r1) and ~np.isnan(r2):
            ry=findpy(r2, l, gi, f)
            return np.array([[r2, ry]])
        
        elif np.isnan(r1) and np.isnan(r2) :
            return np.array([float("nan")])
        
        elif ~np.isnan(r1) and ~np.isnan(r2):
            r1y=findpy(r1, l, gi, f)
            r2y=findpy(r2, l, gi, f)
            return np.array([ [r1, r1y], [r2, r2y] ])
        
        else:
            return np.array([r1, r2])
    
    else:
        z1 = (-1*c/b)
        
        r1=(1/k)*np.arccos(z1/f[0])
        
        if ~np.isnan(r1):
            ry=findpy(r1, l, gi, f)
            return np.array([r1, ry])
        elif np.isnan(r1):
            prx = (1/k)*np.arccos(gi*(1+error)/np.sum(f))
            pry = prx
            return np.array([[prx, pry]])
        else:
            prx = (1/k)*np.arccos(gi*(1+error)/np.sum(f))
            pry = prx
            return np.array([[prx, pry]])

        
# def fn_solveforx_v2(l, gi, f, m, j, error):
    
#     k = 2 * np.pi * l
    
#     i = list(range(j)) + list(range(j+1,len(f)))
    
#     a = (1-m*m)/(f[j]*f[j])
#     b = 2 * m*m * gi /(f[j]*f[j])
#     c = m*m*( 1 - (gi*gi) / (f[j]*f[j]) ) - np.array([ (f[ii]*f[ii]) / (f[j]*f[j]) for ii in i]).sum(axis = 0) 
    
    
#     if a != 0:
        
#         z1 = (-b + np.sqrt(b*b - 4*a*c))/(2*a)
#         z2 = (-b - np.sqrt(b*b - 4*a*c))/(2*a)
        
#         r1=(1/k)*np.arccos(z1/f[0])
#         r2=(1/k)*np.arccos(z2/f[0])
        
#         if ~np.isnan(r1) and np.isnan(r2):
#             ry=findpy(r1, l, gi, f)
#             return np.array([r1, ry])
        
#         elif np.isnan(r1) and ~np.isnan(r2):
#             ry=findpy(r2, l, gi, f)
#             return np.array([r2, ry])
        
#         elif np.isnan(r1) and np.isnan(r2) :
#             return np.array([float("nan")])
        
#         elif ~np.isnan(r1) and ~np.isnan(r2):
#             r1y=findpy(r1, l, gi, f)
#             r2y=findpy(r2, l, gi, f)
#             return np.array([ [r1, r1y], [r2, r2y] ])
        
#         else:
#             return np.array([r1, r2])
    
#     else:
#         z1 = (-1*c/b)
        
#         r1=(1/k)*np.arccos(z1/f[0])
        
#         if ~np.isnan(r1):
#             ry=findpy(r1, l, gi, f)
#             return np.array([r1, ry])
#         elif np.isnan(r1):
#             prx = (1/k)*np.arccos(gi*(1+error)/np.sum(f))
#             pry = prx
#             return np.array([prx, pry])
#         else:
#             prx = (1/k)*np.arccos(gi*(1+error)/np.sum(f))
#             pry = prx
#             return np.array([prx, pry])



# def double_segment_nEPA(gi, l, f, j=1, error=0):
#     pnt = []
#     k   = 2*np.pi*l
#     gi   = np.abs(gi) 
    
#     p1x = (1/k)*np.arccos(gi*(1+error)/np.sum(f))
#     p1y = p1x
#     pnt.append([p1x, p1y])
    
#     p2x = (1/k)*np.arccos((gi*(1+error)-f[1])/f[0])
#     p2y = 0
    
#     if np.isnan(p2x):
#         p2x = 0.5/l
    
#     pnt.append([p2x, p2y])
    
#     m1  = (p2y-p1y)/(p2x-p1x)
#     n1  = find_interception(p2x,p2y,m1)
    
#     p5   = fn_solveforx_v2(l, gi, f, m1, j, error)
    
#     if ~np.all(np.isnan(p5)):
        
#         if len(np.shape(p5)) > 1 :
            
#             pnt = []
            
#             if np.floor(p5[0,1]/p5[0,0]) <= 1 and np.floor(p5[1,1]/p5[1,0]) > 1:
#                 pLB=p5[0]
#                 pUB=p5[1]
#             else:
#                 pLB=p5[1]
#                 pUB=p5[0]
            
#             ### Lower Boundary (LB)
#             nLB = pLB[1] - m1*pLB[0]
            
#             pLB1y = 0.5/l
#             pLB1x = (pLB1y - nLB)/m1
            
#             pLB3x = np.abs(nLB/m1)
#             pLB3y = 0
                        
#             ### Upper Boundary (UB)
            
#             nUB = pUB[1] - m1*pUB[0]
#             pUB4x = np.abs(nUB/m1)
#             pUB4y = 0
            
#             pUB6y = 0.5/l
#             pUB6x = (pUB6y - nUB)/m1
            
#             ### Colloecting point in order
#             pnt.append([pLB1x,  pLB1y])
#             pnt.append([pLB[0], pLB[1]])
#             pnt.append([pLB3x,  pLB3y])
            
#             pnt.append([pUB4x,  pUB4y])
#             pnt.append([pUB[0], pUB[1]])
#             pnt.append([pUB6x,  pUB6y])
                        
#             return np.array(pnt)
        
#         elif ~np.all(np.isnan(p5)) and len(np.shape(p5)) == 1 :
#             p5y = p5[1]
#             p5x = p5[0]
#         else:
#             p5y = 0.5/l
#             p5x  = (1/k)*np.arccos( gi/f[0] - (f[1]/f[0])*np.cos(k*p5y) )
    
#     n2  = find_interception(p5x,p5y,m1)
    
#     p4x = -n2 / (m1-1)
#     p4y = p4x
#     #print("===> p4 is ", p4x, p4y)
    
#     p3x = -n2 / m1
#     p3y = 0
    
#     pnt.append([p3x, p3y])
#     pnt.append([p5x, p5y])
    
    
#     #---> Part 2 linearization 
    
#     p6x  = 0
#     p6y  = (1/k)*np.arccos((gi*(1+error)-f[0])/f[1])
    
#     if np.isnan(p6y):
#         p6y = 0.5/l
        
#     m3   = (-p6y+p1y)/(-p6x+p1x)
    
#     p7 = fn_solveforx_v2(l, gi, f, m3, j, error)
    
#     if ~np.all(np.isnan(p7)):
#         p7x = p7[0]
#         p7y = p7[1]
        
#         n4   = find_interception(p7x,p7y,m3)
#         p8x  = 0
#         if ~np.isnan(n4) and n4<=0.5/l:
#             p8y  = n4 
#         elif ~np.isnan(n4) and n4>=0.5/l:
#             p8y  = 0.5/l
#         else:
#             print("--> from def jonopt_error_v5: do not know what to do for p8y ")
        
#     else:
#         p7y = 0.5/l
#         p7x  = (1/k)*np.arccos( gi/f[0] - (f[1]/f[0])*np.cos(k*p7y) )  # === p7x  = findpx(p7y,l,g,f)
        
#         n4   = find_interception(p7x,p7y,m3)
#         p8x  = 0
#         p8y  = 0.5/l
        
        
#     n4   = find_interception(p7x,p7y,m3)
    
#     p9x = n4/(1-m3)
#     p9y = p9x
    
#     if p9y == p4y and p9x == p4x :
#         pnt.append([p4x, p4y])
#         pnt.append([p9x, p9y])
#     else:
#         if p9y > p4y and p9x > p4x :
                    
#             p9x = (n2-n4) / (m3-m1)
#             p9y = m3 * p9x + n4
            
#             pnt.append([p4x, p4y])
#             pnt.append([p9x, p9y])
            
#             m49   = (p9y-p4y)/(p9x-p4x)
#             n49   = p1y - m49*p1x
#             #pnewx = fn_solveforx(l, g, f, m49, j, xexp) #p1x + ( (0.5/l) - p1y ) / m49
#             pnewy = p9y #(1/k)*np.arccos( g/f[1] - (f[0]/f[1])*np.cos(k*pnewx) )
#             pnewx = (pnewy - n49)/( m49 )
            
#             #pnt.append([pnewx, pnewy])
            
#         else:
            
#             p9x = (n2-n4) / (m3-m1)
#             p9y = m3 * p9x + n4
            
#             #pnt.append([p4x, p4y])
#             pnt.append([p9x, p9y])
        
#             m49   = (p9y-p4y)/(p9x-p4x)
#             n49   = p1y - m49*p1x
#             pnewy = p9y
#             pnewx = (pnewy - n49)/( m49 )
            
#             #pnt.append([pnewx, pnewy])
                
#     pnt.append([p7x, p7y])
#     pnt.append([p8x, p8y])
#     pnt.append([p6x, p6y])
#     pnt.append([p1x, p1y])
    
    
#     return np.array(pnt)