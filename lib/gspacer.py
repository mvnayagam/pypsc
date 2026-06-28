import numpy as np
import polytope as pc


import logging
logger = logging.getLogger(__name__)



class GSpacer:
    
    def __init__(self, reflection, atoms, structurefactor: float, imax: float=0.5):
        
        """ __Collects needed parameter for the class objects__

        Args:
            - reflection (int): XRD reflection order, basically the Miller index of reflection
            - atoms (list): array of atomic positions
            - structurefactor (list): array of scattering factors. simply set to 1 in EPA cases 
        
        Drived parameter: 
            - k (float): the constant term for given reflection index 2 * pi * h
            - dim (int): dimension of given problem         
        """
        
        self.reflection = reflection
        self.atoms = np.asarray(atoms, dtype=float)
        self.sf = np.asarray(structurefactor, dtype=float)
        self.imax = imax
        self.k = 2*np.pi*reflection
        self.dim = len(structurefactor)
    
    @staticmethod
    def Asym(dim: int, imax=0.5) -> object:
        
        temp = np.tril(np.ones(shape=(dim, dim)), 0)
        temp = imax*np.vstack([[0]*dim, temp])
        
        return pc.qhull(np.array(temp))

    def g(self) -> float:
        """ __returns scattered amplitude with fixed scattering strength__ 
        
        Returns:
            - float: scattered amplitude
        """
        return (np.sum([self.sf*np.cos(2*np.pi*self.reflection*self.atoms)]))
    
    def g_vectorized(self) -> float:
        """ __returns scattered amplitude with fixed scattering strength__ 
        
        Returns:
            - float: scattered amplitude
        """
        reflections = np.asarray(self.reflection)
        return np.sum( (self.sf*np.cos(2*np.pi*reflections[:, None]*self.atoms[None, :])) , axis=1)
    
    def F(self) -> float:
        """returns scattered amplitude with individual scattering strengths 
        structurefactor(i) for given atomic positions atoms(i), i - atoms index. Centrosymmetric case.
        
        Args:
            reflection (int): array of possible atomic coordinates
            atoms (list): array of atoms positions
            structurefactor (list): array of scattering factors. simply set to 1.
        Returns:
            float: scattered amplitude
        """
        
        return (np.sum([self.sf*np.cos(self.k*self.atoms) ]))
    
    def F_vectorized(self) -> float:
        """returns scattered amplitude with individual scattering strengths 
        structurefactor(i) for given atomic positions atoms(i), i - atoms index. Centrosymmetric case.
        
        Args:
            reflection (int): array of possible atomic coordinates
            atoms (list): array of atoms positions
            structurefactor (list): array of scattering factors. simply set to 1.
        Returns:
            float: scattered amplitude
        """
        reflections = np.asarray(self.reflection)
        return np.sum( (self.sf*np.cos(2*np.pi*reflections[:, None]*self.atoms[None, :])) , axis=1)        
        
    def hsurf_g(self, amplitude: float, atomindex: int, signfactor: int = 1):
        """ __returns hyper isosurface for given scattering amplitude__
        
        Args:
            - amplitude (float): the amplitude of intensity
            - atomindex (int): for which atoms the hsurf is to be solved
            - signfactor (int, optional): sign of amplitude. always +/- 1
        
        Returns:
            list: returns hyper isosurface for given amplitude gi
        """
        # mask all indices except atomindex
        mask = np.ones(self.dim, dtype=bool)
        mask[atomindex] = False
            
        # vectorized cosine sum
        cos_terms = np.cos(self.k * self.atoms)
        
        argm = (signfactor * amplitude / self.sf[atomindex]) - np.sum((self.sf[mask] / self.sf[atomindex]) * cos_terms)
        
        # numercal safty
        # argm = argm-np.ceil(argm) if argm <0 else argm-np.floor(argm)
        # argm = np.clip(argm, -1.0, 1.0)

        xj = np.arccos(argm) / self.k
        
        return xj

    def hsurf_F(self, Intensity: float, atomindex: int, signfactor1: int=1, signfactor2: int=1) -> float:
        """__returns possible coordinate of 'atomindex'th atom. F = np.sqrt(I) is used__
                
        Args:
            - Intensity (float): intensity
            - atomindex (int): selected coordinate
            - signfactor1 (int, optional): sign of amplitude (1: positive, -1: negative)
            - signfactor2 (int, optional): sign of ordinate (1: positive, -1: negative)
            - nan (str, optional): Defaults to True.
        Returns:
            - list: possible coordinates along atomindex direction
        """
        s1, s2 = np.sign(signfactor1), np.sign(signfactor2)

        inds = np.delete(np.arange(self.dim), atomindex)

        gj = np.sum(self.sf[inds] * np.cos(self.k * self.atoms[inds]))

        return s2 * np.arccos((s1*np.sqrt(Intensity) - gj) / self.sf[atomindex]) / self.k
    
    def hsurf_F2(self, Intensity: float, atomindex: int, signfactor1: int=1, signfactor2: int=1, nan: bool=True) -> np.ndarray:
        """__returns possible coordinate of 'atomindex'th atom. I = |F*F|² is used__
                
        Args:
            - Intensity (float): intensity
            - atomindex (int): selected coordinate
            - signfactor1 (int, optional): sign of amplitude (1: positive, -1: negative)
            - signfactor2 (int, optional): sign of ordinate (1: positive, -1: negative)
            - nan (str, optional): Defaults to True.
        Returns:
            - list: possible coordinates along atomindex direction
        """
        
        s1, s2 = np.sign([signfactor1, signfactor2])

        inds = np.delete(np.arange(self.dim), atomindex)

        cosx = np.cos(self.k * self.atoms[inds])
        sf = self.sf[inds]

        # contribution from atoms except atomindex
        F_rest = np.sum(sf[:,None,None] * cosx, axis=0)

        cj = F_rest * np.conj(F_rest) - Intensity

        # cross term with atomindex
        bj = F_rest * np.conj(self.sf[atomindex]) + np.conj(F_rest) * self.sf[atomindex]

        # self contribution of atomindex
        aj = self.sf[atomindex] * np.conj(self.sf[atomindex])

        zj = (-bj + s1*np.sqrt(bj**2 - 4*aj*cj)) / (2*aj)

        xj = np.real(s2*np.arccos(zj) / self.k)

        if nan:
            invalid = (bj**2 < 4*aj*cj) | (np.abs(zj) > 1.0)
            xj = np.where(invalid, np.nan, xj)

        return xj
        
    def grad_g(self) -> list:
        """Returns the gradient vector at point atoms
        
        Args:
            reflection (int): represent reflection
            atoms (list): test structure 
            structurefactor (list/array): list of atomic scattering factors
            normalize (bool, optional): Defaults to True. This normalizes gradient vector
            
        Returns:
            _array_: Returns the gradient vector as array
        """    
        gradG = np.array([-self.sf*self.k*np.sin(self.k*self.atoms)])
        return gradG/np.linalg.norm(gradG)

    def grad_F(self) -> list:
        """Returns the gradient vector at point atoms

        Args:
            reflection (int): represent reflection
            atoms (list): test structure 
            structurefactor (list/array): list of atomic scattering factors       
        Returns:
            _array_: Returns the gradient vector as array
        """      
        gradF = np.array( [-self.k*self.sf*np.sin(self.k * self.atoms)] )
        
        return gradF/np.linalg.norm(gradF)




def Asym(dim: int, imax=0.5) -> object:
    
    temp = np.tril(np.ones(shape=(dim, dim)), 0)
    temp = imax*np.vstack([[0]*dim, temp])
    
    return pc.qhull(np.array(temp))

def g(h: int, x: list, f: list) -> float:
    """returns scattered amplitude with fixed scattering strength 
       f for given atomic positions x. Centrosymmetric case.
    Args:
        h: reflection index
        f: atomic scattering factor, scalar quantity
        x: array of atom positions
    Returns:
        float: scattered amplitude
    """
    x = np.asarray(x, dtype=float)
    f = np.asarray(f, dtype=float)
    
    # return ( sum([f[i]*np.cos(2*np.pi*h*x[i]) for i in range(len(x))] ))
    return (np.sum([f * np.cos( 2*np.pi * h * x) ] ))

def F(h: int, x: list, f: list) -> float:
    """returns scattered amplitude with individual scattering strengths 
       f(i) for given atomic positions x(i), i - atom index. Centrosymmetric case.
       
    Args:
        h (int): array of possible atomic coordinates
        x (list): array of atom positions
        f (list): array of scattering factors. simply set to 1.
    Returns:
        float: scattered amplitude
    """
    x = np.asarray(x, dtype=float)
    f = np.asarray(f, dtype=float)
    
    # return (sum([f[i]*np.cos(2*np.pi*h*x[i]) for i in range(len(f))]))
    return (np.sum([f * np.cos( 2*np.pi * h * x) ] ))
    
def hsurf_g(h: int, x: list, f: list, gi: float, j: int, s: int=1) -> list:
    """returns hyper isosurface for given amplitude gi
    
    Args:
        h (int): array of possible atomic coordinates
        x (list): array of atom positions
        f (list): array of scattering factors. simply set to 1.
        gi (float): the amplitude of intensity
        j (int): for which atoms the hsurf is to be solved
        s (int, optional): sign of amplitude. always +/- 1

    Returns:
        list: returns hyper isosurface for given amplitude gi
    """
    x = np.asarray(x, dtype=float)
    f = np.asarray(f, dtype=float)
    
    
    ilist = list(range(j)) + list(range(j+1,len(x))) # select parameter indices except index j
    
    k = 2*np.pi*h
    argm = s*gi/f[j] - np.array([(f[i]/f[j])*np.cos(k*x[i]) for i in ilist]).sum(axis = 0)
    xj = (np.arccos(argm))/k
    
    return xj

def hsurf_F(h: int, x: list, f: list, I: float, j: int, s: int =1, s2: int = 1) -> list:
    """ list of possible coordinates along j direction
    
    Args:
        h (int): reflection index
        x (list): array of atom positions
        f (list): array of atomic scattering factors
        I (float): intensity
        j (int): selected coordinate
        s (int, optional): sign of amplitude (1: positive, -1: negative)
        s2 (int, optional): sign of coordinate (1: positive, -1: negative)
        nan (str, optional): Defaults to True.

    Returns:
        list: possible coordinates along j direction
    """
    
    s, s2  = np.sign(s),np.sign(s2) 
    inds = list(range(j)) + list(range(j+1,len(f))) # select parameter indices except index j
    #same as inds = np.delete(np.arange(len(f)), j)
    
    gj = [f[i]*np.cos(2*np.pi*h*x[i]) for i in inds]
    
    return s2*np.arccos((s*np.sqrt(I) - sum(gj))/f[j])/(2*np.pi*h)

def hsurf_F2(h: int, x: list, f: list, I: float, j: int, s: int =1, s2: int = 1, nan: str = True) -> list:
    """_summary_

    Args:
        h (int): reflection index
        x (list): array of atom positions
        f (list): array of atomic scattering factors
        I (float): intensity
        j (int): selected coordinate
        s (int, optional): sign of amplitude (1: positive, -1: negative)
        s2 (int, optional): sign of coordinate (1: positive, -1: negative)
        nan (str, optional): _description_. Defaults to True.

    Returns:
        list: possible coordinates along j direction
    """
    
    s, s2  = np.sign([s, s2])
    inds = list(range(j)) + list(range(j+1,len(f))) # select parameter indices except index j
    #same as inds = np.delete(np.arange(len(f)), j)
    
    # double sum over all indices except index j
    gj = [f[i]*np.conj(f[k])*np.cos(2*np.pi*h*x[i])*np.cos(2*np.pi*h*x[k]) for i in inds for k in inds]
    cj = np.array(gj).sum(axis = 0) - I
    
    # single sum over all indices except index j
    b1j = np.array([f[i]*np.cos(2*np.pi*h*x[i]) for i in inds]).sum(axis = 0)*np.conj(f[j])
    b2j = np.array([np.conj(f[i])*np.cos(2*np.pi*h*x[i]) for i in inds]).sum(axis = 0)*f[j]
    bj = b1j + b2j
    
    # contribution from index j
    aj = f[j]*np.conj(f[j])
    
    # print(cj, bj, aj)
    zj = (-bj + s*np.sqrt(bj**2 - 4*aj*cj))/(2*aj)
    xj = np.real(s2*np.arccos(zj)/(2*np.pi*h))

    if nan and isinstance(xj, list):
        
        ifalse = np.where((bj**2 < 4*aj*cj) + (np.abs(zj) > 1.0))
        xj[ifalse] = np.nan

    return xj

def grad_g(h: int, x: list, f:list) -> list:
    """Returns the gradient vector at point x
    
    Args:
        h (int): represent reflection
        x (list): test structure 
        f (list/array): list of atomic scattering factors
        normalize (bool, optional): Defaults to True. This normalizes gradient vector
        
    Returns:
        _array_: Returns the gradient vector as array
    """    
    k = 2*np.pi*h
    
    if len(x) > 1:
        gradG = np.array([-f[i]*k*np.sin(k*x[i]) for i in range(len(f))])
        return gradG/np.linalg.norm(gradG)
    else:
        gradG = np.array([-f[i]*k*np.sin(k*x) for i in range(len(f))])
        return gradG/np.linalg.norm(gradG)

def grad_F(h: int, x: list, f:list, normalize=True) -> list:
    """Returns the gradient vector at point x

    Args:
        h (int): represent reflection
        x (list): test structure 
        f (list/array): list of atomic scattering factors
        normalize (bool, optional): Defaults to True. This normalizes gradient vector
        
    Returns:
        _array_: Returns the gradient vector as array
    """      
    k = 2*np.pi*h
    
    gradF = np.array( [-k*f[i]*np.sin(k*x[i]) for i in range(len(f))])
    
    return gradF/np.linalg.norm(gradF)

