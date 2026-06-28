import numpy as np
import pandas as pd
import os


import numpy as np
import pandas as pd
import os


class HKLAnalyzer:

    def __init__(self, fpath, fhkl, option=0):
        self.fpath = fpath
        self.fhkl = fhkl
        self.option = option

        self.df = None
        self.results = None

        self.load()

    def find_col(self, names):
        for n in names:
            if n in self.df.columns: return n
        raise ValueError(f"Missing column. Tried: {names}")

    def load(self):

        with open(os.path.join(self.fpath, self.fhkl), "r") as f:
            self.df = pd.read_csv(f, sep=r"\s+", encoding="latin")

        rename = {"H":"h", "K":"k", "L":"l", "F":"|F|", "Freal":"F(real)", "Fimag":"F(imag)"}
        self.df = self.df.rename(columns=rename)

    def analyze(self):
        
        df = self.df
        
        hcol = self.find_col(["h"])
        kcol = self.find_col(["k"])
        lcol = self.find_col(["l"])
        
        fcol = self.find_col(["|F|", "F", "Amplitude"])
        freal = self.find_col(["F(real)", "Real", "Freal"])
        fimag = self.find_col(["F(imag)", "Imag", "Fimag"])
        
        lambdac = self.find_col(["ID(Î»)", "ID(λ)", "Lambda", "Wavelength"])
        dcol = self.find_col(["d(Ã…)", "d(Å)", "d"])
        twotheta = self.find_col(["2Î¸", "2theta", "TwoTheta"])
        
        lambdas = pd.unique(df[lambdac])
        
        if self.option >= len(lambdas): raise ValueError("Invalid wavelength option")
        
        df = df[df[lambdac] == lambdas[self.option]]
        
        if "Phase" in df.columns:
            df = df[df["Phase"] != 0]
        
        hdf = df[(df[hcol]!=0)&(df[kcol]==0)&(df[lcol]==0)]
        kdf = df[(df[hcol]==0)&(df[kcol]!=0)&(df[lcol]==0)]
        ldf = df[(df[hcol]==0)&(df[kcol]==0)&(df[lcol]!=0)]
        
        hkdf = df[(df[hcol]!=0)&(df[kcol]!=0)&(df[lcol]==0)]
        hldf = df[(df[hcol]!=0)&(df[kcol]==0)&(df[lcol]!=0)]
        kldf = df[(df[hcol]==0)&(df[kcol]!=0)&(df[lcol]!=0)]
        hkldf = df[(df[hcol]!=0)&(df[kcol]!=0)&(df[lcol]!=0)]
        
        def phase(x):
            return np.arctan2(x[fimag], x[freal]).to_numpy()
        
        def wavelength(x):
            return 2*x[dcol]*1e-10*np.sin(np.radians(x[twotheta]/2))
        
        hRO = hdf[[hcol]].to_numpy()
        kRO = kdf[[kcol]].to_numpy()
        lRO = ldf[[lcol]].to_numpy()
        
        hkRO = hkdf[[hcol,kcol]].to_numpy()
        hlRO = hldf[[hcol,lcol]].to_numpy()
        klRO = kldf[[kcol,lcol]].to_numpy()
        hklRO = hkldf[[hcol,kcol,lcol]].to_numpy()
        
        
        hsqrtI = hdf[fcol].to_numpy()
        ksqrtI = kdf[fcol].to_numpy()
        lsqrtI = ldf[fcol].to_numpy()
        
        hksqrtI = hkdf[fcol].to_numpy()
        hlsqrtI = hldf[fcol].to_numpy()
        klsqrtI = kldf[fcol].to_numpy()
        hklsqrtI = hkldf[fcol].to_numpy()
        
        
        h_lambda = wavelength(hdf)
        k_lambda = wavelength(kdf)
        l_lambda = wavelength(ldf)
        
        hk_lambda = wavelength(hkdf)
        hl_lambda = wavelength(hldf)
        kl_lambda = wavelength(kldf)
        hkl_lambda = wavelength(hkldf)
        
        for x in [h_lambda,k_lambda,l_lambda,hk_lambda,hl_lambda,kl_lambda,hkl_lambda]:
            if len(x):
                wl = x.iloc[0] if hasattr(x,"iloc") else x[0]
                break
        
        energy = (6.62607015e-34*2.99792458e8)/(wl*1.602176634e-19)
        
        self.results = (
            [hRO, hsqrtI, phase(hdf), h_lambda, energy],
            [kRO, ksqrtI, phase(kdf), k_lambda, energy],
            [lRO, lsqrtI, phase(ldf), l_lambda, energy],

            [hkRO, hksqrtI, phase(hkdf), hk_lambda, energy],
            [hlRO, hlsqrtI, phase(hldf), hl_lambda, energy],
            [klRO, klsqrtI, phase(kldf), kl_lambda, energy],

            [hklRO, hklsqrtI, phase(hkldf), hkl_lambda, energy]
            )

        return self.results


# --- USGAE
# from hklanalyzer import HKLAnalyzer
# hkl = HKLAnalyzer("./data", "sample.hkl", option=0)
# hinfo, kinfo, linfo, hkinfo, hlinfo, klinfo, hklinfo = hkl.analyze()    



def find_col(df, names):
    for n in names:
        if n in df.columns: return n
    raise ValueError(f"Missing column. Tried: {names}")


def analyzehkl(fpath: str, fhkl: str, option: int = 0) -> list:
    '''
    input
        fpath - path to input file fhkl
        fhkl  - experimental hkl file to read
    output
        fout  -  "hklanalysis.txt" file is created and analysis information is written
        return - Returns 7 list corresponding to h, k, l, hk, hl, kl, and hkl projections.
                 Each list contains [RO, |F|, phase, lambda, energy] values for structure determination process.
                 
    Note: use encoding=latin to treat lambda, theta and angstroem symbols  
          df['ID(Î»)']     = df[ ID(lambda) ]
          df['d(Ã\x85)']   = df[ d(angstroem)]
          df['2Î']        = df[ 2*theta]
    '''
    with open(os.path.join(fpath, fhkl), "r") as f:
        df = pd.read_csv(f, sep=r"\s+", encoding="latin")

    rename = {"H":"h", "K":"k", "L":"l", "F":"|F|", "Freal":"F(real)", "Fimag":"F(imag)"}
    df = df.rename(columns=rename)

    hcol = find_col(df, ["h"])
    kcol = find_col(df, ["k"])
    lcol = find_col(df, ["l"])
    fcol = find_col(df, ["|F|", "F", "Amplitude"])
    freal = find_col(df, ["F(real)", "Real", "Freal"])
    fimag = find_col(df, ["F(imag)", "Imag", "Fimag"])
    lambdac = find_col(df, ["ID(Î»)", "ID(λ)", "Lambda", "Wavelength"])
    dcol = find_col(df, ["d(Ã…)", "d(Å)", "d"])
    twotheta = find_col(df, ["2Î¸", "2theta", "TwoTheta"])

    lambdas = pd.unique(df[lambdac])
    if option >= len(lambdas): raise ValueError(f"Invalid lambda option {option}")

    df = df[df[lambdac] == lambdas[option]]
    df = df[df["Phase"] != 0] if "Phase" in df.columns else df

    hdf = df[(df[hcol]!=0)&(df[kcol]==0)&(df[lcol]==0)]
    kdf = df[(df[hcol]==0)&(df[kcol]!=0)&(df[lcol]==0)]
    ldf = df[(df[hcol]==0)&(df[kcol]==0)&(df[lcol]!=0)]

    hkdf = df[(df[hcol]!=0)&(df[kcol]!=0)&(df[lcol]==0)]
    hldf = df[(df[hcol]!=0)&(df[kcol]==0)&(df[lcol]!=0)]
    kldf = df[(df[hcol]==0)&(df[kcol]!=0)&(df[lcol]!=0)]
    hkldf = df[(df[hcol]!=0)&(df[kcol]!=0)&(df[lcol]!=0)]

    if hkldf.empty and hdf.empty and kdf.empty and ldf.empty: raise ValueError("No valid reflections found")

    def get_phase(x):
        return np.arctan2(x[fimag], x[freal]).to_numpy()

    def get_lambda(x):
        return 2*x[dcol]*1e-10*np.sin(np.radians(x[twotheta]/2))

    hRO = hdf[[hcol]].to_numpy()
    kRO = kdf[[kcol]].to_numpy()
    lRO = ldf[[lcol]].to_numpy()

    hkRO = hkdf[[hcol,kcol]].to_numpy()
    hlRO = hldf[[hcol,lcol]].to_numpy()
    klRO = kldf[[kcol,lcol]].to_numpy()
    hklRO = hkldf[[hcol,kcol,lcol]].to_numpy()

    hsqrtI = hdf[fcol].to_numpy()
    ksqrtI = kdf[fcol].to_numpy()
    lsqrtI = ldf[fcol].to_numpy()

    hksqrtI = hkdf[fcol].to_numpy()
    hlsqrtI = hldf[fcol].to_numpy()
    klsqrtI = kldf[fcol].to_numpy()
    hklsqrtI = hkldf[fcol].to_numpy()

    hphase = get_phase(hdf)
    kphase = get_phase(kdf)
    lphase = get_phase(ldf)

    hkphase = get_phase(hkdf)
    hlphase = get_phase(hldf)
    klphase = get_phase(kldf)
    hklphase = get_phase(hkldf)

    h_lambda = get_lambda(hdf)
    k_lambda = get_lambda(kdf)
    l_lambda = get_lambda(ldf)

    hk_lambda = get_lambda(hkdf)
    hl_lambda = get_lambda(hldf)
    kl_lambda = get_lambda(kldf)
    hkl_lambda = get_lambda(hkldf)

    for x in [h_lambda,k_lambda,l_lambda,hk_lambda,hl_lambda,kl_lambda,hkl_lambda]:
        if len(x): 
            wavelength = x.iloc[0] if hasattr(x,"iloc") else x[0]
            break

    energy = (6.62607015e-34*2.99792458e8)/(wavelength*1.602176634e-19)

    hinfo = [hRO, hsqrtI, hphase, h_lambda, energy]
    kinfo = [kRO, ksqrtI, kphase, k_lambda, energy]
    linfo = [lRO, lsqrtI, lphase, l_lambda, energy]

    hkinfo = [hkRO, hksqrtI, hkphase, hk_lambda, energy]
    hlinfo = [hlRO, hlsqrtI, hlphase, hl_lambda, energy]
    klinfo = [klRO, klsqrtI, klphase, kl_lambda, energy]

    hklinfo = [hklRO, hklsqrtI, hklphase, hkl_lambda, energy]

    return hinfo, kinfo, linfo, hkinfo, hlinfo, klinfo, hklinfo