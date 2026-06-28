import numpy as np
import pandas as pd
import unicodedata as ud
import os, sys

def analyzehkl(fpath, fhkl, option=0):
    
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
          df['2Î¸']        = df[ 2*theta]
    '''
           
    with open(os.path.join(fpath, fhkl), "r") as ftoread:
        df = pd.read_csv(ftoread, delim_whitespace=True, encoding='latin')#, decimal='\t')
    
    if df.iloc[0].count() != df.shape[1] : # if df['Phase'].isnull().all(): # checking for nan values in Phase is another option
        print(f'\x1b[1;31m---> There is a problem in reading header and the column values\n')
        print(f'\x1b[1;31m     Total header {df.iloc[0].count()} is different from available column {df.shape[1]}')
        print(f'\x1b[1;31m     Check headers in {fhkl} file. I am quitting now')
        sys.exit()
    
    print(f"df.columns.values : {df.columns.values}")
    #print(df.shape)
    
    dflambda  = pd.unique(df['ID(Î»)'])
    lambdalen = len(dflambda)
    lambdalst = dflambda[option]
        
    ndf = df[df['ID(Î»)'] == dflambda[option]]
    #ndf = df[df['ID(Î»)'] == dflambda[option]]
    ndf_clean = ndf.drop(ndf[ndf.Phase==0].index)
    
    hdf=ndf_clean[(ndf_clean['h']!=0) & (ndf_clean['k']==0) & (ndf_clean['l']==0)]
    kdf=ndf_clean[(ndf_clean['h']==0) & (ndf_clean['k']!=0) & (ndf_clean['l']==0)]
    ldf=ndf_clean[(ndf_clean['h']==0) & (ndf_clean['k']==0) & (ndf_clean['l']!=0)]
    hkdf=ndf_clean[(ndf_clean['h']!=0) & (ndf_clean['k']!=0) & (ndf_clean['l']==0)]
    hldf=ndf_clean[(ndf_clean['h']!=0) & (ndf_clean['k']==0) & (ndf_clean['l']!=0)]
    kldf=ndf_clean[(ndf_clean['h']==0) & (ndf_clean['k']!=0) & (ndf_clean['l']!=0)]
    hkldf=ndf_clean[(ndf_clean['h']!=0) & (ndf_clean['k']!=0) & (ndf_clean['l']!=0)]
    
    with open(os.path.join(fpath, 'hklanalysis.txt'), 'w') as ftowrit:
        ftowrit.write(f"\n\n======================>>  WELCOME TO PARAMETER SPACE ANALYSIS <<======================\n\n")
        ftowrit.write(f"---> The found out lambda IDs are {dflambda} and the lambda with ID {lambdalst} is used in the calculation.\n")
        ftowrit.write(f"     To select other lambdas use the option \'option\' while calling analyzehkl(fpath, fhkl, option=0) routine.\n\n")
        ftowrit.write(f"---> To solve the structure i assume the lambda at index {dflambda[option]}\n")
        ftowrit.write(f"---> Starting to analyze possible h, k, l, hk, hl, kl, and hkl combinations\n")
        ftowrit.write(f'---> The following details found from given hkl file.\n\n')
        ftowrit.write(f"     Total h projection : {hdf.shape[0]}\n")
        ftowrit.write(f"     Total k projection : {kdf.shape[0]}\n")
        ftowrit.write(f"     Total l projection : {ldf.shape[0]}\n")
        ftowrit.write(f"     Total hk projection : {hkdf.shape[0]}\n")
        ftowrit.write(f"     Total hl projection : {hldf.shape[0]}\n")
        ftowrit.write(f"     Total kl projection : {kldf.shape[0]}\n")
        ftowrit.write(f"     Total hkl projection : {hkldf.shape[0]}\n\n")
        ftowrit.write(f"     Total available projections : {hkldf.shape[0]+hkdf.shape[0]+hldf.shape[0]+kldf.shape[0]+hdf.shape[0]+kdf.shape[0]+ldf.shape[0]}\n\n")
                
        if hdf.shape[0] == 0:
            ftowrit.write(f'---> WARNING: No h projection is found. I will not solve for x coordinates of atoms.\n')
            print(f'---> WARNING: No h projection is found. I will not solve for x coordinates of atoms.\n')
        if kdf.shape[0] == 0:
            ftowrit.write(f'---> WARNING: No k projection is found. I will not solve for y coordinates of atoms.\n')
            print(f'---> WARNING: No k projection is found. I will not solve for y coordinates of atoms.\n')
        if ldf.shape[0] == 0:
            ftowrit.write(f'---> WARNING: No l projection is found. I will not solve for z coordinates of atoms.\n')
            print(f'---> WARNING: No l projection is found. I will not solve for z coordinates of atoms.\n')
        if (hdf.shape[0] == 0) & (kdf.shape[0] == 0) & (ldf.shape[0] == 0):
            ftowrit.write(f'---> WARNING: No h or k or l projections are found. I can not solve the structure. I am quitting now ....\n')
            print(f'---> WARNING: No h or k or l projections are found. I can not solve the structure. I am quitting now ....\n')
        
        if hkdf.shape[0] == 0:
            ftowrit.write(f'---> WARNING: No h+k projection is found. I will not solve for x-y coordinates of atoms.\n')
            print(f'---> WARNING: No h+k projection is found. I will not solve for x-y coordinates of atoms.\n')
        if hldf.shape[0] == 0:
            ftowrit.write(f'---> WARNING: No h+l projection is found. I will not solve for x-z coordinates of atoms.\n')
            print(f'---> WARNING: No h+l projection is found. I will not solve for x-z coordinates of atoms.\n')
        if kldf.shape[0] == 0:
            ftowrit.write(f'---> WARNING: No k+l projection is found. I will not solve for y-z coordinates of atoms.\n')
            print(f'---> WARNING: No k+l projection is found. I will not solve for y-z coordinates of atoms.\n')
        
        ftowrit.write(f'---> The following details found from given hkl file.\n')
    
    hRO=hdf['h'].to_numpy()
    kRO=kdf['k'].to_numpy()
    lRO=ldf['l'].to_numpy()
    
    hkRO=hkdf[['h', 'k']].to_numpy()
    hlRO=hldf[['h', 'l']].to_numpy()
    klRO=kldf[['k', 'l']].to_numpy()
    
    hklRO=hkldf[['h', 'k', 'l']].to_numpy()
        
    hsqrtI=hdf['|F|'].to_numpy()
    ksqrtI=kdf['|F|'].to_numpy()
    lsqrtI=ldf['|F|'].to_numpy()
    
    hksqrtI=hkdf['|F|'].to_numpy()
    hlsqrtI=hldf['|F|'].to_numpy()
    klsqrtI=kldf['|F|'].to_numpy()
    
    hklsqrtI=hkdf['|F|'].to_numpy()
    
    # ---> phases
    hphase=np.arctan(hdf['F(imag)']/hdf['F(real)']).to_numpy()
    kphase=np.arctan(kdf['F(imag)']/kdf['F(real)']).to_numpy()
    lphase=np.arctan(ldf['F(imag)']/ldf['F(real)']).to_numpy()
    
    hkphase=np.arctan(hkdf['F(imag)']/hkdf['F(real)']).to_numpy()
    hlphase=np.arctan(hldf['F(imag)']/hldf['F(real)']).to_numpy()
    klphase=np.arctan(kldf['F(imag)']/kldf['F(real)']).to_numpy()
    
    hklphase=np.arctan(hkldf['F(imag)']/hkldf['F(real)']).to_numpy()
    
    # ---> get lambda
       
    h_lambda = 2*hdf[hdf.columns.values[3]]*1E-10*np.sin(np.radians(hdf[hdf.columns.values[7]]/2)) # 2*hdf['d(Ã…)']*1E-10*np.sin(hdf['2Î¸']/2) / hdf['h']
    k_lambda = 2*kdf[kdf.columns.values[3]]*1E-10*np.sin(np.radians(kdf[kdf.columns.values[7]]/2)) # 2*kdf['d(Ã…)']*1E-10*np.sin(kdf['2Î¸']/2) / kdf['k']
    l_lambda = 2*ldf[ldf.columns.values[3]]*1E-10*np.sin(np.radians(ldf[ldf.columns.values[7]]/2)) # 2*ldf['d(Ã…)']*1E-10*np.sin(ldf['2Î¸']/2) / ldf['l'] # previously 'd(Ã\x85)'
    
    hk_lambda = 2*hkdf['d(Ã…)']*1E-10*np.sin(np.radians(hkdf[hkdf.columns.values[7]]/2))
    hl_lambda = 2*hldf['d(Ã…)']*1E-10*np.sin(np.radians(hldf[hldf.columns.values[7]]/2))
    kl_lambda = 2*kldf['d(Ã…)']*1E-10*np.sin(np.radians(kldf[kldf.columns.values[7]]/2))
    
    hkl_lambda = 2*hkldf['d(Ã…)']*1E-10*np.sin(np.radians(hkldf[hkldf.columns.values[7]]/2))
        
    hkllambda = h_lambda.iloc[0] if np.all(h_lambda) else k_lambda.iloc[0] if np.all(k_lambda) else l_lambda.iloc[0]
    
    energy = (6.62607015E-34 * 2.99792458E8) / (hkllambda * 1.602176634E-19)
    
    hinfo = [hRO, hsqrtI, hphase, hkllambda, energy]
    kinfo = [kRO, ksqrtI, kphase, hkllambda, energy]
    linfo = [lRO, lsqrtI, lphase, hkllambda, energy]
    
    hkinfo = [hkRO, hksqrtI, hkphase, hk_lambda, energy]
    hlinfo = [hlRO, hlsqrtI, hlphase, hl_lambda, energy]
    klinfo = [klRO, klsqrtI, klphase, kl_lambda, energy]
    
    hklinfo = [hklRO, hklsqrtI, hklphase, hkl_lambda, energy]
    
    return hinfo, kinfo, linfo, hkinfo, hlinfo, klinfo, hklinfo