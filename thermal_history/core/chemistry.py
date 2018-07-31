#Core chemistry functions
import core_parameters as prm
from thermal_history.core.profiles import melt_T_fe, profiles

import numpy as np
from scipy.optimize import bisect
import pdb

###############################################################################
#Convert mole fraction to mass concentration by equations in Labrosse (2014)
###############################################################################
def mole_frac2mass_conc(mf):
    '''
    mf = mole fractions of light elements (array)

    returns: conc = mass fraction of light elements (array, size of mf) 
    '''
    
    denom = np.dot(mf,prm.mm[1:]) + (1 - np.sum(mf))*prm.mm[0]
    
    conc = mf*prm.mm[1:]/denom
        
    return conc
    
###############################################################################
#Convert mass concentration to mole fraction by equations in Labrosse (2014)
###############################################################################
def mass_conc2mole_frac(conc):
    '''
    conc = mass fraction of light elements (array)

    returns: mf = mole fractions of light elements (array, size of conc) 
    '''
    
    
    denom = np.sum(conc/prm.mm[1:]) + (1-np.sum(conc))/prm.mm[0]

    mf = conc/(prm.mm[1:]*denom)
    
    return mf
    
###############################################################################
#Calculate the mole fraction of light element species in the solid 
###############################################################################
def solid_conc(mf_liq, T_m, ds_fe):
    '''
    mf_liq =  mole fraction of light elements in liquid core (array)
       T_m =  melting temperature of iron at ICB (single value)
     ds_fe =  entropy of melting of iron at ICB (single value)
     
     returns: mf_sol: mole fraction of light elements in the solid core (array, size of mf_liq)
    '''
    
    
    def f(guess,dmu_x,lambda_liq_x,lambda_sol_x,mf_liq_x,T_m,ds_fe,kb):
        return dmu_x + mf_liq_x*lambda_liq_x - guess*lambda_sol_x - kb*T_m*np.log(guess/mf_liq_x)*(1+(guess-mf_liq_x)/(ds_fe/prm.kb))
        
#    
#    import matplotlib.pyplot as plt
#    a = []
#    x = np.linspace(0,1)
#    for i in x:
#        a.append(f(i,prm.dmu[2],prm.lambda_liq[2],prm.lambda_sol[2],mf_liq[2],T_m,ds_fe,prm.kb))
#    plt.plot(x,a)
#    plt.show()
    
    
    #Bisection method to find chemcical equilibrium
    mf_sol = np.zeros(mf_liq.size)
    for i in range(mf_liq.size):
        if mf_liq[i] == 0:
            mf_sol[i] = 0
        else:
            lower = 1e-10
            upper = 1
        
            mf_sol[i] = bisect(f,lower,upper,args=(prm.dmu[i],prm.lambda_liq[i],prm.lambda_sol[i],mf_liq[i],T_m,ds_fe,prm.kb),maxiter=200)
  

    return mf_sol 
    
###############################################################################
#Calculate the entropy of melting for Iron
###############################################################################
def entropy_melting(P):
    '''
    P = pressure (single value or array)
    
    returns: ds_fe = entropy of melting for iron at given P
    '''
    
    ds_fe = prm.ent_mel[0] + prm.ent_mel[1]*P + prm.ent_mel[2]*P**2 + prm.ent_mel[3]*P**3 + prm.ent_mel[4]*P**4

    return ds_fe
    
###############################################################################
#Calculate the melting temperature including the depression due to light elements
#Equation 12 from Alfe et al.(2002)
###############################################################################
def melt_pt_dep(mf_liq,mf_sol,P):
    '''
    mf_liq = mole fraction of light elements in the liquid core (array)
    mf_sol = mole fraction of light elements in the solid core (array)
         P = pressure (single value or array)
    
    returns: dTm = deflection of melting curve due to presence of light elements at given P
    '''
        
    dTm = prm.kb*(melt_T_fe(P)/entropy_melting(P))*np.sum(mf_sol-mf_liq)

    return dTm

###############################################################################
#Calculate the melting temperature depression and fractionation of light elements.
###############################################################################
def LE_frac_dep(P,P_icb,conc_l):
    
    #Melting temperature depression, fractionation of light elements
    Tm = melt_T_fe(P_icb)
    ds_fe = entropy_melting(P_icb)
    mf_l = mass_conc2mole_frac(conc_l)
    mf_s = solid_conc(mf_l, Tm, ds_fe)

    dTm = melt_pt_dep(mf_l,mf_s,P_icb)

    conc_s = mole_frac2mass_conc(mf_s)

    return dTm, conc_s

###############################################################################
def calibrate_melting_curve(model):

    Tcen    = model.core.Tcen
    ri      = model.core.ri
    conc_l  = mole_frac2mass_conc(model.core.mf_l)
    r_upper = model.core.r_upper
    i_rho   = model.core.i_rho
    o_rho   = model.core.o_rho

    for i in range(200):

        r, idx, rho, g, psi, P, Ta, Ta_grad, k, Tm_fe = profiles(r_upper,ri,i_rho,o_rho,Tcen)        

        dTm, conc_s = LE_frac_dep(P,P[idx],conc_l)
        Tm = Tm_fe + dTm

        dT = Tm[idx] - Ta[idx]
        
        if np.abs(dT) < 0.01:
            break
        
        prm.melt_T[0] = prm.melt_T[0] - dT
        
        
        


    print('\n'+'Calibrating melting curve intercept. Error = '+str(dT)+' \N{DEGREE SIGN}K \nprm.melt_T[0] =  '+str(prm.melt_T[0])+'\n')
        


















