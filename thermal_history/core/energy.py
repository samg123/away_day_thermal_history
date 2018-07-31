#Core model functions
import core_parameters as prm

from thermal_history.core.profiles import adiabat

import numpy as np
from scipy.integrate import trapz,cumtrapz
import pdb

###############################################################################
#Calculate inner core radius
###############################################################################
def ic_growth(r,Tcen,Tm):
       
    
    if adiabat(r[0],Tcen) > Tm[0]:
        ri = 0
    elif adiabat(r[-1],Tcen) > Tm[-1]:
        
        dT = adiabat(r,Tcen)-Tm
        
        x1 = np.arange(dT.size)[dT>0][0]-1
        x2 = np.arange(dT.size)[dT>0][0]
        
        dx = dT[x1] / (dT[x1]-dT[x2])
        
        ri = r[x1] + dx*(r[x2]-r[x1])
        
        
#        f_Tm =  interpolate.interp1d(r,Tm)     
#        def f(r,Tcen,f_Tm):
#            return adiabat(r,Tcen)-f_Tm(r)
#        
#        ri = bisect(f,r[0],r[-1],args=(Tcen,f_Tm))
    else:
        assert adiabat(r[-1],Tcen) >= Tm[-1], 'The whole core has frozen!'
        
        
    return ri
    
###############################################################################
#Latent Heat
###############################################################################
def latent_heat(ri,rho,Ta,idx,Cr):
    
    
    Ql_tilda = 4*np.pi*ri**2*rho[idx]*prm.L*Cr
    

    El_tilda = Ql_tilda*(Ta[idx]-Ta[-1])/(Ta[idx]*Ta[-1])
    

    return Ql_tilda, El_tilda
    
###############################################################################
#Gravitational Energy
###############################################################################
def gravitational(r,Ta,rho,psi,idx,Cr,Cc,M_oc):
    
    I = 4*np.pi*integrate(r[idx:],rho[idx:]*psi[idx:]*r[idx:]**2) - M_oc*psi[idx]
    
    Qg_tilda = I*np.sum(prm.alpha_c*Cc)*Cr
    
    Eg_tilda = Qg_tilda/Ta[-1]

    return Qg_tilda, Eg_tilda

###############################################################################
#Secular cooling
###############################################################################
def secular_cool(r,rho,Ta,M):
    #Values are normalised to the cooling rate.
    Tcen = Ta[0]
    Is = 4*np.pi*integrate(r,rho*Ta*r**2)
    Qs = -prm.cp*Is/Tcen    
    
    Ts = adiabat(r[-1],Tcen)
    Es = prm.cp*(M-Is/Ts)/Tcen

    
    return Qs, Es
###############################################################################
#Entropy of conduction
###############################################################################
def cond_entropy(Ta,Ta_grad,k,r):

    Ek = 4*np.pi*integrate(r,k*(Ta_grad/Ta)**2*r**2)
    #pdb.set_trace()
    return Ek    
###############################################################################
#Entropy of Ohmic dissipation
###############################################################################
def ohm_entropy(Es,Eg,El,Ek):

    Ej = Es+El+Eg - Ek

    return Ej
###############################################################################
#Numerical integration
###############################################################################
def integrate(x,y,cumulative=0):       
    if cumulative == 0:
        #Trapezium rule A = np.sum(0.5*(y[:-1]+y[1:])*np.diff(x))    
        A = trapz(y,x)
    else:
        A = cumtrapz(y,x,initial=0)
    return A

