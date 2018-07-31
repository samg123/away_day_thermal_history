import numpy as np

#All variables in SI units please!

######################################################################
# Initial/Boundary Conditions
######################################################################
ys = 60*60*24*365
time = ys*4.5e9

Tcen = 5583.69                    #Temperature at center of core.
mf_l = np.array([0.13,0,0.08])    #Light element mole fraction: O,S,Si

ri = 1221e3              #Inner core radius


keep_Ej_positive = False  #If intergrating in time backwards, set to True to keep Ej positive prior to IC nucleation

# Q_cmb heat flow as a function of time. If constant, simply have the function
# return a constant value.
def Qcmb_func(time,*args):
      
#    Q0=10e12
#    tau = 2*(60*60*24*365)*1e9
#    
#    Q = 13e12 + Q0*np.exp(-time/tau)
    
    #return constant 
    Q = 15e12
    return Q

######################################################################
######################################################################
    

#Radial Domain
r_upper = 3480e3         #Core-Mantle_boundary radius
profiles_np = 500        #Number of points in radial domain 


#Physical Constants (no need to edit)
ev = 1.602e-19       #Electron volt
kb = 1.3806485e-23   #Boltzmanns constant
G = 6.67e-11         #Gravitational Constant

######################################################################
# Core properties
######################################################################
    
#Physical Properties
kappa   = 1e-5                #Thermal diffusivity
alpha_T = 1e-5                #Thermal expansivity
cp      = 715                 #Specific heat capacity (Davies, 2015)
Pcmb    = 135.8e9             #CMB pressure
L       = 750000              #Latent heat of freezing (Davies, 2015)


k = np.array([1.55827303e+02,    #|Polynomial coefficients for conductivity (Davies et al, 2015, d_rho=0.8)
              -1.11873180e-06,
              -4.03433477e-12,
              -8.30870506e-20])

o_rho = 1000*np.array([12.5815,             #|Best fit polynomials to PREM outer core density 
                       -1.2638/6371e3,      #|(Table 1:Dziewonski + Anderson, 1981)
                       -3.6426/(6371e3)**2, #|
                       -5.5281/(6371e3)**3])#|

i_rho = 1000*np.array([13.0885,             #|Best fit polynomials to PREM inner core density 
                       0,                   #|(Table 1:Dziewonski + Anderson, 1981)
                       -8.8381/(6371e3)**2, #|
                       0])                  #|


adiabat = np.array([1,
                    -5.74304315e-09,        #| Polynomial coefficients for the adiabat as a function of radius normalised to Tcen.
                    -2.03329880e-14,        #| Fitted to definition of adiabat with PREM values, see Davies 2015, d_rho=0.8
                    -2.53543963e-22])       #|                                                                            
                                            
######################################################################
######################################################################




######################################################################
# Core chemistry 
######################################################################

#Must be consistent, i.e. first element in D_c corresonds to 1st element in dmu.
#Must be 1D arrays (even if only 1 value!)

D_c = np.array([10e-9, 5e-9, 5e-9])           #Chemical Diffusion rates: O,S,Si (Table 1:Gubbins and Davies, 2013)

lambda_sol = np.array([0, 5.9, 2.7])*ev       #Corrections to chemical potentials (solid phase): O,S,Si (Table 1:Alfe,2002)

lambda_liq = np.array([3.25, 6.15, 3.6])*ev   #Corrections to chemical potentials (liquid phase): O,S,Si (Table 1:Alfe,2002)

dmu = np.array([-2.6, -0.25, -0.05])*ev       #Change in chemcical potential: O,S,Si (Alfe,2002)

mm = np.array([55.84, 16, 32.06, 28.09])      #Molar Mass: Fe,O,S,Si (g/mol)

alpha_c = np.array([1.1, 0.64, 0.87])         #Volume dependence on composition: O,S,Si (Gubbins 2004)

ent_mel = np.array([1.05, 0, 0, 0, 0])*kb        #Best fit polynomials to entropy of melting (Figure 3 Alfe,2002)

melt_T = np.array([1498.55,                #| #Best fit polynomials to melting temperature (d_rho = 0.8) (supp material: Davies et al,2015)
                   27.3351/1e9,            #|
                   -0.0664736/1e18,        #|
                   7.94628e-5/1e27])       #|

######################################################################
######################################################################

            
      
            