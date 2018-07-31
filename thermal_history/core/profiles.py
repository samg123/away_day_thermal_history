#Radial Profiles Functions
import core_parameters as prm

import numpy as np
import pdb


###############################################################################
def eval_poly(x,poly):
    '''
    Assumes poly is iteratable, in increasing order
    '''
    if type(x) == float or type(x) == int:
        y = np.zeros(1)
    else:
        y = np.zeros(x.shape)
    for i in range(poly.size):
        y = y + poly[i]*(x**i)
    return y
###############################################################################
def conductivity(r):
    '''
    r = radius (singluar value or array)
    poly = best-fit conductivity polynomials
    
    returns: conductivity at r
    '''
    poly= prm.k
    #f = poly[3]*P**3 + poly[2]*P**2 + poly[1]*P + poly[0]
    k = eval_poly(r,poly)     
       
    return k
###############################################################################
#Calculate the melting temperature of Iron and it's gradient at a given pressure using best fit polynomials
###############################################################################
def melt_T_fe(P):
    '''
    P = Pressure (singluar value or array)
    
    returns: melting temperature of iron at pressure P
    '''
    Tm = eval_poly(P,prm.melt_T)
    #Tm = prm.melt_T[0] + prm.melt_T[1]*P + prm.melt_T[2]*P**2 + prm.melt_T[3]*P**3
    #dTm_dP = prm.melt_T[1] + 2*prm.melt_T[2]*P + 3*prm.melt_T[3]*P**2    

    return Tm
###############################################################################
#Calculate the melting temperature radial gradient at a given pressure
###############################################################################
def melt_T_grad_r(P,rho,g):
    '''
    P = Pressure (singluar value or array)
    rho = density at pressure P
    g = gravity at pressure P
    
    returns: radial melting temperature gradient of iron at pressure P
    '''
    
    poly = np.arange(1,prm.melt_T.size)*prm.melt_T[1:]
    dTm_dP = eval_poly(P,poly)
    dTm_dr = dTm_dP*-rho*g   #Assume hydrostic pressure gradient    

    return dTm_dr
###############################################################################

def density(r,poly):
    '''
    r = radius (singluar value or array)
    poly = best-fit density polynomials (cubic)
    
    returns: density at r
    '''
    #f = poly[3]*r**3 + poly[2]*r**2 + poly[1]*r + poly[0]
    f = eval_poly(r,poly)    
        
    return f
    
###############################################################################

def gravity(r,poly):
    '''
    r = radius (singluar value or array)
    poly = density polynomials
    
    returns: gravity at r
    '''
    poly = np.concatenate(([0],poly))/np.arange(2,poly.size+3)
    
    f = (4*np.pi*prm.G)*eval_poly(r,poly)  
       
    return f
    
###############################################################################
def grav_potential(r,poly):
    '''
    r = radius (singluar value or array)
    poly = best-fit density polynomials (cubic)
    
    returns: gravitational potential at r, relative to r=0
    '''
    
    f = (4*np.pi*prm.G)*((poly[3]*r**5)/30  +  (poly[2]*r**4)/20  +  (poly[1]*r**3)/12  +  poly[0]*r**2/6)
            
    return f
    
###############################################################################
def pressure(r,poly,p_up,r_up):
    '''
    r = radius (singluar value or array)
    poly = best-fit density polynomials (cubic)
    p_up, r_up = pressure and radius at upper boundary
    
    returns: pressure at r, given p_up at r_up.
    '''
    
    r = np.array(r,dtype='float64')
    
    a8 = poly[3]**2/48
    a7 = poly[3]*poly[2]*11/210
    a6 = poly[3]*poly[1]*5/72   +   poly[2]**2/30
    a5 = poly[3]*poly[0]/10     +   poly[2]*poly[1]*9/100
    a4 = poly[2]*poly[0]*2/15   +   poly[1]**2/16
    a3 = poly[1]*poly[0]*7/36
    a2 = poly[0]**2/6  
    
    P_upper = -4*np.pi*prm.G*(a2*r_up**2 + a3*r_up**3 + a4*r_up**4 + a5*r_up**5 + a6*r_up**6 + a7*r_up**7 + a8*r_up**8)
    P_lower = -4*np.pi*prm.G*(a2*r**2   +    a3*r**3    +   a4*r**4   +   a5*r**5   +   a6*r**6   +   a7*r**7   + a8*r**8)
    
    P = p_up - (P_upper - P_lower)

    #Small contribution arising from different density structures of inner/outer core
    #is much smaller than 10^5 Pa and so is ignored.

    return P   

###############################################################################
def pressure_grad(r,poly):
    '''
    r = radius (singluar value or array)
    poly = best-fit density polynomials (cubic)
    
    returns: hydrostatic pressure gradient at r
    '''
    
    a8 = poly[3]**2/48
    a7 = poly[3]*poly[2]*11/210
    a6 = poly[3]*poly[1]*5/72   +   poly[2]**2/30
    a5 = poly[3]*poly[0]/10     +   poly[2]*poly[1]*9/100
    a4 = poly[2]*poly[0]*2/15   +   poly[1]**2/16
    a3 = poly[1]*poly[0]*7/36
    a2 = poly[0]**2/6  
    
    dP_dr = -4*np.pi*prm.G*(2*a2*r   +    3*a3*r**2    +   4*a4*r**3   +   5*a5*r**4   +   6*a6*r**5   +   7*a7*r**6   + 8*a8*r**7)

    return dP_dr   

###############################################################################
#def pressure_dense(r,idx,o_rho,i_rho):
#    
#    n = prm.profiles_np
#    
#    if idx == 0:
#        r_dense = np.linspace(0,prm.dense_range,n)
#        P_dense = pressure(r_dense,o_rho,prm.Pcmb,prm.r_upper)
#        idx_dense = 0
#    else:
#        if r[idx] < prm.dense_range:
#        
#            r1 = np.linspace(0,r[idx],n)
#            r2 = np.linspace(r[idx],r[idx]+prm.dense_range,n)  
#        else:
#            r1 = np.linspace(r[idx]-prm.dense_range,r[idx],n)
#            r2 = np.linspace(r[idx],r[idx]+prm.dense_range,n)
#            
#        r_dense = np.concatenate((r1,r2))
#        idx_dense = n
#        
#        P2 = pressure(r2,o_rho,prm.Pcmb,prm.r_upper)
#        P1 = pressure(r1,i_rho,P2[0],r2[0])
#        P_dense = np.concatenate((P1,P2))
#        
#    return r_dense, P_dense, idx_dense
           
###############################################################################
def adiabat(r,Tcen):
    '''
    r = radius (singluar value or array)
    Tcen = Temperature at the center of the core
    
    returns: adiabatic temperature at r.
    
    uses the polynomial for the adiabat.
    '''
    
    #f = Tcen*(1 + prm.adiabat[0]*r + prm.adiabat[1]*r**2 + prm.adiabat[2]*r**3)
    f = Tcen*eval_poly(r,prm.adiabat)
    #f = (-5000/prm.r_upper)*r + Tcen

    return f

###############################################################################
def adiabat_grad(r,Tcen):
    '''
    r = radius (singluar value or array)
    Tcen = Temperature at the center of the core
    
    returns: adiabatic temperature gradient wrt radius at r.
    
    uses the cubic polynomial for the adiabat.
    '''
    
    #f = Tcen*(prm.adiabat[0] + 2*prm.adiabat[1]*r + 3*prm.adiabat[2]*r**2)
    
    poly = np.arange(1,prm.adiabat.size)*prm.adiabat[1:]
    f = Tcen*eval_poly(r,poly)
    
    #f = np.ones(r.size)*(-5000/prm.r_upper)
    
    return f
###############################################################################
def adiabat_gradP(P):
    '''
    r = radius (singluar value or array)
    Tcen = Temperature at the center of the core
    
    returns: adiabatic temperature gradient wrt pressure at P.
    
    uses the cubic polynomial for the adiabat.
    '''
    f = prm.adiabat_P[0] + 2*prm.adiabat_P[1]*P + 3*prm.adiabat_P[2]*P**2
    pdb.set_trace() #check to see if function is used: It probably shouldn't
    return f
###############################################################################
def profiles(s,ri,i_rho,o_rho,Tcen):
    '''
    s = radius at top of convecting interior.
    ri = inner core radius
    i_rho = inner core density cubic polynomials
    o_rho = outer core density cubic polynomials
    Tcen = Temperature at the center of the core
    
    returns: r: radius
           idx: index of base of outer core
           rho: density
             g: gravity
           psi: gravitational potential
             P: pressure
             T: temperature
        dTa_dr: temperature gradient
             k: conductivity
    ''' 
    
    n = prm.profiles_np #Number of radial points for profiles
    
    if ri < 0:
        ri = 0
    
    if ri == 0:
        #Radius
        idx = 0        
        r = np.linspace(0,s,n)
        
        if s < prm.r_upper:
            r = np.append(r,np.linspace(s,prm.r_upper,100))
        
        #Density        
        rho = density(r,o_rho)
        
        #Gravity        
        g = gravity(r,o_rho)
        
        #Gravitational Potential
        psi = grav_potential(r,o_rho)
        psi = psi - grav_potential(s,o_rho)  #Shift reference to psi(cmb)=0
        
        #Pressure
        P = pressure(r,o_rho,prm.Pcmb,prm.r_upper)
        
        #Temperature
        T = adiabat(r,Tcen)
        
        #Temperature Gradient
        dTa_dr = adiabat_grad(r,Tcen)
        
        #Conductivity
        k = conductivity(r)
        
        #Melting Temperature of Iron
        Tm_fe = melt_T_fe(P)
    
    else:
        #Radius
        
        n1 = int(2 + (n-2)*ri/s)
        n2 = n-n1

        r1 = np.linspace(0,ri,n1)
        r2 = np.linspace(ri,s,n2)
        r = np.concatenate((r1,r2))
        
        if s < prm.r_upper:
            r2 = np.append(r2,np.linspace(s,prm.r_upper,100))
        
        #Density
        rho1 = density(r1,i_rho)
        rho2 = density(r2,o_rho)

        rho = np.concatenate((rho1,rho2))
        
        #Gravity
        g1 = gravity(r1,i_rho)   
        g2 = gravity(r2,o_rho) + (g1[-1]-gravity(ri,o_rho))*(ri**2/r2**2)

        g = np.concatenate((g1,g2))   
        
        #Gravitational Potential
  
        psi1 = grav_potential(r1,i_rho)
        psi2 = grav_potential(r2,o_rho) + (psi1[-1]-grav_potential(ri,o_rho))*r2/ri
        
        psi_cmb = grav_potential(prm.r_upper,o_rho) + (psi1[-1]-grav_potential(ri,o_rho))*prm.r_upper/ri

        psi1 = psi1 - psi_cmb #Shift reference to psi(cmb)=0
        psi2 = psi2 - psi_cmb

        psi = np.concatenate((psi1,psi2))   
        
        #Pressure
        P2 = pressure(r2,o_rho,prm.Pcmb,prm.r_upper)
        P1 = pressure(r1,i_rho,P2[0],ri)

        P = np.concatenate((P1,P2))    

        #Temperature
        T = adiabat(r,Tcen)
        
        #Temperature Gradient
        dTa_dr_1 = adiabat_grad(r1,Tcen)
        dTa_dr_2 = adiabat_grad(r2,Tcen)
        dTa_dr = np.concatenate([dTa_dr_1,dTa_dr_2])
        
        #Conductivity
        k = conductivity(r)
        
        #Melting Temperature of Iron
        Tm_fe = melt_T_fe(P)
    
        idx = n1
        

        
    return r[:n], idx, rho[:n], g[:n], psi[:n], P[:n], T[:n], dTa_dr[:n], k[:n], Tm_fe[:n]

################################################################################
def mass_correct(i_rho, o_rho, ri, M, r_upper):
    '''
    i_rho = inner core density cubic polynomials
    ri = inner core radius
    M = total core mass
    
    returns: i_rho:  corrected inner core density cubic polynomials
    '''
#    if ri > 0:
#        #Assumes only the first polynomial term for the inner core density is
#        #changed to correct mass of the core
#        dM = mass(i_rho,np.zeros(i_rho.size),ri)[0] - M_ic    
#        d = 3*dM/(4*np.pi*ri**3)   
#        i_rho[0] = i_rho[0] - d
#        
#        if density(ri,i_rho) < density(ri,o_rho):
#            i_rho = o_rho.copy()
            
    ########## NEED TO SORT OUT MASS CONS FOR MOVING LAYER ##############

    M_ic, M_oc = mass(i_rho,o_rho,ri, r_upper)[:2]
    
    M_oc_target = M - M_ic

    dM = M_oc_target - M_oc
    
    drho = 3*dM/(4*np.pi*(prm.r_upper**3-ri**3))
    
    o_rho[0] = o_rho[0] + drho
    
        
    return o_rho
###############################################################################    
def mass(i_rho,o_rho,ri,r_upper):
    '''
    i_rho = inner core density cubic polynomials
    o_rho = outer core density cubic polynomials
    ri = inner core radius
    
    returns M_ic: inner core mass
            M_oc: outer core mass
               M: total core mass
    '''

    M_ic = 4*np.pi*((1/3)*i_rho[0]*ri**3 + (1/4)*i_rho[1]*ri**4 + \
           (1/5)*i_rho[2]*ri**5 + (1/6)*i_rho[3]*ri**6)
           
    M_oc = 4*np.pi*((1/3)*o_rho[0]*(r_upper**3-ri**3) + (1/4)*o_rho[1]*(r_upper**4-ri**4) + \
           (1/5)*o_rho[2]*(r_upper**5-ri**5) + (1/6)*o_rho[3]*(r_upper**6-ri**6))
           
    M = M_ic + M_oc
    
    return M_ic, M_oc, M

###############################################################################
