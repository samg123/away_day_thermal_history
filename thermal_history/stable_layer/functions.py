import sl_parameters as prm
import core_parameters as prm_c

from thermal_history.core.profiles import adiabat, adiabat_grad

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import erfc
from scipy.optimize import bisect

import pdb

#Stable layer functions

def initialise_layer(s,Tcen):
    
    r = np.linspace(s, prm.r_upper, prm.n_points)
    T = adiabat(r,Tcen)
    
    return r, T

def evolve_thermal_layer(r, T, Tcen, dT_dr_cmb, dt):
    
    ub = dT_dr_cmb                #upper boundary condition (fixed gradient)
    lb = adiabat(r[0],Tcen)  #lower boundary condition (fixed gradient)

    ub_type = 1  #upper boundary condition type (0=Fixed value  1=Fixed gradient)    
    lb_type = 0  #upper boundary condition type (0=Fixed value  1=Fixed gradient)    
    
    
    #Diffuse the solution and cool adiabat
    T_new = diffusion(T, r, dt, prm_c.kappa, lb_type, ub_type, lb, ub)

    return T_new

def move_thermal_boundary(r,T,Tcen,n_points):

    #Calculate movement of boundary by matching the gradients
    #s_new = move_bound2(Tcen,lb)
    
    #######################
    #Match temperature

#    if T[-1]-adiabat(r[-1],Tcen) < 0:
#        s_new = r[-1]
#        r_new = np.ones(n_points)*r[-1]
#        T_new = np.ones(n_points)*adiabat(r[-1],Tcen)
#    else:
#        def f(guess,T,Tcen):
#            return T-adiabat(guess,Tcen)
#        
#        s_new = bisect(f,0,3480e3,args=(T[0],Tcen),maxiter=200)
    #######################
    
    
    #Match gradient
    #####################
    dT_dr = np.gradient(T,r[1]-r[0],edge_order=2)
    dT_dr_func = interp1d(r,dT_dr,kind='linear')
    
    
    s_new = move_bound2(Tcen,dT_dr[0])
    if 1 == 1:
    #####################

        T_rel = T - adiabat(r,Tcen)
        
        if s_new < r[0]: #Add grid points below layer
            
            #Append solution        
            r, T_rel = expand_domain(r,s_new,T_rel)
        
        else:  #Find point that is also sub-adiabatic
        
            grad_diff = np.gradient(T,r[1]-r[0],edge_order=2) - adiabat_grad(r,Tcen)
            
            grad_diff_func = interp1d(r,grad_diff,kind='linear')
            
            #If super-adiabatic everywhere
            if len(grad_diff[grad_diff>0]) == 0:
                s_new = r[-1]
                
            #If super-adiabatic at the base of the lauer but still sub-adiabatic at the top, reduce to just sub-adiabatic region
            elif grad_diff[0]<0 and grad_diff[-1]>0:
                    
                s_new = bisect(grad_diff_func,r[0],r[-1],maxiter=200)
    
            
            
        
        #Interpolate onto regular grid        
        r_new = np.linspace(s_new,r[-1],n_points)
        T_rel = interp1d(r,T_rel,kind='linear')(r_new)
    
        T_new = T_rel + adiabat(r_new,Tcen)
    
    
    return r_new, T_new


def expand_domain(r,s_new,X_rel):
    
    s = r[0]
    r_append = np.linspace(s_new,s,prm.n_points)[:-1]
    
    
    #linear fit
    X_append = np.linspace(0,X_rel[0],prm.n_points)[:-1]

    
    r_new = np.insert(r,0,r_append)
    X_new = np.insert(X_rel,0,X_append)

    return r_new, X_new


def layer_growth(r, c, T, Tcen, dTa_dt, dc_int, dc_dr, dc_sl):
            
    if prm.compositional_stratification:

        s_new_c = r[0] + (dc_int - dc_sl)/dc_dr
        
    else:
        s_new_c = r[-1]
        
    if prm.thermal_stratification:
    
        ######## Match Gradients  #######
    
        T_rel = T - adiabat(r,Tcen)

        s_new_T = move_bound2(Tcen,r,T)
    
    else:
        T_rel = T-adiabat(r,Tcen)
        s_new_T = r[-1]
        
    s_new = np.min([s_new_c,s_new_T])
    
    if s_new < r[0]: #Interface moves down
    
        #Append solution
        r, T_rel = append_thermal_solution(r,s_new,T_rel)
        T = T_rel + adiabat(r,Tcen)
    
        r, c = append_compositional_solution(r,s_new,c,dc_dr)
    
        #Interpolate solution onto regular grid
        r_new = np.linspace(s_new,prm_c.r_cmb,prm.n_points)
        T_new = interp1d(r,T,kind='linear')(r_new)
        c_new = interp1d(r,c,kind='linear')(r_new)
    
    else: #Interface moves down  
    
        #Interpolate solution onto regular grid
        r_new = np.linspace(s_new,prm_c.r_cmb,prm.n_points)
        T_new = interp1d(r,T,kind='linear')(r_new)
        c_new = interp1d(r,c,kind='linear')(r_new)
        
    return r_new, T_new, c_new
        
        
def expand_layer(ds,sl_model,Tcen):
    
    T = sl_model.T
    r = sl_model.r
    s_new= r[0]+ds
    
    Ta = adiabat(s_new,Tcen)
    
    r_insert = np.linspace(s_new, r[0],10)
    T_insert = np.linspace(T[0], Ta, 10)
    
    n = np.ceil(1 + len(r)*ds/(r[-1]-r[0]))
    
    T = np.insert(T,0,T_insert)
    r = np.insert(r,0,r_insert)
    
    r_new= np.linspace(r[0],r[-1],n)
    T_new = np.interp(r_new,r,T)
    
    sl_model.r = r_new
    sl_model.T = T_new
    
    
    return sl_model
    
    
def shrink_layer(ds,sl_model,Tcen):
    
    T = sl_model.T
    r = sl_model.r
    s_new= r[0]+ds
    
    n = np.ceil(1 + len(r)*ds/(r[-1]-r[0]))
    
    r_new = np.linspace(s_new,r[-1],n)
    T_new = np.interp(r_new,r,T)
    
    sl_model.r = r_new
    sl_model.T = T_new
    
    return sl_model
        
        
###############################################################################
#Move boundary by matching the gradients
###############################################################################
def move_bound2(Tcen,dT_dr):

    #dT_dr = np.gradient(T,r[1]-r[0],edge_order=2)[0]

    #Solve quadratic equation for radius when adiabatic gradient = gradient in T
    a = Tcen*3*prm_c.adiabat[-1]
    b = Tcen*2*prm_c.adiabat[-2]
    c = Tcen*prm_c.adiabat[-3]-dT_dr

    r_root = np.array([(-b+np.sqrt(b**2-4*a*c))/(2*a),(-b-np.sqrt(b**2-4*a*c))/(2*a)])

    pos_roots = r_root[r_root >= 0]
    core_root = pos_roots[pos_roots <= prm_c.r_upper]

    if not core_root:
        print('No roots of the quadratic in the core')
        pdb.set_trace()

    return core_root

####################
    


def append_layer(r,s_new,T_rel):
    
    size = r.size
    s = r[0]
    r_append = np.linspace(s_new,s,size)[:-1]
    
    ###################
    #linear fit
    T_append = np.linspace(0,T_rel[0],size)[:-1]
    ###################
    #Cubic fit
#    x1, x2 = float(s_new), float(s)
#    f_x1, f_x2 = float(0), float(T_rel[0])
#    f_p_x1, f_p_x2 = float(0),float(0)           
#    m = cubic_fit(x1, x2, f_x1, f_x2, f_p_x1, f_p_x2)
#    
#    T_append = m[0]*r_append**3 + m[1]*r_append**2 + m[2]*r_append + m[3]
    ###################
    
    r_new = np.insert(r,0,r_append)
    T_new = np.insert(T_rel,0,T_append)


    return r_new, T_new

def append_compositional_solution(r,s_new,c,dc_dr):
    
    s = r[0]
    r_append = np.linspace(s_new,s,prm.n_points)[:-1]
    
    #Linear fit
    c_append = np.linspace(0,c[0],prm.n_points)[:-1]
#    
#    #Cubic fit
#    x1, x2 = float(s_new), float(s)
#    f_x1, f_x2 = float(0), float(c[0])
#    f_p_x1, f_p_x2 = float(0),float(dc_dr)
#
#    try:           
#        m = cubic_fit(x1, x2, f_x1, f_x2, f_p_x1, f_p_x2)
#    except:
#        pdb.set_trace()
#        
#    c_append = m[0]*r_append**3 + m[1]*r_append**2 + m[2]*r_append + m[3]
    ###################
    
    r_new = np.insert(r,0,r_append)
    c_new = np.insert(c,0,c_append)

    
    return r_new, c_new

###############################################################################
###############################################################################
def GD_diffusion(lb,ub,D,r,c,t):

            c1 = np.ones(r.size)*lb
            
            b = -ub
            
            h = np.sqrt(D*t)
            y = r[-1]-r[0]
            for i in range(len(r)):
                if i>0:
                    dy = r[i] - r[i-1]
                else:
                    dy=0
        
                y = y - dy
        
                zeta = y/(2*h)
        
                t1 = np.exp(-(zeta**2))/np.sqrt(np.pi)
                t2 = zeta*erfc(zeta)
        
                c[i] = c[i] - 2*b*h*(t1-t2)
                
            dc_dr = -b*erfc((r[-1]-r)/(2*h))
        
            return c1, dc_dr
        
#################################################################################

def diffusion(X,r,dt,D,gp1,gp2,lb,ub,cst=0,coord='sph'):
    
    dr = r[1]-r[0]
    m = D*dt/(2*dr**2)
    n=1

    if m >= 0.5:
        n = int(np.ceil(m/0.5))
        dt = dt/n
        m = m/n
    
    if coord == 'sph':
        a = LHS_lin_eq(r,dr,m,gp1,gp2)
        for i in range(n):
            b = RHS_lin_eq(X,r,dr,m,gp1,gp2,lb,ub,cst=cst)
            X = np.linalg.solve(a,b)
            
    elif coord == 'cart':
        a = LHS_lin_eq2(r.size,m,gp1,gp2)
        for i in range(n):
            b = RHS_lin_eq2(X,dr,m,gp1,gp2,lb,ub,cst=cst)    
            X = np.linalg.solve(a,b)
    
    return X
############################################################################### 
###############################################################################
def RHS_lin_eq(X,r,dr,m,gp1,gp2,lb,ub,cst=0):
    #Set up linear algebra equations in the form Ax=B
    
    
    
    size = r.size
    B = np.zeros(size)
    
    if not type(cst)==np.ndarray:
        cst = np.ones(size)*cst
    

    
    
    for i in range(1,size-1):
        B[i] = m*(1 - dr/r[i])*X[i-1] + (1-2*m)*X[i] + m*(1 + dr/r[i])*X[i+1] + cst[i]
    
    if gp1 == 1:
        B[0]=(1-2*m)*X[0] + 2*m*X[1] + 4*m*dr*lb*(dr/r[0] - 1) + cst[0]
    else:
        B[0] = lb + cst[0]
    
    if gp2 == 1:
        B[-1]=(2*m)*X[-2] + (1-2*m)*X[-1] + 4*m*dr*ub*(1+ dr/r[-1]) + cst[-1]
    else:
        B[-1] = ub + cst[-1]

    return B  
###############################################################################
###############################################################################      
def LHS_lin_eq(r,dr,m,gp1,gp2):
    #Set up linear algebra equations in the form Ax=B
    
    size = r.size
    A = np.zeros([size,size])
    
    for i in range(1,size-1):
        A[i,i] = (1+2*m)

        A[i,i-1] = m*(dr/r[i] - 1)

        A[i,i+1] = m*(-1 -dr/r[i])
    
    
    if gp1 == 1:
        A[0,1] = -2*m
        A[0,0] = (1+2*m)
    elif gp1 == 0:
        A[0,0] = 1
      
    if gp2 == 1:
        A[-1,-2] = -2*m
        A[-1,-1] = (1+2*m)
    elif gp2 == 0:
        A[-1,-1] = 1

    return A
###############################################################################
###############################################################################

###############################################################################
# Cartesian coordinates
###############################################################################
def RHS_lin_eq2(T,dx,m,gp1,gp2,lb,ub,cst=0):
    #Set up B in linear algebra equations in the form Ax=B

    size = T.size
    B = np.zeros(size)
    
    if not type(cst)==np.ndarray:
        cst = np.ones(size)*cst
        

    for i in range(1,size-1):
        B[i] = m*T[i-1] + (1-2*m)*T[i] + m*T[i+1] + cst[i]

    if gp1 == 1:
        B[0]=(1-2*m)*T[0] + 2*m*T[1] - 2*m*dx*(lb+lb) + cst[0]
    else:
        B[0] = lb + cst[0]

    if gp2 == 1:
        B[-1]=(2*m)*T[-2] + (1-2*m)*T[-1] + 2*m*dx*(ub+ub) + cst[-1]
    else:
        B[-1] = ub + cst[-1]

    return B
###############################################################################
###############################################################################

def LHS_lin_eq2(size,m,gp1,gp2):
    #Set up A in linear algebra equations in the form Ax=B

    A = np.zeros([size,size])

    for i in range(1,size-1):
        A[i,i] = (1+2*m)

        A[i,i-1] = -m

        A[i,i+1] = -m


    if gp1 == 1:
        A[0,1] = -2*m
        A[0,0] = (1+2*m)
    elif gp1 == 0:
        A[0,0] = 1

    if gp2 == 1:
        A[-1,-2] = -2*m
        A[-1,-1] = (1+2*m)
    elif gp2 == 0:
        A[-1,-1] = 1

    return A
###############################################################################
###############################################################################
    
def cubic_fit(x1,x2,f_x1,f_x2,f_x1_prime,f_x2_prime):
    
    #Fit a cubic equation f(x) = m1*x^3 + m2*x^2 + m3*x + m4
    #Defined conditions are f(x1), f(x2) and f'(x1), f'(x2)
    
    A = np.zeros([4,4])
    
    A[0,:] = [x1**3, x1**2, x1, 1]
    A[1,:] = [x2**3, x2**2, x2, 1]
    A[2,:] = [3*x1**2, 2*x1, 1, 0]
    A[3,:] = [3*x2**2, 2*x2, 1, 0]
    
    B = [f_x1,f_x2,f_x1_prime,f_x2_prime]
    
    A = np.array(A)
    for i in range(4):
        B[i] = float(B[i])
        
    m = np.linalg.solve(A,B)
    
    return m