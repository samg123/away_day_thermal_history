import numpy as np
import pdb



#Mantle model
def mantle_evolution(model, core_model=False):
    
    #Read in variables from model and parameters file
    time     = model.time

    Tm       = model.Tm       #average mantle temperature

    r_upper  = model.r_upper  #upper radius
    r_lower  = model.r_lower  #lower radius
    
    nu_upper = model.nu_upper #viscosity in upper mantle
    nu_lower = model.nu_lower #viscosity in lower mantle
    
    k_upper  = model.k_upper  #thermal conductivity in upper mantle
    k_lower  = model.k_lower  #thermal conductivity in lower mantle
    
    Tsurf    = model.Tsurf    #Surface temperature
    Qr0      = model.Qr0      #Initial radiogenic heating
    Trad     = model.Trad     #radiogenic heating decay
    mass     = model.mass     #mantle mass
    alpha    = model.alpha    #volumetric expansion
    g        = model.g        #gravity
    kappa    = model.kappa    #thermal diffusivity
    cp       = model.cp       #specific heat capacity
    Rac      = model.Rac      #critical Rayleigh Number
    
    
    if core_model:
        from thermal_history.core.profiles import adiabat
        Tcen = core_model.Tcen
        Tcmb = adiabat(r_lower,Tcen)
    else:
        Tcmb = model.Tcmb_func(model)

    #Calculate boundary layers
    Ta_upper = Tm*0.7
    Ta_lower = Tm*1.3
    
    D = r_upper - r_lower   #Mantle Thickness
    
    delta_upper = D*((kappa*nu_upper*Rac)/((D**3)*alpha*g*(Ta_upper-Tsurf)))**(1/3)
    
    delta_lower = D*((kappa*nu_lower*Rac)/((D**3)*alpha*g*(Tcmb-Ta_lower)))**(1/3)
    
    #Conductive heat flow through boundaries
    
    Qsurface = 4*np.pi*r_upper**2*k_upper*((Ta_upper-Tsurf)/delta_upper)
    
    Qcmb = 4*np.pi*r_lower**2*k_lower*(((Tcmb-Ta_lower))/delta_lower)
    
    #heat produced via radiogenic production
    Qr = Qr0*np.exp(-time/Trad)
    
    #Calculating energy from mantle secular cooling
    Qs = Qsurface - Qcmb - Qr
    
    #Mantle cooling rate
    dTm_dt = -Qs/(mass*cp)
   
    #Save values to model class
    model.delta_upper = delta_upper
    model.delta_lower = delta_lower
    
    model.Qr = Qr
    model.Qs = Qs
    model.Qcmb = Qcmb
    model.Qsurface = Qsurface

    model.dT_dt = dTm_dt
    model.Tcmb  = Tcmb
    
    return model

################################################################################################################################   




#Core model
################################################################################################################################
def core_evolution(model, dt=1e6, mantle_model=False, sl_model=False, debugging=False):
    
    
    from thermal_history.core.profiles import melt_T_grad_r, mass, profiles, mass_correct, adiabat, adiabat_grad
    from thermal_history.core.energy import ic_growth,latent_heat, gravitational, secular_cool, cond_entropy, ohm_entropy
    from thermal_history.core.chemistry import LE_frac_dep, mole_frac2mass_conc, mass_conc2mole_frac
    
    
    ###############################################################################
    #Call relevant functions for the core evolution
    ###############################################################################
    
    #Read in variables from model    
    Tcen    = model.Tcen
    ri      = model.ri
    mf_l    = model.mf_l
    conc_l  = mole_frac2mass_conc(mf_l)
    i_rho   = model.i_rho
    o_rho   = model.o_rho
    r_upper = model.r_upper
    
    time = model.time
    model.dt = dt*model.ys
    
    #Evaluate mass and save initial mass
    M_ic, M_oc, M = mass(i_rho,o_rho,ri,r_upper)
    if model.it == 0:
        model.M0 = M

    
    #Evaluate radial profiles
    r, idx, rho, g, psi, P, Ta, Ta_grad, k, Tm_fe = profiles(r_upper,ri,i_rho,o_rho,Tcen)   
        
    
    #Calculate fractionation of light elements and depression of melting curve
    dTm, conc_s = LE_frac_dep(P,P[idx],conc_l)
    Tm = Tm_fe + dTm

    #############################
    #Set conductive het flow through upper boundary      
    if not mantle_model and not sl_model:
        Q_upper = model.Qcmb_func(time) 
        
    if mantle_model:
        Q_upper = mantle_model.Qcmb
        
    if sl_model and r_upper < sl_model.r_upper:
        Q_upper = -4*np.pi*r_upper**2 * k[-1] * adiabat_grad(r_upper,Tcen)       
    else:
        Q_upper = model.Qcmb_func(time) 
               
    #############################   
    
    
    #############################
    #Find new intersection between adiabat and melting curve
    ri_new = ic_growth(r,Tcen,Tm)
    
    #dri_dt = (ri_new-ri)/dt #time step defined at the end

    if ri_new == 0:
        flag = False
        conc_s = np.zeros(conc_s.size)
    else:
        flag = True
    #############################   
    
    
    #############################
    #Calculate Cc and Cr
    if flag:
        
        dTm_dr = melt_T_grad_r(P[idx],rho[idx],g[idx])
        dTa_dr = Ta_grad[idx]
        
        Cr = (1/(dTm_dr-dTa_dr))*Ta[idx]/Tcen
        Cc = 4*np.pi*ri**2*rho[idx]*(conc_l-conc_s)/M_oc
        
    else:
        Cc = np.zeros(len(conc_l))
        Cr = 0
    #############################

    
    #############################       
    #Calculate normalised energies/entropies and cooling
    Ql_tilda, El_tilda = latent_heat(ri,rho,Ta,idx,Cr)
    
    Qg_tilda, Eg_tilda = gravitational(r,Ta,rho,psi,idx,Cr,Cc,M_oc)

    Qs_tilda, Es_tilda = secular_cool(r,rho,Ta,M)
    #############################
    
#    if not mantle_model and dt < 0 and model.keep_Ej_positive and ri == 0:
#        
#        Ej = model.Ej_last
#        
#        dT_dt = (Ej + model.Ek) / (Es_tilda + El_tilda + Eg_tilda)
#        
#        Q_upper = dT_dt*(Qs_tilda + Ql_tilda + Qg_tilda)
#        
#        
#    else:  
#
#        dT_dt = Q_upper/(Qs_tilda+Ql_tilda+Qg_tilda)
    
    dT_dt = Q_upper/(Qs_tilda+Ql_tilda+Qg_tilda)
    
    #Adjust density profile due to light element release
    dc = conc_l - mole_frac2mass_conc(model.parameters['mf_l'])

    
    o_rho[0] = model.parameters['o_rho'][0]*(1 + np.dot(dc,model.parameters['alpha_c']))
    o_rho = mass_correct(i_rho,o_rho,ri,model.M0,model.parameters['r_upper'])

    
    if debugging:
        return dict([[key,value] for key,value in locals().items() if type(value) in (int,float,str,np.ndarray) and not key.startswith('_')])
    
    else:
    
        #Calculate entropies and energies and save to model.
        
        model.Ek = cond_entropy(Ta,Ta_grad,k,r) #Entropy of adiabat conduction
    
        
        model.Q_upper = Q_upper
        
        
        model.Qs = Qs_tilda*dT_dt
        model.Es = Es_tilda*dT_dt
        
        model.Ql = Ql_tilda*dT_dt
        model.El = El_tilda*dT_dt
        
        model.Qg = Qg_tilda*dT_dt
        model.Eg = Eg_tilda*dT_dt
        
        model.Ej = ohm_entropy(model.Es,
                                    model.Eg,
                                    model.El,
                                    model.Ek)
        
        model.dT_dt  = dT_dt
        model.dri_dt = Cr*dT_dt
        model.ri     = ri_new
        model.dc_dt  = Cc*Cr*dT_dt
        
        model.o_rho = o_rho
        model.conc_l = conc_l
        model.conc_s = conc_s
        
        #Rate of change of temperature is calculated below as it depends on time step used by stable layer calculation
    
        return model

################################################################################################################################


#Stable layer model
################################################################################################################################
def sl_evolution(model, dt=1e6, core_model=False, mantle_model=False, debugging=False):
    
    from thermal_history.core.profiles import conductivity, density, adiabat, adiabat_grad
    from thermal_history.core.numerical import integrate
    from thermal_history.stable_layer.functions import evolve_thermal_layer, append_layer, diffusion, initialise_layer
    
    from scipy.optimize import bisect
    from scipy.interpolate import interp1d
    from scipy.special import erfcinv

    
    #Read in values from model   
    time = model.time
    it   = model.it
    ys = model.ys
    model.dt = dt*ys 
    t_tot = model.dt
    
    r_cmb = model.r_upper
    k_cmb = conductivity(r_cmb)

    diffusivity_c = model.D_c[0]
    diffusivity_T = model.kappa
    alpha_T       = model.alpha_T
    alpha_c       = model.alpha_c[0]
    n_points      = model.n_points
    tolerance     = model.depth_tolerance
    
    if mantle_model:
        Qcmb = mantle_model.Qcmb
    else:
        Qcmb = model.Qcmb_func(time)
        
    if core_model:
        Tcen, dTa_dt = core_model.Tcen, core_model.dT_dt
    else:
        Tcen, dTa_dt = model.Ta_func(model)
        
        
      
    
    #Initialse values on iteration 0:
    if it == 0:
        
        model.r = np.ones(n_points)*r_cmb
        model.T = adiabat(model.r,Tcen)
        
        
        
    
    #for testing purposes
    #Qcmb = 0.99 * -adiabat_grad(r_cmb,Tcen)*k_cmb*4*np.pi*r_cmb**2
    
    
    #Calculate one of 2 cases, thermal/chemical:

    if model.thermal_stratification:
        
        time_gone = 0
        
        test=0
        
        while time_gone < t_tot:
            test = test+1
        
            T = model.T
            r = model.r
            
            s = r[0]
            
            #TESTING
#            if time < 3.5e9*ys:
#                Qcmb = 1.00001-(k_cmb*4*np.pi*r_cmb**2)*adiabat_grad(r_cmb,Tcen)
#            else:
#                Qcmb = 0.999*-(k_cmb*4*np.pi*r_cmb**2)*adiabat_grad(r_cmb,Tcen)
                

            
            #Calculate CMB temp gradient
            dT_dr_cmb = -Qcmb/(k_cmb*4*np.pi*r_cmb**2) 
            
            
                       
            adiabaticity = float(dT_dr_cmb/adiabat_grad(r_cmb,Tcen))
            
            if adiabaticity >= 1 and s > tolerance:
                
                r_new = np.ones(n_points)*r_cmb
                T_new = adiabat(r_new,Tcen)
                
                s_new = r_new[0]
                ds_dt = (s_new-s)/t_tot
                
                time_gone = t_tot
                
                
            else:
                
                if s > tolerance:
                    r, T = initialise_layer(tolerance,Tcen)
                    s = r[0]
                    
                    
                #Variable dt
                dt = (1/diffusivity_T)*((r_cmb-s)/(2*erfcinv(1e-10)))**2
                
                if time_gone + dt > t_tot:
                    dt = t_tot - time_gone
                elif dt < 1000*ys:
                    dt= 1000*ys
            
                lb = adiabat_grad(r[0],Tcen)
                ub = dT_dr_cmb
                
                #Diffuse the solution and cool adiabat
                    
#                if model.it == 881:
#                    print(test)
                
                T_new = diffusion(T,r,dt,diffusivity_T,1,1,lb,ub,coord='sph')
                Tcen = Tcen + dTa_dt*dt
                
        
                assert T_new[0] < adiabat(0,Tcen), "Entire core is stratified"
    
                if T_new[0] < adiabat(r_cmb,Tcen):
                    
                    r_new = np.ones(n_points)*r_cmb
                    T_new = adiabat(r_new,Tcen)
                
                    s_new = r_new[0]
                    ds_dt = (s_new-s)/dt
                    
                else:
    
                    #Calculate movement of boundary
                    def f(guess,T,Tcen):
                        return T-adiabat(guess,Tcen)
                
                    try:
                        s_new = bisect(f,0,3480e3,args=(T_new[0],Tcen),maxiter=200)
                    except:
                        pdb.set_trace()
                        
                    ds_dt = (s_new-s)/dt
                    if dt == 0:
                        pdb.set_trace()
                        
                    T_rel = T_new - adiabat(r,Tcen)
                    #Append solution
                    r, T_rel = append_layer(r,s_new,T_rel)
                    T = T_rel + adiabat(r,Tcen)
                    
                    #Interpolate solution
                    r_new = np.linspace(s_new,r_cmb,n_points)
                    T_new = interp1d(r,T,kind='linear')(r_new)
            
                    time_gone += dt

                    model.r     = r_new
                    model.T     = T_new
            
        #pdb.set_trace()
        

        
#            
#            
#            #Check if layer is thin and superadiabatic (destroy layer):
#            if r_s >= tolerance and adiabaticity >= adiabaticity_limit:
#
#                
#                r_new = np.ones(n_points)*r_cmb
#                T_new = np.ones(n_points)*adiabat(r_cmb,Tcen)
#                Qs = 0
#                ds_dt = 0
#                
#                
#            #layer is thin and sub-adiabatic (initialise values):
#            elif r_s > tolerance and adiabaticity < adiabaticity_limit:
#
#                r_new = np.linspace(tolerance,r_cmb,n_points)
#                T_new = adiabat(r_new,Tcen)
#                Qs = 0
#                ds_dt = 0
#
#            #layer is thick and can be evolved
#            else:
#
#                #Decide on time step
#                dt = 100*model.ys*(model.s_layer.layer_thickness/30)
#                
#                
#                r = model.s_layer.r
#                T = model.s_layer.T
#                
#                T_new = evolve_thermal_layer(r, T, Tcen, dT_dr_cmb, dt)
#                
#                dT_dt = (T_new-T)/dt
#                rho = density(r,o_rho)
#                
#                Qs = 4*np.pi*integrate(r,dT_dt*rho*cp*r**2)
#                
#                Tcen = Tcen + dTa_dt*dt
#                
#                r_new, T_new = move_thermal_boundary(r,T_new,Tcen,n_points)
#                
#                ds_dt = (r_new[0]-r[0])/dt
#
#                if r_new[0] > tolerance:
#                    model.adiabaticity_limit = adiabaticity
#            
#
#            #Save those unique to thermal stratification
#            model.s_layer.Qs = Qs
#            model.s_layer.T = T_new
                         
        
    if model.compositional_stratification:
        
        if it == 0:
            r = np.linspace(tolerance,r_cmb,n_points)
            c = np.ones(r.size)*conc_l
        else:
            r = model.r
            c = model.c
        
        #Barodiffusion
#            dmu_dc = prm_sl.dmu_dc
#            dg_dr = 10/3480e3
#            g = 10
#            a = diffusivity_c*alpha_c/dmu_dc  #Gubbins/Davies table 1
#        
#            #baro_cst = -( 2*g*a/r +a*dg_dr 
        #ub = alpha_c*g/dmu_dc
        baro_cst=0
        
        ###################
        #Chemical Diffusion
        dT_dr = Qcmb/-(k_cmb*4*np.pi*r_s**2)
        
        super_adiabatic_grad = dT_dr - adiabat_grad(r_s,Tcen)
        #super_adiabatic_grad = -1/1000
        
        bc_lower = 1  #lower boundary condition type
        bc_upper = 0  #upper boundary condition type

        ub = model.ub_func(model)  #upper boundary condition
        lb = -(alpha_T/alpha_c)*super_adiabatic_grad  #lower boundary condition
        
        if lb < 0:
            print('Can\'t have this boundary condition!!')
            pdb.set_trace()
    
    
        c_new = diffusion(c,r,dt,diffusivity_c,bc_lower,bc_upper,lb,ub,cst=baro_cst,coord='sph')
        
        dc_dt_sl = (c_new[0]-c[0])/dt
    
        #Calculate movement of boundary
        ds_dt = (dc_dt-dc_dt_sl)/lb


        ds = ds_dt*dt
        s_new = r_s + ds
        
        if ds < 0:
            #Append solution
            r_append = np.linspace(s_new,r[0],10)[:-1]
            c_append = np.ones(9)*conc_l
            
            r_appended = np.append(r_append,r)
            c_appended = np.append(c_append,c_new)
            
            r_new = np.linspace(s_new,r[-1],n_points)
            c_new = np.interp(r_new,r_appended,c_appended)
            
        else:
            
            r_new = np.linspace(s_new,r[-1],n_points)
            c_new = np.interp(r_new,r,c_new)
        
        
        #Save those unique to compositional stratification
        model.c = c_new
    

    if debugging:
        return dict([[key,value] for key,value in locals().items() if type(value) in (int,float,str,np.ndarray) and not key.startswith('_')])
    
    else:
        
#        pdb.set_trace()
        #Save to model
        model.Tcen = Tcen
        model.dTa_dt = dTa_dt
        model.Qcmb = Qcmb
        model.r_lower = s_new
        model.ds_dt = ds_dt
        model.r     = r_new
        model.T     = T_new
    
        model.adiabaticity = adiabaticity
        
        return model

#################################################################################################

def update(mantle_model=False,sl_model=False,core_model=False):
    
    if sl_model:
        
        sl_model.it += 1
        sl_model.time += sl_model.dt
        
        sl_model.r_lower = sl_model.r[0]
        sl_model.layer_thickness = sl_model.r_upper-sl_model.r_lower
        
        if mantle_model:
            mantle_model.dt = sl_model.dt
        if core_model:
            core_model.dt = sl_model.dt
            core_model.r_upper = sl_model.r_lower
        
    if mantle_model:
        
        mantle_model.it += 1
        mantle_model.time += mantle_model.dt
    
        mantle_model.Tm += mantle_model.dT_dt*mantle_model.dt
        
    if core_model:
        
        from thermal_history.core.chemistry import mass_conc2mole_frac
    
        core_model.it += 1
        core_model.time += core_model.dt    
        
        core_model.Tcen   += core_model.dt*core_model.dT_dt 
        core_model.ri     += core_model.dt*core_model.dri_dt
        core_model.conc_l += core_model.dt*core_model.dc_dt
        core_model.mf_l   = mass_conc2mole_frac(core_model.conc_l)
    


def append_solution(r,s_new,T_rel):
    
    s = r[0]
    r_append = np.linspace(s_new,s,10)[:-1]
    
    ###################
    #linear fit
    T_append = np.linspace(0,T_rel[0],10)[:-1]
    ###################
    #Cubic fit
    #x1, x2 = float(s_new), float(s)
    #f_x1, f_x2 = float(0), float(T_rel[0])
    #f_p_x1, f_p_x2 = float(0),float(0)           
    #m = num.cubic_fit(x1, x2, f_x1, f_x2, f_p_x1, f_p_x2)

    #T_append = m[0]*r_append**3 + m[1]*r_append**2 + m[2]*r_append + m[3]
    ###################

    r_new = np.insert(r,0,r_append)
    T_new = np.insert(T_rel,0,T_append)
    
    return r_new, T_new





    