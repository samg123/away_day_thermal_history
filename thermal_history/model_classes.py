import numpy as np
import netCDF4 as nc4
import pdb
import os
from copy import deepcopy
from types import ModuleType

from thermal_history.evolution import core_evolution, mantle_evolution, sl_evolution
    
    


################################
class core_class:
    
    def __init__(self, backwards=False):
        
        import core_parameters as prm_c
        
        self.it = 0                #iteration counter
        self.ys = 60*60*24*365     #seconds in a year
        
        self.filename = 'core_output.nc'  #default filename
        self.save_it = 0                  #save iteration counter
        
        #Add values from parameters file to this class and into a parameters dictionary for saving
        parameters={}
        items = [item for item in dir(prm_c) if not item.startswith("__") and not type(getattr(prm_c,item))==ModuleType]
        for i in items:
            try:
                setattr(self,i,deepcopy(getattr(prm_c,i)))
            except:
                pdb.set_trace()
            
            #Save it to parameters dictionary
            if not callable(getattr(prm_c,i)):
                parameters[i] = deepcopy(getattr(prm_c,i))            
            
        self.parameters = parameters
        
        
    def evolve(self, *args, debugging=False, **kwargs):
        """
        evolve the core model by 1 time step.
        
        any arguments passed in are passed forward to core_evolution
        
        keyword arguments:
            
            dt: time step in years. Can be negative. Default: 1e6 (1 Ma)
            
            mantle_model: pass in mantle_class if coupled to a mantle model. Default: False
            
            sl_model: pass in sl_class if coupled to a stable layer model. Default: False
            
            debugging: If 'True' returns a dictionary of all variables calculated in core_evolution. Default: False
        """
        
        core_evolution(self, *args, debugging=debugging, **kwargs)

        
    def update(self):
        
        """
        update the core model by one time step.
        """
        
        from thermal_history.core.chemistry import mass_conc2mole_frac
        
        self.it += 1
        self.time += self.dt
        
        self.Tcen   += self.dt*self.dT_dt 
        self.ri     += self.dt*self.dri_dt
        self.conc_l += self.dt*self.dc_dt
        self.mf_l   = mass_conc2mole_frac(self.conc_l)
        
        
    def __call__(self):
        """
        Returns a dictionary of all int/float/numpy array attributes of the class.
        """
        types = (int,float,np.int64,np.float64,np.ndarray)
        all_vars = dict([[name,f] for name,f in self.__dict__.items() if type(f) in types])
        return all_vars

    def save(self,filename=None,print_progress=False):
        """
        Saves the int/float/numpy array attributes of the class to the output file.
        
        keyword arguments:
            
            filename: string for the netcdf filename. Default: 'core_output.nc'
            
            print_progress: if True prints the iteration number to screen. Default: false
        """
        
        if filename:
            self.filename = filename
            
        save_model(self)
        
        if print_progress:
            print('\r'+'iteration: ',self.it,end='')

    def read_data(self,filename=None):
        """
        reads the netcdf file and returns a dictionary of all variables within it.
        
        keyword arguments:
            
            filename: string of file to open. If None, defaults to using core_class.filename. Default: None
        """
        if not filename:
            filename = self.filename
        return read_data(filename)
        
        
    
    
def output_file_setup(model):
    """
    sets up the output file by creating the variables within it. Any int/float/numpy array attribute of the model_class
    is set up. Note: IF THE OUTPUT FILE ALREADY EXISTS, IT WILL FIRST BE DELETED.
    
    Input: model_class to be used.
    """
    
    filename = model.filename

    if os.path.isfile(filename):
        print('\n'+filename+' already exists, deleting '+filename+'\n')
        os.remove(filename)
    
    
    
    model.out_file = nc4.Dataset(filename,'w',format='NETCDF4')

    model.data_group = model.out_file.createGroup('data')
    model.data_group.createDimension('time',None)
    
    var_dict = model()
    model.save_dict = {}
    sizes = []
    for key in var_dict.keys():
        
        if type(var_dict[key]) in (int,float,np.int64,np.float64):
            s = 1
        elif not isinstance(type(var_dict[key]), (str,np.ndarray)):
            s = len(var_dict[key])
        else:
            pdb.set_trace()
            
        if s not in sizes:
            model.data_group.createDimension(str(s),s)
            
        sizes.append(s)
        
        if s == 1:
            model.save_dict[key] = model.data_group.createVariable(key,'f8',('time','1'))
        else:
            model.save_dict[key] = model.data_group.createVariable(key,'f8',('time',str(s)))
        
        
    
    
    types = (int, float, np.int, np.float, np.ndarray, str)
        
    parameter_group = model.out_file.createGroup('parameters')

    for key, value in model.parameters.items():
        if type(value) in types:
            setattr(parameter_group, key, value)
        
        
    return model

#####################################
        
def save_model(model):
    
    if not hasattr(model,'save_dict'):
        output_file_setup(model)
     
    file_types = (int,float,np.int64,np.float64,np.ndarray)
    
    var_dict  = model()

    for key in var_dict.keys():
        for t in file_types:
            if type(var_dict[key]) == t:
                try:
                    model.save_dict[key][model.save_it,:] = var_dict[key]
                    break
                except:
                    pdb.set_trace()
                    
    model.save_it += 1

#####################################
def read_data(filename):
        
        f = nc4.Dataset(filename,'r')
        data_dict = {}
        
        #parameters = {} Need to do parameters as well at some point
        data = f.groups['data']
        
        for key in data.variables.keys():
            
            v = data.variables[key][:]
            
            try:
                if len(v.shape) == 1:
                    data_dict[key] = np.array(np.squeeze(v))
                else:
                    data_dict[key] = np.array(v)        
            except:
                pdb.set_trace()
                
                
        f.close()
        
        return data_dict
    
    
    
    
    
    
    
    
    
    
    
    
    
################################    
class sl_class:
    
    def __init__(self):
        import sl_parameters as prm_sl
        import core_parameters as prm_c
                    
        #Set initial values for 1st iteration   
        self.it = 0 #iteration counter
        self.time = 0
        self.ys = 60*60*24*365
        
        self.filename = 'sl_output.nc'
        
        #Add values from parameters file to this class and into a parameters dictionary for saving
        parameters={}
        
        for mod in [prm_c,prm_sl]:
            items = [item for item in dir(mod) if not item.startswith("__")]
            for i in items:
                setattr(self,i,getattr(mod,i))
                
                #Save it to parameters dictionary
                if not callable(getattr(mod,i)):
                    parameters[i] = getattr(mod,i)
        
        self.parameters = parameters

    def evolve(self, *args, debugging=False, **kwargs):
        if not debugging:
            sl_evolution(self, *args, debugging=debugging, **kwargs)
        else:
            return sl_evolution(self, *args, debugging=debugging, **kwargs)


    def update(self):
        
        self.r_lower = self.r[0]
        self.layer_thickness = self.r_upper-self.r_lower
        
    def __call__(self):
        types = (int,float,np.int64,np.float64,np.ndarray)
        all_vars = dict([[name,f] for name,f in self.__dict__.items() if type(f) in types])
        return all_vars

    def save(self,filename=None,print_progress=False):
        save_model(self,filename=filename)

    def read_data(self,filename=None):
        if not filename:
            filename = self.filename
        return read_data(filename)

####################################################### 
   
class mantle_class:
    
    def __init__(self):
        import mantle_parameters as prm_m
        
        #Set initial values for 1st iteration   
        self.it = 0
    
        #Add values from parameters file to this class and into a parameters dictionary for saving
        parameters={}
        items = [item for item in dir(prm_m) if not item.startswith("__")]
        for i in items:
            setattr(self,i,getattr(prm_m,i))
            
            #Save it to parameters dictionary
            if not callable(getattr(prm_m,i)):
                parameters[i] = getattr(prm_m,i)
        
    
    def update(self):

        self.Tm = self.Tm + self.dT_dt*self.dt
        

    def __call__(self):
        types = (int,float,np.int64,np.float64,np.ndarray)
        all_vars = dict([[name,f] for name,f in self.__dict__.items() if type(f) in types])
        return all_vars
################################ 
    
    
    
    
    
    