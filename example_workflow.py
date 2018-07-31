import numpy as np
import matplotlib.pyplot as plt
from thermal_history.model_classes import core_class
from thermal_history.evolution import update

#Initialise core class
core = core_class()

#Iterate for 4.5Ga backwards
for i in range(4500):
    
    core.evolve(dt=-1e6)
    update(core_model=core)
    
    if i in range(0,4500,10):            #Save every 10th data point
        core.save(print_progress=False)  #Sometimes printing the progress doesn't format correctly in jupyter notebooks if it runs too quickly
    
core.save(print_progress=False) #Save the last one
    
#Read data
core_data = core.read_data()


#Plotting the data
time = core_data['time']/(1e6*core.ys)
Tcen = core_data['Tcen']
ri = core_data['ri']/1000
Ej = core_data['Ej']/1e6
O = core_data['conc_l'][:,0]
Si = core_data['conc_l'][:,2]

plt.figure(figsize=(13,8.5))
plt.subplot(2,2,1)

plt.plot(time,Tcen)
plt.xlabel('Time / Ma')
plt.ylabel('˚K')
plt.title('Temperature at the center of the core')

plt.subplot(2,2,2)

plt.plot(time,ri)
plt.xlabel('Time / Ma')
plt.ylabel('km')
plt.title('Inner core radius')

plt.subplot(2,2,3)

plt.plot(time,Ej)
plt.xlabel('Time / Ma')
plt.ylabel('MW/K')
plt.title('Entropy for ohmic dissipation')

plt.subplot(2,2,4)

plt.plot(time,O)
plt.plot(time,Si)
plt.legend(['Oxygen','Silicon'],loc=0)
plt.xlabel('Time, / Ma')
plt.ylabel('wt %')
plt.title('Light element mass fraction')

plt.tight_layout()
plt.show()