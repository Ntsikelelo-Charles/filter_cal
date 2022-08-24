import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, signal
import hera_cal as hc

max_frate =0.6
min_frate=-0.6
# Load some sample visibilities. Note that frf.FRFilter is a subclass of vis_clean.VisClean!
glm_file_deep ='/home/ntsikelelo/noise_data.uvh5'
F_glm_deep = hc.frf.FRFilter(glm_file_deep)
F_glm_deep.read(polarizations=[-5])

F_glm_deep.hd.conjugate_bls()

F_glm_deep.fft_data(data=F_glm_deep.data, assign='d_both_fft_glm_deep', ax='both', window='bh', overwrite=True, ifft=True)

fr_profile_glm_deep = np.ones_like(F_glm_deep.frates)
fr_profile_glm_deep[(min_frate < F_glm_deep.frates) & (F_glm_deep.frates<=max_frate)] = 0
frp_dc_glm_deep = hc.datacontainer.DataContainer({k: fr_profile_glm_deep for k in F_glm_deep.data})

from scipy import integrate

filter_fun=fr_profile_glm_deep
I1 = integrate.simpson(filter_fun**2,F_glm_deep.frates)
I2 = integrate.simpson(np.ones(filter_fun.shape)**2,F_glm_deep.frates)
sigma_frac=I1/I2
print(I1,I2,sigma_frac)
np.save('/home/ntsikelelo/sigma_frac_noise'+str(-max_frate)+'_'+str(-min_frate)+'.npy',sigma_frac)
# this applies the filter to the data, results in F.filt_data object in real space (i.e. freq & LST`)
F_glm_deep.filter_data(F_glm_deep.data, frp_dc_glm_deep, overwrite=True, axis=0, verbose=False) 
F_glm_deep.write_data(F_glm_deep.filt_data,'/home/ntsikelelo/noise_filtered_data_'+str(-max_frate)+'_'+str(-min_frate)+'.uvh5',overwrite=True)