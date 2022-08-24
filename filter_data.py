import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, signal
import hera_cal as hc
from scipy import integrate

max_frate =0.60
min_frate=-0.60
filter_max=int(max_frate*100)
print(filter_max)
path='/net/ike/vault-ike/ntsikelelo/Simulated_data_files/UVH5_files/'

# Load some sample visibilities. Note that frf.FRFilter is a subclass of vis_clean.VisClean!
glm_file_deep =path+'Raw_data_with_noise.uvh5'
F_glm_deep = hc.frf.FRFilter(glm_file_deep)
F_glm_deep.read()

# Load some sample visibilities. Note that frf.FRFilter is a subclass of vis_clean.VisClean!
glm_file =path+'Model_data_complete.uvh5'
F_glm = hc.frf.FRFilter(glm_file)
F_glm.read()


# Load some sample visibilities. Note that frf.FRFilter is a subclass of vis_clean.VisClean!
gsm_file =path+'Model_data_incomplete.uvh5'
F_gsm = hc.frf.FRFilter(gsm_file)
F_gsm.read()



F_glm_deep.attach_data()

F_glm.attach_data()

F_gsm.attach_data()

F_glm_deep.fft_data(data=F_glm_deep.data, assign='d_both_fft_glm_deep', ax='both', window='bh', overwrite=True, ifft=True)
F_glm.fft_data(data=F_glm.data, assign='d_both_fft_glm', ax='both', window='bh', overwrite=True, ifft=True)
F_gsm.fft_data(data=F_gsm.data, assign='d_both_fft_gsm', ax='both', window='bh', overwrite=True, ifft=True)

fr_profile_glm_deep = np.ones_like(F_glm_deep.frates)
fr_profile_glm_deep[(min_frate < F_glm_deep.frates) & (F_glm_deep.frates<=max_frate)] = 0
frp_dc_glm_deep = hc.datacontainer.DataContainer({k: fr_profile_glm_deep for k in F_glm_deep.data})

fr_profile_glm = np.ones_like(F_glm.frates)
fr_profile_glm[(min_frate < F_glm.frates) & (F_glm.frates<=max_frate)] = 0
frp_dc_glm = hc.datacontainer.DataContainer({k: fr_profile_glm for k in F_glm.data})


fr_profile_gsm = np.ones_like(F_gsm.frates)
fr_profile_gsm[(min_frate < F_gsm.frates) & (F_gsm.frates<=max_frate)] = 0
frp_dc_gsm = hc.datacontainer.DataContainer({k: fr_profile_gsm for k in F_gsm.data})

filter_fun=fr_profile_glm_deep
I1 = integrate.simpson(filter_fun**2,F_glm_deep.frates)
I2 = integrate.simpson(np.ones(filter_fun.shape)**2,F_glm_deep.frates)
sigma_frac=I1/I2
print(I1,I2,sigma_frac)
np.save('/net/ike/vault-ike/ntsikelelo/Simulated_data_files/sigma_frac_'+str(filter_max)+'.npy',sigma_frac)

# this applies the filter to the data, results in F.filt_data object in real space (i.e. freq & LST`)
F_glm_deep.filter_data(F_glm_deep.data, frp_dc_glm_deep, overwrite=True, axis=0, verbose=False) 

F_glm.filter_data(F_glm.data, frp_dc_glm, overwrite=True, axis=0, verbose=False)  

# this applies the filter to the data, results in F.filt_data object in real space (i.e. freq & LST`)
F_gsm.filter_data(F_gsm.data, frp_dc_gsm, overwrite=True, axis=0, verbose=False)

F_glm.write_data(F_glm.filt_data,path+'Model_complete_filtered_data_'+str(filter_max)+'.uvh5',overwrite=True)
F_gsm.write_data(F_gsm.filt_data,path+'Model_incomplete_filtered_data_'+str(filter_max)+'.uvh5',overwrite=True)
F_glm_deep.write_data(F_glm_deep.filt_data,path+'Raw_filtered_data_deep_'+str(filter_max)+'.uvh5',overwrite=True)
