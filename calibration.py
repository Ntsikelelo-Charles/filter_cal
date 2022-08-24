
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import signal
import glob
import os
import copy
import hera_sim as hs
from pyuvdata import UVData, UVBeam, utils as uvutils
import hera_pspec as hp
import hera_cal as hc
import healpy

## need for gain keys
filter_type="notch_filter_0.25"
print(filter_type)






mode_array=np.array(["complete_with_filter","complete_with_filter_baseline_cut","incomplete_with_filter_baseline_cut","incomplete_with_filter"])

# mode_array=np.array(["incomplete_with_filter_baseline_cut"])

# mode_array=np.array(["incomplete_no_filter","complete_no_filter","complete_no_filter_baseline_cut"])

sigma_frac=np.load('/home/ntsikelelo/Simulated_data_files/sigma_frac.npy')                   


# t=np.array([0,1])
t=np.zeros((8),dtype=int)
t[0:8]=np.linspace(0,200,8)
chisq_perfect=np.zeros(shape=(200,308))
chisq_perfect_smooth_gains=np.zeros(shape=(200,308))
gains_all_times=np.zeros(shape=(200,308),dtype=complex)
gains_all_times_smooth=np.zeros(shape=(200,308),dtype=complex)

mode=""
model_file=''
raw_file=''
bool_baseline_cut=False
bool_filter_used=False
for step in range (len(mode_array)):
    mode=mode_array[step]
    print(mode)               
    if mode=="incomplete_with_filter_baseline_cut":                    
        model_file="/home/ntsikelelo/Simulated_data_files/UVH5_files/Model_incomplete_filtered_data.uvh5"
        raw_file="/home/ntsikelelo/Simulated_data_files/UVH5_files/Raw_filtered_data_deep.uvh5"
        bool_baseline_cut=True  
        bool_filter_used=True

    if mode=="complete_with_filter" :                   
        model_file="/home/ntsikelelo/Simulated_data_files/UVH5_files/Model_complete_filtered_data.uvh5"
        raw_file="/home/ntsikelelo/Simulated_data_files/UVH5_files/Raw_filtered_data_deep.uvh5"
        bool_baseline_cut=False 
        bool_filter_used=True

    if mode=="complete_with_filter_baseline_cut" :                   
        model_file="/home/ntsikelelo/Simulated_data_files/UVH5_files/Model_complete_filtered_data.uvh5"
        raw_file="/home/ntsikelelo/Simulated_data_files/UVH5_files/Raw_filtered_data_deep.uvh5"                           
        bool_baseline_cut=True 
        bool_filter_used=True
        

    if mode=="incomplete_no_filter":                   
        model_file="/home/ntsikelelo/Simulated_data_files/UVH5_files/Model_data_incomplete.uvh5"
        raw_file="/home/ntsikelelo/Simulated_data_files/UVH5_files/Raw_data_with_noise.uvh5"
        bool_baseline_cut=False         

    if mode=="complete_no_filter":                   
        model_file="/home/ntsikelelo/Simulated_data_files/UVH5_files/Model_data_complete.uvh5"
        raw_file="/home/ntsikelelo/Simulated_data_files/UVH5_files/Raw_data_with_noise.uvh5"
        bool_baseline_cut=False
        
    if mode=="complete_no_filter_baseline_cut":
        model_file="/home/ntsikelelo/Simulated_data_files/UVH5_files/Model_data_incomplete.uvh5"
        raw_file="/home/ntsikelelo/Simulated_data_files/UVH5_files/Raw_data_with_noise.uvh5"
        bool_baseline_cut=True   
        
        
    if mode=="incomplete_with_filter" :                   
        model_file="/home/ntsikelelo/Simulated_data_files/UVH5_files/Model_incomplete_filtered_data.uvh5"
        raw_file="/home/ntsikelelo/Simulated_data_files/UVH5_files/Raw_filtered_data_deep.uvh5"                           
        bool_baseline_cut=False 
        bool_filter_used=True
   
        
        
    for N_time in range (len(t)-1):
        print(t[N_time],t[N_time+1])
        model = hc.io.HERAData(model_file)
        antpos, ants = model.get_ENU_antpos()
        antpos_d = dict(zip(ants, antpos))
        N=0
        times = np.unique(model.time_array)
        model.read(times=times[t[N_time]:t[N_time+1]])

        model.inflate_by_redundancy()
        model.conjugate_bls()
        model._determine_blt_slicing()
        model._determine_pol_indexing()
        model.x_orientation = 'east'

        bl_lens = {bl: np.linalg.norm(antpos_d[bl[1]]-antpos_d[bl[0]]) for bl in model.get_antpairs()}   
        # downselect to maximum baseline length (to reduce computational load in calibration)
        bls = []
        for bl in list(model.get_antpairs()):
            # max 40 m baseline cut (also cut autos)
            bl_len = np.linalg.norm(antpos_d[bl[1]] - antpos_d[bl[0]])
            if bl_len > 0:
                bls.append(bl[:2])   
        model.select(bls=bls)
        
        # load the metadata
        raw = hc.io.HERAData(raw_file)
        times = np.unique(raw.time_array)
        raw.read(times=times[t[N_time]:t[N_time+1]])
        raw.select(bls=bls)
        raw.x_orientation = 'east'
        
        freqs=model.freq_array[0,:]
        gain_keys = []
        for i in range (len(ants)):
            gain_keys.append((i,'Jee'))
        print("now callibrating")

        model_data, _, _ = model.build_datacontainers()
        raw_data, _, _ = raw.build_datacontainers()
        
        
        #Choosing baseline cut
        if mode=="incomplete_no_filter" or mode=="complete_no_filter" or  mode=="complete_no_filter_baseline_cut":
            wgts = hc.datacontainer.DataContainer({k: np.ones_like(raw_data[k], np.float) for k in raw_data})
            bas_norm=[]
            for k in raw_data:
                blvec = antpos[k[0]] - antpos[k[1]]
                bas_norm.append(np.linalg.norm(blvec))
                if np.linalg.norm(blvec) < 40:
                    if bool_baseline_cut:    
                        wgts[k][:] = 1e-40
                    else:
                        wgts[k][:] = 1
        else:
            ## baseline cut filtered data            
            wgts = hc.datacontainer.DataContainer({k: np.ones_like(raw_data[k], np.float) for k in raw_data})
            bas_norm=[]
            for k in raw_data:
                blvec = antpos[k[0]] - antpos[k[1]][0:1][0]
                bas_norm.append(np.linalg.norm(blvec))
                if np.linalg.norm(blvec) < 40:
                    if bool_baseline_cut:    
                        wgts[k][:] = 1
                    else:
                        wgts[k][:] = 1                    
                    
        



        
        # run phase calibration!
        phs_fit = hc.abscal.phs_logcal(model_data, raw_data, wgts=wgts, refant=0)

        # the fits are the delay [ns] and phase [radians], which we need to turn into complex gains
        # g = exp[ 2pi i tau * nu + i phi ]
        phs_gains, avg_phs_gains = {}, {}
        for key in gain_keys:
            phs_gains[key] = np.exp(1j*phs_fit["phi_{}_{}".format(*key)])
            # now average the solutions across frequency
            avg_phs_gains[key] = np.repeat(np.mean(phs_gains[key], axis=-1, keepdims=True), len(freqs), axis=-1)

        cal_data = copy.deepcopy(raw_data)
        hc.apply_cal.calibrate_in_place(cal_data, avg_phs_gains)
        
        amp_fit = hc.abscal.amp_logcal(model_data, raw_data, wgts=wgts)

        amp_gains = {}
        for key in gain_keys:
            amp_gains[key] = np.exp(amp_fit["eta_{}_{}".format(*key)])

        gains_full = hc.abscal.merge_gains([avg_phs_gains, amp_gains])
        cal_data = copy.deepcopy(raw_data)
        hc.apply_cal.calibrate_in_place(cal_data, gains_full)

        smooth_scale=3
        filter_scale=10
        
        gains=gains_full

#         ## Smooth gains
#         smooth_wgts = {k: np.ones_like(gains[k], np.float) for k in gains}
#         smooth_gains = copy.deepcopy(gains)
#         for k in gains:
#             w = np.ones_like(gains[k], np.float)
#             # fit out bandpass amplitude w/ 3rd order polynomial to make fourier filtering easier
            
            
#             bamp=np.zeros((t[N_time+1]-t[N_time],308))
#             for m in range(bamp.shape[0]):
#                 fit = np.polyfit(freqs, np.abs(gains[k][m,:]).T, 3)
#                 bamp[m,:] = np.polyval(fit, freqs)
#             # fourier filter the ratio
#             smooth_gains[k], info = hc.smooth_cal.freq_filter(gains[k] / bamp, w, freqs, filter_scale=filter_scale) ##wgts weight of gains
#             # multiply fit back
#             smooth_gains[k] *= bamp
#         cal_data = copy.deepcopy(raw_data)
#         hc.apply_cal.calibrate_in_place(cal_data, smooth_gains)

#         print("smooth done!")
#         cal_data = copy.deepcopy(raw_data)
#         hc.apply_cal.calibrate_in_place(cal_data,smooth_gains)
        k = (0, 18, 'ee')
        np.save("/home/ntsikelelo/Simulated_data_files/calibration_test_raw_data_"+filter_type+"_"+mode, model_data[k] / raw_data[k])
        np.save("/home/ntsikelelo/Simulated_data_files/calibration_test_cal_data_"+filter_type+"_"+mode, model_data[k] / cal_data[k])
        print("done with cal")


        ant=18
        Nrms= 1e-5
        noise_wgts = {k: sigma_frac*np.ones_like(raw_data[k], dtype=float) / Nrms**2 for k in raw_data}
        gains_all_times[t[N_time]:t[N_time+1],:]=gains[(ant, 'Jee')]
#         gains_all_times_smooth[t[N_time]:t[N_time+1],:]=smooth_gains[(ant, 'Jxx')]
        abs_chisq, nObs, _, _ = hc.utils.chisq(raw_data, model_data, gains=gains_full, data_wgts=noise_wgts)
        chisq_dof = nObs.mean() - len(gains_full)
        red_chisq = abs_chisq / (chisq_dof)
        chisq_perfect[t[N_time]:t[N_time+1],:]=red_chisq
#             chisq_perfect_smooth_gains[t[N_time]:t[N_time+1],:]=compute_chi_square(model_data,raw_data,smooth_gains,noise,bool_filter_used)                    

    np.save("/home/ntsikelelo/Simulated_data_files/gains_"+filter_type+"_"+mode+".npy",gains_all_times)
#     np.save("/home/ntsikelelo/Simulated_data_files/gains_smooth"+filter_type+"_"+mode,gains_all_times_smooth)
    np.save("/home/ntsikelelo/Simulated_data_files/correct_sigma_chisq_sky_model_"+filter_type+"_"+mode+".npy",chisq_perfect)
#     np.save("/home/ntsikelelo/Simulated_data_files/correct_sigma_chisq_sky_model_"+filter_type+"_"+mode+"_smooth_gains.npy",chisq_perfect_smooth_gains) 
    print("done with chi sqaure "+mode)
