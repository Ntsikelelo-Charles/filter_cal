import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pyuvdata import UVData
import hera_cal as hc
import uvtools as uvt
import hera_pspec as hp
import copy
from scipy import integrate
from scipy import optimize

def gauss(x, amp, loc, scale):
    return amp * np.exp(-0.5 * (x-loc)**2 / scale**2)

def chisq(x0, x, y):
    yfit = gauss(x, *x0)
    return np.sum(np.abs(yfit - y)**2)

def gauss_fit(x0, x, y, method='powell'):
    fit = optimize.minimize(chisq, x0, args=(x, y), method=method)
    ypred = gauss(x, *fit.x)
    return fit, ypred

path="/net/ike/vault-ike/ntsikelelo/Simulated_data_files/UVH5_files/"
path2="/net/jake/home/ntsikelelo/Simulated_data_files/UVH5_files/"

file=np.array(["Model_data_complete.uvh5","Model_data_incomplete.uvh5","Raw_data_with_noise.uvh5"])
file_output=np.array(["Model_complete_filtered_Gaussian.uvh5","Model_incomplete_filtered_Gaussian.uvh5","Raw_data_filtered_Gaussian.uvh5"])

for mode in range (len(file)):
    print(file[mode])
    if path2+file[mode]=="Raw_data_with_noise.uvh5":
        Model_complete_file = path+file[mode]
        Model_complete = hc.frf.FRFilter(Model_complete_file)
        uvd = UVData()
        uvd.read(Model_complete_file)
        freqs = Model_complete.freqs/1e6
        times = (Model_complete.times-Model_complete .times.min()) * 24 * 3600  # seconds
        uvd.inflate_by_redundancy()
        uvd.conjugate_bls()

        F = hc.frf.FRFilter(uvd)
        filter_factor     = [1e-8]
        F.fft_data(ax='time', window='blackman', overwrite=True, ifft=True)

        fr_select = (F.frates > 0.5) & (F.frates < 4.0) 
        fr_select_negative=(F.frates<-0.5) & (F.frates>-4)
        m=np.load("/net/jake/home/ntsikelelo/Simulated_data_files/m_slope_filter.npy")
        c=np.load("/net/jake/home/ntsikelelo/Simulated_data_files/c_intercept_filter.npy")

        x0 = np.array([1e-3, 2.0, 0.3])

        x = F.frates[fr_select]
        x_negative = F.frates[fr_select_negative]
        antpos = F.antpos
        filter_fun=np.zeros(F.frates.shape)
        sigma_frac={}
        filt_data = copy.deepcopy(F_model.data)
        filt_data = copy.deepcopy(F_modelhah.data)
        for k in F.data:
            blvec = (antpos[k[1]] - antpos[k[0]])
            bl_len_EW = blvec[0]
            fringe_value=m*bl_len_EW+c
         
            if bl_len_EW>0:
                y = np.abs(F.dfft[k]).mean(1)[fr_select]
                x0[1]=fringe_value
                fit, ypred = gauss_fit(x0, x, y, method='powell')

                filter_fun[fr_select]=ypred/np.max(ypred)
                I1 = integrate.simpson(filter_fun**2,F.frates)
                I2 = integrate.simpson(np.ones(F.frates.shape)**2,F.frates)
                sigma_frac[k]=I1/I2
                # get the gaussian fit parameters
                gmean, gsigma = fit.x[1:]

                # make the filter
                filter_center = -gmean * 1e-3
                filter_half_width = np.abs(gsigma) * 2 * 1e-3
                fitler_factor = 1e-8
                C = uvt.dspec.dayenu_mat_inv(times, filter_center, filter_half_width, filter_factor, no_regularization=False)
                R = np.linalg.pinv(C, rcond=1e-10)
                filt_data[k] = filt_data[k] - R @ filt_data[k]        

            if bl_len_EW<0:
    
                y = np.abs(F.dfft[k]).mean(1)[fr_select_negative]
                x0[1]=fringe_value
                fit, ypred = gauss_fit(x0, x_negative, y, method='powell')

                filter_fun[fr_select]=ypred/np.max(ypred)
                I1 = integrate.simpson(filter_fun**2,F.frates)
                I2 = integrate.simpson(np.ones(F.frates.shape)**2,F.frates)
                sigma_frac[k]=I1/I2

                # get the gaussian fit parameters
                gmean, gsigma = fit.x[1:]

                # make the filter
                filter_center = -gmean * 1e-3
                filter_half_width = np.abs(gsigma) * 2 * 1e-3
                fitler_factor = 1e-8
                C = uvt.dspec.dayenu_mat_inv(times, filter_center, filter_half_width, filter_factor, no_regularization=False)
                R = np.linalg.pinv(C, rcond=1e-10)
                filt_data[k] = filt_data[k] - R @ filt_data[k]
                

        F.write_data(filt_data,path+file_output[mode],overwrite=True)
    
    
    if file[mode]=="Model_data_complete.uvh5" or file[mode]=="Model_data_incomplete.uvh5":
        print("Model filter")
        Model_complete_file = path+file[mode]
        Model_complete = hc.frf.FRFilter(Model_complete_file)
        uvd = UVData()
        uvd.read(Model_complete_file)
        freqs = Model_complete.freqs/1e6
        times = (Model_complete.times-Model_complete .times.min()) * 24 * 3600  # seconds
        uvd.inflate_by_redundancy()
        uvd.conjugate_bls()

        F = hc.frf.FRFilter(uvd)
        filter_factor     = [1e-8]
        F.fft_data(ax='time', window='blackman', overwrite=True, ifft=True)

        fr_select = (F.frates > 0.5) & (F.frates < 4.0) 
        fr_select_negative=(F.frates<-0.5) & (F.frates>-4)
        m=np.load("/net/jake/home/ntsikelelo/Simulated_data_files/m_slope_filter.npy")
        c=np.load("/net/jake/home/ntsikelelo/Simulated_data_files/c_intercept_filter.npy")

        x0 = np.array([175, 2.0, 0.3])

        x = F.frates[fr_select]
        x_negative = F.frates[fr_select_negative]
        antpos = F.antpos
        filter_fun=np.zeros(F.frates.shape)
        sigma_frac={}
        filt_data = copy.deepcopy(F.data)

        for k in F.data:
            blvec = (antpos[k[1]] - antpos[k[0]])
            bl_len_EW = blvec[0]
            fringe_value=m*bl_len_EW+c
            if bl_len_EW>0:
                y = np.abs(F.dfft[k]).mean(1)[fr_select]
                x0[1]=fringe_value
                fit, ypred = gauss_fit(x0, x, y, method='powell')

                filter_fun[fr_select]=ypred/np.max(ypred)
                I1 = integrate.simpson(filter_fun**2,F.frates)
                I2 = integrate.simpson(np.ones(F.frates.shape)**2,F.frates)
                sigma_frac[k]=I1/I2
                # get the gaussian fit parameters
                gmean, gsigma = fit.x[1:]

                # make the filter
                filter_center = -gmean * 1e-3
                filter_half_width = np.abs(gsigma) * 2 * 1e-3
                fitler_factor = 1e-8
                C = uvt.dspec.dayenu_mat_inv(times, filter_center, filter_half_width, filter_factor, no_regularization=False)
                R = np.linalg.pinv(C, rcond=1e-10)
                filt_data[k] = filt_data[k] - R @ filt_data[k]        

            if bl_len_EW<0:
                y = np.abs(F.dfft[k]).mean(1)[fr_select_negative]
                x0[1]=fringe_value
                fit, ypred = gauss_fit(x0, x_negative, y, method='powell')

                filter_fun[fr_select]=ypred/np.max(ypred)
                I1 = integrate.simpson(filter_fun**2,F.frates)
                I2 = integrate.simpson(np.ones(F.frates.shape)**2,F.frates)
                sigma_frac[k]=I1/I2

                # get the gaussian fit parameters
                gmean, gsigma = fit.x[1:]

                # make the filter
                filter_center = -gmean * 1e-3
                filter_half_width = np.abs(gsigma) * 2 * 1e-3
                fitler_factor = 1e-8
                C = uvt.dspec.dayenu_mat_inv(times, filter_center, filter_half_width, filter_factor, no_regularization=False)
                R = np.linalg.pinv(C, rcond=1e-10)
                filt_data[k] = filt_data[k] - R @ filt_data[k]


        F.write_data(filt_data,path+file_output[mode],overwrite=True)




