from scipy.misc import factorial as fac
from sys import argv
import time
from mpl_toolkits.mplot3d import Axes3D  
import numpy as np
import matplotlib.pyplot as plt
from pyrap.tables import table 
import astropy.io.fits as fits
from astropy import wcs
from pyuvdata import UVBeam
import os
import sys
import glob
import argparse
import shutil
import copy
import healpy
import scipy.stats as stats
import pytz
import datetime
import ephem
from scipy import interpolate
from astropy.time import Time
import pylab as plt
from astropy.table import Table
from scipy import interpolate
import healpy as hp
import astropy 







field_name=np.array(["zen.LST.0.50034.sum","zen.LST.0.50175.sum","zen.LST.0.50316.sum","zen.LST.0.50457.sum","zen.LST.0.50598.sum","zen.LST.0.50739.sum","zen.LST.0.50880.sum","zen.LST.0.51021.sum","zen.LST.0.51162.sum","zen.LST.0.51303.sum","zen.LST.0.51444.sum","zen.LST.0.51585.sum","zen.LST.0.51726.sum","zen.LST.0.51867.sum","zen.LST.0.52008.sum","zen.LST.0.52148.sum","zen.LST.0.52289.sum","zen.LST.0.52430.sum","zen.LST.0.52571.sum","zen.LST.0.52712.sum","zen.LST.0.52853.sum","zen.LST.0.52994.sum","zen.LST.0.53135.sum","zen.LST.0.53276.sum","zen.LST.0.53417.sum","zen.LST.0.53558.sum","zen.LST.0.53699.sum","zen.LST.0.53840.sum","zen.LST.0.53981.sum","zen.LST.0.54122.sum","zen.LST.0.54263.sum","zen.LST.0.54404.sum","zen.LST.0.54544.sum","zen.LST.0.54685.sum","zen.LST.0.54826.sum","zen.LST.0.54967.sum","zen.LST.0.55108.sum","zen.LST.0.55249.sum","zen.LST.0.55390.sum","zen.LST.0.55531.sum","zen.LST.0.55672.sum","zen.LST.0.55813.sum","zen.LST.0.55954.sum","zen.LST.0.56095.sum","zen.LST.0.56236.sum","zen.LST.0.56377.sum","zen.LST.0.56518.sum","zen.LST.0.56659.sum","zen.LST.0.56800.sum","zen.LST.0.56940.sum","zen.LST.0.57081.sum","zen.LST.0.57222.sum","zen.LST.0.57363.sum","zen.LST.0.57504.sum","zen.LST.0.57645.sum","zen.LST.0.57786.sum","zen.LST.0.57927.sum","zen.LST.0.58068.sum","zen.LST.0.58209.sum","zen.LST.0.58350.sum","zen.LST.0.58491.sum","zen.LST.0.58632.sum","zen.LST.0.58773.sum","zen.LST.0.58914.sum","zen.LST.0.59055.sum","zen.LST.0.59196.sum","zen.LST.0.59336.sum","zen.LST.0.59477.sum","zen.LST.0.59618.sum","zen.LST.0.59759.sum","zen.LST.0.59900.sum"])


# load hdu     
# load beam
uvb = UVBeam()
uvb.read_beamfits('/net/ike/home/ntsikelelo/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Vivaldi_power_beam_healpix.fits')
uvb.peak_normalize()
pol_ind = np.where(uvb.polarization_array ==-6)[0][0]
# get beam models and beam parameters
beam_maps = np.abs(uvb.data_array[0, 0, pol_ind, :, :])
beam_freqs = uvb.freq_array.squeeze() / 1e6
Nbeam_freqs = len(beam_freqs)
beam_nside = healpy.npix2nside(beam_maps.shape[1])

# construct beam interpolation function
def beam_interp_func(theta, phi):
    # convert to radians
    theta = copy.copy(theta) * np.pi / 180.0
    phi = copy.copy(phi) * np.pi / 180.0
    shape = theta.shape
    beam_interp = [healpy.get_interp_val(m, theta.ravel(), phi.ravel(), lonlat=False).reshape(shape) for m in beam_maps]
    return np.array(beam_interp)



for k in range(len(field_name)):
    point_ra=np.load("/net/ike/vault-ike/ntsikelelo/full-band/point_ra.npy")[k]
    point_dec=np.load("/net/ike/vault-ike/ntsikelelo/full-band/point_dec.npy")[k]
    x=field_name[k]
    ffile="/net/ike/vault-ike/ntsikelelo/full-band/"+x+"_cl_im.fits"
    hdu = fits.open(ffile)

    # get header and data
    head = hdu[0].header
    data = hdu[0].data

    # determine if freq precedes stokes in header
    if head['CTYPE3'] == 'FREQ':
        freq_ax = 3
        stok_ax = 4
    else:
        freq_ax = 4
        stok_ax = 3

    # get axes info
    npix1 = head["NAXIS1"]
    npix2 = head["NAXIS2"]
    nstok = head["NAXIS{}".format(stok_ax)]
    nfreq = head["NAXIS{}".format(freq_ax)]





    # get WCS
    w = wcs.WCS(ffile)

    # get pixel coordinates
    lon_arr, lat_arr = np.meshgrid(np.arange(npix1), np.arange(npix2))
    lon, lat, s, f = w.all_pix2world(lon_arr.ravel(), lat_arr.ravel(), 0, 0, 0)
    lon = lon.reshape(npix2, npix1)
    lat = lat.reshape(npix2, npix1)
    theta = np.sqrt( (lon - point_ra)**2 + (lat - point_dec)**2 )
    phi = np.arctan2((lat-point_dec), (lon-point_ra)) + np.pi

    # get data frequencies
    if freq_ax == 3:
        data_freqs = w.all_pix2world(0, 0, np.arange(nfreq), 0, 0)[2] / 1e6
    else:
        data_freqs = w.all_pix2world(0, 0, 0, np.arange(nfreq), 0)[3] / 1e6
    Ndata_freqs = len(data_freqs)


    pb = beam_interp_func(theta, phi)

    # interpolate primary beam onto data frequencies
    pb_shape = (pb.shape[1], pb.shape[2])
    pb_interp = interpolate.interp1d(beam_freqs, pb.reshape(pb.shape[0], -1).T, fill_value='extrapolate')(data_freqs)
    pb_interp = (pb_interp.T).reshape((Ndata_freqs,) + pb_shape)

    # data shape is [naxis4, naxis3, naxis2, naxis1]
    if freq_ax == 3:
        pb_interp = pb_interp[np.newaxis]
    else:
        pb_interp = pb_interp[:, np.newaxis]

    # divide or multiply by primary beam
#     for f in range(pb_interp.shape[0]):
#         pb_interp[f,0,:,:]=pb_interp[f,0,:,:]/np.max(pb_interp[f,0,:,:])
        
    data_pbcorr = data * pb_interp
    print("normalised with corr pol primary beam applied to model image image "+x) 
    output_fname="/net/ike/vault-ike/ntsikelelo/full-band/"+x+"_cl_im_pb_applied.fits"
    fits.writeto(output_fname, data_pbcorr, head, overwrite=True)
    
    
