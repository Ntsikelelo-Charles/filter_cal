import numpy as np
from pyrap import table 
import astropy.io.fits as fits
from astropy import wcs
from pyuvdata import UVBeam
from astropy.table import Table








field_name=np.array(["zen.LST.0.50034.sum","zen.LST.0.50175.sum","zen.LST.0.50316.sum","zen.LST.0.50457.sum","zen.LST.0.50598.sum","zen.LST.0.50739.sum","zen.LST.0.50880.sum","zen.LST.0.51021.sum","zen.LST.0.51162.sum","zen.LST.0.51303.sum","zen.LST.0.51444.sum","zen.LST.0.51585.sum","zen.LST.0.51726.sum","zen.LST.0.51867.sum","zen.LST.0.52008.sum","zen.LST.0.52148.sum","zen.LST.0.52289.sum","zen.LST.0.52430.sum","zen.LST.0.52571.sum","zen.LST.0.52712.sum","zen.LST.0.52853.sum","zen.LST.0.52994.sum","zen.LST.0.53135.sum","zen.LST.0.53276.sum","zen.LST.0.53417.sum","zen.LST.0.53558.sum","zen.LST.0.53699.sum","zen.LST.0.53840.sum","zen.LST.0.53981.sum","zen.LST.0.54122.sum","zen.LST.0.54263.sum","zen.LST.0.54404.sum","zen.LST.0.54544.sum","zen.LST.0.54685.sum","zen.LST.0.54826.sum","zen.LST.0.54967.sum","zen.LST.0.55108.sum","zen.LST.0.55249.sum","zen.LST.0.55390.sum","zen.LST.0.55531.sum","zen.LST.0.55672.sum","zen.LST.0.55813.sum","zen.LST.0.55954.sum","zen.LST.0.56095.sum","zen.LST.0.56236.sum","zen.LST.0.56377.sum","zen.LST.0.56518.sum","zen.LST.0.56659.sum","zen.LST.0.56800.sum","zen.LST.0.56940.sum","zen.LST.0.57081.sum","zen.LST.0.57222.sum","zen.LST.0.57363.sum","zen.LST.0.57504.sum","zen.LST.0.57645.sum","zen.LST.0.57786.sum","zen.LST.0.57927.sum","zen.LST.0.58068.sum","zen.LST.0.58209.sum","zen.LST.0.58350.sum","zen.LST.0.58491.sum","zen.LST.0.58632.sum","zen.LST.0.58773.sum","zen.LST.0.58914.sum","zen.LST.0.59055.sum","zen.LST.0.59196.sum","zen.LST.0.59336.sum","zen.LST.0.59477.sum","zen.LST.0.59618.sum","zen.LST.0.59759.sum","zen.LST.0.59900.sum"])






def beam_interp_func(theta, phi, data):
                # convert to radians
                theta = copy.copy(theta) * np.pi / 180.0
                phi = copy.copy(phi) * np.pi / 180.0
                shape = theta.shape
                beam_interp = [healpy.get_interp_val(m, theta.ravel(), phi.ravel(), lonlat=False).reshape(shape) for m in data]
                return np.array(beam_interp)

gleam_catalogue = Table.read('/home/ntsikelelo/my_files/GLEAM_models/GLEAM_models_var/GLEAM_EGC_v2.fits')
uvb = UVBeam()
uvb.read_beamfits('/net/ike/home/ntsikelelo/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Vivaldi_power_beam_healpix.fits')
uvb.peak_normalize()
pol_ind = np.where(uvb.polarization_array ==-6)[0][0]

beam_maps = np.abs(uvb.data_array[0, 0, pol_ind, :, :])
beam_freqs = uvb.freq_array.squeeze() / 1e6
Nbeam_freqs = len(beam_freqs)
beam_nside = healpy.npix2nside(beam_maps.shape[1])            

RA_point=np.zeros(field_name.shape)
DEC_point=np.zeros(field_name.shape)
flux_all=[]
RA_source_all=[]
DE_source_all=[]
spectral_index_source_all=[]
for fi in range(len(field_name)):
    #field pointing center 
    fld = table("/net/ike/vault-ike/ntsikelelo/full-band/Field_ms_files/"+field_name[fi]+".ms"+"::FIELD")
    radec0 = fld.getcol('PHASE_DIR').squeeze().reshape(1,2)
    radec0 = np.tile(radec0, (60,1)) 
    fld.close() 
    ms = table("/net/ike/vault-ike/ntsikelelo/full-band/Field_ms_files/"+field_name[fi]+".ms")
    time = ms.getcol('TIME')
    antenna=ms.getcol('ANTENNA1')
    
    n_antenna=antenna.size
    freqs = table("/net/ike/vault-ike/ntsikelelo/full-band/Field_ms_files/"+field_name[fi]+".ms"+'::SPECTRAL_WINDOW').getcol('CHAN_FREQ')[0].astype(np.float64)
    n_freq = freqs.size
    n_time=time.size
    
    point_ra=np.rad2deg(radec0[0,0])
    point_dec=np.rad2deg(radec0[0,1]) 
    RA_point[fi]=point_ra
    DEC_point[fi]=point_dec
    dis=(gleam_catalogue['RAJ2000']-point_ra)**2+(gleam_catalogue['DEJ2000']-point_dec)**2
    
    ## select GLEAM sources
    indices=np.where( (dis<15**2) & (gleam_catalogue['int_flux_151']>10))


    DE=gleam_catalogue['DEJ2000'][indices]
    RA=gleam_catalogue['RAJ2000'][indices]
    flux=gleam_catalogue['int_flux_wide'][indices]
    spectral_index=gleam_catalogue['alpha'][indices]
    source_names=gleam_catalogue['Name'][indices]
    RA_source_all.append(RA)
    DE_source_all.append(DE)
    spectral_index_source_all.append(spectral_index)
    
                #HERA beam
    uvb.read_beamfits('/net/ike/home/ntsikelelo/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Vivaldi_power_beam_healpix.fits')
    pol_ind = np.where(uvb.polarization_array ==-6)[0][0]
    beam_maps = np.abs(uvb.data_array[0, 0, pol_ind, :, :])
    beam_freqs = uvb.freq_array.squeeze() / 1e6
    Nbeam_freqs = len(beam_freqs)
    beam_nside = healpy.npix2nside(beam_maps.shape[1])

    theta = np.sqrt( (RA - point_ra)**2 + (DE - point_dec)**2 ) # center origin at the array dec and RA
    phi = np.arctan2((DE-point_dec), (RA-point_ra)) + np.pi

    pb = beam_interp_func(theta, phi,beam_maps)

    data_freqs=np.linspace(beam_freqs[0],beam_freqs[-1],n_freq) #MHz
    Ndata_freqs = len(data_freqs)
    print(data_freqs[0],beam_freqs[0])
    beam_values=np.zeros(shape=(n_freq, DE_cen.shape[0]))
    for source in range (pb.shape[1]):
        pb_interp = interpolate.interp1d(beam_freqs,pb[:,source], kind='cubic')
        beam_values[:,source]=pb_interp(data_freqs)

    gains=np.zeros(shape=(n_time,n_antenna,n_freq,DE_cen.shape[0],1), dtype=complex)
    gains[:,:,:,:,0]=np.sqrt(beam_values)
    np.save('/net/ike/vault-ike/ntsikelelo/Gains_'+field_name[fi]+'_.npy', gains)
    




    print(flux.shape)
    flux_all.append(flux)
    
print(np.array(RA_source_all).shape)    
np.save("/net/ike/vault-ike/ntsikelelo/full-band/apparent_flux.npy",np.array(flux_all)) 
np.save("/net/ike/vault-ike/ntsikelelo/full-band/point_ra.npy",RA_point)
np.save("/net/ike/vault-ike/ntsikelelo/full-band/point_dec.npy",DEC_point)
np.save("/net/ike/vault-ike/ntsikelelo/full-band/source_ra.npy",np.array(RA_source_all))
np.save("/net/ike/vault-ike/ntsikelelo/full-band/source_dec.npy",np.array(DE_source_all))
np.save("/net/ike/vault-ike/ntsikelelo/full-band/source_spectral_index.npy",np.array(spectral_index_source_all))