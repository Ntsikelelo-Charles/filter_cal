import numpy as np



def convert_degrees_hours(RA,scale,h):
    RA_cordinate_hours=[""]*len(RA)
    for i in range(len(RA)):
        RAh=((RA[i]/scale)*24) #degrees to hours
        RAm=(RAh-int(RAh))*60  #hours to sec
        RAs=(RAm-int(RAm))*60
        RAH=int(RAh)
        RAM=np.abs(int(RAm))
        RAS=np.abs(int(RAs))

        RA_hours=str(RAH)+h
        RA_minutes=str(RAM)+"m"
        RA_seconds=str(RAS)+"s"
        RA_cordinate_hours[i]=(RA_hours+RA_minutes+RA_seconds)
    
    return RA_cordinate_hours

def convert_degrees_hours_DE(DE,h):
    
    DE_cordinate_degress_min=[""]*len(DE)
    for i in range(len(DE)):
        DEh=(DE[i])
        DEm=(DEh-int(DEh))*60
        DEs=(DEm-int(DEm))*60
        DEH=int(DEh)
        DEM=np.abs(int(DEm))
        DES=np.abs(DEs)

        DE_degrees=str(DEH)+h
        DE_minutes=str(DEM)+"m"
        DE_seconds=str(DES)+"s"
        DE_cordinate_degress_min[i]=(DE_degrees+DE_minutes+DE_seconds)
    
    return DE_cordinate_degress_min

field_name=np.array(["zen.LST.0.50034.sum","zen.LST.0.50175.sum","zen.LST.0.50316.sum","zen.LST.0.50457.sum","zen.LST.0.50598.sum","zen.LST.0.50739.sum","zen.LST.0.50880.sum","zen.LST.0.51021.sum","zen.LST.0.51162.sum","zen.LST.0.51303.sum","zen.LST.0.51444.sum","zen.LST.0.51585.sum","zen.LST.0.51726.sum","zen.LST.0.51867.sum","zen.LST.0.52008.sum","zen.LST.0.52148.sum","zen.LST.0.52289.sum","zen.LST.0.52430.sum","zen.LST.0.52571.sum","zen.LST.0.52712.sum","zen.LST.0.52853.sum","zen.LST.0.52994.sum","zen.LST.0.53135.sum","zen.LST.0.53276.sum","zen.LST.0.53417.sum","zen.LST.0.53558.sum","zen.LST.0.53699.sum","zen.LST.0.53840.sum","zen.LST.0.53981.sum","zen.LST.0.54122.sum","zen.LST.0.54263.sum","zen.LST.0.54404.sum","zen.LST.0.54544.sum","zen.LST.0.54685.sum","zen.LST.0.54826.sum","zen.LST.0.54967.sum","zen.LST.0.55108.sum","zen.LST.0.55249.sum","zen.LST.0.55390.sum","zen.LST.0.55531.sum","zen.LST.0.55672.sum","zen.LST.0.55813.sum","zen.LST.0.55954.sum","zen.LST.0.56095.sum","zen.LST.0.56236.sum","zen.LST.0.56377.sum","zen.LST.0.56518.sum","zen.LST.0.56659.sum","zen.LST.0.56800.sum","zen.LST.0.56940.sum","zen.LST.0.57081.sum","zen.LST.0.57222.sum","zen.LST.0.57363.sum","zen.LST.0.57504.sum","zen.LST.0.57645.sum","zen.LST.0.57786.sum","zen.LST.0.57927.sum","zen.LST.0.58068.sum","zen.LST.0.58209.sum","zen.LST.0.58350.sum","zen.LST.0.58491.sum","zen.LST.0.58632.sum","zen.LST.0.58773.sum","zen.LST.0.58914.sum","zen.LST.0.59055.sum","zen.LST.0.59196.sum","zen.LST.0.59336.sum","zen.LST.0.59477.sum","zen.LST.0.59618.sum","zen.LST.0.59759.sum","zen.LST.0.59900.sum"])

point_ra_all=np.load("/net/ike/vault-ike/ntsikelelo/full-band/point_ra.npy")
point_dec_all=np.load("/net/ike/vault-ike/ntsikelelo/full-band/point_dec.npy")
apparent_flux_all=np.load("/net/ike/vault-ike/ntsikelelo/full-band/apparent_flux.npy",allow_pickle=True)
RA_all=np.load("/net/ike/vault-ike/ntsikelelo/full-band/source_ra.npy",allow_pickle=True)
DE_all=np.load("/net/ike/vault-ike/ntsikelelo/full-band/source_dec.npy",allow_pickle=True)
spectral_index_all=np.load("/net/ike/vault-ike/ntsikelelo/full-band/source_spectral_index.npy",allow_pickle=True)

for k in range(len(field_name)):
    #field pointing center 
    point_ra=point_ra_all[k]
    point_dec=point_dec_all[k]
    apparent_flux=apparent_flux_all[k]
    RA=RA_all[k]
    DE=DE_all[k]
    spectral_index=spectral_index_all[k]
    apparent_flux=apparent_flux_all[k]
    print("total number of sources "+ str(len(DE)))
    print(" max is flux= "+str(np.max(apparent_flux)))
    RA_cordinate_hours=convert_degrees_hours(RA,360.0,"h")
    DE_cordinate_degress_min=convert_degrees_hours_DE(DE,"d")
     
    for i in range(len(RA)):
         cl.addcomponent(fluxunit="Jy",shape="point", flux=apparent_flux[i],freq="151MHz",spectrumtype='spectral index',index=spectral_index[i],dir=(RA_cordinate_hours[i],DE_cordinate_degress_min[i]))
          
    ones=np.ones(shape=(2))
    ref=(convert_degrees_hours(ones*point_ra,360.0,"h")[0],convert_degrees_hours_DE(ones*point_dec,"d")[0])
    x=field_name[k]
    freqs=np.linspace(46.9207763671875,234.2987060546875,1536)
    cell='120arcsec'                                                                             
    Nfreqs = len(freqs)                                                   
    print("..saving "+x+"_cl.im")                                         
    ia.fromshape("/net/ike/vault-ike/ntsikelelo/full-band/"+x+"_cl.im", [512,512, 1, Nfreqs], overwrite=True)        
    cs = ia.coordsys()                                                    
    cs.setunits(['rad','rad','','Hz'])                                    
    # set pixel properties                                                
    cell_rad = qa.convert(qa.quantity(cell),"rad")['value']               
    cs.setincrement([-cell_rad, cell_rad], type='direction')              
    cs.setreferencevalue([qa.convert(ref[0],'rad')['value'], qa.convert(ref[1],'rad')['value']], type="direction")                      
    # set freq properties                                  
    qa_freqs = qa.quantity(freqs, 'MHz')                    
    cs.setspectral(frequencies=qa_freqs)                                  
    # set flux properties, make image, export to fits                     
    ia.setcoordsys(cs.torecord())                                         
    ia.setbrightnessunit("Jy/pixel")    
    ia.modify(cl.torecord(), subtract=False)                      
    print("..saving "+x+"_cl.fits")                  
    exportfits(imagename="/net/ike/vault-ike/ntsikelelo/full-band/"+x+"_cl.im", fitsimage="/net/ike/vault-ike/ntsikelelo/full-band/"+x+"_cl_im.fits", overwrite=True, stokeslast=False)         
    ia.close()
    cl.close() 
   
    
