import numpy as np

field_name=np.array(["zen.LST.0.50034.sum","zen.LST.0.50175.sum","zen.LST.0.50316.sum","zen.LST.0.50457.sum","zen.LST.0.50598.sum","zen.LST.0.50739.sum","zen.LST.0.50880.sum","zen.LST.0.51021.sum","zen.LST.0.51162.sum","zen.LST.0.51303.sum","zen.LST.0.51444.sum","zen.LST.0.51585.sum","zen.LST.0.51726.sum","zen.LST.0.51867.sum","zen.LST.0.52008.sum","zen.LST.0.52148.sum","zen.LST.0.52289.sum","zen.LST.0.52430.sum","zen.LST.0.52571.sum","zen.LST.0.52712.sum","zen.LST.0.52853.sum","zen.LST.0.52994.sum","zen.LST.0.53135.sum","zen.LST.0.53276.sum","zen.LST.0.53417.sum","zen.LST.0.53558.sum","zen.LST.0.53699.sum","zen.LST.0.53840.sum","zen.LST.0.53981.sum","zen.LST.0.54122.sum","zen.LST.0.54263.sum","zen.LST.0.54404.sum","zen.LST.0.54544.sum","zen.LST.0.54685.sum","zen.LST.0.54826.sum","zen.LST.0.54967.sum","zen.LST.0.55108.sum","zen.LST.0.55249.sum","zen.LST.0.55390.sum","zen.LST.0.55531.sum","zen.LST.0.55672.sum","zen.LST.0.55813.sum","zen.LST.0.55954.sum","zen.LST.0.56095.sum","zen.LST.0.56236.sum","zen.LST.0.56377.sum","zen.LST.0.56518.sum","zen.LST.0.56659.sum","zen.LST.0.56800.sum","zen.LST.0.56940.sum","zen.LST.0.57081.sum","zen.LST.0.57222.sum","zen.LST.0.57363.sum","zen.LST.0.57504.sum","zen.LST.0.57645.sum","zen.LST.0.57786.sum","zen.LST.0.57927.sum","zen.LST.0.58068.sum","zen.LST.0.58209.sum","zen.LST.0.58350.sum","zen.LST.0.58491.sum","zen.LST.0.58632.sum","zen.LST.0.58773.sum","zen.LST.0.58914.sum","zen.LST.0.59055.sum","zen.LST.0.59196.sum","zen.LST.0.59336.sum","zen.LST.0.59477.sum","zen.LST.0.59618.sum","zen.LST.0.59759.sum","zen.LST.0.59900.sum"])



for k in range(len(field_name)):
    x=field_name[k]
    importfits(fitsimage="/net/ike/vault-ike/ntsikelelo/full-band/"+x+"_cl_im_pb_applied.fits",imagename="/net/ike/vault-ike/ntsikelelo/full-band/"+x+"_cl_im_pb_applied.im", overwrite=True)  

    ft("/net/ike/vault-ike/ntsikelelo/full-band/Field_ms_files/"+field_name[k]+".ms",model="/net/ike/vault-ike/ntsikelelo/full-band/"+x+"_cl_im_pb_applied.im",usescratch=True)

    print("Now splitting")
    split(vis="/net/ike/vault-ike/ntsikelelo/full-band/Field_ms_files/"+field_name[k]+".ms",outputvis="/net/ike/vault-ike/ntsikelelo/full-band/Field_ms_files_model/"+field_name[k]+'.ms', datacolumn='model') 
