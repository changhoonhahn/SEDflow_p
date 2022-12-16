'''

script to compile all of the training data into some easily useable form 

'''
import os, sys
import numpy as np


dat_dir = '/scratch/network/chhahn/sedflow/training_sed/'
################################################################
# compile thetas and obs
################################################################
fsps = os.path.join(dat_dir, 'train.v0.1.101.thetas_sps.npy')
funt = os.path.join(dat_dir, 'train.v0.1.101.thetas_unt.npy')

f_Aspec = os.path.join(dat_dir, 'train.v0.1.101.norm_spec.nde_noise.npy') 
f_hspec = os.path.join(dat_dir, 'train.v0.1.101.h_spec.reg_comp.nde_noise.npy') 
f_Aivar = os.path.join(dat_dir, 'train.v0.1.101.norm_ivar.nde_noise.npy') 
f_hivar = os.path.join(dat_dir, 'train.v0.1.101.h_ivar.nde_noise.npy') 

f_zred = os.path.join(dat_dir, 'train.v0.1.101.redshifts.npy') 

A_spec = np.load(f_Aspec)
zero_norm = (A_spec == 0)

theta_sps = np.load(fsps)[~zero_norm]
theta_unt = np.load(funt)[~zero_norm]

A_spec = A_spec[~zero_norm]
h_spec = np.load(f_hspec)
A_ivar = np.load(f_Aivar)[~zero_norm]
h_ivar = np.load(f_hivar)[~zero_norm]

zred = np.load(f_zred)[~zero_norm]

################################################################
# slight clean up  
################################################################
not_nan = (
        ~(np.sum(np.isnan(A_spec)).astype(bool)) &
        ~(np.sum(np.isnan(h_spec), axis=1).astype(bool)) & 
        ~(np.sum(np.isnan(A_ivar)).astype(bool)) & 
        ~(np.sum(np.isnan(h_ivar), axis=1).astype(bool))) 
print('%i of %i have no NaNs' % (np.sum(not_nan), len(not_nan)))

np.save('/scratch/network/chhahn/sedflow/sedflow_p.test.v0.2.theta_sps.npy',
        theta_sps[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.test.v0.2.theta_unt.npy', 
        theta_unt[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.test.v0.2.norm_spec.npy', 
        A_spec[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.test.v0.2.h_spec.npy', 
        h_spec[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.test.v0.2.norm_ivar.npy', 
        A_ivar[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.test.v0.2.h_ivar.npy', 
        h_ivar[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.test.v0.2.zred.npy', 
        zred[not_nan]) 


'''
dat_dir = '/scratch/network/chhahn/sedflow/training_sed/'
################################################################
# compile thetas and obs
################################################################
fsps = os.path.join(dat_dir, 'train.v0.1.101.thetas_sps.npy')
funt = os.path.join(dat_dir, 'train.v0.1.101.thetas_unt.npy')

f_Aspec = os.path.join(dat_dir, 'train.v0.1.101.norm_spec.nde_noise.npy') 
f_hspec = os.path.join(dat_dir, 'train.v0.1.101.h_spec.nde_noise.npy') 
f_Aivar = os.path.join(dat_dir, 'train.v0.1.101.norm_ivar.nde_noise.npy') 
f_hivar = os.path.join(dat_dir, 'train.v0.1.101.h_ivar.nde_noise.npy') 

theta_sps = np.load(fsps)
theta_unt = np.load(funt)

A_spec = np.load(f_Aspec)
h_spec = np.load(f_hspec)
A_ivar = np.load(f_Aivar)
h_ivar = np.load(f_hivar)

f_zred = os.path.join(dat_dir, 'train.v0.1.101.redshifts.npy') 
zred = np.load(f_zred)

################################################################
# slight clean up  
################################################################
not_nan = (
        ~(np.sum(np.isnan(A_spec)).astype(bool)) &
        ~(np.sum(np.isnan(h_spec), axis=1).astype(bool)) & 
        ~(np.sum(np.isnan(A_ivar)).astype(bool)) & 
        ~(np.sum(np.isnan(h_ivar), axis=1).astype(bool))) 
print('%i of %i have no NaNs' % (np.sum(not_nan), len(not_nan)))

np.save('/scratch/network/chhahn/sedflow/sedflow_p.test.v0.1.theta_sps.npy',
        theta_sps[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.test.v0.1.theta_unt.npy', 
        theta_unt[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.test.v0.1.norm_spec.npy', 
        A_spec[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.test.v0.1.h_spec.npy', 
        h_spec[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.test.v0.1.norm_ivar.npy', 
        A_ivar[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.test.v0.1.h_ivar.npy', 
        h_ivar[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.test.v0.1.zred.npy', 
        zred[not_nan]) 
'''
