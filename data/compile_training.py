'''

script to compile all of the training data into some easily useable form 

* v0.2 with regression compresson encoded spectra 

'''
import os, sys
import numpy as np


dat_dir = '/scratch/network/chhahn/sedflow/training_sed/'

################################################################
# compile thetas and observables 
################################################################
theta_sps, theta_unt = [], [] 
A_spec, h_spec, Ah_ivar, zred = [], [], [], []
for ibatch in range(10): 
    fsps = os.path.join(dat_dir, f'train.v0.1.{ibatch}.thetas_sps.npy')
    funt = os.path.join(dat_dir, f'train.v0.1.{ibatch}.thetas_unt.npy')

    f_Aspec = os.path.join(dat_dir, f'train.v0.1.{ibatch}.norm_spec.nde_noise.spender.npy') 
    f_hspec = os.path.join(dat_dir, f'train.v0.1.{ibatch}.h_spec.nde_noise.spender.npy') 

    f_Ahivar = f'/scratch/network/chhahn/sedflow/nde_noise/Ah_ivar.nde.{ibatch}.npy'
    
    f_zred = os.path.join(dat_dir, f'train.v0.1.{ibatch}.redshifts.npy') 
    
    theta_sps.append(np.load(fsps))
    theta_unt.append(np.load(funt))

    A_spec.append(np.load(f_Aspec))
    h_spec.append(np.load(f_hspec))
    Ah_ivar.append(np.load(f_Ahivar))

    zred.append(np.load(f_zred))

theta_sps = np.concatenate(theta_sps, axis=0) 
theta_unt = np.concatenate(theta_unt, axis=0) 

A_spec = np.concatenate(A_spec, axis=0)
h_spec = np.concatenate(h_spec, axis=0)
Ah_spec = np.concatenate([A_spec[:,None], h_spec], axis=1) 
Ah_ivar = np.concatenate(Ah_ivar, axis=0)
zred = np.concatenate(zred, axis=0) 

np.save('/scratch/network/chhahn/sedflow/sedflow_p.train.v0.3.theta_sps.npy', theta_sps) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.train.v0.3.theta_unt.npy', theta_unt) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.train.v0.3.Ah_spec.npy', Ah_spec) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.train.v0.3.Ah_ivar.npy', Ah_ivar) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.train.v0.3.zred.npy', zred) 
