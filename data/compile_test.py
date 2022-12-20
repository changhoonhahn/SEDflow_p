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

f_zred = os.path.join(dat_dir, 'train.v0.1.101.redshifts.npy') 

f_Aspec = os.path.join(dat_dir, 'train.v0.1.101.norm_spec.nde_noise.spender.npy') 
f_hspec = os.path.join(dat_dir, 'train.v0.1.101.h_spec.nde_noise.spender.npy') 
f_Ahivar = '/scratch/network/chhahn/sedflow/nde_noise/Ah_ivar.nde.101.npy'

theta_sps = np.load(fsps)
theta_unt = np.load(funt)

zred = np.load(f_zred)

A_spec = np.load(f_Aspec)
h_spec = np.load(f_hspec)
Ah_spec = np.concatenate([A_spec[:,None], h_spec], axis=1) 
Ah_ivar = np.load(f_Ahivar)

################################################################
# slight clean up  
################################################################
np.save('/scratch/network/chhahn/sedflow/sedflow_p.test.v0.3.theta_sps.npy', theta_sps) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.test.v0.3.theta_unt.npy', theta_unt) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.test.v0.3.Ah_spec.npy', Ah_spec) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.test.v0.3.Ah_ivar.npy', Ah_ivar) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.test.v0.3.zred.npy', zred) 
