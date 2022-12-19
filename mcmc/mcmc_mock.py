'''

script to run MCMC on mock observations 


'''
import os, sys
import numpy as np

from provabgs import infer as Infer
from provabgs import models as Models

####################################################
# input
####################################################
niter   = int(sys.argv[1])
i0      = int(sys.argv[2])
i1      = int(sys.argv[3])

####################################################
# read in mock observations 
####################################################
dat_dir = '/scratch/network/chhahn/sedflow/training_sed/'
wobs = np.load('/scratch/network/chhahn/sedflow/sdss.clean.wave.npy')
spec = np.load(os.path.join(dat_dir, f'train.v0.1.101.seds.nde_noise.npy')).astype(np.float32)[:100]
ivar = np.load(os.path.join(dat_dir, f'train.v0.1.101.ivar.nde_noise.npy')).astype(np.float32)[:100]
zred = np.load(os.path.join(dat_dir, f'train.v0.1.101.redshifts.npy'))[:100]

####################################################
# setup inference 
####################################################
# SPS parameter priors (same as the priors used to generate the training data
# -- don't touch!)
prior_sps = Infer.load_priors([
        Infer.UniformPrior(7., 12.5, label='sed'),
        Infer.FlatDirichletPrior(4, label='sed'),           # flat dirichilet priors
        Infer.UniformPrior(0., 1., label='sed'),            # burst fraction
        Infer.UniformPrior(1e-2, 13.27, label='sed'),    # tburst
        Infer.LogUniformPrior(4.5e-5, 1.5e-2, label='sed'), # log uniform priors on ZH coeff
        Infer.LogUniformPrior(4.5e-5, 1.5e-2, label='sed'), # log uniform priors on ZH coeff
        Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust1
        Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust2
        Infer.UniformPrior(-2., 1., label='sed')     # uniform priors on dust_index
    ])

# SPS model 
m_sps = Models.NMF(burst=True, emulator=True)

def run_mcmc(i_obs): 
    # MCMC object
    mcmc = Infer.specphotoMCMC(model=m_sps, prior=prior_sps)
    
    fmcmc = os.path.join('/scratch/network/chhahn/sedflow/mcmc_spectra/', 
            'mcmc.mock.%i.hdf5' % i_obs)

    if not os.path.isfile(fmcmc): 
        print('%s running' % os.path.basename(fmcmc))
    
        # run MCMC
        zeus_chain = mcmc.run(
                wave_obs=wobs, # observed wavelength
                flux_obs=spec[i_obs], # observed flux of spectrum
                flux_ivar_obs=ivar[i_obs], # no noise in this example
                zred=zred[i_obs],
                vdisp=0.,
                sampler='zeus',
                nwalkers=30,
                burnin=0,
                opt_maxiter=2000,
                niter=niter,
                progress=True,
                writeout=fmcmc)
    else: 
        print('%s already exists' % os.path.basename(fmcmc))
    return None 


for i in range(i0, i1+1): 
    run_mcmc(i)
