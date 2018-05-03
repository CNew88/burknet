# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 21:27:47 2018

@author: Burkay
"""

import burknet as bn

# Load data
#data = bn.Data()
#data.load('./data/celeba Ny48 Nx32 Ntrain202599.pkl.gz')
#data.load('./data/celeba Ny80 Nx64 zoom0.6 Ntrain202599.pkl.gz')
#data.load('./data/caleb2 Ny80 Nx64 zoomy1 zoomx0.8 Ntrain819.pkl.gz')

# Build network
vae = bn.VAE()
vae.create_C(data, n_latent=100, beta=.5)

# Train
#vae.fit(data, n_epochs=250)
vae.fit(data, n_epochs=500, sessionname='S_180425_2019')
