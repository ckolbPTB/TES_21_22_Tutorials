#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A collection of auxiliary functions for manipulating data.

"""
import numpy as np

import torch

from numpy.lib.stride_tricks import as_strided
from numpy.fft import ifftshift

import  sys
sys.path.append('../')

def random_phase(img):
	
	""" 
	function for generating a random phase-profile
	
	"""
	
	#get shape of in-plane image
	Nx,Ny = img.shape[:2]
	x = np.linspace(-np.pi, np.pi, Nx)
	y = np.linspace(-np.pi, np.pi, Ny)
	
	#generate parameters to create a random phase profile with values 
	xx, yy = np.meshgrid(x, y)
	
	a, b,c,d,e = np.random.random(5)
	z = a*np.sin(b*xx-c) + (1-a)*np.cos(d*yy-e) 
	
	#bring to [-np.pi, np.pi]
	z = (np.pi- (-np.pi))*(z-np.min(z))/(np.max(z)-np.min(z)) + (-np.pi)
	
	return z


def cplx_np2torch(x,dim):
	"""
	functon for converting a complex-valued np.array x
	to a complex-valued torch-tensor, where the 2 channels 
	for the real and imaginary parts are inserted as "dim" dimension
	"""
	
	x = torch.stack([torch.tensor(np.real(x)),torch.tensor(np.imag(x))],dim=dim)
			
	return x
	
def cplx_torch2np(x,dim):
	
	"""
	functon for converting a complex-valued torch-tensors to a complex-valued numpy array
	the parameter "dim" indicates which dimension is used to stre the real 
	and the imaginary part in the torch-tensor
	
	first, the tensor is transposed, such that we can access te real  and the imaginry parts
	
	the output is a complex-valued numpy array where the dimension "dim" is dropped
	"""
		
	#permutes the axis "dim" and 0
	#now, the 0-th axis contains the real and imainary parts
	x = torch.transpose(x,dim,0) 
	
	if x.is_cuda:
		x = x.cpu()
	
	#get the real and imaginry parts
	xr = x[0,...].numpy()
	xi = x[1,...].numpy()

	x = xr+1j*xi
	
	#expand dimensions in order to be able to get back to original shape
	x = np.expand_dims(x,axis=0)
	
	x = np.swapaxes(x,0,dim)
	
	#drop the dimension "dim"
	x = np.squeeze(x,axis=dim)
	
	return x


def random_mask(img, acc_factor=4):
	
	"""
	function for generating a random binary massk.
	For each time point, the mask is different for incoherent undersampling.
	"""
	
	

def normal_pdf(length, sensitivity):
	return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


def cartesian_mask(shape, acc, sample_n=10):

	"""
	Sampling density estimated from implementation of kt FOCUSS

	shape: tuple - of form (..., nx, ny)
	acc: float - doesn't have to be integer 4, 8, etc..
	
	Note:
		function borrowed from Jo Schlemper from
		https://github.com/js3611/Deep-MRI-Reconstruction/blob/master/utils/compressed_sensing.py

	"""
	N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
	pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
	lmda = Nx/(2.*acc)
	n_lines = int(Nx / acc)

	# add uniform distribution
	pdf_x += lmda * 1./Nx

	if sample_n:
		pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
		pdf_x /= np.sum(pdf_x)
		n_lines -= sample_n

	mask = np.zeros((N, Nx))
	for i in range(N):
		idx = np.random.choice(Nx, n_lines, False, pdf_x)
		mask[i, idx] = 1

	if sample_n:
		mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

	size = mask.itemsize
	mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

	mask = mask.reshape(shape)

	return mask

def cine_cartesian_mask(shape,acc_factor,mode='numpy'):
	
	"""
	create a binary mask for a 2d cine MR image sequence:
		
	N.B. for numpy, the binary mask is only real-valued, but this suffices
	as when computing the product of a complex-valued array with a ral-valued
	one, the output is complex-valued. This is because, for \mathbb{C}, the one-element
	is given by 1=1+0*j.
	In contrast, for pytorch, where complex-valued arrays are stored as two-channeled 
	signals, the mask has replicate the support of the indices for both channels 
	"""
	
	nx,ny,nt = shape
	
	mask = np.zeros(shape)
	
	for kt in range(nt):
		
		mask[:,:,kt] = cartesian_mask((nx,ny),acc_factor)
		
	
	if mode=='pytorch':
		
		mask = ifftshift(mask,axes=[0,1])
		mask = (1+1j)*mask #has shape (nx,ny,nt)
		
		#make a torch-tensor of shape (1,2,nx,ny,nt) out of it
		mask = cplx_np2torch(mask, 0).unsqueeze(0)
	
	return mask
	

def load_data(Ids):
	
	"""
	function for loading the image data of patients indexed by the set
	Ids and stack all different slices to have (N,Nx,Ny,Nt), where 
	N is the total number of cine MR images.
	"""
	
	#initalize list of images
	pats_list = []
	
	for pid in Ids:
		#load image
		img = np.load('data/np_arrays/xf_pat{}.npy'.format(pid))
		img = np.moveaxis(img,(0,1,2,3),(1,2,0,3))
		pats_list.append(img)
		print(img.shape)
		
	xf = np.concatenate(pats_list,axis=0)
	
	return xf
		
	