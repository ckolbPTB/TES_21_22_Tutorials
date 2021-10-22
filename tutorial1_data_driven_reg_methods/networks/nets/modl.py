import torch
import torch.nn as nn

from tutorial1_data_driven_reg_methods.networks.nets.unet import UNet

class MODL(nn.Module):
	
	"""
	An implementation of a network which alternates between
	the application of a CNN-block and a daata-consistency step
	which yields the minimizer of a CNN-regularized functional.
		
	The CNN-block is either a 2D or a 3D U-Net which is parametrized
	by 	the number of encoding stages, the number of conv layers 
	per	stage and the initial number of filters.
	
	For pre-training, the length of the network is set to one, rather
	than to the chosen length.
	"""

	def __init__(self, 
				nu = 4, 
				n_enc_stages = 1,
				n_convs_per_stage = 4,
				n_filters = 16,
				CNN_type = '3D',
				mode = 'pre_training'):

		super(MODL, self).__init__()
		
		#length of the network
		self.nu = nu
		
		
		if CNN_type == '3D':
			
			dim=3
			
			#3D UNet with two input/output channels for real/imag part
			self.cnn = UNet(dim,
				   n_ch_in = 2,
				   n_ch_out = 2,
				   n_enc_stages = n_enc_stages,
				   n_convs_per_stage = n_convs_per_stage,
				   n_filters = n_filters,
				   res_connection=True
				   )
			
		elif CNN_type == '2D':
			
			dim=2
			self.cnn = UNet(dim,
				   n_ch_in = 40,
				   n_ch_out = 40,
				   n_enc_stages = n_enc_stages,
				   n_convs_per_stage = n_convs_per_stage,
				   n_filters = n_filters,
				   res_connection=True
				   )
			 
		#whether to pre-train or to fine-tune
		self.mode = mode
		self.CNN_type = CNN_type
		
		#FFT is an isometry
		self.normalized = True
		
		#the reg-parameter
		self.log_lambda_reg = nn.Parameter(torch.tensor(0.5),requires_grad=True)
			
	def apply_forward(self, x):
		
		"""
		FFT for k-space data acquisition 
		"""
		#permute axes
		x = x.permute(0, 4, 2, 3, 1)
		
		#apply fft
		k = torch.fft(x, 2, normalized=self.normalized)
		
		return k
	
	def apply_mask(self, k, mask):
		"""
		apply mask for simulating undersampling,
		k has to have shape (mb, Nt, Nx, Ny, 2)
		"""
		
		#permute mask
		mask = mask.permute(0, 4, 2, 3, 1)
		
		return mask * k
	
	def apply_adjoint(self, k):
		"""
		compute inverse fft
		"""
		x = torch.ifft(k, 2, normalized=self.normalized)
		x = x.permute(0, 4, 2, 3, 1)
		
		return x
	
	def data_consistency_layer(self, xcnn, k, mask):
		
		"""
		#yields the minimizer of the cnn-regularized functional 
		"""
		
		#permute axes to bring channels to last dimension
		xcnn = xcnn.permute(0, 4, 2, 3, 1)
		k = k.permute(0, 4, 2, 3, 1)
		mask = mask.permute(0, 4, 2, 3, 1)
			
		#estimated k-space data
		kcnn = torch.fft(xcnn, 2, normalized=self.normalized)
		
		kest = 1. / (1+ torch.exp(self.log_lambda_reg) ) * k + \
				(torch.exp(self.log_lambda_reg)/(1 + torch.exp(self.log_lambda_reg) ) ) * mask * kcnn + \
				(1- mask) * kcnn
		
	
		#apply IFFT
		x = torch.ifft(kest, 2, normalized=self.normalized)
		
		#re-permute
		x = x.permute(0, 4, 2, 3, 1)

		return x

	def forward(self, x, k, mask):
		
		"""
		input image x 	- shape (mb, 2, Nx, Ny, Nt)
		k-space data k 	- shape (mb, 2, Nx, Ny, Nt)
		binary mask 	- shape (mb, 2, Nx, Ny, Nt)
		"""
		
		#iterate several times, expect when pre-training
		if self.mode == 'pre_training':
			nu = 1
		elif self.mode in ['fine_tuning','testing']:
			nu = self.nu
		
		for ku in range(nu):
			print('iteration {} out of {}'.format(ku+1, nu))
			
			if self.CNN_type == '2D':
				mb,nch,nx,ny,nt = x.shape
				x = x.permute(0,4,1,2,3).reshape(mb,2*nt,nx,ny)
			
			zk = self.cnn(x)
			
			if self.CNN_type == '2D':
				zk = zk.reshape(1,2,nt,nx,ny).permute(0,1,3,4,2)
				x = x.reshape(1,2,nt,nx,ny).permute(0,1,3,4,2)
			
			x = self.data_consistency_layer(zk, k, mask)
			
		return x
