# TGV denoising using the primal-dual method of Chambolle-Pock


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg
import scipy as sp
#from scipy.sparse import diags, hstack, vstack, identity
#from scipy.sparse.linalg import spsolve
#from scipy import sparse
#from scipy.linalg import norm

def psnr(u1,u2):

    (n,m)=u1.shape
    mse=(1/(n*m))*(np.sum((u1-u2)**2))
    mpsnr=10*np.log10(1/mse)

    return mpsnr


def reproject(p, alpha):
    y1 = np.zeros(p.shape)
    (k, n, m) = y1.shape

    if k == 2:
        y1[0, :, :] = p[0, :, :] / np.maximum(np.sqrt(p[0, :, :] ** 2 + p[1, :, :] ** 2) / alpha, 1)
        y1[1, :, :] = p[1, :, :] / np.maximum(np.sqrt(p[0, :, :] ** 2 + p[1, :, :] ** 2) / alpha, 1)
    elif k == 3:
        y1[0, :, :] = p[0, :, :] / np.maximum(np.sqrt(p[0, :, :] ** 2 + p[1, :, :] ** 2 + 2 * p[2, :, :] ** 2) / alpha,
                                              1)
        y1[1, :, :] = p[1, :, :] / np.maximum(np.sqrt(p[0, :, :] ** 2 + p[1, :, :] ** 2 + 2 * p[2, :, :] ** 2) / alpha,
                                              1)
        y1[2, :, :] = p[2, :, :] / np.maximum(np.sqrt(p[0, :, :] ** 2 + p[1, :, :] ** 2 + 2 * p[2, :, :] ** 2) / alpha,
                                              1)

    return y1

def dxm(u):
    dx=np.zeros(u.shape)
    (n,m)=dx.shape
    dx=np.append( u[:,0:m-1], np.zeros((n,1)), axis=1) - np.append( np.zeros((n,1)), u[:,0:m-1], axis=1  )
    return dx

def dym(u):
    dy=np.zeros(u.shape)
    (n,m)=dy.shape
    dy=np.append( u[0:n-1,:], np.zeros((1,m)), axis=0) - np.append( np.zeros((1,m)), u[0:n-1,:], axis=0  )
    return dy

def dxp(u):
    dx=np.zeros(u.shape)
    (n,m)=dx.shape
    dx=np.append( u[:,1:m], u[:,[m-1]], axis=1) - u
    return dx

def dyp(u):
    dx=np.zeros(u.shape)
    (n,m)=dx.shape
    dx=np.append( u[1:n,:], u[[n-1],:], axis=0) - u
    return dx


#mat_contents = sio.loadmat('parrot')
#clean=mat_contents['parrot']
#f=mat_contents['parrot_noisy_01']

def function_TGV_denoising_CP(f, clean, alpha, beta, iter):

    (n,m) = f.shape


    alpha=alpha*np.ones((n,m))
    beta=beta*np.ones((n,m))
    alpha_w=0.01
    max_its=iter

    u=f.copy()
    uold=np.zeros((n,m))
    w=np.zeros((2, n, m))
    wold=np.zeros((2, n, m))
    um=np.zeros((2, n, m))
    p=np.zeros((2, n, m))
    q=np.zeros((3, n, m))
    b=np.zeros((3, n, m))

    L=np.sqrt(12)
    sigma=1/L
    tau=1/L
    gamma=0.7*alpha_w

    for k in range(0,max_its):

        uold=u.copy()
        wold=w.copy()

        div_p = dxm(p[0,:,:]) + dym(p[1,:,:])
        u =  (u+tau*(f+div_p))/(1+tau)

        div_q1 = dxm(q[0,:,:]) + dym(q[2,:,:])
        div_q2 = dxm(q[2, :, :]) + dym(q[1, :, :])
        w[0,:,:] = w[0,:,:] + tau*(p[0,:,:] + div_q1)
        w[1,:,:] = w[1,:,:] + tau*(p[1,:,:] + div_q2)
        w = w/(1+tau*alpha_w)

        theta=1/np.sqrt(1+2*gamma*tau)
        tau=theta*tau
        sigma=sigma/theta

        wold = w + theta * (w - wold)
        uold = u + theta * (u - uold)

        um[0,:,:] = dxp(uold) - wold[0,:,:]
        um[1,:,:] = dyp(uold) - wold[1,:,:]
        p=reproject(p+sigma*um,alpha)

        b[0,:,:]=dxp(wold[0,:,:])
        b[1,:,:]=dyp(wold[1,:,:])
        b[2,:,:] = 0.5*(dyp(wold[0, :, :]) + dxp(wold[1, :, :]))
        q = reproject(q + sigma * b, beta)


        if np.mod(k+1,100)==0:

            print('Iteration ', k+1, ': The PSNR is', "{:.2f}".format(psnr(u,clean)))
            #plt.pause(0.05)

    plt.figure(figsize=(7, 7))
    imgplot2 = plt.imshow(u)
    imgplot2.set_cmap('gray')


    return u

def P_a_Huber(v1,v2,a, gam, sig):

    normv_over_a=(1/(1+sig*(gam/a)))*np.sqrt(v1**2 + v2**2)
    denom= np.maximum(1,normv_over_a/a)

    v1_star=((1/(1+sig*(gam/a)))*v1)/denom;
    v2_star=((1/(1+sig*(gam/a)))*v2)/denom;

    return v1_star, v2_star


def function_HuberTV_denoising_CP(f,clean, alpha, gamma, iter):
    (n, m) = f.shape

    alpha = alpha * np.ones((n, m))
    max_its = iter

    L2=8

    uold=f.copy()
    u = np.zeros((n, m))
    ubar_old=uold.copy()
    ubar=np.zeros((n, m))
    p1_old=np.zeros((n, m))
    p1=np.zeros((n, m))
    p2_old=np.zeros((n, m))
    p2=np.zeros((n, m))

    for k in range(1, max_its+1):

        tau=1/(k+1)
        sigma=(1/tau)/L2

        p1=p1_old + sigma * dxp(ubar_old)
        p2=p2_old + sigma * dyp(ubar_old)

        (p1,p2)= P_a_Huber(p1,p2,alpha, gamma, sigma)

        u = (1/(1+tau))*(uold + tau*(dxm(p1)+dym(p2)+f))

        ubar=2*u-uold

        uold=u
        ubar_old=ubar
        p1_old=p1
        p2_old=p2

        if np.mod(k + 1, 100) == 0:
            print('Iteration ', k + 1, ': The PSNR is', "{:.2f}".format(psnr(u, clean)))

    plt.figure(figsize=(7, 7))
    imgplot2 = plt.imshow(u)
    imgplot2.set_cmap('gray')

    return u

def normalize(image):
    """
    Gets a grey valued image $x = (x_{ij})_{i,j}$ wich pixel values in [a,b] = [ \min_{ij} x_{ij}, \max_{ij} x_{ij}]
    """
    #image = image - np.min(image)
    image = image / np.max(image)
    return image

def subsampling(image,mask):
    """
    Gets an image and a binary mask with values in [0,1] and gives back the image values at the location where the
    mask is one.
    """
    indexes = mask > 0.5
    return image[indexes]

def subsampling_transposed(data_vec,mask):
    """
    Transposed of the subsampling operator
    """
    indexes = mask>0.5
    height, width = mask.shape

    M = height
    N = width
    result = np.zeros((M,N),dtype = complex)
    result[indexes] = data_vec
    return result


def compute_differential_operators(N, M, h):
    """
    This method creates sparse representation matrices of forward finite differences
    as in the book by christian bredies[Bredies20]
    """
    # compute D_y as a sparse NM x NM matrix
    one_vec = np.ones((M - 1) * N)
    diag = np.array(one_vec)
    offsets = np.array([0])
    sp_help_matr_1 = - sparse.dia_matrix((diag, offsets), shape=(N * M, N * M))
    sp_help_matr_2 = sparse.eye(N * M, k=N)
    D_x_sparse = (1 / h) * (sp_help_matr_1 + sp_help_matr_2)

    # compute D_x as a sparse NM x NM  matrix
    E_M = sparse.eye(M)
    one_vec = - np.ones(N - 1)
    offsets = np.array([0])
    sp_matr_help_1 = sparse.dia_matrix((one_vec, offsets), shape=(N, N))
    sp_matr_help_2 = sparse.eye(N, k=1)
    sp_matr_help_3 = sp_matr_help_1 + sp_matr_help_2
    D_y_sparse = (1 / h) * sparse.kron(E_M, sp_matr_help_3)

    # compute gradient as a sparse NM x 2NM
    grad = sparse.vstack([D_x_sparse, D_y_sparse])

    # compute divergence as a sparse 2NM x NM
    sparse_div_x = - D_x_sparse.transpose()
    sparse_div_y = - D_y_sparse.transpose()
    div = sparse.hstack([sparse_div_x, sparse_div_y])

    return D_x_sparse, D_y_sparse, grad, div

def reprojectmri(y,alpha):
    (N,M,p) = np.shape(y)
    y1 = np.zeros((N,M,p))
    if p == 2:
        reproject = np.maximum(1.0, np.sqrt(np.absolute(y[:,:,0])**2 + np.absolute(y[:,:,1])**2)/alpha)
        y1[:,:,0] = y[:,:,0]/ reproject
        y1[:,:,1] = y[:,:,1]/ reproject
    elif p == 3:
        reproject = (np.maximum(1.0, np.sqrt(np.absolute(y[:,:,0])**2 +
                                            np.absolute(y[:,:,1])**2 + 2*np.absolute(y[:,:,2])**2)/alpha))
        y1[:,:,0] = y[:,:,0]/ reproject
        y1[:,:,1] = y[:,:,1]/ reproject
        y1[:,:,2] = y[:,:,2]/ reproject
    return y1

def function_TV_MRI_CP(data, orig, mask, x_0, tau, sigma, h, max_it, tol, alpha):
    """ Chambolle pock algorithm for TV MRI reconstruction as in the book by kristian Bredies
    """
    convergence = np.zeros(max_it)
    (N, M) = np.shape(orig)

    # initialization
    x = x_0
    x_bar = np.zeros((N, M))
    y = data
    z = np.zeros((N, M, 2))
    tens = np.zeros((N, M, 2))
    D_x_sparse, D_y_sparse, grad, div = compute_differential_operators(N, M, h)
    div_x = - D_x_sparse.transpose()
    div_y = - D_y_sparse.transpose()
    i = 0
    while i < max_it:
        x_old = x
        # help_vec_1 = np.real(uifft2(subsampling_transposed(y,mask)))
        help_vec_1 = np.real(np.fft.ifft2(subsampling_transposed(y, mask), norm='ortho'))
        z_1 = z[:, :, 0]
        z_2 = z[:, :, 1]
        help_vec = div_x.dot(z_1.ravel()) + div_y.dot(z_2.ravel()) - help_vec_1.ravel()
        help_mat = tau * np.reshape(help_vec, (N, M))
        x = x_old + help_mat

        x_bar = 2 * x - x_old
        y = (y + sigma * (subsampling(np.fft.fft2(x_bar, norm='ortho'), mask) - data)) / (1 + sigma)

        help_vec2 = grad.dot(x_bar.ravel())
        tens[:, :, 0] = np.reshape(help_vec2[0:N * M], (N, M))
        tens[:, :, 1] = np.reshape(help_vec2[N * M:2 * N * M], (N, M))

        w = z + sigma * tens
        z = reprojectmri(w, alpha)
        convergence[i] = np.linalg.norm(x - x_old)

       # if np.mod(i, 100) == 0:
       #     print("Current Iterate: " + str(i))
       #     print("Current stepize: " + str(convergence[i]))
       #     # plt.imshow(np.hstack([x,x_0]),cmap='gray')
       #     # plt.show()

        if np.mod(i + 1, 100) == 0:
            print('Iteration ', i + 1, ': The PSNR is', "{:.2f}".format(psnr(x, orig)))

        if convergence[i] < tol:
            break

        i = i + 1

    return x


def function_TGV_MRI_CP(data, orig, mask, x_0, tau, sigma, lambda_prox, h, max_it, tol, alpha_0, alpha_1):
    convergence = np.zeros(max_it)
    (N, M) = np.shape(x_0);
    # ititialization
    x = np.zeros((N, M));
    x = x_0
    x_bar = np.zeros((N, M))
    w = np.zeros((N, M, 2))
    w_bar = np.zeros((N, M, 2))
    p = np.zeros((N, M, 2))
    q = np.zeros((N, M, 3))
    r = np.zeros(np.shape(data), dtype=complex)

    tens = np.zeros((N, M, 2))
    tens2 = np.zeros((N, M, 3))
    div_q = tens

    D_x_sparse, D_y_sparse, grad, div = compute_differential_operators(N, M, h)
    div_x = - D_x_sparse.transpose()
    div_y = - D_y_sparse.transpose()

    i = 0
    while i < max_it:

        tens[:, :, 0] = p[:, :, 0] + sigma * (np.reshape(D_x_sparse.dot(np.ravel(x_bar)), (N, M)) - w_bar[:, :, 0])
        tens[:, :, 1] = p[:, :, 1] + sigma * (np.reshape(D_y_sparse.dot(np.ravel(x_bar)), (N, M)) - w_bar[:, :, 1])
        p = reprojectmri(tens, alpha_1);

        w_bar1 = w_bar[:, :, 0]
        w_bar2 = w_bar[:, :, 1]
        tens2[:, :, 0] = q[:, :, 0] + sigma * np.reshape(D_x_sparse.dot(np.ravel(w_bar1)), (N, M))
        tens2[:, :, 1] = q[:, :, 1] + sigma * np.reshape(D_y_sparse.dot(np.ravel(w_bar2)), (N, M))
        tens2[:, :, 2] = (q[:, :, 2] + 0.5 * sigma * np.reshape(D_x_sparse.dot(np.ravel(w_bar2)), (N, M))
                          + 0.5 * sigma * np.reshape(D_y_sparse.dot(np.ravel(w_bar1)), (N, M)))

        q = reprojectmri(tens2, alpha_0)

        mat = r + sigma * (subsampling(np.fft.fft2(x_bar, norm='ortho'), mask) - data)
        r = mat / (sigma * lambda_prox + 1)

        x_old = x

        p1 = p[:, :, 0]
        p2 = p[:, :, 1]
        vec = div_x.dot(np.ravel(p1)) + div_y.dot(np.ravel(p2))
        vec = np.reshape(vec, (N, M))
        vec = vec - np.real(np.fft.ifft2(subsampling_transposed(r, mask), norm='ortho'))
        x = x + tau * vec

        x_bar = 2 * x - x_old
        w_old = w

        q1 = q[:, :, 0]
        q2 = q[:, :, 1]
        q3 = q[:, :, 2]

        div_q[:, :, 0] = np.reshape(div_x.dot(np.ravel(q1)) + div_y.dot(np.ravel(q3)), (N, M))
        div_q[:, :, 1] = np.reshape(div_y.dot(np.ravel(q2)) + div_x.dot(np.ravel(q3)), (N, M))
        w = tau * (p + div_q) + w
        w_bar = 2 * w - w_old

        convergence[i] = np.linalg.norm(x - x_old)
        #if np.mod(i, 50) == 0:
        #    print("Current Iterate: " + str(i))
        #    print("Current stepsize: " + str(convergence[i]))
        #    # plt.imshow(np.hstack([x,x_0]),cmap='gray')
        #    # plt.show()

        if np.mod(i + 1, 100) == 0:
            print('Iteration ', i + 1, ': The PSNR is', "{:.2f}".format(psnr(x, orig)))

        if convergence[i] < tol:
            break

        i = i + 1
    return x