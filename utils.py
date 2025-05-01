import autograd.numpy as anp
import autograd.numpy.random as anpr
import numpy as np
from autograd.extend import primitive, defvjp
import scipy.linalg as sla
from autograd.scipy.signal import convolve
from scipy.linalg import block_diag
from tqdm import trange
import torch
import scipy.linalg as sla
from torch.linalg import inv, eig, cholesky, svd
from torch import matmul, eye, ones, diag, cat, reshape, nonzero, sum, hstack, eye

def factorial(n):
    return torch.prod(torch.arange(1, n+1, dtype=torch.float32))

def complex_taylor_coefficient(sigma, mu, num_taylor):
    coeffs = []
    if num_taylor == 2:
        q = torch.sqrt(2 * torch.pi / sigma) * 8 * torch.pow(sigma, 2)
        coeffs.append(torch.pow(mu, 4) + 4 * sigma * torch.pow(mu, 2) + 8 * torch.pow(sigma, 2))
        coeffs.append((-1j) * (-4 * torch.pow(mu, 3) - 8 * mu * sigma))
        coeffs.append((-1) * (6 * torch.pow(mu, 2) + 4 * sigma))
        coeffs.append((1j) * (-4 * mu))
        
    elif num_taylor == 3:
        q = torch.sqrt(2 * torch.pi / sigma) * 48 * torch.pow(sigma, 3)
        coeffs.append(torch.pow(mu, 6) + 6 * sigma * torch.pow(mu, 4) + 24 * torch.pow(sigma, 2) * torch.pow(mu, 2) + 48 * torch.pow(sigma, 3))
        coeffs.append((-1j) * (-6 * torch.pow(mu, 5) - 24 * sigma * torch.pow(mu, 3) - 48 * torch.pow(sigma, 2) * mu))
        coeffs.append((-1.0) * (15 * torch.pow(mu, 4) + 36 * sigma * torch.pow(mu, 2) + 24 * torch.pow(sigma, 2)))
        coeffs.append((1j) * (-20 * torch.pow(mu, 3) - 24 * sigma * mu))
        coeffs.append((1.0) * 15 * torch.pow(mu, 2) + 6 * sigma)
        coeffs.append((-1j) * (-6 * mu))
    elif num_taylor == 4:
        q = torch.sqrt(2 * torch.pi / sigma) * 384 * torch.pow(sigma, 4)
        coeffs.append(torch.pow(mu, 8) + 8 * sigma * torch.pow(mu, 6) + 48 * torch.pow(sigma, 2) * torch.pow(mu, 4) +
                      192 * torch.pow(sigma, 3) * torch.pow(mu, 2) + 384 * torch.pow(sigma, 4))
        coeffs.append((-1j) * (-8 * torch.pow(mu, 7) - 48 * sigma * torch.pow(mu, 5) -
                             192 * torch.pow(sigma, 2) * torch.pow(mu, 3) - 384 * torch.pow(sigma, 3) * mu))
        coeffs.append((-1.0) * (28 * torch.pow(mu, 6) + 120 * sigma * torch.pow(mu, 4) +
                              288 * torch.pow(sigma, 2) * torch.pow(mu, 2) + 192 * torch.pow(sigma, 3)))
        coeffs.append((1j) * (-56 * torch.pow(mu, 5) - 160 * sigma * torch.pow(mu, 3) - 192 * torch.pow(sigma, 2) * mu))
        coeffs.append((1.0) * (70 * torch.pow(mu, 4) + 120 * sigma * torch.pow(mu, 2) + 48 * torch.pow(sigma, 2)))
        coeffs.append((-1j) * (-56 * torch.pow(mu, 3) - 48 * sigma * mu))
        coeffs.append((-1.0) * (28 * torch.pow(mu, 2) + 8 * sigma))
        coeffs.append((1j) * (-8 * mu))
    else:
        raise NotImplementedError

    return torch.tensor(coeffs, dtype=torch.complex64), q

def complex_approximation(sigma, mu, num_taylor):
    coeffs, q = complex_taylor_coefficient(sigma, mu, num_taylor)

    first_value = 1.0 if num_taylor % 2 == 0 else -1.0

    p = torch.cat([torch.tensor([first_value]), coeffs.flip(0)])

    root = myroots(p)  # Assuming myroots is implemented
    p_negative = mypoly(root[torch.where(torch.real(root) < 0)])

    return p_negative, q, root

def square_exp_approximation(sigma, mu, num_taylor):
    fn = factorial(num_taylor)
    coeffs = []
    for i in range(0, num_taylor):
        coeffs.append(fn * torch.pow((2 * sigma), num_taylor - i) * torch.pow(-1, i) / factorial(i))
        coeffs.append(0.0)
    first_value = 1.0 if num_taylor % 2 == 0 else -1.0
    p = torch.cat([torch.tensor([first_value]), coeffs[::-1]])
    root = myroots(p)  # Assuming myroots is implemented
    p_negative = mypoly(root[torch.where(torch.real(root) < 0)])
    q = torch.sqrt(2 * torch.pi / sigma) * factorial(num_taylor) * torch.pow(2 * sigma, num_taylor)    

    return p_negative, q

def get_block_idxs(group_dims):
    num_groups = len(group_dims)
    block_idxs = []
    startIdx = 0
    for i in range(num_groups):
        group_dim = group_dims[i]
        endIdx = startIdx + group_dim
        block_idxs.append([startIdx, endIdx])
        startIdx = endIdx
    return block_idxs

def create_block_mask(group_dims):
    num_groups = len(group_dims)
    blocks = []
    for i in range(num_groups):
        blocks.append(torch.ones((group_dims[i], group_dims[i])))
    block_mask = block_diag(*blocks)
    return block_mask

def mat2blocks(A, block_idx):
    num_blocks = len(block_idx)
    blocks = []
    for i in range(num_blocks):
        curr_block = block_idx[i]
        idx = torch.arange(curr_block[0], curr_block[1])
        blocks.append(A[idx[:, None], idx[None, :]])
    return blocks

def em_pcca(y, T, num_groups, xdim_across, xdim_within, ydims, maxIters=1e3, tolLL=1e-5):
    ydim = torch.sum(torch.tensor(ydims))

    block_idxs = get_block_idxs(ydims)
    block_mask = create_block_mask(ydims)

    cY = torch.cov(y)
    if torch.linalg.matrix_rank(cY) == ydim:
        scale = torch.exp(2 * torch.sum(torch.log(torch.diagonal(torch.linalg.cholesky(cY)))) / ydim)
    else:
        r = torch.linalg.matrix_rank(cY)
        e, _ = torch.linalg.eig(cY)
        s = -torch.sort(-e).values
        s = s[:r]
        scale = torch.prod(s)**(1.0 / len(s))

    C = torch.randn(int(ydim), int(xdim_across)) * torch.sqrt(scale / xdim_across)

    Rs = []
    for i in range(num_groups):
        y_i = y[int(torch.sum(torch.tensor(ydims[0:i]))): int(torch.sum(torch.tensor(ydims[0:i+1]))), :]
        Rs.append(torch.cov(y_i))
    R = block_diag(*Rs)
    d = torch.mean(y, axis=1)

    I = torch.eye(xdim_across)
    const = (-ydim / 2) * torch.log(2 * torch.tensor(torch.pi))

    LLi = 0
    LL = []
    LLold = -torch.inf
    print("Initialize by fitting pcca")
    max_ll = -torch.inf
    max_C = None
    max_Rs = None
    for i in range(int(maxIters)):
        iRs = []
        for j in range(num_groups):
            if torch.linalg.cond(Rs[j]) > 1e5:  # A very large condition number indicates singularity
                # Add small regularization to Rs[j] if it's singular
                Rs[j] = Rs[j] + 1e-3 * torch.eye(Rs[j].shape[0], device=Rs[j].device)
            # Compute the inverse or pseudoinverse
            iRs.append(torch.linalg.pinv(Rs[j]))
        iR = block_diag(*iRs)
        iR = 0.5 * (iR + iR.T)
        iRC = iR @ C

        MM = iR - iRC @ torch.linalg.pinv(I + C.T @ iRC) @ iRC.T

        beta = C.T @ MM

        cY_beta = cY @ beta.T
        Exx = I - beta @ C + beta @ cY_beta

        # calculate LL
        regularization = 1e-3
        MM_reg = MM + regularization * torch.eye(MM.shape[0], device=MM.device)

        # Use SVD for a more numerically stable calculation of the log determinant
        u, s, v = torch.svd(MM_reg)
        ldM = torch.sum(torch.log(s))

        if LLi != 0:
            LLold = LLi
        LLi = T * const + T * ldM - 0.5 * T * torch.sum(MM * cY)
        LL.append(LLi)

        C = torch.linalg.lstsq(Exx.T, cY_beta.T)[0].T

        R = cY - cY_beta @ C.T
        R = 0.5 * (R + R.T)
        R = R * block_mask
        R = torch.real(R)
        Rs = mat2blocks(R, block_idxs)

        if not torch.isnan(LLi):
            if LLi > max_ll:
                max_ll = LLi
                max_C = C
                max_Rs = Rs

    C_across = []
    for i in range(num_groups):
        cur_group = block_idxs[i]
        C_across.append(max_C[cur_group[0]:cur_group[1], :])
    Rs = max_Rs

    C_within = []
    if xdim_within[0] != 0:
        for i in range(num_groups):
            y_i = y[int(torch.sum(torch.tensor(ydims[0:i]))): int(torch.sum(torch.tensor(ydims[0:i+1]))), :]
            C_i = C_across[i]
            covY = torch.cov(y_i)
            _, _, C_uncorr = torch.svd(C_i.T @ covY)
            C_uncorr = C_uncorr[:, xdim_across:xdim_across + xdim_within[i]]
            C_within.append(C_uncorr)

    return C_across, C_within, d, Rs

def pcca_x(y, T, num_groups, xdim_across, xdim_within, ydims, num_trials, C_across, C_within, Rs, d):
    ydim = torch.sum(torch.tensor(ydims))

    C = cat(C_across, dim=0)

    y = reshape(y, (ydim, T, num_trials))
    x_latents_across = torch.zeros((xdim_across, T, num_trials))
    
    for i in range(num_trials):
        y0 = y[:, :, i] - d[:, None].expand(-1, T)
        I = eye(xdim_across)

        iRs = []
        for j in range(num_groups):
            if torch.linalg.cond(Rs[j]) > 1e5:  # A very large condition number indicates singularity
                # Add small regularization to Rs[j] if it's singular
                Rs[j] = Rs[j] + 1e-3 * torch.eye(Rs[j].shape[0], device=Rs[j].device)
            # Compute the inverse or pseudoinverse
            iRs.append(torch.linalg.pinv(Rs[j]))
        iR = block_diag(*iRs)
        iR = 0.5 * (iR + iR.T)
        iRC = matmul(iR, C)
        MM = iR - matmul(iRC, inv(I + matmul(C.T, iRC))) @ iRC.T

        beta = matmul(C.T, MM)

        x_latents_across[:, :, i] = matmul(beta, y0)

    x_latents = []
    if xdim_within[0] != 0:
        for i in range(num_groups):
            C_uncorr = C_within[i]
            y_i = y[int(torch.sum(torch.tensor(ydims[0:i]))): int(torch.sum(torch.tensor(ydims[0:i+1]))), :, :]
            y_i = reshape(y_i, (y_i.shape[0], T, num_trials))
            x_latents_within = torch.zeros((xdim_within[i], T, num_trials))
            for j in range(num_trials):
                d_i = torch.mean(y_i[:, :, j], axis=1, keepdim=True)
                y0 = y_i[:, :, j] - d_i.expand(-1, T)
                I = eye(xdim_within[i])

                iR = inv(Rs[i])
                iR = 0.5 * (iR + iR.T)
                iRC = matmul(iR, C_uncorr)
                # Ensure the dimensions of the identity matrix I match the expected size
                I = eye(C_uncorr.shape[1], device=y.device)  # Identity matrix of shape (q, q)

                # Compute the intermediate result, ensuring compatible shapes
                iRC = matmul(iR, C_uncorr)  # iRC should have shape (n, q)

                # Ensure matmul(C_uncorr.T, iRC) results in a matrix of shape (q, q)
                # Regularize the matrix and compute the inverse term
                regularization = 1e-3
                regularized_matrix = I + matmul(C_uncorr.T, iRC) + regularization * torch.eye(C_uncorr.shape[1], device=y.device)

                # Ensure this matrix is invertible, and compute the inverse term
                inv_term = inv(regularized_matrix)

                # Compute MM using the inverse term
                MM = iR - matmul(iRC, matmul(inv_term, iRC.T))


                beta = matmul(C_uncorr.T, MM)
                x_latents_within[:, :, j] = matmul(beta, y0)
            x_latents.append(cat((x_latents_across, x_latents_within), dim=0))
        x_latents = cat(x_latents, dim=0)

        return x_latents, x_latents_across
    else:
        return x_latents_across, x_latents_across


def myexpm(x):
    return sla.expm(x)

def myexpm_vjp(g, ans, x):
    return sla.expm_frechet(x.T, g, compute_expm=False)


def myroots(p):
    # find non-zero array entries
    non_zero = nonzero(torch.ravel(p))[0]

    # Return an empty array if polynomial is all zeros
    if len(non_zero) == 0:
        return torch.tensor([])

    # find the number of trailing zeros -- this is the number of roots at 0.
    trailing_zeros = len(p) - non_zero[-1] - 1

    # strip leading and trailing zeros
    p = p[int(non_zero[0]):int(non_zero[-1])+1]

    # casting: if incoming array isn't floating point, make it floating point.
    if not torch.is_floating_point(p):
        p = p.float()

    N = len(p)
    if N > 1:
        # build companion matrix and find its eigenvalues (the roots)
        A = torch.diag(torch.ones((N-2,), p.dtype), -1)
        first_row = -p[1:] / p[0]
        A = torch.vstack((first_row[None, :], A[1:, :]))
        roots, _ = eig(A)
    else:
        roots = torch.tensor([])

    # tack any zeros onto the back of the array
    roots = hstack((roots, torch.zeros(trailing_zeros, roots.dtype)))
    return roots


def mypoly(roots):
    sh = roots.shape

    if len(sh) == 2 and sh[0] == sh[1] and sh[0] != 0:
        roots, _ = eig(roots)
    elif len(sh) == 1:
        dt = roots.dtype
        if dt != object:
            roots = roots.to(torch.mintypecode(dt.char))
    else:
        raise ValueError("input must be 1d or non-empty square 2d array.")

    if len(roots) == 0:
        return 1.0
    dt = roots.dtype
    a = torch.ones((1,), dtype=dt)
    for root in roots:
        a = torch.conv1d(a, torch.tensor([1, -root], dtype=dt), padding='same')

    return a


def block_diag(*arrs):
    acc = arrs[0]
    for a in arrs[1:]:
        _, c = a.shape
        a = torch.nn.functional.pad(a, (0, acc.shape[1]), "constant", value=0.0)
        acc = torch.nn.functional.pad(acc, (0, c), "constant", value=0.0)
        acc = torch.cat((acc, a), dim=0)
    return acc


def softplus(x, beta=1.0):
    if isinstance(x, list):
        return [1.0/beta * torch.log(1 + torch.exp(beta * xi)) for xi in x]
    else:
        return 1.0/beta * torch.log(1 + torch.exp(beta * x))


def reverse_softplus(x, beta=1.0):
    if isinstance(x, list):
        return [1.0/beta * torch.log(-1 + torch.exp(beta * xi)) for xi in x]
    else:
        return 1.0/beta * torch.log(-1 + torch.exp(beta * x))


def svdsolve(A):
    u, s, v = svd(A)
    Ainv = matmul(v.T, matmul(diag(s**-1), u.T))
    return Ainv


def min_max_normalize(x, axis=0):
    return 2 * (x - torch.min(x, axis)) / (torch.max(x, axis) - torch.min(x, axis)) - 1


def inv_cholesky(x):
    L = cholesky(x)
    inv_L = inv(L)
    return matmul(inv_L.T, inv_L)


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return torch.allclose(a, a.T, rtol=rtol, atol=atol)

