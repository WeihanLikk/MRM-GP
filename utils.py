import autograd.numpy as anp
import autograd.numpy.random as anpr
import numpy as np
from autograd.extend import primitive, defvjp
import scipy.linalg as sla
from autograd.scipy.signal import convolve
from scipy.linalg import block_diag
from tqdm import trange

def factorial(n):
    return anp.prod(range(1, n+1))

def complex_taylor_coefficient(sigma, mu, num_taylor):
    coeffs = []
    if num_taylor == 2:
        q = anp.sqrt(2*anp.pi/sigma) * 8 * anp.power(sigma, 2)
        coeffs.append(anp.power(mu, 4) + 4*sigma *
                      anp.power(mu, 2) + 8 * anp.power(sigma, 2))
        coeffs.append((-1j)*(-4*anp.power(mu, 3)-8*mu*sigma))
        coeffs.append((-1)*(6*anp.power(mu, 2) + 4*sigma))
        coeffs.append((1j)*(-4*mu))
        
    elif num_taylor == 3:
        q = anp.sqrt(2*anp.pi/sigma) * 48 * anp.power(sigma, 3)
        coeffs.append(anp.power(mu, 6) + 6*sigma*anp.power(mu, 4) + 24 *
                      anp.power(sigma, 2)*anp.power(mu, 2) + 48 * anp.power(sigma, 3))
        coeffs.append((-1j)*(-6*anp.power(mu, 5)-24*sigma *
                      anp.power(mu, 3)-48*anp.power(sigma, 2)*mu))
        coeffs.append((-1.0)*(15*anp.power(mu, 4)+36*sigma *
                      anp.power(mu, 2)+24*anp.power(sigma, 2)))
        coeffs.append((1j)*(-20*anp.power(mu, 3)-24*sigma*mu))
        coeffs.append((1.0)*15*anp.power(mu, 2)+6*sigma)
        coeffs.append((-1j)*(-6*mu))
    elif num_taylor == 4:
        q = anp.sqrt(2*anp.pi/sigma) * 384 * anp.power(sigma, 4)
        coeffs.append(anp.power(mu, 8)+8*sigma*anp.power(mu, 6)+48*anp.power(sigma, 2)*anp.power(mu, 4) +
                      192*anp.power(sigma, 3)*anp.power(mu, 2)+384*anp.power(sigma, 4))
        coeffs.append((-1j)*(-8*anp.power(mu, 7)-48*sigma*anp.power(mu, 5) -
                             192*anp.power(sigma, 2)*anp.power(mu, 3)-384*anp.power(sigma, 3)*mu))
        coeffs.append((-1.0)*(28*anp.power(mu, 6)+120*sigma*anp.power(mu, 4) +
                              288*anp.power(sigma, 2)*anp.power(mu, 2)+192*anp.power(sigma, 3)))
        coeffs.append((1j)*(-56*anp.power(mu, 5)-160*sigma *
                      anp.power(mu, 3)-192*anp.power(sigma, 2)*mu))
        coeffs.append((1.0)*(70*anp.power(mu, 4)+120*sigma *
                      anp.power(mu, 2)+48*anp.power(sigma, 2)))
        coeffs.append((-1j)*(-56*anp.power(mu, 3)-48*sigma*mu))
        coeffs.append((-1.0)*(28*anp.power(mu, 2)+8*sigma))
        coeffs.append((1j)*(-8*mu))
    else:
        raise NotImplementedError

    return anp.array(coeffs), q

def complex_approximation(sigma, mu, num_taylor):
    coeffs, q = complex_taylor_coefficient(sigma, mu, num_taylor)

    first_value = 1.0 if anp.mod(num_taylor, 2) == 0 else -1.0

    p = anp.append([first_value], coeffs[::-1])

    root = myroots(p)
    p_negative = mypoly(root[anp.where(anp.real(root) < 0)])

    return p_negative, q, root

def square_exp_approximation(sigma, mu, num_taylor):
    fn = factorial(num_taylor)
    coeffs = []
    for i in range(0, num_taylor):
        coeffs.append(fn * anp.power((2 * sigma), num_taylor-i)
                      * anp.power(-1, i) / factorial(i))
        coeffs.append(0.0)
    first_value = 1.0 if anp.mod(num_taylor, 2) == 0 else -1.0
    p = anp.append([first_value], coeffs[::-1])
    root = myroots(p)
    p_negative = mypoly(root[anp.where(anp.real(root) < 0)])
    q = anp.sqrt(2*anp.pi/sigma) * factorial(num_taylor) * \
        anp.power(2 * sigma, num_taylor)    

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
        blocks.append(np.ones((group_dims[i], group_dims[i])))
    block_mask = block_diag(*blocks)
    return block_mask

def mat2blocks(A, block_idx):
    num_blocks = len(block_idx)
    blocks = []
    for i in range(num_blocks):
        curr_block = block_idx[i]
        idx = np.array(range(curr_block[0], curr_block[1]))
        blocks.append(A[np.ix_(idx, idx)])
    return blocks

def em_pcca(y, T, num_groups, xdim_across, xdim_within, ydims, maxIters=1e3, tolLL=1e-5):
    ydim = np.sum(ydims)

    block_idxs = get_block_idxs(ydims)
    block_mask = create_block_mask(ydims)

    cY = np.cov(y)
    if np.linalg.matrix_rank(cY) == ydim:
        scale = np.exp(
            2 * np.sum(np.log(np.diag(np.linalg.cholesky(cY))))/ydim)
    else:
        r = np.linalg.matrix_rank(cY)
        e, _ = np.linalg.eig(cY)
        s = -np.sort(-e)
        s = s[0:r]
        scale = s.prod()**(1.0/len(s))

    C = np.random.randn(int(ydim), int(xdim_across)) * \
        np.sqrt(scale / xdim_across)

    Rs = []
    for i in range(num_groups):
        y_i = y[int(np.sum(ydims[0:i])): int(np.sum(ydims[0:i+1])), :]
        Rs.append(np.cov(y_i))
    R = block_diag(*Rs)
    d = np.mean(y, axis=1)

    I = np.eye(xdim_across)
    const = (-ydim / 2) * np.log(2 * np.pi)

    LLi = 0
    LL = []
    LLold = -np.inf
    print("Initialize by fitting pcca")
    max_ll = -np.inf
    max_C = None
    max_Rs = None
    for i in trange(int(maxIters)):
        iRs = []
        for j in range(num_groups):
            iRs.append(np.linalg.inv(Rs[j]))
        iR = block_diag(*iRs)
        iR = 0.5 * (iR + iR.T)
        iRC = iR @ C

        MM = iR - iRC @ np.linalg.pinv(I + C.T @ iRC) @ iRC.T

        beta = C.T @ MM

        cY_beta = cY @ beta.T
        Exx = I - beta @ C + beta @ cY_beta

        # calculate LL
        ldM = np.sum(np.log(np.diag(np.linalg.cholesky(MM+1e-3*np.eye(MM.shape[0])))))

        if LLi != 0:
            LLold = LLi
        LLi = T * const + T * ldM - 0.5 * T * np.sum(MM * cY)
        LL.append(LLi)

        C = np.linalg.lstsq(Exx.T, cY_beta.T,rcond=None)[0].T

        R = cY - cY_beta @ C.T
        R = 0.5 * (R + R.T)
        R = R * block_mask
        R = np.real(R)
        Rs = mat2blocks(R, block_idxs)

        if not np.isnan(LLi):
            if LLi > max_ll:
                max_ll = LLi
                max_C = C
                max_Rs = Rs

    C_across = []
    # ds = []
    for i in range(num_groups):
        cur_group = block_idxs[i]
        C_across.append(max_C[cur_group[0]:cur_group[1], :])
    Rs = max_Rs

    C_within = []
    if xdim_within[0] != 0:
        for i in range(num_groups):
            y_i = y[int(np.sum(ydims[0:i])): int(np.sum(ydims[0:i+1])), :]
            C_i = C_across[i]
            covY = np.cov(y_i)
            _, _, C_uncorr = np.linalg.svd(C_i.T @ covY)
            C_uncorr = C_uncorr[:, xdim_across:xdim_across + xdim_within[i]]
            C_within.append(C_uncorr)

    return C_across, C_within, d, Rs


def pcca_x(y, T, num_groups, xdim_across, xdim_within, ydims, num_trials, C_across, C_within, Rs, d):
    ydim = np.sum(ydims)

    C = np.concatenate(C_across, axis=0)

    y = np.reshape(y, (ydim, T, num_trials), order="F")
    x_latents_across = np.zeros((xdim_across, T, num_trials))
    for i in range(num_trials):
        y0 = y[:, :, i] - np.tile(d[:, None], T)
        I = np.eye(xdim_across)

        iRs = []
        for j in range(num_groups):
            iRs.append(np.linalg.inv(Rs[j]))
        iR = block_diag(*iRs)
        iR = 0.5 * (iR + iR.T)
        iRC = iR @ C
        MM = iR - iRC @ np.linalg.inv(I + C.T @ iRC) @ iRC.T

        beta = C.T @ MM

        x_latents_across[:, :, i] = beta @ y0

    x_latents = []
    if xdim_within[0] != 0:
        for i in range(num_groups):
            C_uncorr = C_within[i]
            y_i = y[int(np.sum(ydims[0:i])): int(np.sum(ydims[0:i+1])), :, :]
            y_i = np.reshape(y_i, (y_i.shape[0], T, num_trials), order="F")
            x_latents_within = np.zeros((xdim_within[i], T, num_trials))
            for j in range(num_trials):
                d = np.expand_dims(np.mean(y_i[:, :, j], axis=1), axis=1)
                y0 = y_i[:, :, j] - np.tile(d, T)
                I = np.eye(xdim_within[i])

                iR = np.linalg.inv(Rs[i])
                iR = 0.5 * (iR + iR.T)
                iRC = iR @ C_uncorr
                MM = iR - iRC @ np.linalg.inv(I + C_uncorr.T @ iRC) @ iRC.T

                beta = C_uncorr.T @ MM
                x_latents_within[:, :, j] = beta @ y0
            x_latents.append(np.concatenate(
                (x_latents_across, x_latents_within), axis=0))
        x_latents = np.concatenate(x_latents, axis=0)

        return x_latents, x_latents_across
    else:
        return x_latents_across, x_latents_across


@primitive
def myexpm(x):
    return sla.expm(x)


def myexpm_vjp(g, ans, x):
    return sla.expm_frechet(x.T, g, compute_expm=False)


defvjp(myexpm, lambda ans, x: lambda g: myexpm_vjp(g, ans, x))

def myroots(p):
    # find non-zero array entries
    non_zero = anp.nonzero(anp.ravel(p))[0]

    # Return an empty array if polynomial is all zeros
    if len(non_zero) == 0:
        return anp.array([])

    # find the number of trailing zeros -- this is the number of roots at 0.
    trailing_zeros = len(p) - non_zero[-1] - 1

    # strip leading and trailing zeros
    p = p[int(non_zero[0]):int(non_zero[-1])+1]

    # casting: if incoming array isn't floating point, make it floating point.
    if not issubclass(p.dtype.type, (anp.floating, anp.complexfloating)):
        p = p.astype(float)

    N = len(p)
    if N > 1:
        # build companion matrix and find its eigenvalues (the roots)
        A = anp.diag(anp.ones((N-2,), p.dtype), -1)
        # A[0,:] = -p[1:] / p[0]
        first_row = -p[1:] / p[0]
        A = anp.vstack((first_row[None, :], A[1:, :]))
        roots, _ = anp.linalg.eig(A)
    else:
        roots = anp.array([])

    # tack any zeros onto the back of the array
    roots = anp.hstack((roots, anp.zeros(trailing_zeros, roots.dtype)))
    return roots


def mypoly(roots):
    sh = roots.shape

    if len(sh) == 2 and sh[0] == sh[1] and sh[0] != 0:
        roots, _ = anp.linalg.eig(roots)
    elif len(sh) == 1:
        dt = roots.dtype
        if dt != object:
            roots = roots.astype(anp.mintypecode(dt.char))
    else:
        raise ValueError("input must be 1d or non-empty square 2d array.")

    if len(roots) == 0:
        return 1.0
    dt = roots.dtype
    a = anp.ones((1,), dtype=dt)
    for root in roots:
        a = convolve(a, anp.array([1, -root], dtype=dt), mode='full')

    return a

def block_diag(*arrs):
    acc = arrs[0]
    for a in arrs[1:]:
        _, c = a.shape
        a = anp.pad(a, ((0, 0), (acc.shape[-1], 0)),
                    'constant', constant_values=0.0)
        acc = anp.pad(acc, ((0, 0), (0, c)), 'constant', constant_values=0.0)
        acc = anp.concatenate((acc, a), axis=0)
    return acc


def softplus(x, beta=1.0):
    if type(x) == list:
        return [1.0/beta * anp.log(1 + anp.exp(beta * xi)) for xi in x]
    else:
        return 1.0/beta * anp.log(1 + anp.exp(beta * x))


def reverse_softplus(x, beta=1.0):
    if type(x) == list:
        return [1.0/beta * anp.log(-1 + anp.exp(beta * xi)) for xi in x]
    else:
        return 1.0/beta * anp.log(-1 + anp.exp(beta * x))


def svdsolve(A):
    u, s, v = anp.linalg.svd(A)
    Ainv = anp.dot(v.transpose(), anp.dot(anp.diag(s**-1), u.transpose()))
    return Ainv


def min_max_normalize(x, axis=0):
    return 2*(x - anp.min(x, axis)) / (anp.max(x, axis) - anp.min(x, axis)) - 1


def inv_cholesky(x):
    L = anp.linalg.cholesky(x)
    inv_L = anp.linalg.inv(L)
    return inv_L.T @ inv_L

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return anp.allclose(a, a.T, rtol=rtol, atol=atol)
