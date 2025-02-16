import autograd.numpy as anp
import autograd.numpy.random as anpr
from utils import myexpm, block_diag, complex_approximation
from autograd.scipy.linalg import solve_sylvester
import ssm.stats as stats
from mrmgp.optimizers import adam, bfgs, rmsprop, sgd, lbfgs

import torch
from torch import nn 
from torch.nn.functional import normalize

class Kernel():
    def __init__(self, K, D, M=0):
        self.K, self.D, self.M = K, D, M


class SpectralKernelDynamics(Kernel):
    def __init__(self, K, D, x_across, x_within, num_groups, num_derivatives, num_dims, num_times, M=0, lags=1, dtype=anp.float64):
        super(SpectralKernelDynamics, self).__init__(K, D, M)
        # kernel parameters
        self.Rq = 2
        self.sigma_across = 0.05 * torch.ones((K, x_across), dtype=dtype)
        self.constant_across = (torch.rand((K, x_across, 2, self.Rq), dtype=dtype) * 1.5) + 0.5
        self.sigma_within = []
        self.constant_within = []
        self.mus_within = []
        for i in range(num_groups):
            self.sigma_within.append(0.05 * torch.ones((1, x_within[i]), dtype=dtype))
            self.constant_within.append((torch.rand((1, x_within[i], self.Rq), dtype=dtype) * 1.5) + 0.5)
            self.mus_within.append(0.5 * torch.ones((1, x_within[i]), dtype=dtype))


        self.delays = 0.0 * torch.ones((K, x_across), dtype=dtype)
        self.mus = 0.5 * torch.ones((K, x_across), dtype=dtype)
        self.constant = torch.tensor([[1.0, 1.2], [1.2, 1.0]])
        self.I = torch.eye(num_groups)
        self.num_derivatives = num_derivatives
        self.num_dims = num_dims
        self.coeff_indexes_across = torch.arange(0, num_derivatives[0], 2)
        self.coeff_indexes_within = torch.arange(0, num_derivatives[1], 2)
        self.tvec = torch.linspace(0, num_times, num_times)
        
        self.Lt_across = torch.zeros((self.num_dims[0], 1))
        self.Lt_across[-1, 0] = 1.0  
        self.Lt_across_zero = torch.zeros((self.num_dims[0], self.num_dims[0]))
        self.Lt_within = torch.zeros((self.num_dims[1], 1))
        self.Lt_within[-1, 0] = 1.0 
        self.Lt_within_zero = torch.zeros((self.num_dims[1], self.num_dims[1]))
        self.blank = torch.zeros((self.num_dims[1], self.num_dims[1]))

        # AR parameters
        assert lags > 0
        self.lags = lags
        self.K = K
        self.D = D
        self.x_across = x_across
        self.x_within = x_within
        self.num_groups = num_groups

        self.mu_init = torch.zeros((K, D))
        self.Vs = torch.zeros((K, D, M))
        self.bs = torch.zeros((K, D))
        Ft0 = torch.eye(self.num_dims[1] - 1)
        self.Ft0_within = torch.cat((torch.zeros((self.num_dims[1] - 1, 1)), Ft0), dim=1).to(torch.complex128)
        # create mask matrix to extract across-region latent variables
        self.valid_across_mat = torch.zeros((self.D, 2 * num_groups * x_across))
        count_across = 0
        for i in range(2*(num_groups*x_across+sum(x_within))):
            if i >= x_across and i < x_across + x_within[0]:
                continue
            if i >= num_groups * x_across + x_within[0] and i < num_groups * x_across + sum(x_within):
                continue
            if i >= num_groups * x_across + sum(x_within) + x_across and i < num_groups * x_across + sum(x_within) + x_across + x_within[0]:
                continue
            if i >= num_groups * x_across + sum(x_within) + num_groups * x_across + x_within[0] and i < num_groups * x_across + sum(x_within) + num_groups * x_across + sum(x_within):
                continue
            self.valid_across_mat[num_dims[0]*i, count_across] = 1
            count_across = count_across + 1


    @property
    def params(self):
        return self.delays, torch.log(self.mus), torch.log(self.sigma_across), torch.log(self.mus_within), torch.log(self.sigma_within)

    @params.setter
    def params(self, value):
        self.sigma_across = torch.exp(self.sigma_across)
        self.sigma_within = torch.exp(self.sigma_within)
        self.mus = torch.exp(self.mus)
        self.mus_within = torch.exp(self.mus_within)

    @property
    def Sigmas_init(self):
        return self.Qs

    @property
    def Sigmas(self):
        return self.Qs

    def params_bounds(self):
        mu_bound = [(torch.log(torch.tensor(1e-4)), torch.log(torch.tensor(5.0)))] * len(self.mus.flatten())  # Use torch.tensor for the bounds
        mus_within_bound = [(torch.log(torch.tensor(1e-4)), torch.log(torch.tensor(5.0)))] * (len(self.mus_within[0].flatten()) + len(self.mus_within[1].flatten()))
        sigma_across_bound = [(torch.log(torch.tensor(1e-4)), torch.log(torch.tensor(5.0)))] * len(self.sigma_across.flatten())
        sigma_within_bound = [(torch.log(torch.tensor(1e-4)), torch.log(torch.tensor(5.0)))] * (len(self.sigma_within[0].flatten()) + len(self.sigma_within[1].flatten()))
        return delay_bound + mu_bound + sigma_across_bound + mus_within_bound + sigma_within_bound


    def initialize(self):
        self.As, self.Qs, self.inv_QrQis = self.update_params(self.params)

    def log_prior(self):
        return 0

    def kernel_cc(self, mu, delay, sigma, constant):
        a11 = torch.tensor(0.0)  # Initialize as tensors
        a12 = torch.tensor(0.0)
        a21 = torch.tensor(0.0)
        a22 = torch.tensor(0.0)


        for i in range(constant.shape[1]):
            a11 = a11 + torch.square(constant[0, i])
            a12 = a12 + constant[0, i] * constant[1, i]
            a21 = a21 + constant[0, i] * constant[1, i]
            a22 = a22 + torch.square(constant[1, i])

        exp1 = torch.polar(torch.ones_like(mu), -mu * delay) 
        exp2 = torch.polar(torch.ones_like(mu), torch.zeros_like(mu))

        kcc = torch.tensor(
            [[a11, a12 * exp2 * torch.conj(exp1)],
            [a21 * exp1 * torch.conj(exp2), a22]]
        ).to(torch.complex128)

        return kcc

    def compute_across(self, params):
        sigma = torch.exp(sigma_across)
        mu = torch.exp(mu)
        constant = self.constant
        dt = 1.0

        coeff, q, _ = complex_approximation(sigma, mu, int(self.num_derivatives[1] / 2))
        coeff = coeff[1:][::-1]

        F = torch.cat((self.Ft0_within, -1.0 * coeff[None, :]), dim=1)  

        LQcL = (self.Lt_across * q) @ self.Lt_across.T 
        Pinf = torch.tensor(solve_sylvester(F.cpu().numpy(), np.conj(F.cpu().numpy()).T, -1.0 * LQcL.cpu().numpy()), dtype=torch.complex128, device=self.Lt_across.device) 

        kcc = self.kernel_cc(mu, delay, sigma, constant)

        At = torch.matrix_exp(dt * F)
        Qt = Pinf - At @ Pinf @ torch.conj(At.T)
        Qs = torch.kron(kcc, Qt)

        return At, Qs

    def compute_within(self, params):
        mus_within, sigma_within = params
        sigma = torch.exp(sigma_within)
        mu = torch.exp(mus_within)
        coeff, q, _ = complex_approximation(sigma, mu, int(self.num_derivatives[1] / 2))

        coeff = coeff[1:].flip(0)
        dt = 1.0
        LQcL = (self.Lt_within @ q) @ self.Lt_within.T
        F = torch.cat((self.Ft0_within, -1.0 * coeff[None, :]), dim=0)
        
        Pinf = solve_sylvester(F, torch.conj(F).T, -1.0 * LQcL)
        At = myexpm(dt * F)
        Qt = Pinf - At @ Pinf @ torch.conj(At).T
        
        return At, Qt

    def update_params(self, params, *args):
        delays, mus, sigma_across, mus_within, sigma_within = params
        num_dims = self.num_dims
        K = self.K
        x_across = self.x_across
        x_within = self.x_within
        num_groups = self.num_groups
        blank = self.blank
        new_As = []
        new_Qs = []
        inv_QrQis = []
        for k in range(K):
            As = [[] for _ in range(num_groups)]
            LQcLs = [[] for _ in range(num_groups)]

            for i in range(x_across):

                params = delays[k, i], mus[k, i], sigma_across[k, i]
                At, Qs = self.compute_across(params)

                for j in range(num_groups):
                    As[j].append(At)
                    LQcLs[j].append(Qs)

            for i in range(num_groups):
                for j in range(x_within[i]):
                    params = mus_within[i][0, j], sigma_within[i][0, j]
                    At, Qt = self.compute_within(params)

                    As[i].append(At)
                    LQcLs[i].append(Qt)

            As_all = []
            for i in range(num_groups):
                As_all.extend(As[i])
            As = block_diag(*As_all)

            LQcLs_part11_real = []
            LQcLs_part12_real = []
            LQcLs_part21_real = []
            LQcLs_part22_real = []

            for i in range(x_across+x_within[0]):
                if i < x_across:
                    LQcLs_part11_real.append(
                        LQcLs[0][i][0:num_dims[0], 0:num_dims[0]])
                    LQcLs_part12_real.append(
                        LQcLs[0][i][0:num_dims[0], num_dims[0]:])
                    LQcLs_part21_real.append(
                        LQcLs[0][i][num_dims[0]:, 0:num_dims[0]])
                    LQcLs_part22_real.append(
                        LQcLs[0][i][num_dims[0]:, num_dims[0]:])
                else:
                    LQcLs_part11_real.append(LQcLs[0][i])
                    LQcLs_part12_real.append(blank)
                    LQcLs_part21_real.append(blank)
                    LQcLs_part22_real.append(LQcLs[1][i])

            new_LQcL11_real = block_diag(*LQcLs_part11_real)
            new_LQcL12_real = block_diag(*LQcLs_part12_real)
            new_LQcL21_real = block_diag(*LQcLs_part21_real)
            new_LQcL22_real = block_diag(*LQcLs_part22_real)
            Qs = torch.vstack([torch.hstack([new_LQcL11_real, new_LQcL12_real]), torch.hstack([new_LQcL21_real, new_LQcL22_real])])

            Qr = torch.real(Qs)
            Qi = torch.imag(Qs)
            Ar = torch.real(As)
            Ai = torch.imag(As)
            inv_Qr = torch.linalg.inv(Qr)
            inv_term_11 = torch.linalg.inv(Qr + Qi @ inv_Qr @ Qi)
            inv_term_12 = inv_Qr @ Qi @ inv_term_11

            new_Q = torch.vstack([torch.hstack([Qr, -Qi]), torch.hstack([Qi, Qr])])
            new_A = torch.vstack([torch.hstack([Ar, -Ai]), torch.hstack([Ai, Ar])])

            inv_newQ = torch.vstack([torch.hstack([inv_term_11, inv_term_12]), torch.hstack([-inv_term_12, inv_term_11])])

            new_As.append(new_A)
            new_Qs.append(new_Q)
            inv_QrQis.append(inv_newQ)

        return torch.stack(new_As), torch.stack(new_Qs), torch.stack(inv_QrQis)

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        """
        Modified from SSM package.
        """
        D, As, bs, Vs = self.D, self.As, self.bs, self.Vs
        if xhist.shape[0] < self.lags:
            # Sample from the initial distribution
            S = torch.linalg.cholesky(self.Sigmas_init[z]) if with_noise else 0
            return self.mu_init[z] + torch.matmul(S, torch.randn(D))
        else:
            # Sample from the autoregressive distribution
            mu = (Vs[z] @ input[:self.M] + bs[z]).to(torch.complex128)
            for l in range(self.lags):
                Al = As[z][:, l*D:(l+1)*D]
                mu += Al @ xhist[-l-1]
            S = torch.linalg.cholesky(self.Sigmas[z]) if with_noise else 0
            return mu + torch.matmul(S, torch.randn(D))

    def _compute_mus(self, data, input, mask, tag):
        """
        Modified from SSM package.
        """
        K, M = self.K, self.M
        T, D = data.shape
        As, bs, Vs, mu0s = self.As, self.bs, self.Vs, self.mu_init

        mus = []
        for k, (A, b, V, mu0) in enumerate(zip(As, bs, Vs, mu0s)):
            mus_k_init = mu0 * torch.ones((self.lags, D))
            mus_k_ar = input[self.lags:, :M] @ V.T
            for l in range(self.lags):
                Al = A[:, l*D:(l + 1)*D]
                mus_k_ar = mus_k_ar + data[self.lags-l-1:-l-1] @ Al.T
            mus_k_ar = mus_k_ar + b
            mus.append(torch.vstack((mus_k_init, mus_k_ar)))
        return torch.stack(mus)

    def neg_hessian_expected_log_dynamics_prob(self, Ez, data, input, mask, tag=None):
        """
        Modified from SSM package.
        """
        assert torch.all(mask), "Cannot compute negative Hessian of autoregressive observations with missing data."
        assert self.lags == 1, "Does not compute negative Hessian of autoregressive observations with lags > 1"

        inv_new_Qs = self.inv_QrQis
        J_ini = torch.sum(Ez[0, :, None, None] * inv_new_Qs, axis=0)

        dynamics_terms = torch.stack([A.T @ inv_Sigma @ A for A, inv_Sigma in zip(self.As, inv_new_Qs)])
        J_dyn_11 = torch.sum(Ez[1:, :, None, None] * dynamics_terms[None, :], axis=1)

        J_dyn_22 = torch.sum(Ez[1:, :, None, None] * inv_new_Qs[None, :], axis=1)

        off_diag_terms = torch.stack([inv_Sigma @ A for A, inv_Sigma in zip(self.As, inv_new_Qs)])
        J_dyn_21 = -1 * torch.sum(Ez[1:, :, None, None] * off_diag_terms[None, :], axis=1)

        return J_ini, J_dyn_11, J_dyn_21, J_dyn_22

    def log_likelihoods(self, data, input, mask, tag, across_only=False):
        assert torch.all(mask), "Cannot compute likelihood of autoregressive observations with missing data."
        L = self.lags
        mus = self._compute_mus(data, input, mask, tag)
        if across_only:
            Sigmas_init_valid = self.valid_across_mat.T @ self.Sigmas_init @ self.valid_across_mat
            Sigmas_valid = self.valid_across_mat.T @ self.Sigmas @ self.valid_across_mat
            data_valid = data @ self.valid_across_mat
            mus_valid = mus @ self.valid_across_mat
        else:
            Sigmas_init_valid = self.Sigmas_init
            Sigmas_valid = self.Sigmas
            data_valid = data
            mus_valid = mus

        ll_init = torch.stack([multivariate_normal_logpdf(data_valid[:L], mu[:L], Sigma)
                            for mu, Sigma in zip(mus_valid, Sigmas_init_valid)], dim=1)

        ll_ar = torch.stack([multivariate_normal_logpdf(data_valid[L:], mu[L:], Sigma)
                            for mu, Sigma in zip(mus_valid, Sigmas_valid)], dim=1)

        return torch.cat((ll_init, ll_ar), dim=0)

    def m_step(self, expectations, datas, inputs, masks, tags, optimizer="lbfgs", **kwargs):
        optimizer = dict(adam=adam, bfgs=bfgs, lbfgs=lbfgs,
                         rmsprop=rmsprop, sgd=sgd)[optimizer]

        # expected log joint
        def _expected_log_joint(expectations):
            elbo = self.log_prior()
            for data, input, mask, tag, (expected_states, _, _) \
                    in zip(datas, inputs, masks, tags, expectations):
                lls = self.log_likelihoods(
                    data, input, mask, tag)
                elbo += torch.sum(expected_states * lls)

            return elbo

        # define optimization target
        T = sum([data.shape[0] for data in datas])

        def _objective(params, *args):
            # update As, Ws based on params
            del self.As, self.Qs, self.inv_QrQis
            self.As, self.Qs, self.inv_QrQis = self.update_params(
                params, *args)
            obj = _expected_log_joint(expectations)
            return -obj / T

        self.params = optimizer(_objective, self.params, 
                                bounds=self.params_bounds(), 
                                num_iters=100, 
                                tol=1e-4, 
                                verbose=False,
                                args=(), **kwargs)

        self.As, self.Qs, self.inv_QrQis = self.update_params(self.params)