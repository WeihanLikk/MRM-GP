import autograd.numpy as anp
import autograd.numpy.random as anpr
from utils import myexpm, block_diag, complex_approximation
from autograd.scipy.linalg import solve_sylvester
import ssm.stats as stats
from mrmgp.optimizers import adam, bfgs, rmsprop, sgd, lbfgs


class Kernel():
    def __init__(self, K, D, M=0):
        self.K, self.D, self.M = K, D, M


class SpectralKernelDynamics(Kernel):
    def __init__(self, K, D, x_across, x_within, num_groups, num_derivatives, num_dims, num_times, M=0, lags=1, dtype=anp.float64):
        super(SpectralKernelDynamics, self).__init__(K, D, M)
        # kernel parameters
        self.Rq = 2
        self.sigma_across = 0.05 * anp.ones((K, x_across)).astype(dtype)
        self.constant_across = anpr.uniform(
            0.5, 2.0, (K, x_across, 2, self.Rq)).astype(dtype)
        self.sigma_within = []
        self.constant_within = []
        self.mus_within = []
        for i in range(num_groups):
            self.sigma_within.append(0.05 * anp.ones((1, x_within[i])).astype(dtype))
            self.constant_within.append(anpr.uniform(0.5, 2.0, (1, x_within[i], self.Rq)).astype(dtype))
            self.mus_within.append(0.5 * anp.ones((1, x_within[i])).astype(dtype))

        self.delays = 0.0 * anp.ones((K, x_across)).astype(dtype)
        self.mus = 0.5 * anp.ones((K, x_across)).astype(dtype)
        self.constant = anp.array([[1.0, 1.2], [1.2, 1.0]]) # or any random postive number
        self.I = anp.eye(num_groups)
        self.num_derivatives = num_derivatives
        self.num_dims = num_dims
        self.coeff_indexes_across = anp.arange(0, num_derivatives[0], 2)
        self.coeff_indexes_within = anp.arange(0, num_derivatives[1], 2)
        self.tvec = anp.linspace(0, num_times, num_times)
        
        self.Lt_across = anp.zeros((self.num_dims[0], 1))
        self.Lt_across[-1, 0] = 1
        self.Lt_across_zero = anp.zeros((self.num_dims[0], self.num_dims[0]))
        self.Lt_within = anp.zeros((self.num_dims[1], 1))
        self.Lt_within[-1, 0] = 1
        self.Lt_within_zero = anp.zeros((self.num_dims[1], self.num_dims[1]))
        self.blank = anp.zeros((self.num_dims[1], self.num_dims[1]))

        # AR parameters
        assert lags > 0
        self.lags = lags
        self.K = K
        self.D = D
        self.x_across = x_across
        self.x_within = x_within
        self.num_groups = num_groups

        self.mu_init = anp.zeros((K, D))
        self.Vs = anp.zeros((K, D, M))
        self.bs = anp.zeros((K, D))
        Ft0 = anp.eye(self.num_dims[1]-1)
        self.Ft0_within = anp.concatenate(
            (anp.zeros((self.num_dims[1]-1, 1)), Ft0), axis=1).astype(anp.complex128)

        # create mask matrix to extract across-region latent variables
        self.valid_across_mat = anp.zeros((self.D, 2*num_groups*x_across))
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
        return self.delays, anp.log(self.mus), anp.log(self.sigma_across), anp.log(self.mus_within), anp.log(self.sigma_within)

    @params.setter
    def params(self, value):
        self.delays, mus, sigma_across, mus_within, sigma_within = value
        self.sigma_across = anp.exp(sigma_across)
        self.sigma_within = anp.exp(sigma_within)
        self.mus = anp.exp(mus)
        self.mus_within = anp.exp(mus_within)

    @property
    def Sigmas_init(self):
        return self.Qs

    @property
    def Sigmas(self):
        return self.Qs

    def params_bounds(self):
        delay_bound = [(-10, 10)] * len(self.delays.flatten())
        mu_bound = [(anp.log(1e-4), anp.log(5))] * len(self.mus.flatten())
        mus_within_bound = [(anp.log(1e-4), anp.log(5))] * (
            len(self.mus_within[0].flatten()) + len(self.mus_within[1].flatten()))
        sigma_across_bound = [(anp.log(1e-4), anp.log(5.0))] * \
            len(self.sigma_across.flatten())
        sigma_within_bound = [(anp.log(1e-4), anp.log(5.0))] * (
            len(self.sigma_within[0].flatten()) + len(self.sigma_within[1].flatten()))
        return delay_bound + mu_bound + sigma_across_bound + mus_within_bound + sigma_within_bound

    def initialize(self):
        self.As, self.Qs, self.inv_QrQis = self.update_params(self.params)

    def log_prior(self):
        return 0

    def kernel_cc(self, mu, delay, sigma, constant):
        a11 = 0
        a12 = 0
        a21 = 0
        a22 = 0

        for i in range(self.Rq):
            a11 = a11 + anp.square(constant[0, i])
            a12 = a12 + constant[0, i] * constant[1, i]
            a21 = a21 + constant[0, i] * constant[1, i]
            a22 = a22 + anp.square(constant[1, i])

        kcc = anp.array(
            [[a11, a12*anp.exp(-1j*0.0)*anp.conj(anp.exp(-1j*mu*delay))],
             [a21*anp.exp(-1j*mu*delay)*anp.conj(anp.exp(-1j*0.0)), a22]]
        )
        return kcc

    def compute_across(self, params):
        delay, mu, sigma_across = params
        sigma = anp.exp(sigma_across)
        mu = anp.exp(mu)
        constant = self.constant
        dt = 1.0
        coeff, q, _ = complex_approximation(sigma, mu, int(self.num_derivatives[1]/2))
        coeff = coeff[1:][::-1]
        F = anp.concatenate((self.Ft0_within, -1.0*coeff[None, :]))
        LQcL = (self.Lt_across * q) @ self.Lt_across.T
        Pinf = solve_sylvester(F, anp.conj(F).T, -1.0*LQcL)

        kcc = self.kernel_cc(mu, delay, sigma, constant)

        At = myexpm(dt * F)
        Qt = Pinf - At @ Pinf @ (anp.conj(At).T)
        Qs = anp.kron(kcc, Qt)

        return At, Qs

    def compute_within(self, params):
        mus_within, sigma_within = params
        sigma = anp.exp(sigma_within)
        mu = anp.exp(mus_within)
        coeff, q, _ = complex_approximation(sigma, mu, int(self.num_derivatives[1]/2))
        coeff = coeff[1:][::-1]
        dt = 1.0
        LQcL = (self.Lt_within * q) @ self.Lt_within.T
        F = anp.concatenate((self.Ft0_within, -1.0*coeff[None, :]))
        Pinf = solve_sylvester(F, anp.conj(F).T, -1.0*LQcL)
        At = myexpm(dt * F)
        Qt = Pinf - At @ Pinf @ (anp.conj(At).T)

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
            Qs = anp.vstack([anp.hstack([new_LQcL11_real, new_LQcL12_real]), anp.hstack([new_LQcL21_real, new_LQcL22_real])])

            Qr = anp.real(Qs)
            Qi = anp.imag(Qs)
            Ar = anp.real(As)
            Ai = anp.imag(As)
            inv_Qr = anp.linalg.inv(Qr)
            inv_term_11 = anp.linalg.inv(Qr + Qi @ inv_Qr @ Qi)
            inv_term_12 = inv_Qr @ Qi @ inv_term_11

            new_Q = anp.vstack([anp.hstack([Qr, -Qi]), anp.hstack([Qi, Qr])])
            new_A = anp.vstack([anp.hstack([Ar, -Ai]), anp.hstack([Ai, Ar])])

            inv_newQ = anp.vstack(
                [anp.hstack([inv_term_11, inv_term_12]), anp.hstack([-inv_term_12, inv_term_11])])

            new_As.append(new_A)
            new_Qs.append(new_Q)
            inv_QrQis.append(inv_newQ)

        return anp.array(new_As), anp.array(new_Qs), anp.array(inv_QrQis)

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        """
        Modified from SSM package.
        """
        D, As, bs, Vs = self.D, self.As, self.bs, self.Vs
        if xhist.shape[0] < self.lags:
            # Sample from the initial distribution
            S = anp.linalg.cholesky(self.Sigmas_init[z]) if with_noise else 0
            return self.mu_init[z] + anp.dot(S, anpr.randn(D))
        else:
            # Sample from the autoregressive distribution
            mu = (Vs[z].dot(input[:self.M]) + bs[z]).astype(anp.complex128)
            for l in range(self.lags):
                Al = As[z][:, l*D:(l+1)*D]

                mu += Al.dot(xhist[-l-1])
            S = anp.linalg.cholesky(self.Sigmas[z]) if with_noise else 0
            return mu + anp.dot(S, anpr.randn(D))

    def _compute_mus(self, data, input, mask, tag):
        """
        Modified from SSM package.
        """
        K, M = self.K, self.M
        T, D = data.shape
        As, bs, Vs, mu0s = self.As, self.bs, self.Vs, self.mu_init

        mus = []
        for k, (A, b, V, mu0) in enumerate(zip(As, bs, Vs, mu0s)):
            # Initial condition
            mus_k_init = mu0 * anp.ones((self.lags, D))
            # Subsequent means are determined by the AR process
            mus_k_ar = anp.dot(input[self.lags:, :M], V.T)
            for l in range(self.lags):
                Al = A[:, l*D:(l + 1)*D]
                mus_k_ar = mus_k_ar + anp.dot(data[self.lags-l-1:-l-1], Al.T)

            mus_k_ar = mus_k_ar + b

            # Append concatenated mean
            mus.append(anp.vstack((mus_k_init, mus_k_ar)))

        return anp.array(mus)

    def neg_hessian_expected_log_dynamics_prob(self, Ez, data, input, mask, tag=None):
        """
        Modified from SSM package.
        """
        assert anp.all(mask), "Cannot compute negative Hessian of autoregressive obsevations with missing data."
        assert self.lags == 1, "Does not compute negative Hessian of autoregressive observations with lags > 1"

        inv_new_Qs = self.inv_QrQis
        J_ini = anp.sum(Ez[0, :, None, None] * inv_new_Qs, axis=0)

        dynamics_terms = anp.array(
            [A.T@inv_Sigma@A for A, inv_Sigma in zip(self.As, inv_new_Qs)])  # A^T Qinv A terms
        J_dyn_11 = anp.sum(Ez[1:, :, None, None] *
                           dynamics_terms[None, :], axis=1)

        J_dyn_22 = anp.sum(Ez[1:, :, None, None] * inv_new_Qs[None, :], axis=1)

        off_diag_terms = anp.array(
            [inv_Sigma@A for A, inv_Sigma in zip(self.As, inv_new_Qs)])
        J_dyn_21 = -1 * anp.sum(Ez[1:, :, None, None]
                                * off_diag_terms[None, :], axis=1)

        return J_ini, J_dyn_11, J_dyn_21, J_dyn_22

    def log_likelihoods(self, data, input, mask, tag, across_only=False):
        assert anp.all(
            mask), "Cannot compute likelihood of autoregressive obsevations with missing data."
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

        # Compute the likelihood of the initial data and remainder separately
        ll_init = anp.column_stack([stats.multivariate_normal_logpdf(data_valid[:L], mu[:L], Sigma)
                                   for mu, Sigma in zip(mus_valid, Sigmas_init_valid)])

        ll_ar = anp.column_stack([stats.multivariate_normal_logpdf(data_valid[L:], mu[L:], Sigma)
                                 for mu, Sigma in zip(mus_valid, Sigmas_valid)])

        # Compute the likelihood of the initial data and remainder separately
        return anp.row_stack((ll_init, ll_ar))

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
                elbo += anp.sum(expected_states * lls)

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