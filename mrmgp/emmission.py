import autograd.numpy as anp
import autograd.numpy.random as npr
from ssm.util import ensure_args_are_lists
from ssm.preprocessing import interpolate_data, pca_with_imputation
from ssm.util import ensure_args_are_lists, logistic, logit, softplus, inv_softplus
from autograd.scipy.special import gammaln
from mrmgp.optimizers import adam, bfgs, rmsprop, sgd, lbfgs
from autograd import hessian
from utils import block_diag
from warnings import warn
from ssm.regression import fit_linear_regression

"""
These functions are modified from SSM package.
"""

class Emissions(object):
    def __init__(self, N, K, D, M=0, single_subspace=True):
        self.N, self.K, self.D, self.M, self.single_subspace = \
            N, K, D, M, single_subspace

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, value):
        raise NotImplementedError

    def permute(self, perm):
        pass

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, num_em_iters=25):
        pass

    def initialize_from_arhmm(self, arhmm, pca):
        pass

    def log_prior(self):
        return 0

    def log_likelihoods(self, data, input, mask, tag, x):
        raise NotImplementedError

    def forward(self, x, input=None, tag=None):
        raise NotImplementedError

    def invert(self, data, input=None, mask=None, tag=None):
        raise NotImplementedError

    def sample(self, z, x, input=None, tag=None):
        raise NotImplementedError

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        raise NotImplementedError

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        if self.single_subspace is False:
            raise Exception(
                "Multiple subspaces are not supported for this Emissions class.")
        warn("Analytical Hessian is not implemented for this Emissions class. \
              Optimization via Laplace-EM may be slow. Consider using an \
              alternative posterior and inference method.")
        # Return (T, D, D) array of blocks for the diagonal of the Hessian

        def obj(xt, datat, inputt, maskt): return \
            self.log_likelihoods(
                datat[None, :], inputt[None, :], maskt[None, :], tag, xt[None, :])[0, 0]
        hess = hessian(obj)
        terms = anp.array([anp.squeeze(hess(xt, datat, inputt, maskt))
                          for xt, datat, inputt, maskt in zip(x, data, input, mask)])
        return -1 * terms

    def m_step(self, discrete_expectations, continuous_expectations,
               datas, inputs, masks, tags,
               optimizer="lbfgs", maxiter=100, **kwargs):
        """
        If M-step in Laplace-EM cannot be done in closed form for the emissions, default to SGD.
        """
        optimizer = dict(adam=adam, bfgs=bfgs, lbfgs=lbfgs,
                         rmsprop=rmsprop, sgd=sgd)[optimizer]

        # expected log likelihood
        T = sum([data.shape[0] for data in datas])

        def _objective(params, *itr):
            self.params = params
            obj = 0
            obj += self.log_prior()
            for data, input, mask, tag, x, (Ez, _, _) in \
                    zip(datas, inputs, masks, tags, continuous_expectations, discrete_expectations):
                obj += anp.sum(Ez * self.log_likelihoods(data,
                               input, mask, tag, x))
            return -obj / T

        # Optimize emissions log-likelihood
        self.params = optimizer(_objective, self.params,
                                num_iters=maxiter,
                                **kwargs)


class _LinearEmissions(Emissions):
    """
    A simple linear mapping from continuous states x to data y.

        E[y | x] = Cx + d + Fu

    where C is an emission matrix, d is a bias, F an input matrix,
    and u is an input.
    """

    def __init__(self, N, K, D, Hs, x_across, x_within, num_dims, M=0, single_subspace=True):
        super(_LinearEmissions, self).__init__(
            N, K, D, M=M, single_subspace=single_subspace)

        # Initialize linear layer.  Set _Cs to be private so that it can be
        # changed in subclasses.
        self._Cs = npr.randn(
            1, N, D) if single_subspace else npr.randn(K, N, D)
        self.Fs = npr.randn(1, N, M) if single_subspace else npr.randn(K, N, M)
        self.ds = npr.randn(1, N) if single_subspace else npr.randn(K, N)
        self.Hs = Hs
        self.x_across = x_across
        self.x_within = x_within
        self.num_dims = num_dims

    @property
    def Cs(self):
        return self._Cs

    @Cs.setter
    def Cs(self, value):
        K, N, D = self.K, self.N, self.D
        assert value.shape == (1, N, D) if self.single_subspace else (K, N, D)
        self._Cs = value

    @property
    def params(self):
        return self.Cs, self.Fs, self.ds

    @params.setter
    def params(self, value):
        self.Cs, self.Fs, self.ds = value

    def permute(self, perm):
        if not self.single_subspace:
            self.Cs = self.Cs[perm]
            self.Fs = self.Fs[perm]
            self.ds = self.ds[perm]

    def _invert(self, data, input=None, mask=None, tag=None):
        """
        Approximate invert the linear emission model with the pseudoinverse

        y = Cx + d + noise; C orthogonal.
        xhat = (C^T C)^{-1} C^T (y-d)
        """
        # Invert with the average emission parameters
        C = anp.mean(self.Cs, axis=0)
        F = anp.mean(self.Fs, axis=0)
        d = anp.mean(self.ds, axis=0)
        C_pseudoinv = anp.linalg.solve(C.T.dot(C), C.T).T

        # Account for the bias
        bias = input.dot(F.T) + d

        if not anp.all(mask):
            data = interpolate_data(data, mask)
            for itr in range(25):
                mu = (data - bias).dot(C_pseudoinv)
                data[:, ~mask[0]] = (mu.dot(C.T) + bias)[:, ~mask[0]]

        # Project data to get the mean
        xhat = (data - bias).dot(C_pseudoinv)
        return xhat

    def forward(self, x, input, tag, index=None):
        Hx = (self.Hs @ x.T).T
        if index is None:
            return anp.matmul(self.Cs[None, ...], Hx[:, None, :, None])[:, :, :, 0] \
                + anp.matmul(self.Fs[None, ...], input[:, None, :, None])[:, :, :, 0] \
                + self.ds
        else:
            return anp.matmul(self.Cs[None, :, index, :], Hx[:, None, :, None])[:, :, :, 0] \
                + anp.matmul(self.Fs[None, :, index, :], input[:, None, :, None])[:, :, :, 0] \
                + self.ds[:, index]

    @ensure_args_are_lists
    def _initialize_with_pca(self, datas, inputs=None, masks=None, tags=None, num_iters=20):
        Keff = 1 if self.single_subspace else self.K

        # First solve a linear regression for data given input
        if self.M > 0:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression(fit_intercept=False)
            lr.fit(anp.vstack(inputs), anp.vstack(datas))
            self.Fs = anp.tile(lr.coef_[None, :, :], (Keff, 1, 1))

        # Compute residual after accounting for input
        resids = [data - anp.dot(input, self.Fs[0].T)
                  for data, input in zip(datas, inputs)]

        # Run PCA to get a linear embedding of the data with the maximum effective dimension
        pca, xs, ll = pca_with_imputation(min(self.D * Keff, self.N),
                                          resids, masks, num_iters=num_iters)

        # Assign each state a random projection of these dimensions
        Cs, ds = [], []
        for k in range(Keff):
            weights = npr.randn(self.D, self.D * Keff)
            weights = anp.linalg.svd(weights, full_matrices=False)[2]
            Cs.append((weights @ pca.components_).T)
            ds.append(pca.mean_)

        # Find the components with the largest power
        self.Cs = anp.array(Cs)
        self.ds = anp.array(ds)

        return pca


class _CompoundLinearEmissions(Emissions):
    def __init__(self, N, K, D, num_groups, x_across, x_within, num_dims, M=0, single_subspace=True,
                 N_vec=None, D_vec=None, **kwargs):
        """
        N_vec, D_vec are the sizes of the constituent emission models.
        Assume N_vec and D_vec are lists/tuples/arrays of length G and

        N_vec = [N_1, ..., N_P] indicates that the first group of neurons
        is size N_1, the P-th populations is size N_P.  Likewise for D_vec.
        We will assume that the data is grouped in the same way.

        We require sum(N_vec) == N and sum(D_vec) == D.
        """
        super(_CompoundLinearEmissions, self).__init__(
            N, K, D, M=M, single_subspace=single_subspace)

        assert isinstance(N_vec, (anp.ndarray, list, tuple))
        N_vec = anp.array(N_vec, dtype=int)
        assert anp.sum(N_vec) == N

        assert isinstance(D_vec, (anp.ndarray, list, tuple)
                          ) and len(D_vec) == len(N_vec)
        D_vec = anp.array(D_vec, dtype=int)
        assert anp.sum(D_vec) == D

        self.N_vec, self.D_vec = N_vec, D_vec

        # Save the number of subpopulations
        self.P = len(self.N_vec)
        self.num_dims = num_dims
        self.Dxs = [0, x_across*num_dims[0]+x_within[0] *
                    num_dims[1], x_across*num_dims[0]+x_within[1]*num_dims[1]]
        self.x_across = x_across
        self.x_within = x_within

        self.Hs = anp.zeros((D, 2*sum(self.Dxs)))
        # self.Hs = anp.zeros((D, sum(self.Dxs)))
        for i in range(D):
            if i < x_across:
                index = i*num_dims[0]
            elif x_across <= i < x_across+x_within[0]:
                index = x_across*num_dims[0] + (i-x_across)*num_dims[1]
            elif x_across+x_within[0] <= i < num_groups*x_across + x_within[0]:
                index = x_across * \
                    num_dims[0]+x_within[0]*num_dims[1] + \
                    (i-x_across-x_within[0])*num_dims[0]
            else:
                index = num_groups*x_across * \
                    num_dims[0]+x_within[0]*num_dims[1] + \
                    (i-num_groups*x_across-x_within[0])*num_dims[1]
                # for 2 num_groups
            self.Hs[i, index] = 1.0

        # The main purpose of this class is to wrap a bunch of emissions instances
        self.emissions_models = []
        count = 0
        for n, d in zip(N_vec, D_vec):
            Hs_i = self.Hs[count*d:(count+1)*d, self.Dxs[count]:self.Dxs[count]+self.Dxs[count+1]]
            Hs_i = anp.hstack((Hs_i, anp.zeros_like(Hs_i)))
            self.emissions_models.append(_LinearEmissions(
                n, K, d, Hs_i, x_across, x_within[count], num_dims, M=M))
            count += 1
        # self.emissions_models = [_LinearEmissions(n, K, d, num_dim, M=M) for n, d in zip(N_vec, D_vec)]

    @property
    def Cs(self):
        if self.single_subspace:
            return anp.array([block_diag(*[em.Cs[0] for em in self.emissions_models])])
        else:
            return anp.array([block_diag(*[em.Cs[k] for em in self.emissions_models])
                             for k in range(self.K)])

    @property
    def ds(self):
        return anp.concatenate([em.ds for em in self.emissions_models], axis=1)

    @property
    def Fs(self):
        return anp.concatenate([em.Fs for em in self.emissions_models], axis=1)

    @property
    def params(self):
        return tuple(em.params for em in self.emissions_models)

    @params.setter
    def params(self, value):
        assert len(value) == self.P
        for em, v in zip(self.emissions_models, value):
            em.params = v

    def permute(self, perm):
        for em in self.emissions_models:
            em.permute(perm)

    def _invert(self, data, input, mask, tag):
        assert data.shape[1] == self.N
        N_offsets = anp.cumsum(self.N_vec)[:-1]
        states = []
        for em, dp, mp in zip(self.emissions_models,
                              anp.split(data, N_offsets, axis=1),
                              anp.split(mask, N_offsets, axis=1)):
            states.append(em._invert(dp, input, mp, tag))
        return anp.column_stack(states)

    def forward(self, x, input, tag, index=None):
        assert x.shape[1] == sum(self.Dxs) * 2
        Dx_offsets = anp.cumsum(self.Dxs[1:])[:-1]
        datas = []
        x_real = x[:, 0:int(x.shape[1]/2)]
        x_imag = x[:, int(x.shape[1]/2):]
        if index is None:
            for em, xp_r, xp_i in zip(self.emissions_models, anp.split(x_real, Dx_offsets, axis=1), anp.split(x_imag, Dx_offsets, axis=1)):
                xp = anp.hstack((xp_r, xp_i))
                datas.append(em.forward(xp, input, tag))
        else:
            for em, xp_r, xp_i, ind in zip(self.emissions_models, anp.split(x_real, Dx_offsets, axis=1), anp.split(x_imag, Dx_offsets, axis=1), index):
                xp = anp.hstack((xp_r, xp_i))
                datas.append(em.forward(xp, input, tag, ind))
        return anp.concatenate(datas, axis=2)

    @ensure_args_are_lists
    def _initialize_with_pca(self, datas, inputs=None, masks=None, tags=None, num_iters=20):
        for data in datas:
            assert data.shape[1] == self.N

        N_offsets = anp.cumsum(self.N_vec)[:-1]
        pcas = []

        split_datas = list(
            zip(*[anp.split(data, N_offsets, axis=1) for data in datas]))
        split_masks = list(
            zip(*[anp.split(mask, N_offsets, axis=1) for mask in masks]))
        assert len(split_masks) == len(split_datas) == self.P

        for em, dps, mps in zip(self.emissions_models, split_datas, split_masks):
            pcas.append(em._initialize_with_pca(dps, inputs, mps, tags))

        # Combine the PCA objects
        from sklearn.decomposition import PCA
        pca = PCA(self.D)
        pca.components_ = block_diag(*[p.components_ for p in pcas])
        pca.mean_ = anp.concatenate([p.mean_ for p in pcas])
        # Not super pleased with this, but it should work...
        pca.noise_variance_ = anp.concatenate([p.noise_variance_ * anp.ones(n)
                                              for p, n in zip(pcas, self.N_vec)])
        return pca

# Observation models for SLDS


class _GaussianEmissionsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, **kwargs):
        super(_GaussianEmissionsMixin, self).__init__(
            N, K, D, M=M, single_subspace=single_subspace, **kwargs)
        self.inv_etas = -1 + \
            npr.randn(1, N) if single_subspace else npr.randn(K, N)

    @property
    def params(self):
        return super(_GaussianEmissionsMixin, self).params + (self.inv_etas,)

    @params.setter
    def params(self, value):
        self.inv_etas = value[-1]
        super(_GaussianEmissionsMixin, self.__class__).params.fset(
            self, value[:-1])

    def permute(self, perm):
        super(_GaussianEmissionsMixin, self).permute(perm)
        if not self.single_subspace:
            self.inv_etas = self.inv_etas[perm]

    def log_likelihoods(self, data, input, mask, tag, x, index=None):
        mus = self.forward(x, input, tag, index)
        if index is None:
            etas = anp.exp(self.inv_etas)
        else:
            indexes = anp.concatenate((index[0], index[1]+self.N_vec[0]))
            etas = anp.exp(self.inv_etas[:, indexes])
        lls = -0.5 * anp.log(2 * anp.pi * etas) - 0.5 * \
            (data[:, None, :] - mus)**2 / etas
        return anp.sum(lls * mask[:, None, :], axis=2)

    def invert(self, data, input=None, mask=None, tag=None):
        return self._invert(data, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = anp.zeros_like(z, dtype=int) if self.single_subspace else z
        mus = self.forward(x, input, tag)
        etas = anp.exp(self.inv_etas)
        return mus[anp.arange(T), z, :] + 0.1 * anp.sqrt(etas[z]) * npr.randn(T, self.N)

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        mus = self.forward(variational_mean, input, tag)
        yhat = mus[:, 0, :] if self.single_subspace else anp.sum(
            mus * expected_states[:, :, None], axis=1)
        return yhat


class GaussianCompoundEmissions(_GaussianEmissionsMixin, _CompoundLinearEmissions):

    # @ensure_args_are_lists
    def initialize(self, datas, C, Rs, d, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask)
                 for data, mask in zip(datas, masks)]

        etas = anp.concatenate((anp.diag(Rs[0]), anp.diag(Rs[1])))
        self.inv_etas[:, ...] = anp.log(etas)
        self.emissions_models[0].Cs = C[0][None, :, :]
        self.emissions_models[1].Cs = C[1][None, :, :]
        self.emissions_models[0].ds = d[None, 0:self.N_vec[0]]
        self.emissions_models[1].ds = d[None, self.N_vec[0]:]

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez, index):
        assert self.single_subspace, "Only implemented for a single emission model"
        T, D = data.shape
        new_Cs = self.Cs[0] @ self.Hs
        if index is None:
            hess = -1.0 * \
                new_Cs.T@anp.diag(1.0 / anp.exp(self.inv_etas[0]))@new_Cs
        else:
            indexes = anp.concatenate((index[0], index[1]+self.N_vec[0]))
            hess = -1.0 * new_Cs[indexes, :].T@anp.diag(
                1.0 / anp.exp(self.inv_etas[0][indexes]))@new_Cs[indexes, :]
        return -1 * anp.tile(hess[None, :, :], (T, 1, 1))

    def m_step(self, discrete_expectations, continuous_expectations,
               datas, inputs, masks, tags,
               optimizer="bfgs", maxiter=100, **kwargs):

        Xs = [[] for _ in range(len(self.D_vec))]
        ys = [[] for _ in range(len(self.N_vec))]
        N_vec_offsets = anp.cumsum(self.N_vec)[:-1]
        D_vec_offsets = anp.cumsum(self.D_vec)[:-1]
        for x, u, y in zip(continuous_expectations, inputs, datas):
            x_i = x @ self.Hs.T
            y_i = y
            for i, (x_i_s, y_i_s) in enumerate(zip(anp.split(x_i, D_vec_offsets, axis=1), anp.split(y_i, N_vec_offsets, axis=1))):
                Xs[i].append(anp.column_stack([x_i_s, u]))
                ys[i].append(y_i_s)

        inv_etas = []
        for i, (em, n, d) in enumerate(zip(self.emissions_models, self.N_vec, self.D_vec)):
            CF, ds, Sigma = fit_linear_regression(Xs[i], ys[i], prior_ExxT=1e-4 * anp.eye(
                d + self.M + 1), prior_ExyT=anp.zeros((d + self.M + 1, n)))
            em.Cs = CF[None, :, :d]
            em.Fs = CF[None, :, d:]
            em.ds = ds[None, :]
            inv_etas.append(anp.log(anp.diag(Sigma)))
        self.inv_etas = anp.hstack(inv_etas)[None, :]


class _PoissonEmissionsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, link="softplus", bin_size=1.0, **kwargs):

        super(_PoissonEmissionsMixin, self).__init__(
            N, K, D, M=M, single_subspace=single_subspace, **kwargs)

        self.link_name = link
        self.bin_size = bin_size
        mean_functions = dict(
            log=self._log_mean,
            softplus=self._softplus_mean
        )
        self.mean = mean_functions[link]
        link_functions = dict(
            log=self._log_link,
            softplus=self._softplus_link
        )
        self.link = link_functions[link]

        # Set the bias to be small if using log link
        if link == "log":
            self.ds = -3 + .5 * \
                npr.randn(1, N) if single_subspace else npr.randn(K, N)

    def _log_mean(self, x):
        return anp.exp(x) * self.bin_size

    def _softplus_mean(self, x):
        return softplus(x) * self.bin_size

    def _log_link(self, rate):
        return anp.log(rate) - anp.log(self.bin_size)

    def _softplus_link(self, rate):
        return inv_softplus(rate / self.bin_size)

    def log_likelihoods(self, data, input, mask, tag, x, index=None):
        assert data.dtype == int
        lambdas = self.mean(self.forward(x, input, tag, index))
        mask = anp.ones_like(data, dtype=bool) if mask is None else mask
        lls = -gammaln(data[:, None, :] + 1) - lambdas + \
            data[:, None, :] * anp.log(lambdas)
        return anp.sum(lls * mask[:, None, :], axis=2)

    def invert(self, data, input=None, mask=None, tag=None):
        yhat = self.link(anp.clip(data, .1, anp.inf))
        return self._invert(yhat, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = anp.zeros_like(z, dtype=int) if self.single_subspace else z
        lambdas = self.mean(self.forward(x, input, tag))
        y = npr.poisson(lambdas[anp.arange(T), z, :])
        return y

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        lambdas = self.mean(self.forward(variational_mean, input, tag))
        return lambdas[:, 0, :] if self.single_subspace else anp.sum(lambdas * expected_states[:, :, None], axis=1)


class PoissonCompoundEmissions(_PoissonEmissionsMixin, _CompoundLinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask)
                 for data, mask in zip(datas, masks)]
        yhats = [self.link(anp.clip(d, .1, anp.inf)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez, index):
        """
        d/dx log p(y | x) = d/dx [y * (Cx + Fu + d) - exp(Cx + Fu + d)
                          = y * C - lmbda * C
                          = (y - lmbda) * C

        d/dx  (y - lmbda)^T C = d/dx -exp(Cx + Fu + d)^T C
            = -C^T exp(Cx + Fu + d)^T C
        """
        if self.link_name == "log":
            assert self.single_subspace
            new_Cs = self.Cs[0] @ self.Hs
            lambdas = self.mean(self.forward(x, input, tag))
            return -anp.einsum('tn, ni, nj ->tij', -lambdas[:, 0, :], new_Cs, new_Cs)

        elif self.link_name == "softplus":
            assert self.single_subspace
            if index is None:
                new_Cs = self.Cs[0] @ self.Hs
                ds = self.ds[0]
            else:
                indexes = anp.concatenate((index[0], index[1]+self.N_vec[0]))
                new_Cs = self.Cs[0, indexes, :] @ self.Hs
                ds = self.ds[0, indexes]
            lambdas = anp.log1p(
                anp.exp(anp.dot(x, new_Cs.T)+ds))
            expterms = anp.exp(-anp.dot(x, new_Cs.T)-ds)
            diags = (data / lambdas * (expterms - 1.0 / lambdas) -
                     expterms) / (1.0+expterms)**2
            return -anp.einsum('tn, ni, nj ->tij', diags, new_Cs, new_Cs)
