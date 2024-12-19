import autograd.numpy as anp
import autograd.numpy.random as anpr
from autograd import grad
import ssm.transitions as trans
import ssm.init_state_distns as isd
import ssm.hmm as hmm
from ssm.util import ssm_pbar, ensure_args_are_lists, ensure_slds_args_not_none
from ssm.messages import viterbi, hmm_expected_states
from mrmgp.variational import SLDSStructuredMeanFieldVariationalPosterior
from mrmgp.emmission import GaussianCompoundEmissions
from mrmgp.optimizers import lbfgs, newtons_method_block_tridiag_hessian
from mrmgp.kernel import SpectralKernelDynamics
from utils import em_pcca, pcca_x
import copy
import warnings

class MRMGP(object):
    """
    The inference functions are modified from SSM package.
    """
    def __init__(self, N, K, num_derivative, x_across, x_within, num_groups, num_times, ydims, init_state_distn=None):
        if num_groups != 2:
            warnings.warn("currently only support two brain region case")
            raise NotImplementedError
        
        num_derivatives = anp.array([num_derivative, num_derivative]) # for across-region and within-region latent variables
        num_dims = anp.ceil(num_derivatives/2).astype(anp.int32)
        D = (num_groups * x_across * num_dims[0] + sum(x_within) * num_dims[1]) * 2

        self.ydims = ydims
        self.num_times = num_times
        self.num_dims = num_dims
        self.num_groups = num_groups
        self.x_across = x_across
        self.x_within = x_within
        D_vec = []
        for i in range(num_groups):
            D_vec.append(x_across+x_within[i])

        if init_state_distn is None:
            init_state_distn = isd.InitialStateDistribution(K, D)
        assert isinstance(init_state_distn, isd.InitialStateDistribution)

        transitions = trans.StationaryTransitions(K, D)
        dynamics = SpectralKernelDynamics(K, D, x_across, x_within, num_groups, num_derivatives, num_dims, num_times)
        emissions = GaussianCompoundEmissions(N, K, num_groups*x_across+sum(x_within),
                                                num_groups=num_groups, x_across=x_across, x_within=x_within, num_dims=num_dims,
                                                single_subspace=True,
                                                N_vec=ydims,
                                                D_vec=D_vec)

        self.N, self.K, self.D, self.M = N, K, D, 0
        self.init_state_distn = init_state_distn
        self.transitions = transitions
        self.dynamics = dynamics
        self.emissions = emissions

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None,
                   num_init_restarts=1,
                   num_init_iters=50,
                   verbose=0):
        
        # run PCCA to initialize the across latent variables
        y = anp.zeros((datas[0].shape[1], datas[0].shape[0], len(datas)))
        for i in range(len(datas)):
            y[:, :, i] = datas[i].T
        y = anp.reshape(y, (y.shape[0], y.shape[1] * y.shape[2]), order="F")

        C_across, C_within, d, Rs = em_pcca(
            y, self.num_times, self.num_groups, self.x_across, self.x_within, self.ydims)
        x, x_acorss = pcca_x(y, self.num_times, self.num_groups, self.x_across,
                                self.x_within, self.ydims, len(datas), C_across, C_within, Rs, d)

        C = []
        for i in range(len(C_across)):
            if self.x_within[0] != 0:
                C.append(anp.concatenate(
                    (C_across[i], C_within[i]), axis=1))
            else:
                C.append(C_across[i])
        self.emissions.initialize(datas, C, Rs, d, inputs, masks, tags)
        xs = [x[:, :, i].T for i in range(len(datas))]
        xs_across = [x_acorss[:, :, i].T for i in range(len(datas))]

        xmasks = [anp.ones_like(x, dtype=bool) for x in xs_across]

        pbar = ssm_pbar(num_init_restarts, verbose, "ARHMM Initialization restarts", [''])

        best_lp = -anp.inf
        num_dims = xs_across[0].shape[1]
        if num_dims != 0:
            for i in pbar:
                if verbose > 0:
                    print("Initializing with an ARHMM using {} steps of EM.".format(
                        num_init_iters))

                arhmm = hmm.HMM(self.K, num_dims, M=self.M,
                                init_state_distn=copy.deepcopy(
                                    self.init_state_distn),
                                transitions=copy.deepcopy(self.transitions),
                                observations='diagonal_autoregressive')

                arhmm.fit(xs_across, masks=xmasks,
                          verbose=verbose,
                          method="em",
                          num_iters=num_init_iters,
                          init_method="random")

                # Keep track of the arhmm that led to the highest log probability
                current_lp = arhmm.log_probability(xs_across)
                if current_lp > best_lp:
                    best_lp = copy.deepcopy(current_lp)
                    best_arhmm = copy.deepcopy(arhmm)
            self.init_state_distn = copy.deepcopy(best_arhmm.init_state_distn)
            self.transitions = copy.deepcopy(best_arhmm.transitions)
        self.dynamics.initialize()

        return xs

    @ensure_slds_args_not_none
    def expected_states(self, variational_mean, data, input=None, mask=None, tag=None):
        pi0 = self.init_state_distn.initial_state_distn
        x_mask = anp.ones_like(variational_mean, dtype=bool)
        Ps = self.transitions.transition_matrices(
            variational_mean, input, x_mask, tag)
        log_likes = self.dynamics.log_likelihoods(
            variational_mean, input, x_mask, tag, across_only=True)
        return hmm_expected_states(pi0, Ps, log_likes)

    @ensure_slds_args_not_none
    def most_likely_states(self, variational_mean, data, input=None, mask=None, tag=None, index=None):
        pi0 = self.init_state_distn.initial_state_distn
        Ps = self.transitions.transition_matrices(
            variational_mean, input, mask, tag)
        log_likes = self.dynamics.log_likelihoods(
            variational_mean, input, anp.ones_like(variational_mean, dtype=bool), tag, across_only=True)
        return viterbi(pi0, Ps, log_likes)

    @ensure_slds_args_not_none
    def smooth(self, variational_mean, data, input=None, mask=None, tag=None, index=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        Ez, _, _ = self.expected_states(
            variational_mean, data, input, mask, tag)
        return self.emissions.smooth(Ez, variational_mean, data, input, tag, index)

    def sample_continuous_states(self, z):
        T = z.shape[0]
        x = anp.zeros((T, self.D))
        inputs = anp.zeros((T, ))
        for t in range(T):
            x[t, :] = anp.real(self.dynamics.sample_x(
                int(z[t, 0]-1), x[:t], input=inputs, tag=None, with_noise=True))
        return x

    def sample(self, T, input=None, tag=None, prefix=None, with_noise=True, input_z=False):
        N = self.N
        K = self.K
        D = (self.D,)
        M = (self.M,)
        assert isinstance(D, tuple)
        assert isinstance(M, tuple)
        self.dynamics.initialize()
        # If prefix is given, pad the output with it
        if prefix is None:
            pad = 1
            z = anp.zeros(T+1, dtype=int)
            x = anp.zeros((T+1, self.D), dtype=anp.complex128)
            # input = anp.zeros((T+1,) + M) if input is None else input
            input = anp.zeros(
                (T+1,) + M) if input is None else anp.concatenate((anp.zeros((1,) + M), input))
            xmask = anp.ones((T+1,) + D, dtype=bool)

            # Sample the first state from the initial distribution
            pi0 = self.init_state_distn.initial_state_distn
            z[0] = anpr.choice(self.K, p=pi0)
            x[0] = self.dynamics.sample_x2(
                z[0], x[:0], tag=tag, with_noise=with_noise)

        else:
            zhist, xhist, yhist = prefix
            pad = len(zhist)
            assert zhist.dtype == int and zhist.min() >= 0 and zhist.max() < K
            assert yhist.shape == (pad, N)

            if not input_z:
                z = anp.concatenate((zhist, anp.zeros(T, dtype=int)))
            x = anp.concatenate((xhist, anp.zeros((T,) + D)))
            input = anp.zeros(
                (T+pad,) + M) if input is None else anp.concatenate((anp.zeros((pad,) + M), input))
            xmask = anp.ones((T+pad,) + D, dtype=bool)

        # Sample z and x
        input_z = True
        first_z = z[0]
        z = anp.zeros(T+1, dtype=int)
        z[0:int(T/2)] = first_z
        z[int(T/2):] = K - first_z - 1
        for t in range(pad, T+pad):
            Pt = anp.exp(self.transitions.log_transition_matrices(
                None, input[t-1:t+1], mask=xmask[t-1:t+1], tag=tag))[0]
            if not input_z:
                z[t] = anpr.choice(self.K, p=Pt[z[t-1]])
            x[t] = self.dynamics.sample_x2(
                z[t], x[:t], input=input[t], tag=tag, with_noise=with_noise)
    
        y = self.emissions.sample(z, x, input=input, tag=tag)

        return z[pad:], x[pad:], y[pad:]

    def log_prior(self):
        """
        Compute the log prior probability of the model parameters
        """
        return self.init_state_distn.log_prior() + \
            self.transitions.log_prior() + \
            self.dynamics.log_prior() + \
            self.emissions.log_prior()

    def laplace_em_elbo(self, posterior, datas, inputs, masks, tags, n_samples=1, index=None):
        exp_log_joint = 0.0
        for sample in range(n_samples):
            # sample continuous states
            continuous_samples = posterior.sample_continuous_states()
            discrete_expectations = posterior.discrete_expectations

            exp_log_joint += self.log_prior()

            for x, (Ez, Ezzp1, _), data, input, mask, tag in \
                    zip(continuous_samples, discrete_expectations, datas, inputs, masks, tags):
                # The "mask" for x is all ones
                x_mask = anp.ones_like(x, dtype=bool)
                log_pi0 = self.init_state_distn.log_initial_state_distn
                log_Ps = self.transitions.log_transition_matrices(
                    x, input, x_mask, tag)
                log_likes = self.dynamics.log_likelihoods(
                    x, input, x_mask, tag, across_only=False)
                log_likes = log_likes + \
                    self.emissions.log_likelihoods(
                        data, input, mask, tag, x, index)

                # Compute the expected log probability
                exp_log_joint += anp.sum(Ez[0] * log_pi0)
                exp_log_joint += anp.sum(Ezzp1 * log_Ps)
                exp_log_joint += anp.sum(Ez * log_likes)

        return exp_log_joint / n_samples + posterior.entropy()

    def update_posterior_discreate_state(self, posterior, datas, inputs, masks, tags, num_samples, index=None):
        # 0. Draw samples of q(x) for Monte Carlo approximating expectations
        x_sampless = [posterior.sample_continuous_states()
                      for _ in range(num_samples)]
        x_sampless = list(zip(*x_sampless))

        # 1. Update the variational posterior on q(z) for fixed q(x)
        discrete_state_params = []
        for x_samples, data, input, mask, tag in zip(x_sampless, datas, inputs, masks, tags):
            # Make a mask for the continuous states
            x_mask = anp.ones_like(x_samples[0], dtype=bool)

            # Compute expected log initial distribution, transition matrices, and likelihoods
            pi0 = anp.mean(
                [self.init_state_distn.initial_state_distn
                 for x in x_samples], axis=0)
            Ps = anp.mean(
                [self.transitions.transition_matrices(x, input, x_mask, tag)
                 for x in x_samples], axis=0)

            log_likes = anp.mean(
                [self.dynamics.log_likelihoods(x, input, x_mask, tag, across_only=True)
                 for x in x_samples], axis=0)
            
            if not self.emissions.single_subspace:
                log_likes = log_likes + anp.mean(
                    [self.emissions.log_likelihoods(data, input, mask, tag, x)
                     for x in x_samples], axis=0)

            discrete_state_params.append(dict(pi0=pi0,
                                              Ps=Ps,
                                              log_likes=log_likes))
        if index is None:
            posterior.discrete_state_params = discrete_state_params
        else:
            posterior.inferred_discrete_state_params = discrete_state_params

    def expected_log_joint(self, data, input, mask, tag, x, Ez, Ezzp1, scale, index):
        x_mask = anp.ones_like(x, dtype=bool)
        log_pi0 = self.init_state_distn.log_initial_state_distn
        log_Ps = self.transitions.log_transition_matrices(
            x, input, x_mask, tag)
        log_likes = self.dynamics.log_likelihoods(x, input, x_mask, tag)
        log_likes = log_likes + \
            self.emissions.log_likelihoods(data, input, mask, tag, x, index)

        # Compute the expected log probability
        elp = anp.sum(Ez[0] * log_pi0)
        elp += anp.sum(Ezzp1 * log_Ps)
        elp += anp.sum(Ez * log_likes)
        # elp += dynamics_log_like
        assert anp.all(anp.isfinite(elp))
        return -1 * elp / scale

    def hessian_params_to_hs(self, x,
                             J_ini,
                             J_dyn_11,
                             J_dyn_21,
                             J_dyn_22,
                             J_obs):
        h_ini = J_ini @ x[0]

        h_dyn_1 = (J_dyn_11 @ x[:-1][:, :, None])[:, :, 0]
        h_dyn_1 += (anp.swapaxes(J_dyn_21, -1, -2)
                    @ x[1:][:, :, None])[:, :, 0]

        h_dyn_2 = (J_dyn_22 @ x[1:][:, :, None])[:, :, 0]
        h_dyn_2 += (J_dyn_21 @ x[:-1][:, :, None])[:, :, 0]

        h_obs = (J_obs @ x[:, :, None])[:, :, 0]
        return h_ini, h_dyn_1, h_dyn_2, h_obs

    def hessian_params(self, data, input, mask, tag, x, Ez, Ezzp1, index=None):
        T, D = anp.shape(x)
        x_mask = anp.ones((T, D), dtype=bool)
        J_transitions = self.transitions.neg_hessian_expected_log_trans_prob(
            x, input, x_mask, tag, Ezzp1)

        J_ini, J_dyn_11, J_dyn_21, J_dyn_22 = self.dynamics.\
            neg_hessian_expected_log_dynamics_prob(Ez, x, input, x_mask, tag)
        J_dyn_11 += J_transitions

        J_obs = self.emissions.\
            neg_hessian_log_emissions_prob(
                data, input, mask, tag, x, Ez, index)

        return J_ini, J_dyn_11, J_dyn_21, J_dyn_22, J_obs

    def hessian_expected_log_joint(self, data, input, mask, tag, x, Ez, Ezzp1, scale=1, index=None):

        J_ini, J_dyn_11, J_dyn_21, J_dyn_22, J_obs = self.hessian_params(
            data, input, mask, tag, x, Ez, Ezzp1, index)

        hessian_diag = anp.zeros_like(J_obs)
        hessian_diag[:] += J_obs
        hessian_diag[0] += J_ini
        hessian_diag[:-1] += J_dyn_11
        hessian_diag[1:] += J_dyn_22
        hessian_lower_diag = J_dyn_21

        # Return the scaled negative hessian, which is positive definite
        return hessian_diag / scale, hessian_lower_diag / scale

    def update_posterior_latent_dynamics(self, posterior, datas, inputs, masks, tags,
                                         continuous_optimizer,
                                         continuous_tolerance,
                                         continuous_maxiter,
                                         index=None):

        # Optimize the expected log joint for each data array to find the mode
        # and the curvature around the mode.  This gives a  Laplace approximation
        # for q(x)

        continuous_state_params = []
        x0s = posterior.mean_continuous_states if index is None else posterior.mean_inferred_continuous_states
        z0s = posterior.discrete_expectations if index is None else posterior._inferred_discrete_expectations
        for (Ez, Ezzp1, _), x0, data, input, mask, tag in zip(z0s, x0s, datas, inputs, masks, tags):

            scale = x0.size
            kwargs = dict(data=data, input=input, mask=mask,
                          tag=tag, Ez=Ez, Ezzp1=Ezzp1, scale=scale, index=index)

            def _objective(
                x, *args): return self.expected_log_joint(x=x, **kwargs)

            def _grad_obj(x): return grad(self.expected_log_joint, argnum=4)(
                data, input, mask, tag, x, Ez, Ezzp1, scale, index)

            def _hess_obj(x): return self.hessian_expected_log_joint(
                x=x, **kwargs)

            if continuous_optimizer == "newton":
                x = newtons_method_block_tridiag_hessian(
                    x0, lambda x: _objective(x, None), _grad_obj, _hess_obj,
                    tolerance=continuous_tolerance, maxiter=continuous_maxiter)

            elif continuous_optimizer == "lbfgs":
                x = lbfgs(_objective, x0, num_iters=continuous_maxiter, args=(),
                          tol=continuous_tolerance)

            else:
                raise Exception(
                    "Invalid continuous_optimizer: {}".format(continuous_optimizer))

            # Evaluate the Hessian at the mode
            assert anp.all(anp.isfinite(_objective(x, -1)))

            J_ini, J_dyn_11, J_dyn_21, J_dyn_22, J_obs = self.hessian_params(
                data, input, mask, tag, x, Ez, Ezzp1, index)
            h_ini, h_dyn_1, h_dyn_2, h_obs = self.hessian_params_to_hs(x, J_ini, J_dyn_11,
                                                                       J_dyn_21, J_dyn_22, J_obs)

            continuous_state_params.append(dict(J_ini=J_ini,
                                                J_dyn_11=J_dyn_11,
                                                J_dyn_21=J_dyn_21,
                                                J_dyn_22=J_dyn_22,
                                                J_obs=J_obs,
                                                h_ini=h_ini,
                                                h_dyn_1=h_dyn_1,
                                                h_dyn_2=h_dyn_2,
                                                h_obs=h_obs))

        if index is None:
            posterior.continuous_state_params = continuous_state_params
        else:
            posterior.inferred_continuous_state_params = continuous_state_params

    def update_params(self, posterior, datas, inputs, masks, tags,
                      emission_optimizer,
                      emission_optimizer_maxiter,
                      kernel_optimizer,
                      alpha):
        # Compute necessary expectations either analytically or via samples
        continuous_samples = posterior.sample_continuous_states()
        discrete_expectations = posterior.discrete_expectations

        xmasks = [anp.ones_like(x, dtype=bool) for x in continuous_samples]
        if self.K > 1:
            for distn in [self.init_state_distn, self.transitions]:
                curr_prms = copy.deepcopy(distn.params)
                if curr_prms == tuple():
                    continue
                distn.m_step(discrete_expectations,
                             continuous_samples, inputs, xmasks, tags)

        self.dynamics.m_step(expectations=discrete_expectations,
                            datas=continuous_samples,
                            inputs=inputs,
                            masks=xmasks,
                            tags=tags,
                            optimizer=kernel_optimizer)


        self.emissions.m_step(discrete_expectations, continuous_samples,
                              datas, inputs, masks, tags,
                              optimizer=emission_optimizer,
                              maxiter=emission_optimizer_maxiter)


    @ensure_args_are_lists
    def test_inference(self, datas, inputs=None, masks=None, tags=None, num_samples=1,
                  continuous_optimizer="newton",
                  continuous_tolerance=1e-6,
                  continuous_maxiter=1000,
                  index=None):

        posterior = self.posterior

        y = anp.zeros((datas[0].shape[1], datas[0].shape[0], len(datas)))
        for i in range(len(datas)):
            y[:, :, i] = datas[i].T
        y = anp.reshape(
            y, (y.shape[0], y.shape[1] * y.shape[2]), order="F")

        ydims = []
        for i in range(self.num_groups):
            ydims.append(len(index[i]))
        C_across, C_within, d, Rs = em_pcca(
            y, self.num_times, self.num_groups, self.x_across, self.x_within, ydims)
        x, _ = pcca_x(y, self.num_times, self.num_groups, self.x_across,
                                self.x_within, ydims, len(datas), C_across, C_within, Rs, d)
        xs = [x[:, :, i].T for i in range(len(datas))]

        posterior.inferred_continuous_state_params = [posterior._initialize_continuous_state_params(data, x, input, mask, tag)
                                                    for data, x, input, mask, tag in zip(datas, xs, inputs, masks, tags)]

        posterior.inferred_discrete_state_params = [posterior._initialize_discrete_state_params(data, input, mask, tag)
                                                    for data, input, mask, tag in zip(datas, inputs, masks, tags)]
        elbos = []

        if self.K > 1:
            self.update_posterior_discreate_state(
                posterior, datas, inputs, masks, tags, num_samples, index)

        self.update_posterior_latent_dynamics(
            posterior, datas, inputs, masks, tags, continuous_optimizer, continuous_tolerance, continuous_maxiter, index)

        elbos.append(self.laplace_em_elbo(
            posterior, datas, inputs, masks, tags, index=index))

        return anp.array(elbos), posterior

    @ensure_args_are_lists
    def set_posterior(self, datas, inputs=None, masks=None, tags=None,
                      num_init_iters=1,
                      num_init_restarts=1,
                      verbose=2):
        # Initialize the model parameters by a hmm
        xs = self.initialize(datas, inputs, masks, tags,
                             num_init_restarts=num_init_restarts,
                             verbose=verbose,
                             num_init_iters=num_init_iters)

        # Initialize the variational posterior
        self.posterior = SLDSStructuredMeanFieldVariationalPosterior(
            self, datas, xs, inputs, masks, tags)

    @ensure_args_are_lists
    def fit(self, datas, inputs=None, masks=None, tags=None,
            num_init_iters=25,
            num_init_restarts=1,
            num_samples=1,
            num_iters=100,
            verbose=2,
            continuous_optimizer="newton",
            continuous_tolerance=1e-5,
            continuous_maxiter=100,
            kernel_optimizer="lbfgs",
            emission_optimizer="lbfgs",
            emission_optimizer_maxiter=100,
            alpha=0.0):

        # Initialize the model parameters by a hmm
        xs = self.initialize(datas, inputs, masks, tags,
                             num_init_restarts=num_init_restarts,
                             verbose=verbose,
                             num_init_iters=num_init_iters)

        # Initialize the variational posterior
        posterior = SLDSStructuredMeanFieldVariationalPosterior(self, datas, xs, inputs, masks, tags)
        self.posterior = posterior

        elbos = [self.laplace_em_elbo(posterior, datas, inputs, masks, tags)]

        pbar = ssm_pbar(num_iters, verbose, "ELBO: {:.1f}", [elbos[-1]])

        for itr in pbar:
            # 1. Update the discrete state posterior q(z) if K>1
            if self.K > 1:
                self.update_posterior_discreate_state(
                    posterior, datas, inputs, masks, tags, num_samples)

            # 2. Update the continuous state posterior q(x)
            self.update_posterior_latent_dynamics(
                posterior, datas, inputs, masks, tags, continuous_optimizer, continuous_tolerance, continuous_maxiter)

            self.update_params(
                posterior, datas, inputs, masks, tags, emission_optimizer, emission_optimizer_maxiter, kernel_optimizer, alpha)

            elbos.append(self.laplace_em_elbo(
                posterior, datas, inputs, masks, tags))

            if verbose == 2:
                pbar.set_description("ELBO: {:.1f}".format(elbos[-1]))

        return anp.array(elbos), posterior
