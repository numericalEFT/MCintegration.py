import normflows as nf
import torch
from warnings import warn
from scipy.stats import kstest
import time
def calculate_correlation(x):
    x_centered = x[1:] - x.mean()
    y_centered = x[:-1] - x.mean()
    cov = torch.sum(x_centered * y_centered) / (len(x) - 1)
    return torch.abs(cov / torch.var(x))

class MCFlow(nf.NormalizingFlow):

    def __init__(self, q0, flows, p=None):
        super().__init__(q0, flows, p)
    def forward(self, z, rev=False):
        """Transforms latent variable z to the flow variable x

        Args:
          z: Batch in the latent space

        Returns:
          Batch in the space of the target distribution
        """
        if rev:
            log_q = torch.zeros(len(z), device=z.device)
            for i in range(len(self.flows) - 1, -1, -1):
                z, log_det = self.flows[i].inverse(z)
                log_q += log_det
            return z, log_q
        else:
            for flow in self.flows:
                z, _ = flow(z)
            return z
    def IS_chi2(self, num_samples=1):
        z, log_q_ = self.q0(num_samples)
        log_q = torch.zeros_like(log_q_)
        log_q += log_q_
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        # utils.set_requires_grad(self, False)
        prob = torch.abs(self.p.prob(z))
        q = torch.exp(log_q)
        pmean = torch.mean(prob / q)
        prob = prob / pmean
        # print("test:", prob, "\n", ISratio, "\n", log_q, "\n")
        # print( -torch.mean(ISratio.detach()*log_q))
        # utils.set_requires_grad(self, True)
        return torch.mean(torch.square(prob.detach() - q) / q / q.detach())

    def IS_forward_kld(self, num_samples=None):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        if num_samples is None:
            num_samples = self.p.batchsize
        nf.utils.set_requires_grad(self, False)
        z, _ = self.q0(num_samples)
        for flow in self.flows:
            z, _ = flow(z)
        nf.utils.set_requires_grad(self, True)
        z_, log_q = self.inverse_and_log_det(z)
        # utils.set_requires_grad(self, False)
        prob = torch.abs(self.p.prob(z))
        q = torch.exp(log_q)
        pmean = torch.mean(prob / q)
        prob = prob / pmean
        logp = torch.where(prob > 1e-16, torch.log(prob), torch.log(prob + 1e-16))
        ISratio = prob / q
        # print("test:", prob, "\n", ISratio, "\n", log_q, "\n")
        # print( -torch.mean(ISratio.detach()*log_q))
        # utils.set_requires_grad(self, True)
        return torch.mean(ISratio.detach() * (logp.detach() - log_q))

    
    def IS_forward_kld_direct(self,  z):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution

        Returns:
          Estimate of forward KL divergence averaged over batch
        """

        z_, log_q = self.inverse_and_log_det(z)
        # utils.set_requires_grad(self, False)
        prob = torch.abs(self.p.prob(z))
        q = torch.exp(log_q)
        pmean = torch.mean(prob / q)
        prob = prob / pmean
        logp = torch.where(prob > 1e-16, torch.log(prob), torch.log(prob + 1e-16))
        ISratio = prob / q
        # print("test:", prob, "\n", ISratio, "\n", log_q, "\n")
        # print( -torch.mean(ISratio.detach()*log_q))
        # utils.set_requires_grad(self, True)
        return torch.mean(ISratio.detach() * (logp.detach() - log_q))
    def MCvar(self, num_samples=1):
        z, log_q_ = self.q0(num_samples)
        log_J = torch.zeros_like(log_q_)
        # log_J += log_q_
        for flow in self.flows:
            z, log_det = flow(z)
            log_J += log_det
        log_p = self.p.log_prob(z)
        return torch.mean(torch.exp(2 * log_p + 2 * log_J))

    @torch.no_grad()
    def integrate(self):
        """Importance sampling integration with flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw

        Returns:
          mean, variance
        """
        num_samples = self.p.batchsize
        z, log_q = self.sample(num_samples)
        q = torch.exp(log_q)
        func = self.p.prob(z)
        return torch.mean(func / q, dim=0)
    
    @torch.no_grad()
    def integrate_block(
        self, num_blocks, bins=25, hist_range=(0.0, 1.0), correlation_threshold=0.2
    ):
        print("Estimating integral from trained network")

        device = self.p.samples.device
        num_samples = self.p.batchsize
        means_t = torch.zeros(num_blocks, device=device)
        means_abs = torch.empty_like(means_t)
        # Pre-allocate tensor for storing means and histograms
        # num_vars = self.p.ndims
        # with torch.device("cpu"):
        #     if isinstance(bins, int):
        #         histr = torch.zeros(bins, num_vars)
        #         histr_weight = torch.zeros(bins, num_vars)
        #     else:
        #         histr = torch.zeros(bins.shape[0], num_vars)
        #         histr_weight = torch.zeros(bins.shape[0], num_vars)

        partition_z = torch.tensor(0.0, device=device)
        for i in range(num_blocks):
            self.p.samples[:], self.p.log_q[:] = self.q0(num_samples)
            for flow in self.flows:
                self.p.samples[:], log_det = flow(self.p.samples)
                self.p.log_q -= log_det
            self.p.val[:] = self.p.prob(self.p.samples)
            q = torch.exp(self.p.log_q)
            res = self.p.val / q
            means_t[i] = torch.mean(res, dim=0)
            means_abs[i] = torch.mean(res.abs(), dim=0)

            partition_z += torch.mean(torch.abs(self.p.val) / q, dim=0)
            # log_p = torch.log(torch.clamp(prob_abs, min=1e-16))
            # loss += prob_abs / q / z * (log_p - self.p.log_q - torch.log(z))

            # z = self.p.samples.detach().cpu()
            # weights = (res / res.abs()).detach().cpu()
            # for d in range(num_vars):
            #     hist, bin_edges = torch.histogram(
            #         z[:, d], bins=bins, range=hist_range, density=True
            #     )
            #     histr[:, d] += hist
            #     hist, bin_edges = torch.histogram(
            #         z[:, d],
            #         bins=bins,
            #         range=hist_range,
            #         weight=weights,
            #         density=True,
            #     )
            #     histr_weight[:, d] += hist
        partition_z /= num_blocks

        print("Start statistical analysis...")
        start_time = time.time()
        # while calculate_correlation(means_t) > correlation_threshold:
        while (
            kstest(
                means_t.cpu(),
                "norm",
                args=(means_t.mean().item(), means_t.std().item()),
            )[1]
            < 0.05
        ):
            print("correlation too high, merge blocks")
            if num_blocks <= 64:
                warn(
                    "blocks too small, increase burn-in or reduce thinning",
                    category=UserWarning,
                )
                break
            num_blocks //= 2
            means_t = (
                means_t[torch.arange(0, num_blocks * 2, 2, device=device)]
                + means_t[torch.arange(1, num_blocks * 2, 2, device=device)]
            ) / 2.0
            means_abs = (
                means_abs[torch.arange(0, num_blocks * 2, 2, device=device)]
                + means_abs[torch.arange(1, num_blocks * 2, 2, device=device)]
            ) / 2.0
        print("new block number: ", num_blocks)

        statistic, p_value = kstest(
            means_t.cpu(), "norm", args=(means_t.mean().item(), means_t.std().item())
        )
        print(f"K-S test: statistic {statistic}, p-value {p_value}.")

        mean_combined = torch.mean(means_t)
        error_combined = torch.std(means_t) / num_blocks**0.5

        mean_abs = means_abs.mean().item()
        std_abs = means_abs.std().item()
        statistic, p_value = kstest(
            means_abs.cpu(),
            "norm",
            args=(mean_abs, std_abs),
        )
        print(
            f"K-S test for absolute values: statistic {statistic}, p-value {p_value}."
        )
        print(f"Integrated |f|: Mean: {mean_abs}, std: {std_abs/num_blocks**0.5}.")

        print(f"Statistical analysis time: {time.time() - start_time} s")
        return (
            mean_combined,
            error_combined,
            # bin_edges,
            # histr / num_blocks,
            # histr_weight / num_blocks,
            partition_z,
        )

    @torch.no_grad()
    def loss_block(self, num_blocks, partition_z=1.0):
        num_samples = self.p.batchsize

        loss = torch.tensor(0.0, device=self.p.samples.device)
        for _ in range(num_blocks):
            self.p.samples, self.p.log_q = self.q0(num_samples)
            for flow in self.flows:
                self.p.samples, log_det = flow(self.p.samples)
                self.p.log_q -= log_det
            self.p.val = self.p.prob(self.p.samples)

            prob_abs = torch.abs(self.p.val)
            log_p = torch.log(torch.clamp(prob_abs, min=1e-16))
            loss += torch.mean(
                prob_abs
                / torch.exp(self.p.log_q)
                / partition_z
                * (log_p - self.p.log_q - torch.log(partition_z))
            )
        return loss / num_blocks


    @torch.no_grad()
    def histogram(self, extvar_dim, bins, range=(0.0, 1.0), has_weight=True):
        """Plots histogram of samples from flow-based approximate distribution

        Args:
          extvar_dim: Dimension of variable to plot histogram for
          bins: int or 1D Tensor. If int, defines the number of equal-width bins. If tensor, defines the sequence of bin edges including the rightmost edge.
          range: Range of the bins.
          has_weight: Flag whether to use weights for histogram. If True, weights are proportional to the probability of each sample. If False, weights are all equal.
        """
        num_samples = self.p.batchsize
        z, log_q = self.sample(num_samples)
        weights = self.p.prob(z) / torch.abs(self.p.prob(z))

        z = self.p.samples.detach().cpu()
        weights = weights.detach().cpu()

        if has_weight:
            histr, bins = torch.histogram(
                z[:, extvar_dim], bins=bins, range=range, weight=weights, density=True
            )
        else:
            histr, bins = torch.histogram(
                z[:, extvar_dim], bins=bins, range=range, density=True
            )

        return histr, bins

    @torch.no_grad()
    def mcmc_sample(self, steps=1, init=False, alpha=0.1):
        batch_size = self.p.batchsize
        device = self.p.samples.device
        vars_shape = self.p.samples.shape
        epsilon = 1e-16  # Small value to ensure numerical stability

        proposed_samples = torch.empty(vars_shape, device=device)
        proposed_log_q = torch.empty(batch_size, device=device)
        if init:  # Initialize chains
            self.p.samples[:], self.p.log_q[:] = self.q0(batch_size)
            for flow in self.flows:
                self.p.samples, log_det = flow(self.p.samples)
                self.p.log_q -= log_det

        current_weight = alpha * torch.exp(self.p.log_q) + (1 - alpha) * torch.abs(
            self.p.prob(self.p.samples)
        )
        new_weight = torch.empty(batch_size, device=device)
        acceptance_probs = torch.empty(batch_size, device=device)
        accept = torch.empty(batch_size, device=device, dtype=torch.bool)

        for _ in range(steps):
            # Propose new samples using the normalizing flow
            proposed_samples[:], proposed_log_q[:] = self.q0(batch_size)
            for flow in self.flows:
                proposed_samples, proposed_log_det = flow(proposed_samples)
                proposed_log_q -= proposed_log_det

            new_weight[:] = alpha * torch.exp(proposed_log_q) + (1 - alpha) * torch.abs(
                self.p.prob(proposed_samples)
            )
            current_weight[:] = torch.clamp(current_weight, min=epsilon)
            new_weight[:] = torch.clamp(new_weight, min=epsilon)
            # Compute acceptance probabilities
            acceptance_probs[:] = (
                new_weight / current_weight * torch.exp(self.p.log_q - proposed_log_q)
            )  # Pi(x') / Pi(x) * q(x) / q(x')

            # Accept or reject the proposals
            accept[:] = torch.rand(batch_size, device=device) <= acceptance_probs
            self.p.samples = torch.where(
                accept.unsqueeze(1), proposed_samples, self.p.samples
            )
            current_weight = torch.where(accept, new_weight, current_weight)
            self.p.log_q = torch.where(accept, proposed_log_q, self.p.log_q)
        return self.p.samples
    @torch.no_grad()
    def mcmc_integration(
        self,
        len_chain=1000,
        burn_in=None,
        thinning=1,
        alpha=0.1,
        step_size=0.2,  # uniform random walk step size
        mix_rate=0.0,  # mix global random sampling with random walk
        adaptive=False,
        adapt_acc_rate=0.4,
    ):
        """
        Perform MCMC integration using batch processing. Using the Metropolis-Hastings algorithm to sample the distribution:
        Pi(x) = alpha * q(x) + (1 - alpha) * p(x),
        where q(x) is the learned distribution by the normalizing flow, and p(x) is the target distribution.

        Args:
            len_chain: Number of samples to draw.
            burn_in: Number of initial samples to discard.
            thinning: Interval to thin the chain.
            alpha: Annealing parameter.
            step_size: random walk step size.

        Returns:
            mean, error: Mean and standard variance of the integrated samples.
        """
        epsilon = 1e-16  # Small value to ensure numerical stability
        batch_size = self.p.batchsize
        device = self.p.samples.device
        vars_shape = self.p.samples.shape
        if burn_in is None:
            burn_in = len_chain // 4

        # Initialize chains
        start_time = time.time()
        proposed_z = torch.empty(vars_shape, device=device)
        proposed_samples = torch.empty(vars_shape, device=device)
        proposed_log_q = torch.empty(batch_size, device=device)

        current_z, self.p.log_q[:] = self.q0(batch_size)
        self.p.samples[:] = current_z
        for flow in self.flows:
            self.p.samples[:], log_det = flow(self.p.samples)
            self.p.log_q -= log_det

        current_weight = alpha * torch.exp(self.p.log_q) + (1 - alpha) * torch.abs(
            self.p.prob(self.p.samples)
        )  # Pi(x) = alpha * q(x) + (1 - alpha) * p(x)
        torch.clamp(current_weight, min=epsilon, out=current_weight)
        new_weight = torch.empty(batch_size, device=device)
        acceptance_probs = torch.empty(batch_size, device=device)
        accept = torch.empty(batch_size, device=device, dtype=torch.bool)

        bool_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        print(f"Initialization time: {time.time() - start_time} s")

        # burn-in
        start_time = time.time()
        for i in range(burn_in):
            # Propose new samples using the normalizing flow
            bool_mask[:] = torch.rand(batch_size, device=device) > mix_rate
            proposed_z[:], proposed_log_q[:] = self.q0(batch_size)
            proposed_z[bool_mask, :] = (
                current_z[bool_mask, :] + (proposed_z[bool_mask, :] - 0.5) * step_size
            ) % 1.0
            # proposed_z[bool_mask, :] = (
            #     current_z
            #     + torch.normal(mu, step_size, size=vars_shape, device=device)
            # ) % 1.0
            proposed_samples[:] = proposed_z
            for flow in self.flows:
                proposed_samples[:], proposed_log_det = flow(proposed_samples)
                proposed_log_q -= proposed_log_det

            new_weight[:] = alpha * torch.exp(proposed_log_q) + (1 - alpha) * torch.abs(
                self.p.prob(proposed_samples)
            )
            torch.clamp(new_weight, min=epsilon, out=new_weight)
            # Compute acceptance probabilities
            acceptance_probs[:] = (
                new_weight / current_weight * torch.exp(self.p.log_q - proposed_log_q)
            )  # Pi(x') / Pi(x) * q(x) / q(x')

            # Accept or reject the proposals
            accept[:] = torch.rand(batch_size, device=device) <= acceptance_probs
            self.p.samples = torch.where(
                accept.unsqueeze(1), proposed_samples, self.p.samples
            )
            current_z = torch.where(accept.unsqueeze(1), proposed_z, current_z)
            current_weight = torch.where(accept, new_weight, current_weight)
            self.p.log_q = torch.where(accept, proposed_log_q, self.p.log_q)
            # self.p.log_q[accept] = proposed_log_q[accept]

            if adaptive and i % 100 == 0 and i > 0:
                accept_rate = accept.sum().item() / batch_size
                if accept_rate < adapt_acc_rate:
                    step_size *= 0.9
                else:
                    step_size *= 1.1
        print("Adjusted step size: ", step_size)
        print(f"Burn-in time: {time.time() - start_time} s")

        self.p.val[:] = self.p.prob(self.p.samples)
        new_prob = torch.empty_like(self.p.val)

        ref_values = torch.zeros(batch_size, device=device)
        values = torch.zeros(batch_size, device=device)
        abs_values = torch.zeros(batch_size, device=device)
        var_p = torch.zeros(batch_size, device=device)
        var_q = torch.zeros(batch_size, device=device)
        cov_pq = torch.zeros(batch_size, device=device)
        num_measure = 0

        start_time = time.time()
        for i in range(len_chain):
            # Propose new samples using the normalizing flow
            bool_mask[:] = torch.rand(batch_size, device=device) > mix_rate
            proposed_z[:], proposed_log_q[:] = self.q0(batch_size)
            proposed_z[bool_mask, :] = (
                current_z[bool_mask, :] + (proposed_z[bool_mask, :] - 0.5) * step_size
            ) % 1.0
            # proposed_z[bool_mask, :] = (
            #     current_z[bool_mask, :]
            #     + torch.normal(mu, step_size, size=vars_shape, device=device)
            # ) % 1.0
            proposed_samples[:] = proposed_z
            for flow in self.flows:
                proposed_samples[:], proposed_log_det = flow(proposed_samples)
                proposed_log_q -= proposed_log_det

            new_prob[:] = self.p.prob(proposed_samples)
            new_weight[:] = alpha * torch.exp(proposed_log_q) + (1 - alpha) * torch.abs(
                new_prob
            )
            torch.clamp(new_weight, min=epsilon, out=new_weight)
            # Compute acceptance probabilities
            acceptance_probs[:] = (
                new_weight / current_weight * torch.exp(self.p.log_q - proposed_log_q)
            )  # Pi(x') / Pi(x) * q(x) / q(x')

            # Accept or reject the proposals
            accept = torch.rand(batch_size, device=device) <= acceptance_probs
            if i % 400 == 0:
                print("acceptance rate: ", accept.sum().item() / batch_size)

            current_z = torch.where(accept.unsqueeze(1), proposed_z, current_z)
            self.p.val = torch.where(accept, new_prob, self.p.val)
            current_weight = torch.where(accept, new_weight, current_weight)
            self.p.log_q = torch.where(accept, proposed_log_q, self.p.log_q)
            # self.p.log_q[accept] = proposed_log_q[accept]

            # Measurement
            if i % thinning == 0:
                num_measure += 1

                values += self.p.val / current_weight
                ref_values += torch.exp(self.p.log_q) / current_weight
                abs_values += torch.abs(self.p.val / current_weight)

                var_p += (self.p.val / current_weight) ** 2
                var_q += (torch.exp(self.p.log_q) / current_weight) ** 2
                cov_pq += self.p.val * torch.exp(self.p.log_q) / current_weight**2
        values /= num_measure
        ref_values /= num_measure
        abs_values /= num_measure
        var_p /= num_measure
        var_q /= num_measure
        cov_pq /= num_measure
        print(f"MCMC with measurement time: {time.time() - start_time} s")

        # Statistical analysis
        print("Start statistical analysis...")
        start_time = time.time()
        total_num_measure = num_measure * batch_size
        while (
            kstest(
                values.cpu(), "norm", args=(values.mean().item(), values.std().item())
            )[1]
            < 0.05
            or kstest(
                ref_values.cpu(),
                "norm",
                args=(ref_values.mean().item(), ref_values.std().item()),
            )[1]
            < 0.05
        ):
            print("correlation too high, merge blocks")
            if batch_size <= 64:
                warn(
                    "blocks too small, increase burn-in or reduce thinning",
                    category=UserWarning,
                )
                break
            batch_size //= 2
            even_idx = torch.arange(0, batch_size * 2, 2, device=device)
            odd_idx = torch.arange(1, batch_size * 2, 2, device=device)
            values = (values[even_idx] + values[odd_idx]) / 2.0
            abs_values = (abs_values[even_idx] + abs_values[odd_idx]) / 2.0
            ref_values = (ref_values[even_idx] + ref_values[odd_idx]) / 2.0
            var_p = (var_p[even_idx] + var_p[odd_idx]) / 2.0
            var_q = (var_q[even_idx] + var_q[odd_idx]) / 2.0
            cov_pq = (cov_pq[even_idx] + cov_pq[odd_idx]) / 2.0
        print("new block number: ", batch_size)

        statistic, p_value = kstest(
            values.cpu(), "norm", args=(values.mean().item(), values.std().item())
        )
        print(f"K-S test of values: statistic {statistic}, p-value {p_value}")

        statistic, p_value = kstest(
            ref_values.cpu(),
            "norm",
            args=(ref_values.mean().item(), ref_values.std().item()),
        )
        print(f"K-S test of ref_values: statistic {statistic}, p-value {p_value}")

        ratio_mean = torch.mean(values) / torch.mean(ref_values)
        abs_val_mean = torch.mean(abs_values) / torch.mean(ref_values)

        cov_matrix = torch.cov(torch.stack((values, ref_values)))
        print("covariance matrix: ", cov_matrix)
        ratio_var = (
            cov_matrix[0, 0]
            - 2 * ratio_mean * cov_matrix[0, 1]
            + ratio_mean**2 * cov_matrix[1, 1]
        ) / torch.mean(ref_values) ** 2
        ratio_err = (ratio_var / batch_size) ** 0.5

        values /= ref_values
        # print("correlation of combined values: ", calculate_correlation(values))
        _mean = torch.mean(values)
        _std = torch.std(values)
        error = _std / batch_size**0.5

        print("old result: {:.5e} +- {:.5e}".format(_mean.item(), error.item()))

        statistic, p_value = kstest(
            values.cpu(), "norm", args=(_mean.item(), _std.item())
        )
        print(f"K-S test of ratio values: statistic {statistic}, p-value {p_value}")

        abs_values /= ref_values
        err_absval = torch.std(abs_values) / batch_size**0.5
        print(
            "|f(x)| Integration results: {:.5e} +/- {:.5e}".format(
                abs_val_mean.item(), err_absval.item()
            )
        )

        err_var_p = torch.std(var_p) / batch_size**0.5
        err_var_q = torch.std(var_q) / batch_size**0.5
        err_cov_pq = torch.std(cov_pq) / batch_size**0.5
        print(
            "variance of p: {:.5e} +/- {:.5e}".format(
                var_p.mean().item(), err_var_p.item()
            )
        )
        print(
            "variance of q: {:.5e} +/- {:.5e}".format(
                var_q.mean().item(), err_var_q.item()
            )
        )
        print(
            "covariance of pq: {:.5e} +/- {:.5e}".format(
                cov_pq.mean().item(), err_cov_pq.item()
            )
        )

        I_alpha = alpha + (1 - alpha) * abs_val_mean.item()
        print("I_alpha: ", I_alpha)

        var_pq_mean = I_alpha**2 * (
            var_p.mean() + var_q.mean() * ratio_mean**2 - 2 * ratio_mean * cov_pq.mean()
        )
        var_pq = (alpha + (1 - alpha) * abs_values) ** 2 * (
            var_p + var_q * values**2 - 2 * values * cov_pq
        )
        var_pq_err = torch.std(var_pq) / batch_size**0.5
        print(
            "Composite variance: {:.5e} +/- {:.5e}  ".format(
                var_pq_mean.item(), var_pq_err.item()
            ),
            var_pq.mean(),
        )
        print(
            "theoretical estimated error : {:.5e}".format(
                (var_pq_mean / total_num_measure).item() ** 0.5
            )
        )

        print(f"Statistical analysis time: {time.time() - start_time} s")

        if adaptive:
            return ratio_mean, ratio_err, step_size
        else:
            return ratio_mean, ratio_err
