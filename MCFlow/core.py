import normflows as nf
import torch

class MCFlow(nf.NormalizingFlow):

    def __init__(self, q0, flows, p=None):
        super().__init__(q0, flows, p)
    
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

    def IS_forward_kld(self, num_samples=1, beta=1.0):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
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
    def integrate_block(
        self, num_blocks, bins=25, hist_range=(0.0, 1.0), correlation_threshold=0.2
    ):
        print("Estimating integral from trained network")

        num_samples = self.p.batchsize
        num_vars = self.p.ndims
        # Pre-allocate tensor for storing means and histograms
        means_t = torch.zeros(num_blocks)
        with torch.device("cpu"):
            if isinstance(bins, int):
                histr = torch.zeros(bins, num_vars)
                histr_weight = torch.zeros(bins, num_vars)
            else:
                histr = torch.zeros(bins.shape[0], num_vars)
                histr_weight = torch.zeros(bins.shape[0], num_vars)

        partition_z = torch.tensor(0.0, device=self.p.samples.device)
        for i in range(num_blocks):
            self.p.samples, self.p.log_q = self.q0(num_samples)
            for flow in self.flows:
                self.p.samples, self.p.log_det = flow(self.p.samples)
                self.p.log_q -= self.p.log_det
            self.p.val = self.p.prob(self.p.samples)
            q = torch.exp(self.p.log_q)
            res = self.p.val / q
            means_t[i] = torch.mean(res, dim=0)

            partition_z += torch.mean(torch.abs(self.p.val) / q, dim=0)
            # log_p = torch.log(torch.clamp(prob_abs, min=1e-16))
            # loss += prob_abs / q / z * (log_p - self.p.log_q - torch.log(z))

            z = self.p.samples.detach().cpu()
            weights = (res / res.abs()).detach().cpu()
            for d in range(num_vars):
                hist, bin_edges = torch.histogram(
                    z[:, d], bins=bins, range=hist_range, density=True
                )
                histr[:, d] += hist
                hist, bin_edges = torch.histogram(
                    z[:, d],
                    bins=bins,
                    range=hist_range,
                    weight=weights,
                    density=True,
                )
                histr_weight[:, d] += hist

        print("correlation of values: ", calculate_correlation(means_t))
        while calculate_correlation(means_t) > correlation_threshold:
            print("correlation too high, merge blocks")
            if num_blocks <= 64:
                warn(
                    "blocks too small, increase burn-in or reduce thinning",
                    category=UserWarning,
                )
                break
            num_blocks //= 2
            k = 0
            for j in range(0, num_blocks):
                means_t[j] = (means_t[k] + means_t[k + 1]) / 2.0
                k += 2
        mean_combined = torch.mean(means_t[:num_blocks])
        std_combined = torch.std(means_t[:num_blocks]) / num_blocks**0.5

        return (
            mean_combined,
            std_combined,
            bin_edges,
            histr / num_blocks,
            histr_weight / num_blocks,
            partition_z / num_blocks,
        )

    @torch.no_grad()
    def loss_block(self, num_blocks, partition_z=1.0):
        num_samples = self.p.batchsize

        loss = torch.tensor(0.0, device=self.p.samples.device)
        for i in range(num_blocks):
            self.p.samples, self.p.log_q = self.q0(num_samples)
            for flow in self.flows:
                self.p.samples, self.p.log_det = flow(self.p.samples)
                self.p.log_q -= self.p.log_det
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
    def mcmc_sample(self, steps=1, init=False):
        batch_size = self.p.batchsize
        device = self.p.samples.device
        vars_shape = self.p.samples.shape
        if init:  # Initialize chains
            self.p.samples[:], self.p.log_q[:] = self.q0(batch_size)
            for flow in self.flows:
                self.p.samples[:], self.p.log_det[:] = flow(self.p.samples)
                self.p.log_q -= self.p.log_det

        proposed_samples = torch.empty(vars_shape, device=device)
        proposed_log_det = torch.empty(batch_size, device=device)
        proposed_log_q = torch.empty(batch_size, device=device)

        current_prob = torch.abs(self.p.prob(self.p.samples))
        new_prob = torch.empty(batch_size, device=device)

        for _ in range(steps):
            # Propose new samples using the normalizing flow
            proposed_samples[:], proposed_log_q[:] = self.q0(batch_size)
            for flow in self.flows:
                proposed_samples[:], proposed_log_det[:] = flow(proposed_samples)
                proposed_log_q -= proposed_log_det

            new_prob[:] = torch.abs(self.p.prob(proposed_samples))

            # Compute acceptance probabilities
            acceptance_probs = torch.clamp(
                new_prob
                / current_prob  # Pi(x') / Pi(x)
                * torch.exp(
                    self.p.log_q - proposed_log_q  # q(x) / q(x')
                ),
                max=1,
            )

            # Accept or reject the proposals
            accept = torch.rand(batch_size, device=device) <= acceptance_probs
            self.p.samples = torch.where(
                accept.unsqueeze(1), proposed_samples, self.p.samples
            )
            current_prob = torch.where(accept, new_prob, current_prob)
            self.p.log_q = torch.where(accept, proposed_log_q, self.p.log_q)
        return self.p.samples

    @torch.no_grad()
    def mcmc_integration(
        self,
        num_blocks=100,
        len_chain=1000,
        burn_in=None,
        thinning=1,
        alpha=1.0,
        correlation_threshold=0.2,
    ):
        """
        Perform MCMC integration using batch processing. Using the Metropolis-Hastings algorithm to sample the distribution:
        Pi(x) = alpha * q(x) + (1 - alpha) * p(x),
        where q(x) is the learned distribution by the normalizing flow, and p(x) is the target distribution.

        Args:
            num_blocks: Number of blocks to divide the batch into.
            len_chain: Number of samples to draw.
            burn_in: Number of initial samples to discard.
            thinning: Interval to thin the chain.
            alpha: Annealing parameter.

        Returns:
            mean, error: Mean and standard variance of the integrated samples.
        """
        # epsilon = 1e-10  # Small value to ensure numerical stability
        batch_size = self.p.batchsize
        device = self.p.samples.device
        vars_shape = self.p.samples.shape
        if burn_in is None:
            burn_in = len_chain // 5

        # Initialize chains
        self.p.samples[:], self.p.log_q[:] = self.q0(batch_size)
        for flow in self.flows:
            self.p.samples[:], self.p.log_det[:] = flow(self.p.samples)
            self.p.log_q -= self.p.log_det
        proposed_samples = torch.empty(vars_shape, device=device)
        proposed_log_det = torch.empty(batch_size, device=device)
        proposed_log_q = torch.empty(batch_size, device=device)

        current_prob = alpha * torch.exp(self.p.log_q) + (1 - alpha) * torch.abs(
            self.p.prob(self.p.samples)
        )  # Pi(x) = alpha * q(x) + (1 - alpha) * p(x)
        new_prob = torch.empty(batch_size, device=device)

        for _ in range(burn_in):
            # Propose new samples using the normalizing flow
            proposed_samples[:], proposed_log_q[:] = self.q0(batch_size)
            for flow in self.flows:
                proposed_samples[:], proposed_log_det[:] = flow(proposed_samples)
                proposed_log_q -= proposed_log_det

            new_prob[:] = alpha * torch.exp(proposed_log_q) + (1 - alpha) * torch.abs(
                self.p.prob(proposed_samples)
            )

            # Compute acceptance probabilities
            acceptance_probs = torch.clamp(
                new_prob
                / current_prob  # Pi(x') / Pi(x)
                * torch.exp(
                    self.p.log_q - proposed_log_q  # q(x) / q(x')
                ),
                max=1,
            )

            # Accept or reject the proposals
            accept = torch.rand(batch_size, device=device) <= acceptance_probs
            self.p.samples = torch.where(
                accept.unsqueeze(1), proposed_samples, self.p.samples
            )
            current_prob = torch.where(accept, new_prob, current_prob)
            self.p.log_q = torch.where(accept, proposed_log_q, self.p.log_q)
            # self.p.log_q[accept] = proposed_log_q[accept]

        ref_values = torch.zeros(num_blocks, device=device)
        values = torch.zeros(num_blocks, device=device)
        abs_values = torch.zeros(num_blocks, device=device)
        block_size = batch_size // num_blocks
        num_measure = 0
        for i in range(len_chain):
            # Propose new samples using the normalizing flow
            proposed_samples[:], proposed_log_q[:] = self.q0(batch_size)
            for flow in self.flows:
                proposed_samples[:], proposed_log_det[:] = flow(proposed_samples)
                proposed_log_q -= proposed_log_det

            new_prob[:] = alpha * torch.exp(proposed_log_q) + (1 - alpha) * torch.abs(
                self.p.prob(proposed_samples)
            )

            # Compute acceptance probabilities
            acceptance_probs = torch.clamp(
                new_prob
                / current_prob  # Pi(x') / Pi(x)
                * torch.exp(
                    self.p.log_q - proposed_log_q  # q(x) / q(x')
                ),
                max=1,
            )
            # Accept or reject the proposals
            accept = torch.rand(batch_size, device=device) <= acceptance_probs
            self.p.samples = torch.where(
                accept.unsqueeze(1), proposed_samples, self.p.samples
            )
            current_prob = torch.where(accept, new_prob, current_prob)
            self.p.log_q = torch.where(accept, proposed_log_q, self.p.log_q)
            # self.p.log_q[accept] = proposed_log_q[accept]

            # Measurement
            if i % thinning == 0:
                num_measure += 1
                self.p.val = self.p.prob(self.p.samples) / current_prob

                for j in range(num_blocks):
                    start = j * block_size
                    end = (j + 1) * block_size
                    values[j] += torch.mean(self.p.val[start:end])
                    ref_values[j] += torch.mean(
                        torch.exp(self.p.log_q[start:end]) / current_prob[start:end]
                    )
                    abs_values[j] += torch.mean(torch.abs(self.p.val[start:end]))

        print("correlation of values: ", calculate_correlation(values))
        print("correlation of ref_values: ", calculate_correlation(ref_values))
        while (
            calculate_correlation(values) > correlation_threshold
            or calculate_correlation(abs_values) > correlation_threshold
        ):
            print("correlation too high, merge blocks")
            if num_blocks <= 64:
                warn(
                    "blocks too small, increase burn-in or reduce thinning",
                    category=UserWarning,
                )
                break
            num_blocks //= 2
            k = 0
            for j in range(0, num_blocks):
                values[j] = (values[k] + values[k + 1]) / 2.0
                abs_values[j] = (abs_values[k] + abs_values[k + 1]) / 2.0
                ref_values[j] = (ref_values[k] + ref_values[k + 1]) / 2.0
                k += 2
        values = values[:num_blocks]
        abs_values = abs_values[:num_blocks]
        ref_values = ref_values[:num_blocks]

        mean = torch.mean(values) / torch.mean(ref_values)
        abs_val_mean = torch.mean(abs_values) / torch.mean(ref_values)

        values /= ref_values
        print("correlation of combined values: ", calculate_correlation(values))
        error = torch.norm(values - mean) / num_blocks

        abs_values /= ref_values
        err_absval = torch.norm(abs_values - abs_val_mean) / num_blocks
        print(
            "|f(x)| Integration results: {:.5e} +/- {:.5e}",
            abs_val_mean,
            err_absval,
        )

        return mean, error
