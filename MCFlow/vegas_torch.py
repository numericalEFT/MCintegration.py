import torch
import vegas
from warnings import warn
from scipy.stats import kstest
import time
import traceback
import numpy as np


class VegasMap(torch.nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        target,
        # vegas_map,
        num_input_channels,
        integration_region,
        batchsize,
        num_adapt_samples=1000000,
        num_increments=1000,
        niters=20,
        alpha=1.0,
    ):
        super().__init__()

        vegas_map = vegas.AdaptiveMap(integration_region, ninc=num_increments)

        nblock = num_adapt_samples // batchsize
        num_adapt_samples = nblock * batchsize
        # y = torch.rand(num_adapt_samples, num_input_channels, dtype=torch.float64)
        y_np = np.random.uniform(0.0, 1.0, (num_adapt_samples, num_input_channels))
        fx = torch.empty(num_adapt_samples, dtype=torch.float64)

        # @vegas.batchintegrand
        # def func(x):
        #     return torch.Tensor.numpy(target.prob(torch.Tensor(x)))
        # vegas_map.adapt_to_samples(y_np, func, nitn=niters)

        x = torch.empty(num_adapt_samples, num_input_channels, dtype=torch.float64)
        jac = torch.empty(num_adapt_samples, dtype=torch.float64)
        f2 = torch.empty(num_adapt_samples, dtype=torch.float64)
        for _ in range(niters):
            # vegas_map.map(y.numpy(), x.numpy(), jac.numpy())
            vegas_map.map(y_np, x.numpy(), jac.numpy())
            for i in range(nblock):
                fx[i * batchsize : (i + 1) * batchsize] = target.prob(
                    x[i * batchsize : (i + 1) * batchsize]
                )
            f2 = (jac * fx) ** 2
            # vegas_map.add_training_data(y.numpy(), f2.numpy())
            vegas_map.add_training_data(y_np, f2.numpy())
            vegas_map.adapt(alpha=alpha)

        self.register_buffer("y", torch.empty(batchsize, num_input_channels))
        self.register_buffer("grid", torch.Tensor(vegas_map.grid))
        self.register_buffer("inc", torch.Tensor(vegas_map.inc))
        self.register_buffer("ninc", torch.tensor(num_increments))
        self.register_buffer("dim", torch.tensor(num_input_channels))
        self.register_buffer("x", torch.empty(batchsize, num_input_channels))
        self.register_buffer("jac", torch.ones(batchsize))

        self.target = target

    @torch.no_grad()
    def forward(self, y):
        y_ninc = y * self.ninc
        iy = torch.floor(y_ninc).long()
        dy_ninc = y_ninc - iy

        x = torch.empty_like(y)
        jac = torch.ones(y.shape[0], device=x.device)
        # self.jac.fill_(1.0)
        for d in range(self.dim):
            # Handle the case where iy < ninc
            mask = iy[:, d] < self.ninc
            if mask.any():
                x[mask, d] = (
                    self.grid[d, iy[mask, d]]
                    + self.inc[d, iy[mask, d]] * dy_ninc[mask, d]
                )
                jac[mask] *= self.inc[d, iy[mask, d]] * self.ninc

            # Handle the case where iy >= ninc
            mask_inv = ~mask
            if mask_inv.any():
                x[mask_inv, d] = self.grid[d, self.ninc]
                jac[mask_inv] *= self.inc[d, self.ninc - 1] * self.ninc

        return x, jac

    @torch.no_grad()
    def inverse(self, x):
        # self.jac.fill_(1.0)
        y = torch.empty_like(x)
        jac = torch.ones(x.shape[0], device=x.device)
        for d in range(self.dim):
            iy = torch.searchsorted(self.grid[d, :], x[:, d].contiguous(), right=True)

            mask_valid = (iy > 0) & (iy <= self.ninc)
            mask_lower = iy <= 0
            mask_upper = iy > self.ninc

            # Handle valid range (0 < iy <= self.ninc)
            if mask_valid.any():
                iyi_valid = iy[mask_valid] - 1
                y[mask_valid, d] = (
                    iyi_valid
                    + (x[mask_valid, d] - self.grid[d, iyi_valid])
                    / self.inc[d, iyi_valid]
                ) / self.ninc
                jac[mask_valid] *= self.inc[d, iyi_valid] * self.ninc

            # Handle lower bound (iy <= 0)\
            if mask_lower.any():
                y[mask_lower, d] = 0.0
                jac[mask_lower] *= self.inc[d, 0] * self.ninc

            # Handle upper bound (iy > self.ninc)
            if mask_upper.any():
                y[mask_upper, d] = 1.0
                jac[mask_upper] *= self.inc[d, self.ninc - 1] * self.ninc

        return y, jac

    @torch.no_grad()
    def integrate_block(self, num_blocks):
        print("Estimating integral from trained network")

        num_samples = self.y.shape[0]
        num_vars = self.y.shape[1]
        # Pre-allocate tensor for storing means and histograms
        means_t = torch.empty(num_blocks, device=self.y.device)
        means_abs = torch.empty_like(means_t)

        # Loop to fill the tensor with mean values
        for i in range(num_blocks):
            self.y[:] = torch.rand(num_samples, num_vars, device=self.y.device)
            self.x[:], self.jac[:] = self.forward(self.y)

            res = self.target.prob(self.x) * self.jac
            means_t[i] = torch.mean(res, dim=0)
            means_abs[i] = torch.mean(res.abs(), dim=0)

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
                    "blocks too small, try increasing num_blocks",
                    category=UserWarning,
                )
                break
            num_blocks //= 2
            means_t = (
                means_t[torch.arange(0, num_blocks * 2, 2, device=self.y.device)]
                + means_t[torch.arange(1, num_blocks * 2, 2, device=self.y.device)]
            ) / 2.0
            means_abs = (
                means_abs[torch.arange(0, num_blocks * 2, 2, device=self.y.device)]
                + means_abs[torch.arange(1, num_blocks * 2, 2, device=self.y.device)]
            ) / 2.0
        print("Final number of blocks: ", num_blocks)

        statistic, p_value = kstest(
            means_t.cpu(), "norm", args=(means_t.mean().item(), means_t.std().item())
        )
        print(f"K-S test: statistic {statistic}, p-value {p_value}.")

        # Compute mean and standard deviation directly on the tensor
        mean_combined = torch.mean(means_t)
        std_combined = torch.std(means_t) / num_blocks**0.5

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

        return (
            mean_combined,
            std_combined,
        )

    @torch.no_grad()
    def integrate_block_histr(self, num_blocks, bins=25, hist_range=(0.0, 1.0)):
        print("Estimating integral from trained network")

        num_samples = self.y.shape[0]
        num_vars = self.y.shape[1]
        means_t = torch.empty(num_blocks, device=self.y.device)
        # Pre-allocate tensor for storing means and histograms
        with torch.device("cpu"):
            if isinstance(bins, int):
                histr = torch.zeros(bins, num_vars, device=self.y.device)
                histr_weight = torch.zeros(bins, num_vars, device=self.y.device)
            else:
                histr = torch.zeros(bins.shape[0], num_vars, device=self.y.device)
                histr_weight = torch.zeros(
                    bins.shape[0], num_vars, device=self.y.device
                )

        # Loop to fill the tensor with mean values
        for i in range(num_blocks):
            self.y[:] = torch.rand(num_samples, num_vars, device=self.y.device)
            self.x[:], self.jac[:] = self.forward(self.y)

            res = self.target.prob(self.x) * self.jac
            means_t[i] = torch.mean(res, dim=0)

            z = self.x.detach().cpu()
            weights = res.detach().cpu()
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
        # Compute mean and standard deviation directly on the tensor
        mean_combined = torch.mean(means_t)
        std_combined = torch.std(means_t) / num_blocks**0.5

        return (
            mean_combined,
            std_combined,
            bin_edges,
            histr / num_blocks,
            histr_weight / num_blocks,
        )

    @torch.no_grad()
    def mcmc(
        self,
        len_chain=1000,
        burn_in=None,
        thinning=1,
        alpha=1.0,
        step_size=0.2,
        mu=0.0,
        type=None,  # None, "gaussian" or "uniform"
        mix_rate=0.0,
        adaptive=False,
        adapt_acc_rate=0.2,
    ):
        """
        Perform MCMC integration using batch processing. Using the Metropolis-Hastings algorithm to sample the distribution:
        Pi(x) = alpha * q(x) + (1 - alpha) * p(x),
        where q(x) is the learned distribution by the VEGAS map, and p(x) is the target distribution.

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
        device = self.y.device
        vars_shape = self.y.shape
        batch_size = vars_shape[0]
        if burn_in is None:
            burn_in = len_chain // 4

        # Initialize chains
        start_time = time.time()
        self.y[:] = torch.rand(vars_shape, device=device)
        current_samples, current_qinv = self.forward(self.y)
        current_weight = alpha / current_qinv + (1 - alpha) * torch.abs(
            self.target.prob(current_samples)
        )  # Pi(x) = alpha * q(x) + (1 - alpha) * p(x)
        torch.clamp(current_weight, min=epsilon, out=current_weight)

        proposed_y = torch.empty(vars_shape, device=device)
        proposed_samples = torch.empty(vars_shape, device=device)
        proposed_qinv = torch.empty(batch_size, device=device)
        new_weight = torch.empty(batch_size, device=device)

        bool_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        print("Initialziation time: ", time.time() - start_time)

        # burn-in
        start_time = time.time()
        for i in range(burn_in):
            # Propose new samples
            proposed_y[:] = torch.rand(vars_shape, device=device)
            # if type == "gaussian":
            #     bool_mask[:] = torch.rand(batch_size, device=device) > mix_rate
            #     proposed_y[bool_mask, :] = (
            #         self.y[bool_mask, :]
            #         + torch.normal(
            #             mu,
            #             step_size,
            #             size=[bool_mask.sum().item(), num_vars],
            #             device=device,
            #         )
            #     ) % 1.0
            # elif type == "uniform":
            bool_mask[:] = torch.rand(batch_size, device=device) > mix_rate
            proposed_y[bool_mask, :] = (
                self.y[bool_mask, :] + (proposed_y[bool_mask, :] - 0.5) * step_size
            ) % 1.0

            proposed_samples[:], proposed_qinv[:] = self.forward(proposed_y)

            new_weight[:] = alpha / proposed_qinv + (1 - alpha) * torch.abs(
                self.target.prob(proposed_samples)
            )
            torch.clamp(new_weight, min=epsilon, out=new_weight)
            # Compute acceptance probabilities
            acceptance_probs = (
                new_weight / current_weight * proposed_qinv / current_qinv
            )

            # Accept or reject the proposals
            accept = torch.rand(batch_size, device=device) <= acceptance_probs
            self.y = torch.where(accept.unsqueeze(1), proposed_y, self.y)
            current_samples = torch.where(
                accept.unsqueeze(1), proposed_samples, current_samples
            )
            current_weight = torch.where(accept, new_weight, current_weight)
            current_qinv = torch.where(accept, proposed_qinv, current_qinv)
            # self.p.log_q[accept] = proposed_log_q[accept]

            if adaptive and i % 100 == 0 and i > 0:
                accept_rate = accept.sum().item() / batch_size
                if accept_rate < adapt_acc_rate:
                    step_size *= 0.9
                else:
                    step_size *= 1.1
        print("Adjusted step size: ", step_size)
        print("Burn-in time: ", time.time() - start_time)

        current_prob = self.target.prob(current_samples)
        new_prob = torch.empty_like(current_prob)
        values = torch.zeros(batch_size, device=device)
        ref_values = torch.zeros_like(values)
        abs_values = torch.zeros_like(values)
        var_p = torch.zeros_like(values)
        var_q = torch.zeros_like(values)
        cov_pq = torch.zeros_like(values)
        num_measure = 0

        start_time = time.time()
        for i in range(len_chain):
            # Propose new samples
            proposed_y[:] = torch.rand(vars_shape, device=device)
            # if type == "gaussian":
            #     bool_mask[:] = torch.rand(batch_size, device=device) > mix_rate
            #     proposed_y[bool_mask, :] = (
            #         self.y[bool_mask, :]
            #         + torch.normal(
            #             mu,
            #             step_size,
            #             size=[bool_mask.sum().item(), num_vars],
            #             device=device,
            #         )
            #     ) % 1.0
            # elif type == "uniform":
            bool_mask[:] = torch.rand(batch_size, device=device) > mix_rate
            proposed_y[bool_mask, :] = (
                self.y[bool_mask, :] + (proposed_y[bool_mask, :] - 0.5) * step_size
            ) % 1.0

            proposed_samples[:], proposed_qinv[:] = self.forward(proposed_y)
            new_prob[:] = self.target.prob(proposed_samples)
            new_weight[:] = alpha / proposed_qinv + (1 - alpha) * torch.abs(new_prob)
            torch.clamp(new_weight, min=epsilon, out=new_weight)
            # Compute acceptance probabilities
            acceptance_probs = (
                new_weight / current_weight * proposed_qinv / current_qinv
            )

            # Accept or reject the proposals
            accept = torch.rand(batch_size, device=device) <= acceptance_probs
            if i % 400 == 0:
                print("acceptance rate: ", accept.sum().item() / batch_size)

            self.y = torch.where(accept.unsqueeze(1), proposed_y, self.y)
            current_prob = torch.where(accept, new_prob, current_prob)
            current_weight = torch.where(accept, new_weight, current_weight)
            current_qinv = torch.where(accept, proposed_qinv, current_qinv)

            # Measurement
            if i % thinning == 0:
                num_measure += 1

                values += current_prob / current_weight
                ref_values += 1 / (current_qinv * current_weight)
                abs_values += torch.abs(current_prob) / current_weight

                var_p += (current_prob / current_weight) ** 2
                var_q += 1 / (current_qinv * current_weight) ** 2
                cov_pq += current_prob / current_qinv / current_weight**2
        values /= num_measure
        abs_values /= num_measure
        ref_values /= num_measure
        var_p /= num_measure
        var_q /= num_measure
        cov_pq /= num_measure
        print("MCMC with measurement time: ", time.time() - start_time)

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
        print("new batch size: ", batch_size)

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
        print("correlation of ratio values: ", calculate_correlation(values).item())
        _mean = torch.mean(values)
        _std = torch.std(values)
        error = _std / batch_size**0.5

        print("old result: {:.5e} +- {:.5e}".format(_mean.item(), error.item()))

        statistic, p_value = kstest(
            values.cpu(), "norm", args=(_mean.item(), _std.item())
        )
        print(
            "K-S test of ratio values: statistic {:.5e}, p-value {:.5e}",
            statistic,
            p_value,
        )

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

        print("Statistical analysis time: ", time.time() - start_time)

        if adaptive:
            return ratio_mean, ratio_err, step_size
        else:
            return ratio_mean, ratio_err


# Function to calculate correlation between adjacent blocks
def calculate_correlation(x):
    x_centered = x[1:] - x.mean()
    y_centered = x[:-1] - x.mean()
    cov = torch.sum(x_centered * y_centered) / (len(x) - 1)
    return torch.abs(cov / torch.var(x))
