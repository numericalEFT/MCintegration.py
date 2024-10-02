import torch
import math
import numpy as np
import gvar
import sys

MINVAL = 10 ** (sys.float_info.min_10_exp + 50)
MAXVAL = 10 ** (sys.float_info.max_10_exp - 50)


class RAvg(gvar.GVar):
    def __init__(self, weighted=True, itn_results=None, sum_neval=0):
        if weighted:
            self._wlist = []
            self.weighted = True
        else:
            self._sum = 0.0
            self._varsum = 0.0
            self.weighted = False
            self._count = 0
        self._mlist = []

        self.itn_results = []
        if itn_results is None:
            super(RAvg, self).__init__(
                *gvar.gvar(0.0, 0.0).internaldata,
            )
        else:
            if isinstance(itn_results, bytes):
                itn_results = gvar.loads(itn_results)
            for r in itn_results:
                self.add(r)
        self.sum_neval = sum_neval

    def add(self, res):
        self.itn_results.append(res)
        if isinstance(res, gvar.GVarRef):
            return
        self._mlist.append(res.mean)
        if self.weighted:
            self._wlist.append(1 / (res.var if res.var > MINVAL else MINVAL))
            var = 1.0 / np.sum(self._wlist)
            sdev = np.sqrt(var)
            mean = np.sum([w * m for w, m in zip(self._wlist, self._mlist)]) * var
            super(RAvg, self).__init__(*gvar.gvar(mean, sdev).internaldata)
        else:
            self._sum += res.mean
            self._varsum += res.var
            self._count += 1
            mean = self._sum / self._count
            var = self._varsum / self._count**2
            super(RAvg, self).__init__(*gvar.gvar(mean, np.sqrt(var)).internaldata)

    def extend(self, ravg):
        """Merge results from :class:`RAvg` object ``ravg`` after results currently in ``self``."""
        for r in ravg.itn_results:
            self.add(r)
        self.sum_neval += ravg.sum_neval

    def _chi2(self):
        if len(self.itn_results) <= 1:
            return 0.0
        if self.weighted:
            wavg = self.mean
            ans = 0.0
            for m, w in zip(self._mlist, self._wlist):
                ans += (wavg - m) ** 2 * w
            return ans
        else:
            wavg = self.mean
            ans = np.sum([(m - wavg) ** 2 for m in self._mlist]) / (
                self._varsum / self._count
            )
            return ans

    chi2 = property(_chi2, None, None, "*chi**2* of weighted average.")

    def _dof(self):
        return len(self.itn_results) - 1

    dof = property(
        _dof, None, None, "Number of degrees of freedom in weighted average."
    )

    def _nitn(self):
        return len(self.itn_results)

    nitn = property(_nitn, None, None, "Number of iterations.")

    def _Q(self):
        return (
            gvar.gammaQ(self.dof / 2.0, self.chi2 / 2.0)
            if self.dof > 0 and self.chi2 >= 0
            else float("nan")
        )

    Q = property(
        _Q,
        None,
        None,
        "*Q* or *p-value* of weighted average's *chi**2*.",
    )

    def _avg_neval(self):
        return self.sum_neval / self.nitn if self.nitn > 0 else 0

    avg_neval = property(
        _avg_neval, None, None, "Average number of integrand evaluations per iteration."
    )

    def summary(self, weighted=None):
        """Assemble summary of results, iteration-by-iteration, into a string.

        Args:
            weighted (bool): Display weighted averages of results from different
                iterations if ``True``; otherwise show unweighted averages.
                Default behavior is determined by |vegas|.
        """
        if weighted is None:
            weighted = self.weighted
        acc = RAvg(weighted=weighted)
        linedata = []
        for i, res in enumerate(self.itn_results):
            acc.add(res)
            if i > 0:
                chi2_dof = acc.chi2 / acc.dof
                Q = acc.Q
            else:
                chi2_dof = 0.0
                Q = 1.0
            itn = "%3d" % (i + 1)
            integral = "%-15s" % res
            wgtavg = "%-15s" % acc
            chi2dof = "%8.2f" % (acc.chi2 / acc.dof if i != 0 else 0.0)
            Q = "%8.2f" % (acc.Q if i != 0 else 1.0)
            linedata.append((itn, integral, wgtavg, chi2dof, Q))
        nchar = 5 * [0]
        for data in linedata:
            for i, d in enumerate(data):
                if len(d) > nchar[i]:
                    nchar[i] = len(d)
        fmt = "%%%ds   %%-%ds %%-%ds %%%ds %%%ds\n" % tuple(nchar)
        if weighted:
            ans = fmt % ("itn", "integral", "wgt average", "chi2/dof", "Q")
        else:
            ans = fmt % ("itn", "integral", "average", "chi2/dof", "Q")
        ans += len(ans[:-1]) * "-" + "\n"
        for data in linedata:
            ans += fmt % data
        return ans

    def converged(self, rtol, atol):
        return self.sdev < atol + rtol * abs(self.mean)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
