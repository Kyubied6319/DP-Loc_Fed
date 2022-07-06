from math import exp, pi, log, log1p

import numpy as np
import six
from scipy import special


class MomentsAccountant:
    # delta is pre-fixed, the goal is to compute eps (could be reversed but not now)
    def __init__(self, delta, sigma, sampling_prob, sensitivity=1.0):
        self.SENSITIVITY = sensitivity
        self.SIGMA = sigma
        self.DELTA = delta
        self.Q = sampling_prob

        # only for iterator mode (see make_iter() below)
        self.iter_num = 0

        # Keep it fixed:
        self.l_max = 32  # lambda

        # DGM: should be kept fixed
        self.NU = 1e-4

        # Alpha computation for different values of lambda from 1 to l_max
        self.alpha_values_gm = []
        self.alpha_values_dgm = []

        # only for discrete gaussian
        kappa = lambda x: 2 * exp(-2 * pi ** 2 * x ** 2) / (1 - exp(-6 * pi ** 2 * x ** 2))

        for l in range(1, self.l_max + 1):
            alpha_val = self.compute_gauss_alpha(self.SIGMA, self.Q, l=l)
            # print ("lambda = ", l, " alpha = ", alpha_val)
            self.alpha_values_gm.append(alpha_val)
            diff = log((1 + kappa(self.SIGMA * self.SENSITIVITY)) ** l / (
                    1 - kappa(self.SIGMA * self.SENSITIVITY) ** (l + 1))) + 3 * log((1 + self.NU) / (1 - self.NU))
            self.alpha_values_dgm.append(alpha_val + diff)

    def _log_add(self, logx, logy):
        """Add two numbers in the log space."""
        a, b = min(logx, logy), max(logx, logy)
        if a == -np.inf:  # adding 0
            return b
        # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
        return log1p(exp(a - b)) + b  # log1p(x) = log(x + 1)

    def _compute_log_a_int(self, q, sigma, alpha):
        """Compute log(A_alpha) for integer alpha. 0 < q < 1."""
        assert isinstance(alpha, six.integer_types)

        # Initialize with 0 in the log space.
        log_a = -np.inf

        for i in range(alpha + 1):
            log_coef_i = (
                    log(special.binom(alpha, i)) + i * log(q) +
                    (alpha - i) * log(1 - q))

            s = log_coef_i + (i * i - i) / (2 * (sigma ** 2))
            log_a = self._log_add(log_a, s)

        return float(log_a)

    def _compute_log_a(self, q, sigma, alpha):
        """Compute log(A_alpha) for any positive finite alpha."""
        if float(alpha).is_integer():
            return self._compute_log_a_int(q, sigma, int(alpha))
        else:
            raise Exception('Fractional alpha is not supported')

    def compute_gauss_alpha(self, p_sigma, q, l, p_sens=1.0):
        '''
        import scipy.integrate as integrate
        from scipy.stats import norm

        e_1 = lambda x: norm.pdf(x,0,p_sens*p_sigma)* (norm.pdf(x,0,p_sens*p_sigma) / ((1-q)*norm.pdf(x,0,p_sens*p_sigma) +
            q*norm.pdf(x,p_sens,p_sens*p_sigma)))**l
        e_2 = lambda x: ((1-q)*norm.pdf(x,0,p_sens*p_sigma) + q*norm.pdf(x,p_sens,p_sens*p_sigma))*\
            ( ( (1-q)*norm.pdf(x,0,p_sens*p_sigma) + q*norm.pdf(x,p_sens,p_sens*p_sigma)) /
                    norm.pdf(x,0,p_sens*p_sigma) ) **l

        E_1, _ = integrate.quad(e_1,-d, d)
        E_2, _ = integrate.quad(e_2,-d, d)
        return np.log( max([abs(E_1), abs(E_2)]))
        '''
        return self._compute_log_a(q, p_sigma, l + 1)

    def _compute_eps(self, alpha_values, T):
        eps_values = []

        for i in range(T):
            epsilon_values = [((i + 1) * alpha_values[l - 1] - np.log(self.DELTA)) / float(l) for l in
                              range(1, self.l_max + 1)]
            eps_values.append(min(epsilon_values))

        return eps_values

    # Epsilon computation from alphas (we return two arrays one for Gaussian Mechanism and the other to Discrete Gaussian, we have eps value per SGD iteration)
    def get_eps(self, iterations):
        return self.get_eps_gm(iterations)

    def get_eps_dgm(self, iterations):
        return self._compute_eps(self.alpha_values_dgm, iterations)

    def get_eps_gm(self, iterations):
        return self._compute_eps(self.alpha_values_gm, iterations)

    def make_iter(self, iterable, mechanism='gm'):
        alpha_values = self.alpha_values_gm if mechanism == 'gm' else self.alpha_values_dgm
        for item in iterable:
            epsilon_values = [((self.iter_num + 1) * alpha_values[l - 1] - np.log(self.DELTA)) / float(l) for l in
                              range(1, self.l_max + 1)]
            self.iter_num += 1

            yield min(epsilon_values), item
