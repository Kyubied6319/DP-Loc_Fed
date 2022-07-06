import numpy as np
from math import exp, pi, log, ceil, log1p
import numpy as np
from scipy import special
import six

#SENSITIVITY = 1
#SENSITIVITY_tk= 69  # trace max hossza, l2-t felulrol becsuljuk l1-gyel
SIGMA_tk = 8
SIGMA_vae = 2.5
SIGMA_ffn = 3.1
DELTA = 4*(10**-5)
#BATCH_SIZE = 100
BATCH_SIZE_VAE = 200
BATCH_SIZE_FFN = 200
DATASET_SIZE = 400000
#EPOCHS = 15
EPOCHS_VAE = 15
EPOCHS_FFN = 15
#Q = BATCH_SIZE/DATASET_SIZE #0.000909091  # 200/(2.2 * 10**5) #(BATCH_SIZE/DATASET_SIZE)
Q_VAE = BATCH_SIZE_VAE/DATASET_SIZE
Q_FFN = BATCH_SIZE_FFN/DATASET_SIZE 
#T = int(EPOCHS * (DATASET_SIZE / BATCH_SIZE))  #16500  # 15*220000/200 # number of SGD iterations: epoch_num * DATASET_SIZE / BATCH_SIZE
T_VAE = int(EPOCHS_VAE * (DATASET_SIZE / BATCH_SIZE_VAE))
T_FFN = int(EPOCHS_FFN * (DATASET_SIZE / BATCH_SIZE_FFN))
print("BATCH_SIZE_VAE:", BATCH_SIZE_VAE)
print("BATCH_SIZE_FFN:", BATCH_SIZE_FFN)
print("T_VAE:", T_VAE)
print("T_FFN:", T_FFN)
l_max = 32 #lambda


## Auxiliary functions to compute alpha (taken from Google's tensorflow)

def _log_add(logx, logy):
    """Add two numbers in the log space."""
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:  # adding 0
        return b
    # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
    return log1p(exp(a - b)) + b  # log1p(x) = log(x + 1)

def _compute_log_a_int(q, sigma, alpha):
    """Compute log(A_alpha) for integer alpha. 0 < q < 1."""
    assert isinstance(alpha, six.integer_types)

    # Initialize with 0 in the log space.
    log_a = -np.inf

    for i in range(alpha + 1):
      log_coef_i = (
          log(special.binom(alpha, i)) + i * log(q) +
          (alpha - i) * log(1 - q))

      s = log_coef_i + (i * i - i) / (2 * (sigma**2))
      log_a = _log_add(log_a, s)

    return float(log_a)

def _compute_log_a(q, sigma, alpha):
    """Compute log(A_alpha) for any positive finite alpha."""
    if float(alpha).is_integer():
      return _compute_log_a_int(q, sigma, int(alpha))
    else:
      raise Exception('Fractional alpha is not supported')

###  Main part

'''
Alternative way to compute alpha (numerically unstable and much slower), it replaces _compute_log_a above:

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
#alpha topk:
alpha_topk = []
for l in range(1, l_max+1):
    alpha_val =  (l**2 + l)/(4*SIGMA_tk**2)
    alpha_topk.append(alpha_val)

# Computing alpha FFN
alpha_ffn = []
for l in range(1, l_max+1):
    alpha_val = _compute_log_a(Q_FFN, SIGMA_ffn, l+1)
    alpha_ffn.append(alpha_val)

# VAE
alpha_vae = []
for l in range(1, l_max+1):
    alpha_val = _compute_log_a(Q_VAE, SIGMA_vae, l+1)
    alpha_vae.append(alpha_val)

# only if vae and ffn epochs are the same:
eps_values = [(T_FFN * alpha_ffn[l-1] + T_VAE * alpha_vae[l-1] + alpha_topk[l-1] - np.log(DELTA)) / float(l) for l in range(1,l_max+1)]

print ("Iteration (VAE+FFN): %d Epsilon: %.2f" % (T_VAE + T_FFN, min(eps_values)))

#for i in range(T):
#    epsilon_values = [((i+1) * (alpha_values_ffn[l-1] + alpha_values_vae[l-1]) + alpha_topk[l-1] - np.log(DELTA)) / float(l) for l in range(1,l_max+1)]
#    eps_values.append(min(epsilon_values))

#print ("Iteration: %d Epsilon: %.2f" % (i + 1, eps_values[-1]))

