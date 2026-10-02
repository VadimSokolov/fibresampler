"""Mixture / extended-fibre sampler (Hazelton's construction).

Target                f(x)  propto  prod 1/x_ij!      supported on F   (x >= 0)
Inflated companion    g(x)  propto  prod 1/Gamma(x_ij+1+delta)  supported on
                                    the extended fibre F^(-1) (x >= -1)
Mixture               h(x) = pi f(x) + (1-pi) g(x)

The chain runs Metropolis-Hastings on h using a *lattice* basis of moves over
the extended fibre.  Because g charges the rim, the walk can leave F, cross a
region that the lattice basis cannot traverse inside F, and re-enter.  The
latent indicator z is drawn conditionally, and the sub-chain {x : z = 1} is an
exact sample from f.

Note.  This is a generic mixture demonstrator: log_g below is a uniform
Gamma(delta) inflation.  Paper II's rim SLEMs (Table tab:rim, computed by
fig_rim.py) instead score g at the *shifted count* x_j+1 with the cell-specific
means, and measure the g-chain gap on the extended fibre, not this mixture
chain.  The paper reports the mixture sampler as specified, not evaluated; this
module is not tied to a reported number.
"""
import numpy as np
from scipy.special import gammaln, logsumexp
from fibre_core import swap_moves


UNIFORM = False          # module-level switch: target uniform-on-fibre


def log_f(X):
    X = np.asarray(X)
    if X.min() < 0:
        return -np.inf
    return 0.0 if UNIFORM else -gammaln(X + 1.0).sum()


def log_g(X, delta):
    X = np.asarray(X)
    if X.min() < -1:
        return -np.inf
    if UNIFORM:
        # flat on the fibre, gently damped on the rim so that the walk returns
        return -delta * float((X < 0).sum())
    return -gammaln(X + 1.0 + delta).sum()


def log_h(X, pi, delta, lZf, lZg):
    a = np.log(pi) + log_f(X) - lZf
    b = np.log1p(-pi) + log_g(X, delta) - lZg
    return logsumexp([a, b])


def normalisers(F, Fx, delta):
    lZf = logsumexp([log_f(X) for X in F])
    lZg = logsumexp([log_g(X, delta) for X in Fx])
    return lZf, lZg


def run(X0, moves, n_iter, pi, delta, lZf, lZg, rng, target="h"):
    """MH sweep.  target='h' is the mixture chain; target='f' restricts to F."""
    X = X0.copy()
    lp = log_h(X, pi, delta, lZf, lZg) if target == "h" else log_f(X)
    keep, zs = [], []
    for _ in range(n_iter):
        M = moves[rng.integers(len(moves))] * (1 if rng.random() < .5 else -1)
        Y = X + M
        lq = log_h(Y, pi, delta, lZf, lZg) if target == "h" else log_f(Y)
        if np.isfinite(lq) and np.log(rng.random()) < lq - lp:
            X, lp = Y, lq
        keep.append(X.copy())
        if target == "h":
            la = np.log(pi) + log_f(X) - lZf
            lb = np.log1p(-pi) + log_g(X, delta) - lZg
            zs.append(1 if np.log(rng.random()) < la - logsumexp([la, lb]) else 2)
        else:
            zs.append(1)
    return keep, np.array(zs)


def tv_to_truth(draws, F, delta=None):
    """Total variation distance between the empirical law of `draws` and f."""
    idx = {X.tobytes(): k for k, X in enumerate(F)}
    p = np.exp([log_f(X) for X in F]); p /= p.sum()
    cnt = np.zeros(len(F))
    for X in draws:
        k = idx.get(np.asarray(X).tobytes())
        if k is not None:
            cnt[k] += 1
    if cnt.sum() == 0:
        return 1.0
    return .5 * np.abs(cnt / cnt.sum() - p).sum()
