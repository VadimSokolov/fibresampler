"""Figure: tempering, rim, and their composition on a disconnecting move set."""
import sys, numpy as np
sys.path.insert(0, '.')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fibre_core import enumerate_fibre, swap_moves, components
from vs_hazelton import poisson_fibre_logw, tempering_gap

plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'figure.dpi': 200,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.bbox': 'tight'})
INK, ACC, WARM = '#22303c', '#2f6f9f', '#b5533c'

r, c = [4, 1, 4], [4, 1, 4]
F = enumerate_fibre(r, c, 0); Fx = enumerate_fibre(r, c, -1)
L = swap_moves(3, 3, True)
mus = np.logspace(-1, -8, 8)
gt, gr, gc = [], [], []
for mu in mus:
    th = np.ones(9); th[4] = mu; th = th.reshape(3, 3)
    lw = lambda X: poisson_fibre_logw(X, th)
    lwx = lambda X: poisson_fibre_logw(np.asarray(X) + 1, th)
    gt.append(max(tempering_gap(F, L, lw, [1., .5, 0.]), 1e-18))
    gr.append(max(tempering_gap(Fx, L, lwx, [1.], lower=-1), 1e-18))
    gc.append(max(tempering_gap(Fx, L, lwx, [1., .5, 0.], lower=-1), 1e-18))

fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.plot(mus, gc, 'k-*', lw=1.6, ms=8, label=r'rim $+$ ladder $(1, 1/2, 0)$: constant floor')
ax.plot(mus, gr, '-s', color=ACC, lw=1.3, ms=4, label=r'rim only: $\sim\mu$ (corridor created, not flattened)')
ax.plot(mus, gt, '--o', color=WARM, lw=1.3, ms=4, label=r'ladder $(1, 1/2, 0)$ on $\mathcal{F}$: reducible, gap $=0$')
ax.plot(mus, 4e-2 * mus, ':', color='#888', lw=1, label=r'slope-1 guide')
ax.set_xscale('log'); ax.set_yscale('log')
ax.invert_xaxis()
ax.set_ylim(1e-11, 1e-1)
ax.set_xlabel(r'bottleneck mean $\mu_{j^\star}$')
ax.set_ylabel(r'exact joint-chain spectral gap $1-\lambda_\star$')
ax.legend(frameon=False, fontsize=7.2, loc='lower left')
plt.savefig('../results/fig_rim_escape.pdf'); plt.close()
print('rim figure written'); print('floor', gc[-1])
