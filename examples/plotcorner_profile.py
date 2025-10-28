#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import matplotlib.style as mplstyle
import numpy as np
import time
from askcarl.lightgmm import LightBaggingGMM
from askcarl.plot import plot_gmm_corner_pdf

#mplstyle.use('fast')
#mpl.use('Agg')
#mpl.rcParams['path.simplify'] = True
#mpl.rcParams['path.simplify_threshold'] = 1.0
#mpl.rcParams['agg.path.chunksize'] = 2000000 # break long paths for faster Agg rendering

for ndim, scale in (2, 2.0), (5, 1.5), (10, 1.0), (25, 0.8):
        N = 3000
        for i, (corr, M) in enumerate([(0, 30), (0, 300), (0.2, N), (0.4, N), (0.6, N), (0.8, N), (-0.2, N), (-0.4, N), (-0.6, N)]):
            print(f"[{i}] corr={corr} M={M}")
            cov = np.eye(ndim)
            cov[cov == 0] = corr

            for D in 0, 3:
                # Example GMM
                X = np.vstack([
                    np.random.multivariate_normal(0 + np.arange(ndim), cov, abs(M) * ndim),
                    np.random.multivariate_normal(D + np.arange(ndim), cov, abs(M) * ndim)
                ])

                t0 = time.time()
                lgmm = LightBaggingGMM(n_gmms=20, n_components=10)
                lgmm.fit(X)
                gmm = lgmm.to_sklearn()
                print('  LightBaggingGMM fit:', time.time() - t0)

                t1 = time.time()
                plot_gmm_corner_pdf(
                        f'plotcorner_{"mono" if D == 0 else "dual"}_{ndim}d_N{abs(M)}_corr{i}.pdf',
                        gmm, levels=[0.393, 0.675, 0.864], scale=scale,
                        truths=np.arange(ndim), linewidth=0.5 * scale, truthlinewidth=1.0 * scale,
                        margin=40. / scale,
                        )
                #fig, axes = plot_gmm_corner_pdf(gmm, levels=[0.393, 0.675, 0.864], max_err_frac=0.1)
                #fig.savefig(f'plotcorner_{"mono" if D == 0 else "dual"}_{ndim}d_N{abs(M)}_corr{i}.pdf')
                #plt.close(fig)
                print('  GMM plot:', time.time() - t1)
