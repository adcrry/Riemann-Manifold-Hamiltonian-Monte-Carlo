import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def draw_metric_ellipses(ax, metric_func, grid_limits: tuple[float, float, float, float], scale: float = 0.25):
    x_min, x_max, y_min, y_max = grid_limits
    x_grid = np.linspace(x_min, x_max, 8)
    y_grid = np.linspace(y_min, y_max, 8)

    for x in x_grid:
        for y in y_grid:
            q = np.array([x, y])
            G = metric_func(q)
            G_inv = np.linalg.inv(G)
            vals, vecs = np.linalg.eigh(G_inv)
            
            # Angle et Dimensions
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            width, height = 2 * scale * np.sqrt(vals)
            
            ell = patches.Ellipse((x, y), width, height, angle=angle, 
                                  fill=False, color='green', alpha=0.6, lw=1.5)
            ax.add_patch(ell)

def draw_euclidean_circles(ax, grid_limits: tuple[float, float, float, float], radius: float = 0.15):
    x_min, x_max, y_min, y_max = grid_limits
    x_grid = np.linspace(x_min, x_max, 8)
    y_grid = np.linspace(y_min, y_max, 8)

    for x in x_grid:
        for y in y_grid:
            circ = patches.Circle((x, y), radius=radius, 
                                  fill=False, color='blue', alpha=0.6, lw=1.5)
            ax.add_patch(circ)

def plot_mcmc_comparison(
    samples_rm: np.ndarray, 
    samples_std: np.ndarray, 
    potential_func,
    metric_func,
    param_names: list[str] = [r'$\theta_1$', r'$\theta_2$'],
    steps_to_plot: int = 100,
    title_suffix: str = ""
):
    sns.set_theme(style="white", font_scale=1.1)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)

    all_samples = np.vstack([samples_rm, samples_std])
    x_min, x_max = np.min(all_samples[:,0]), np.max(all_samples[:,0])
    y_min, y_max = np.min(all_samples[:,1]), np.max(all_samples[:,1])
    
    margin_x = (x_max - x_min) * 0.2
    margin_y = (y_max - y_min) * 0.2
    limits = (x_min - margin_x, x_max + margin_x, y_min - margin_y, y_max + margin_y)

    xx = np.linspace(limits[0], limits[1], 60)
    yy = np.linspace(limits[2], limits[3], 60)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = np.zeros_like(XX)
    
    for i in range(60):
        for j in range(60):
            q = np.array([XX[i,j], YY[i,j]])
            ZZ[i,j] = potential_func(q)
    ax_rm = fig.add_subplot(gs[0, 0])
    ax_rm.contourf(XX, YY, -ZZ, levels=15, cmap='Greys', alpha=0.3)
    
    draw_metric_ellipses(ax_rm, metric_func, limits)
    
    ax_rm.plot(samples_rm[:steps_to_plot, 0], samples_rm[:steps_to_plot, 1], 'r.-', lw=0.5, markersize=3, label='RM-HMC Path')
    ax_rm.set_title(f"RM-HMC: Adapted Geometry {title_suffix}")
    ax_rm.set_xlabel(param_names[0]); ax_rm.set_ylabel(param_names[1])
    ax_rm.set_xlim(limits[0], limits[1]); ax_rm.set_ylim(limits[2], limits[3])
    ax_rm.legend(loc='lower right')

    ax_std = fig.add_subplot(gs[0, 1])
    ax_std.contourf(XX, YY, -ZZ, levels=15, cmap='Greys', alpha=0.3)
    
    radius = (limits[1] - limits[0]) * 0.04
    draw_euclidean_circles(ax_std, limits, radius=radius)
    
    ax_std.plot(samples_std[:steps_to_plot, 0], samples_std[:steps_to_plot, 1], 'b.-', lw=0.5, markersize=3, label='Std HMC Path')
    ax_std.set_title(f"Standard HMC: Isotropic Geometry {title_suffix}")
    ax_std.set_xlabel(param_names[0]); ax_std.set_ylabel(param_names[1])
    ax_std.set_xlim(limits[0], limits[1]); ax_std.set_ylim(limits[2], limits[3])
    ax_std.legend(loc='lower right')

    ax_acf = fig.add_subplot(gs[1, 0])
    
    def compute_autocorr(x):
        x = x - np.mean(x)
        r = np.correlate(x, x, mode='full')
        return r[r.size//2:] / r[r.size//2]
    
    lags = 40
    acf_rm = compute_autocorr(samples_rm[:, 0])[:lags]
    acf_std = compute_autocorr(samples_std[:, 0])[:lags]
    
    ax_acf.bar(np.arange(lags)-0.15, acf_rm, width=0.3, color='red', label='RM-HMC', alpha=0.7)
    ax_acf.bar(np.arange(lags)+0.15, acf_std, width=0.3, color='blue', label='Std HMC', alpha=0.7)
    ax_acf.axhline(0, color='black', lw=1)
    ax_acf.set_title(f"Autocorrelation on {param_names[0]} (Lower is Better)")
    ax_acf.legend()

    ax_kde = fig.add_subplot(gs[1, 1])
    sns.kdeplot(samples_rm[:, 0], ax=ax_kde, color='red', fill=True, alpha=0.3, label='RM-HMC')
    sns.kdeplot(samples_std[:, 0], ax=ax_kde, color='blue', fill=True, alpha=0.3, label='Std HMC')
    ax_kde.set_title(f"Marginal Posterior Density of {param_names[0]}")
    ax_kde.legend()

    plt.tight_layout()
    plt.show()
