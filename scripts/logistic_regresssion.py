import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.special import expit

from rmhmc.samplers import RMHMCSampler


# -- riemannian --
def metric_tensor_logreg(beta, X, alpha):
    logits = X @ beta
    probs = expit(logits)
    
    W = probs * (1 - probs)
    W = np.maximum(W, 1e-10) 
    
    # Fisher + Prior
    return (X.T * W) @ X + (1.0/alpha) * np.eye(len(beta))

def gradient_metric_logreg(beta, X):
    logits = X @ beta
    probs = expit(logits)
    S = probs * (1 - probs) * (1 - 2 * probs)
    grad_M = np.zeros((len(beta), len(beta), len(beta)))
    for i in range(len(beta)):
        grad_M[i] = (X.T * (X[:, i] * S)) @ X
    return grad_M

def hamiltonian_logreg(beta, p, X, y, alpha):
    logits = X @ beta
    ll = np.dot(y, logits) - np.sum(np.logaddexp(0, logits))
    prior = -0.5/alpha * np.dot(beta, beta)
    U = -ll - prior
    
    M = metric_tensor_logreg(beta, X, alpha)
    _, logdet = np.linalg.slogdet(M)
    term_quad = 0.5 * p @ np.linalg.solve(M, p)
    return U + term_quad + 0.5 * logdet

def grad_h_q_logreg(beta, p, X, y, alpha):
    logits = X @ beta
    # STABILITÉ: expit
    probs = expit(logits)
    grad_U = -(X.T @ (y - probs) - beta/alpha)
    
    M = metric_tensor_logreg(beta, X, alpha)
    M_inv = np.linalg.inv(M)
    dM = gradient_metric_logreg(beta, X)
    
    term_trace = 0.5 * np.einsum('jk,ijk->i', M_inv, dM)
    v = M_inv @ p
    term_quad = -0.5 * np.einsum('j,ijk,k->i', v, dM, v)
    return grad_U + term_trace + term_quad

def grad_h_p_logreg(beta, p, X, alpha):
    M = metric_tensor_logreg(beta, X, alpha)
    return np.linalg.solve(M, p)


# -- euclidian --
def metric_const(beta): return np.eye(len(beta))
def grad_h_p_const(beta, p): return p
def hamiltonian_const(beta, p, X, y, alpha):
    logits = X @ beta
    ll = np.dot(y, logits) - np.sum(np.logaddexp(0, logits))
    U = -ll - (-0.5/alpha * np.dot(beta, beta))
    return U + 0.5 * np.dot(p, p)
def grad_h_q_const(beta, p, X, y, alpha):
    logits = X @ beta
    probs = expit(logits)
    return -(X.T @ (y - probs) - beta/alpha)

# -- plotting utilities --
def draw_metric_ellipses(ax, X, alpha, grid_range):
    """Draws ellipses representing the Riemannian Metric (Curvature)"""
    x_grid = np.linspace(grid_range[0][0], grid_range[0][1], 8)
    y_grid = np.linspace(grid_range[1][0], grid_range[1][1], 8)
    
    scale = 0.25

    for x in x_grid:
        for y in y_grid:
            beta = np.array([x, y])
            G = metric_tensor_logreg(beta, X, alpha)
            G_inv = np.linalg.inv(G) 
            
            vals, vecs = np.linalg.eigh(G_inv)
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            width, height = 2 * scale * np.sqrt(vals)
            
            ell = patches.Ellipse((x, y), width, height, angle=angle, 
                                  fill=False, color='green', alpha=0.6, lw=1)
            ax.add_patch(ell)

def draw_euclidean_circles(ax, grid_range):
    """Draws circles representing the Euclidean Metric (Identity)"""
    x_grid = np.linspace(grid_range[0][0], grid_range[0][1], 8)
    y_grid = np.linspace(grid_range[1][0], grid_range[1][1], 8)
    
    radius = 0.15 

    for x in x_grid:
        for y in y_grid:
            circ = patches.Circle((x, y), radius=radius, 
                                  fill=False, color='blue', alpha=0.6, lw=1)
            ax.add_patch(circ)

def plot_final_comparison(samples_rm, samples_std, X, y, alpha):
    sns.set_theme(style="white", font_scale=1.1)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2) 

    ax_rm = fig.add_subplot(gs[0, 0])
    
    b1_lim = (np.min(samples_rm[:,0])-1, np.max(samples_rm[:,0])+1)
    b2_lim = (np.min(samples_rm[:,1])-1, np.max(samples_rm[:,1])+1)
    xx = np.linspace(b1_lim[0], b1_lim[1], 50)
    yy = np.linspace(b2_lim[0], b2_lim[1], 50)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = np.zeros_like(XX)
    for i in range(50):
        for j in range(50):
            beta = np.array([XX[i,j], YY[i,j]])
            ll = np.dot(y, X @ beta) - np.sum(np.logaddexp(0, X @ beta))
            ZZ[i,j] = ll - 0.5 * np.dot(beta, beta) 
            
    ax_rm.contourf(XX, YY, ZZ, levels=15, cmap='Greys', alpha=0.3)
    draw_metric_ellipses(ax_rm, X, alpha, (b1_lim, b2_lim))
    
    steps = 100 
    ax_rm.plot(samples_rm[:steps, 0], samples_rm[:steps, 1], 'r.-', lw=0.25, markersize=3, label='RM-HMC Path')
    ax_rm.set_title("RM-HMC: Adapted Geometry (Ellipses)")
    ax_rm.set_xlabel(r"$\beta_0$"); ax_rm.set_ylabel(r"$\beta_1$")
    ax_rm.legend(loc='lower right')

    ax_std = fig.add_subplot(gs[0, 1])
    ax_std.contourf(XX, YY, ZZ, levels=15, cmap='Greys', alpha=0.3)
    
    draw_euclidean_circles(ax_std, (b1_lim, b2_lim))
    
    ax_std.plot(samples_std[:steps, 0], samples_std[:steps, 1], 'b.-', lw=0.25, markersize=3, label='Std HMC Path')
    ax_std.set_title("Standard HMC: Isotropic Geometry (Balls)")
    ax_std.set_xlabel(r"$\beta_0$"); ax_std.set_ylabel(r"$\beta_1$")
    ax_std.legend(loc='lower right')
    
    ax_std.set_xlim(ax_rm.get_xlim())
    ax_std.set_ylim(ax_rm.get_ylim())

    ax_acf = fig.add_subplot(gs[1, 0])
    def autocorr(x):
        r = np.correlate(x-np.mean(x), x-np.mean(x), mode='full')
        return r[r.size//2:] / r[r.size//2]
    
    lags = 30
    ax_acf.bar(np.arange(lags)-0.15, autocorr(samples_rm[:,0])[:lags], width=0.3, color='red', label='RM-HMC', alpha=0.7)
    ax_acf.bar(np.arange(lags)+0.15, autocorr(samples_std[:,0])[:lags], width=0.3, color='blue', label='Std HMC', alpha=0.7)
    ax_acf.axhline(0, color='black', lw=1)
    ax_acf.set_title("Autocorrelation (Lower is Better)")
    ax_acf.legend()

    ax_kde = fig.add_subplot(gs[1, 1])
    sns.kdeplot(samples_rm[:,0], ax=ax_kde, color='red', fill=True, alpha=0.3, label='RM-HMC')
    sns.kdeplot(samples_std[:,0], ax=ax_kde, color='blue', fill=True, alpha=0.3, label='Std HMC')
    ax_kde.set_title(r"Marginal Posterior Density ($\beta_0$)")
    ax_kde.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    cov = [[1.0, 0.98], [0.98, 1.0]]
    X = np.random.multivariate_normal([0, 0], cov, size=200)
    beta_true = np.array([1.5, -1.5])
    y = np.random.binomial(1, 1/(1+np.exp(-X @ beta_true)))
    alpha = 100.0

    beta_init = np.array([0.0, 0.0])

    print("Sampling RM-HMC...")
    rm_sampler = RMHMCSampler(
        lambda q,p: hamiltonian_logreg(q,p,X,y,alpha),
        lambda q,p: grad_h_q_logreg(q,p,X,y,alpha),
        lambda q,p: grad_h_p_logreg(q,p,X,alpha),
        lambda q: metric_tensor_logreg(q,X,alpha)
    )
    s_rm = rm_sampler.sample(1000, beta_init, step_size=1.1, num_leapfrog_steps=25, num_fixed_point_steps=20)

    print("\nSampling Standard HMC...")
    std_sampler = RMHMCSampler(
        lambda q,p: hamiltonian_const(q,p,X,y,alpha),
        lambda q,p: grad_h_q_const(q,p,X,y,alpha),
        lambda q,p: grad_h_p_const(q,p),
        metric_const 
    )
    s_std = std_sampler.sample(1000, beta_init, step_size=0.2, num_leapfrog_steps=25, num_fixed_point_steps=1)

    # Burn-in for plotting
    plot_final_comparison(s_rm[500:], s_std[500:], X, y, alpha)
