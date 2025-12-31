import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from rmhmc.samplers import RMHMCSampler

EPS = 1e-6

def plot_custom_1d_analysis(q_rm, p_rm, q_std, p_std, y_obs, 
                            potential_func, metric_func, 
                            hamiltonian_std_func, hamiltonian_rm_func, crossing_traj=None):
    """
    Args:
        ...
        hamiltonian_std_func: function(q, p) -> float (Energy Std)
        hamiltonian_rm_func: function(q, p) -> float (Energy RM)
    """
    q_rm, p_rm = q_rm.flatten(), p_rm.flatten()
    q_std, p_std = q_std.flatten(), p_std.flatten()
    
    true_mode = np.sqrt(y_obs)
    bound = max(np.max(np.abs(q_rm)), true_mode * 1.5)
    theta_grid = np.linspace(-bound, bound, 1000)

    _, (ax_trace, ax_hist) = plt.subplots(1, 2, figsize=(16, 5))
    
    ax_trace.plot(q_rm, label='RM-HMC', alpha=0.7, linewidth=0.8, color='blue')
    ax_trace.plot(q_std, label='Std-HMC', alpha=0.5, linewidth=0.8, color='orange')
    ax_trace.axhline(true_mode, color='green', linestyle='--', alpha=0.5)
    ax_trace.axhline(-true_mode, color='green', linestyle='--', alpha=0.5)
    ax_trace.set_title("1. Trace Plot: Position")
    ax_trace.legend(loc='upper right')

    ax_hist.hist(q_rm, bins=60, density=True, alpha=0.4, color='blue', label='RM-HMC')
    ax_hist.hist(q_std, bins=60, density=True, alpha=0.4, color='orange', label='Std-HMC')
    
    U_vals = np.array([potential_func(np.array([t])) for t in theta_grid])
    prob_unnorm = np.exp(-U_vals + np.min(U_vals))
    
    norm_const = np.trapezoid(prob_unnorm, theta_grid)
        
    ax_hist.plot(theta_grid, prob_unnorm / norm_const, 'k--', label='True Posterior')
    ax_hist.set_title("2. Posterior Density")
    ax_hist.legend()
    plt.tight_layout()
    plt.show()

    _, ax_mech = plt.subplots(figsize=(12, 6))
    
    U_nominal = U_vals - np.min(U_vals)
    G_vals = np.array([metric_func(np.array([t]))[0,0] for t in theta_grid])
    log_det_G = 0.5 * np.log(G_vals + 1e-12)
    U_effective = U_nominal + log_det_G
    U_effective -= np.min(U_effective)

    ax_mech.plot(theta_grid, U_nominal, color='red', linestyle='--', linewidth=2, label='Nominal Barrier (Std-HMC)')
    ax_mech.plot(theta_grid, U_effective, color='purple', linewidth=3, label='Effective Landscape (RM-HMC)')
    ax_mech.fill_between(theta_grid, 0, U_nominal, where=(np.abs(theta_grid) < true_mode), color='red', alpha=0.1)
    
    max_disp = max(np.max(U_nominal[np.abs(theta_grid) < true_mode * 1.2]), 50)
    ax_mech.set_ylim(-5, max_disp * 1.2)
    ax_mech.set_title(f"3. Tunneling Effect (Barrier H={np.max(U_nominal):.1f})")
    ax_mech.legend()
    ax_mech.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    _, ax_phase = plt.subplots(figsize=(14, 8))
    q_vals = np.linspace(-max(np.max(np.abs(q_rm)), 15), max(np.max(np.abs(q_rm)), 15), 200)
    p_vals = np.linspace(-30, 30, 200)
    Q, P = np.meshgrid(q_vals, p_vals)
    
    H_std_grid = np.zeros_like(Q)
    H_rm_grid = np.zeros_like(Q)
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            q_vec, p_vec = np.array([Q[i, j]]), np.array([P[i, j]])
            H_std_grid[i, j] = hamiltonian_std_func(q_vec, p_vec)
            H_rm_grid[i, j] = hamiltonian_rm_func(q_vec, p_vec)
            
    barrier_E = hamiltonian_std_func(np.array([0.0]), np.array([0.0]))
    ax_phase.contour(Q, P, H_std_grid, levels=[barrier_E], colors='black', linewidths=2, linestyles='--')
    ax_phase.contour(Q, P, H_std_grid, levels=15, colors='gray', alpha=0.3)
    
    levels_rm = np.linspace(np.min(H_rm_grid), np.percentile(H_rm_grid, 90), 25)
    ax_phase.contour(Q, P, H_rm_grid, levels=levels_rm, cmap='cool', linewidths=1.5, alpha=0.5)
    
    idx_std = np.random.choice(len(q_std), size=min(2000, len(q_std)), replace=False)
    ax_phase.scatter(q_std[idx_std], p_std[idx_std], s=10, color='orange', alpha=0.6, label='Std Samples')
    
    idx_rm = np.random.choice(len(q_rm), size=min(2000, len(q_rm)), replace=False)
    p_rm_plot = np.clip(p_rm[idx_rm], -30, 30)
    ax_phase.scatter(q_rm[idx_rm], p_rm_plot, s=10, color='blue', alpha=0.3, label='RM Samples')

    if crossing_traj is not None:
        ax_phase.plot(crossing_traj[:, 0], crossing_traj[:, 1], color='red', linewidth=2, label='Crossing Trajectory')
        step = max(1, len(crossing_traj) // 100)
        ax_phase.scatter(crossing_traj[::step, 0], crossing_traj[::step, 1], color='black', s=10, zorder=10)

    ax_phase.set_title("3. Phase Portrait with Continuous Crossing Path", fontsize=14)
    ax_phase.set_xlabel(r"Position $\theta$")
    ax_phase.set_ylabel(r"Momentum $p$")
    ax_phase.set_ylim(-30, 30)
    ax_phase.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def generate_data(theta: np.ndarray, sig_noise: float = 1.) -> float:
    np.random.seed(42)
    return float(theta[0]**2 + sig_noise * np.random.normal(0, 1))

def log_likelihood(theta: np.ndarray, y: float, sig_noise: float) -> float:
    return -(1/(2 * sig_noise**2)) * (y - theta[0]**2)**2

def grad_log_likelihood(theta: np.ndarray, y: float, sig_noise: float) -> np.ndarray:
    grad = (y - theta**2) * 2 * theta / (sig_noise**2)
    return grad

def tensor_metric(theta: np.ndarray, sig_noise: float) -> np.ndarray:
    val = 4 * (theta / sig_noise)**2 + EPS
    return val.reshape(1, 1)

def rm_hamiltonian(theta: np.ndarray, p: np.ndarray, y: float, sig_noise: float) -> float:
    U = -log_likelihood(theta, y, sig_noise)
    G = tensor_metric(theta, sig_noise)
    term_quad = 0.5 * p.T @ np.linalg.solve(G, p)
    term_log = 0.5 * np.linalg.slogdet(G)[1]
    
    return U + float(term_quad + term_log)

def rm_grad_hamiltonian_position(theta: np.ndarray, p: np.ndarray, y: float, sig_noise: float) -> np.ndarray:
    grad_U = -grad_log_likelihood(theta, y, sig_noise)
    G = tensor_metric(theta, sig_noise)
    G_inv = np.linalg.inv(G)
    dG = (8 * theta / sig_noise**2).reshape(1, 1)
    p_mat = p.reshape(1, 1)
    term_quad = -0.5 * (p_mat.T @ G_inv @ dG @ G_inv @ p_mat)
    term_trace = 0.5 * np.trace(G_inv @ dG)
    total_grad = grad_U + term_quad.flatten() + term_trace
    return total_grad

def rm_grad_hamiltonian_momentum(theta: np.ndarray, p: np.ndarray, sig_noise: float) -> np.ndarray:
    G = tensor_metric(theta, sig_noise)
    return np.linalg.solve(G, p).flatten()


if __name__ == "__main__":
    sig_noise = 5.0
    theta_true = np.array([10.0])
    y_obs = generate_data(theta_true, sig_noise)
    theta_init = np.array([-15.0]) 

    print(f"Observation y = {y_obs:.4f}")
    
    # 1. RM-HMC
    print("Sampling RM-HMC...")
    rm_sampler = RMHMCSampler(
        hamiltonian=lambda q,p: rm_hamiltonian(q, p, y_obs, sig_noise),
        gradient_hamiltonian_position=lambda q,p: rm_grad_hamiltonian_position(q, p, y_obs, sig_noise),
        gradient_hamiltonian_momentum=lambda q,p: rm_grad_hamiltonian_momentum(q, p, sig_noise),
        metric_tensor=lambda q: tensor_metric(q, sig_noise)
    )
    
    # Enable save_trajectories=True to populate rm_sampler.trajectories
    positions_rm, momentums_rm = rm_sampler.sample(
        num_samples=10000, 
        init_position=theta_init, 
        trajectory_length=5.0, 
        initial_step_size=0.5, 
        num_burnin_steps=10000,
        adapt_step_size=True, 
        return_burnin=False,
        save_trajectories=True 
    )
    
    # Find a crossing trajectory
    # We look for a path that starts on one side (< -2) and ends on the other (> 2)
    crossing_path = None
    for traj in rm_sampler.trajectories:
        if (traj[0, 0] < -2.0 and traj[-1, 0] > 2.0) or \
           (traj[0, 0] > 2.0 and traj[-1, 0] < -2.0):
            crossing_path = traj
            print(f"Found crossing trajectory! Length: {len(traj)} steps.")
            break
            
    if crossing_path is None:
        print("No crossing detected in the stored trajectories (check burn-in or step size).")

    # 2. Std-HMC
    print("Sampling Std-HMC...")
    std_sampler = RMHMCSampler(
        hamiltonian=lambda q,p: -log_likelihood(q, y_obs, sig_noise) + 0.5 * np.dot(p, p),
        gradient_hamiltonian_position=lambda q,p: -grad_log_likelihood(q, y_obs, sig_noise),
        gradient_hamiltonian_momentum=lambda q,p: p,
        metric_tensor=lambda q: np.eye(1),
    )
    
    positions_std, momentums_std = std_sampler.sample(
        num_samples=10000, 
        init_position=theta_init, 
        trajectory_length=5.0, 
        initial_step_size=0.02,
        num_burnin_steps=10000, 
        adapt_step_size=True, 
        return_burnin=False,
        num_fixed_point_steps=1,
        save_trajectories=False,
    )

    plot_custom_1d_analysis(
        positions_rm[2000:5000], 
        momentums_rm[2000:5000], 
        positions_std[2000:5000], 
        momentums_std[2000:5000], 
        y_obs, 
        potential_func=lambda q: -log_likelihood(q, y_obs, sig_noise),
        metric_func=lambda q: tensor_metric(q, sig_noise),
        hamiltonian_std_func=lambda q, p: -log_likelihood(q, y_obs, sig_noise) + 0.5 * np.dot(p, p),
        hamiltonian_rm_func=lambda q, p: rm_hamiltonian(q, p, y_obs, sig_noise),
        crossing_traj=crossing_path
    )
