import numpy as np
from scipy.special import beta as beta_function
from scipy.special import gamma
from scipy.stats import norm

from rmhmc.samplers import RMHMCSampler
from rmhmc.utils import plot_mcmc_comparison


def log_prior_beta(beta):
    if beta <= 0:
        return -np.inf
    return beta

def grad_prior_beta(beta):
    if beta <= 0:
        return 0.0
    return 1.0

def neg_hess_prior_beta(beta):
    if beta <= 0:
        return 0.0
    return 0.0

def third_deriv_prior_beta(beta):
    return 0.0


A_PHI, B_PHI = 20.0, 1.5
def log_prior_phi(phi):
    if np.abs(phi) >= 1.0:
        return -np.inf
    
    log_term = (A_PHI - 1) * np.log((phi + 1) / 2) + (B_PHI - 1) * np.log((1 - phi) / 2)
    log_const = -np.log(beta_function(A_PHI, B_PHI)) - np.log(2)
    return log_term + log_const

def grad_prior_phi(phi):
    if np.abs(phi) >= 1.0:
        return 0.0
    return (A_PHI - 1) / (phi + 1) - (B_PHI - 1) / (1 - phi)

def neg_hess_prior_phi(phi):
    if np.abs(phi) >= 1.0:
        return 0.0
    
    return (A_PHI - 1) / ((phi + 1) ** 2) + (B_PHI - 1) / ((1 - phi) ** 2)

def third_deriv_prior_phi(phi):
    if np.abs(phi) >= 1.0:
        return 0.0
    
    return -2 * (A_PHI - 1) / ((phi + 1) ** 3) + 2 * (B_PHI - 1) / ((1 - phi) ** 3)

NU_SIGMA, S2_SIGMA = 10.0, 0.05

def log_prior_sigma(sigma):
    if sigma <= 0:
        return -np.inf
    
    log_term = -(NU_SIGMA + 1) * np.log(sigma) - (NU_SIGMA * S2_SIGMA) / (2 * sigma ** 2)
    log_const = (NU_SIGMA/2) * np.log(NU_SIGMA * S2_SIGMA / 2) - np.log(gamma(NU_SIGMA/2)) + np.log(2)
    
    return log_term + log_const

def grad_prior_sigma(sigma):
    if sigma <= 0:
        return 0.0
    
    return -(NU_SIGMA + 1) / sigma + (NU_SIGMA * S2_SIGMA) / (sigma ** 3)

def neg_hess_prior_sigma(sigma):
    if sigma <= 0:
        return 0.0
    B = NU_SIGMA * S2_SIGMA
    return -(NU_SIGMA + 1) / (sigma ** 2) + 3 * B / (sigma ** 4)

def third_deriv_prior_sigma(sigma):
    if sigma <= 0:
        return 0.0
    B = NU_SIGMA * S2_SIGMA
    return 2 * (NU_SIGMA + 1) / (sigma ** 3) - 12 * B / (sigma ** 5)

def generate_ar1_data(T: int = 2000, phi: float = 0.98, sigma: float = 0.15, beta: float = 0.65):
    np.random.seed(42)
    x = np.zeros(T)
    x[0] = np.random.normal(0, sigma / np.sqrt(1 - phi**2))
    for t in range(1, T):
        x[t] = phi * x[t-1] + np.random.normal(0, sigma)
    
    y = np.random.normal(0, 1, size=T) * beta * np.exp(x / 2)
    return x, y

def log_likelihood(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    phi, sigma, beta = theta

    log_likelihood = 0.0
    log_likelihood += np.sum(norm.logpdf(y, loc=0, scale=beta * np.exp(x / 2)))
    log_likelihood += norm.logpdf(x[0], loc=0, scale=sigma / np.sqrt(1 - phi**2))
    log_likelihood += np.sum(norm.logpdf(x[1:], loc=phi * x[:-1], scale=sigma))
    
    return log_likelihood

def grad_log_likelihood(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    phi, sigma, beta = theta
    T = len(x)
    grad = np.zeros(3)
    
    res = x[1:] - phi * x[:-1]

    grad[0] = -phi/(1-phi**2) + np.sum(res * x[:-1]) / sigma**2 + phi * x[0]**2 / (sigma**2)
    grad[1] = -T/sigma + np.sum(res**2) / sigma**3 + (1 - phi**2) * x[0]**2 / sigma**3
    grad[2] = -T/beta + np.sum(y**2 * np.exp(-x)) / beta**3
    
    return grad

def tensor_metric(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    phi, sigma, beta = theta
    T = len(x)

    G = np.zeros((3, 3))
    
    G[0, 0] = (T - 1) / (1 - phi**2) + 2 * phi**2 / (1 - phi**2)**2 # Phi
    G[1, 1] = 2 * T / sigma**2 # Sigma
    G[2, 2] = 2 * T / beta**2  # Beta
    G[0, 1] = 2 * phi / (sigma**3 * (1 - phi**2))
    G[1, 0] = G[0, 1]

    # prior contributions
    G[0, 0] += neg_hess_prior_phi(phi)
    G[1, 1] += neg_hess_prior_sigma(sigma)
    G[2, 2] += neg_hess_prior_beta(beta)
    
    return G # + np.eye(3) * 1e-6  # for numerical stability

def jacobian_metric_phi(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    phi, sigma, _= theta
    T = len(x)
    J = np.zeros((3, 3))
    
    term_fisher = 2 * phi * (1 + T) / (1 - phi**2)**2 + 8 * phi**3 / (1 - phi**2)**3
    term_prior = third_deriv_prior_phi(phi)
    
    J[0, 0] = term_fisher + term_prior
    J[0, 1] = 2 / (sigma**3 * (1 - phi**2)) + 4 * phi**2 / (sigma**3 * (1 - phi**2)**2)
    J[1, 0] = J[0, 1]
    
    return J

def jacobian_metric_sigma(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    phi, sigma, _= theta
    T = len(x)
    J = np.zeros((3, 3))
    
    J[1, 1] = -4 * T / sigma**3 + third_deriv_prior_sigma(sigma)
    J[0, 1] = -6 * phi / (sigma**4 * (1 - phi**2))
    J[1, 0] = J[0, 1]
    
    return J

def jacobian_metric_beta(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    _, _, beta = theta
    T = len(x)
    J = np.zeros((3, 3))
    
    J[2, 2] = -4 * T / beta**3 + third_deriv_prior_beta(beta)
    return J

def rm_hamiltonian(theta: np.ndarray, p: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    log_post = log_likelihood(theta, x, y) + log_prior_phi(theta[0]) + log_prior_sigma(theta[1]) + log_prior_beta(theta[2])
               
    U = -log_post
    G = tensor_metric(theta, x)
    return U + 0.5 * p @ np.linalg.solve(G, p) + 0.5 * np.linalg.slogdet(G)[1]

def rm_grad_hamiltonian_position(theta: np.ndarray, p: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    grad_U = -grad_log_likelihood(theta, x, y)
    
    grad_U[0] -= grad_prior_phi(theta[0])
    grad_U[1] -= grad_prior_sigma(theta[1])
    grad_U[2] -= grad_prior_beta(theta[2])

    G = tensor_metric(theta, x)
    
    G_inv = np.linalg.solve(G, np.eye(3))

    dG_phi = jacobian_metric_phi(theta, x)
    dG_sigma = jacobian_metric_sigma(theta, x)
    dG_beta = jacobian_metric_beta(theta, x)

    term_trace = 0.5 * np.array([
        np.trace(G_inv @ dG_phi),
        np.trace(G_inv @ dG_sigma),
        np.trace(G_inv @ dG_beta)
    ])
    
    v = G_inv @ p
    term_quad = -0.5 * np.array([
        v @ dG_phi @ v,  
        v @ dG_sigma @ v,
        v @ dG_beta @ v  
    ])
    
    return grad_U + term_trace + term_quad

def rm_grad_hamiltonian_momentum(theta: np.ndarray, p: np.ndarray, x: np.ndarray) -> np.ndarray:
    G = tensor_metric(theta, x)
    return np.linalg.solve(G, p)

if __name__ == "__main__":
    phi, sigma, beta = 0.98, 0.15, 0.65
    x, y = generate_ar1_data(T=2000, phi=phi, sigma=sigma, beta=beta)
    
    # [phi, sigma, beta]
    theta_init = np.array([0.80, 0.5, 0.65]) 

    print("Sampling RM-HMC...")
    rm_sampler = RMHMCSampler(
        hamiltonian=lambda q,p: rm_hamiltonian(q, p, x, y),
        gradient_hamiltonian_position=lambda q,p: rm_grad_hamiltonian_position(q, p, x, y),
        gradient_hamiltonian_momentum=lambda q,p: rm_grad_hamiltonian_momentum(q, p, x),
        metric_tensor=lambda q: tensor_metric(q, x)
    )
    s_rm = rm_sampler.sample(10000, theta_init, trajectory_length=3.0, initial_step_size=0.1, num_burnin_steps=10000, adapt_step_size=True, num_fixed_point_steps=20, return_burnin=False)
    
    print("Sampling Std-HMC...")
    std_sampler = RMHMCSampler(
        hamiltonian=lambda q,p: -(log_likelihood(q,x,y) + log_prior_phi(q[0]) + log_prior_sigma(q[1]) + log_prior_beta(q[2])) + 0.5*np.dot(p,p),
        gradient_hamiltonian_position=lambda q,p: -(grad_log_likelihood(q,x,y) + np.array([grad_prior_phi(q[0]), grad_prior_sigma(q[1]), grad_prior_beta(q[2])])),
        gradient_hamiltonian_momentum=lambda q,p: p,
        metric_tensor=lambda q: np.eye(3),
    )
    s_std = std_sampler.sample(10000, theta_init, trajectory_length=1.0, initial_step_size=0.015, num_burnin_steps=10000, adapt_step_size=True, num_fixed_point_steps=1, return_burnin=False)

    plot_mcmc_comparison(
        s_rm[:,:2], s_std[:, :2], 
        potential_func=lambda q: -log_likelihood(np.array([q[0], q[1], beta]), x, y) - log_prior_phi(q[0]) - log_prior_sigma(q[1]) - log_prior_beta(beta),
        metric_func=lambda q: tensor_metric(np.array([q[0], q[1], beta]), x)[:2, :2],
        param_names=[r'$\phi$', r'$\sigma$'],
    )
