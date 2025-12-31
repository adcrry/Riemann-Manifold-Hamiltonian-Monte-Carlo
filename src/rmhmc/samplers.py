from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from tqdm import tqdm

HamiltonianFunc = Callable[[np.ndarray, np.ndarray], float]
GradientFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]
MetricFunc = Callable[[np.ndarray], np.ndarray]

class Sampler(ABC):
    @abstractmethod
    def sample(self, num_samples: int, init_state: np.ndarray, *args, **kwargs) -> np.ndarray:
        pass

class RMHMCSampler(Sampler):
    def __init__(
        self, 
        hamiltonian: HamiltonianFunc, 
        gradient_hamiltonian_position: GradientFunc, 
        gradient_hamiltonian_momentum: GradientFunc, 
        metric_tensor: MetricFunc
    ) -> None: 
        """
        Args:
            hamiltonian: function H(q, p) computing the Hamiltonian value.
            gradient_hamiltonian_position: function dH/dq(q, p).
            gradient_hamiltonian_momentum: function dH/dp(q, p).
            metric_tensor: function G(q) computing the metric tensor.
        """
        self.hamiltonian: HamiltonianFunc = hamiltonian
        self.gradient_hamiltonian_position: GradientFunc = gradient_hamiltonian_position
        self.gradient_hamiltonian_momentum: GradientFunc = gradient_hamiltonian_momentum
        self.metric_tensor: MetricFunc = metric_tensor

    def _fixed_point_step_momentum(self, position: np.ndarray, momentum: np.ndarray, num_step: int, step_size: float) -> np.ndarray:
        new_momentum = momentum.copy()
        for _ in range(num_step):
            grad = self.gradient_hamiltonian_position(position, new_momentum)
            new_momentum = momentum - 0.5 * step_size * grad
        return new_momentum

    def _fixed_point_step_position(self, position: np.ndarray, momentum: np.ndarray, num_step: int, step_size: float) -> np.ndarray:
        new_position = position.copy()
        for _ in range(num_step):
            grad_old = self.gradient_hamiltonian_momentum(position, momentum)
            grad_new = self.gradient_hamiltonian_momentum(new_position, momentum)
            new_position = position + 0.5 * step_size * (grad_old + grad_new)
        return new_position

    def _leapfrog_step(self, position: np.ndarray, momentum: np.ndarray, num_fixed_point_step: int, step_size: float) -> tuple[np.ndarray, np.ndarray]:
        momentum = self._fixed_point_step_momentum(position, momentum, num_fixed_point_step, step_size)
        position = self._fixed_point_step_position(position, momentum, num_fixed_point_step, step_size)
        momentum -= 0.5 * step_size * self.gradient_hamiltonian_position(position, momentum)
        
        return position, momentum
            
    def _sample_momentum(self, position: np.ndarray) -> np.ndarray:
        metric = self.metric_tensor(position)
        return np.random.multivariate_normal(
            mean=np.zeros_like(position),
            cov=metric
        )

    def sample(
        self, 
        num_samples: int, 
        init_position: np.ndarray, 
        trajectory_length: float = 3.0, 
        initial_step_size: float = 0.5,
        target_acceptance: float = 0.75, 
        num_burnin_steps: int = 200, 
        adaptation_window: int = 50, 
        num_fixed_point_steps: int = 6,
        adapt_step_size: bool = True,
        return_burnin: bool = False
    ) -> np.ndarray:
        
        positions: list[np.ndarray] = []
        position = init_position.copy()
        
        # Initialisation
        step_size = initial_step_size
        
        num_leapfrog_steps = max(1, int(np.ceil(trajectory_length / step_size)))
        
        momentum = self._sample_momentum(position)
        current_hamiltonian = self.hamiltonian(position, momentum)

        total_accepted = 0
        window_accepted = 0
        
        total_iterations = num_samples + num_burnin_steps
        
        print(f"Starting RM-HMC (Adapt={adapt_step_size}): Burn-in={num_burnin_steps}, Samples={num_samples}")
        print(f"Params: step_size={step_size:.4f}, leapfrog_steps={num_leapfrog_steps}, trajectory={trajectory_length}")

        for i in tqdm(range(total_iterations), desc="Sampling", unit="it"):
            momentum = self._sample_momentum(position)
            current_hamiltonian = self.hamiltonian(position, momentum)
            
            q_prop, p_prop = position.copy(), momentum.copy()
            
            for _ in range(num_leapfrog_steps):
                q_prop, p_prop = self._leapfrog_step(
                    q_prop, p_prop, num_fixed_point_steps, step_size
                )
            
            proposed_hamiltonian = self.hamiltonian(q_prop, p_prop)
            
            if not np.isfinite(proposed_hamiltonian):
                energy_diff = np.inf
            else:
                energy_diff = proposed_hamiltonian - current_hamiltonian

            if np.log(np.random.rand()) < -energy_diff:
                position = q_prop
                if i >= num_burnin_steps:
                    total_accepted += 1
                window_accepted += 1
            
            if adapt_step_size and i < num_burnin_steps and (i + 1) % adaptation_window == 0:
                current_rate = window_accepted / adaptation_window
                scale_factor = np.exp(current_rate - target_acceptance)
                
                # Update
                step_size = np.clip(step_size * scale_factor, 1e-4, 5.0)
                num_leapfrog_steps = max(1, int(np.ceil(trajectory_length / step_size)))
                
                print(f"  [Burn-in {i+1}] Rate: {current_rate:.2f} -> New eps: {step_size:.4f}, L: {num_leapfrog_steps}")
                window_accepted = 0


            if i >= num_burnin_steps:
                positions.append(position.copy())
            elif return_burnin:
                positions.append(position.copy())

        final_rate = total_accepted / num_samples
        print(f"Sampling Finished. Final Acceptance Rate: {final_rate:.2%}")
        
        return np.array(positions)
