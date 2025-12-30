from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

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
            gradient_hamiltonian_position: function dH/dq(q, p) computing the gradient
                of the Hamiltonian with respect to position.
            gradient_hamiltonian_momentum: function dH/dp(q, p) computing the
                gradient of the Hamiltonian with respect to momentum.
            metric_tensor: function G(q) computing the metric tensor at position q.
        """
        self.hamiltonian: HamiltonianFunc = hamiltonian
        self.gradient_hamiltonian_position: GradientFunc = gradient_hamiltonian_position
        self.gradient_hamiltonian_momentum: GradientFunc= gradient_hamiltonian_momentum
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
        step_size: float = 0.1,
        num_leapfrog_steps: int = 10, 
        num_fixed_point_steps: int = 5,
    ) -> np.ndarray:
        
        positions: list[np.ndarray] = []
        position = init_position.copy()
        
        momentum = self._sample_momentum(position)
        current_hamiltonian = self.hamiltonian(position, momentum)

        accepted_count = 0

        for _ in range(num_samples):
            momentum = self._sample_momentum(position)
            current_hamiltonian = self.hamiltonian(position, momentum)
            
            q_prop, p_prop = position.copy(), momentum.copy()
            for _ in range(num_leapfrog_steps):
                q_prop, p_prop = self._leapfrog_step(
                    q_prop, p_prop, num_fixed_point_steps, step_size
                )
            
            proposed_hamiltonian = self.hamiltonian(q_prop, p_prop)
            
            energy_diff = proposed_hamiltonian - current_hamiltonian

            if np.log(np.random.rand()) < -energy_diff:
                position = q_prop
                accepted_count += 1
            
            positions.append(position.copy())

        print(f"Acceptance rate: {accepted_count / num_samples:.2f}")
        return np.array(positions)
