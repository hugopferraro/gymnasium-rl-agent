"""Fixed-size replay buffer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class TransitionBatch:
    states: NDArray[np.float32]
    actions: NDArray[np.int64]
    rewards: NDArray[np.float32]
    next_states: NDArray[np.float32]
    terminated: NDArray[np.float32]


class ReplayBuffer:
    """Preallocated circular buffer for off-policy learning."""

    def __init__(self, capacity: int, state_shape: tuple[int, ...], seed: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._states = np.empty((capacity, *state_shape), dtype=np.float32)
        self._actions = np.empty(capacity, dtype=np.int64)
        self._rewards = np.empty(capacity, dtype=np.float32)
        self._next_states = np.empty((capacity, *state_shape), dtype=np.float32)
        self._terminated = np.empty(capacity, dtype=np.float32)
        self._position = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        state: NDArray[np.floating],
        action: int,
        reward: float,
        next_state: NDArray[np.floating],
        terminated: bool,
    ) -> None:
        index = self._position
        self._states[index] = state
        self._actions[index] = action
        self._rewards[index] = reward
        self._next_states[index] = next_state
        self._terminated[index] = float(terminated)
        self._position = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        if batch_size > self._size:
            raise ValueError("Cannot sample more transitions than the buffer contains")
        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        return TransitionBatch(
            states=self._states[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_states=self._next_states[indices],
            terminated=self._terminated[indices],
        )
