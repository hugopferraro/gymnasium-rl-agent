import numpy as np
import pytest

from gymnasium_rl_agent.memory.replay_buffer import ReplayBuffer


def test_buffer_overwrites_old_transitions() -> None:
    buffer = ReplayBuffer(capacity=2, state_shape=(2,), seed=7)
    for value in range(3):
        state = np.asarray([value, value], dtype=np.float32)
        buffer.add(state, value, float(value), state + 1, terminated=False)
    assert len(buffer) == 2
    batch = buffer.sample(2)
    assert set(batch.actions.tolist()) == {1, 2}


def test_buffer_refuses_oversized_sample() -> None:
    buffer = ReplayBuffer(capacity=2, state_shape=(1,), seed=7)
    with pytest.raises(ValueError, match="more transitions"):
        buffer.sample(1)
