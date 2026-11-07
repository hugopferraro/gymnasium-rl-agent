import numpy as np
import pytest
import tensorflow as tf

from gymnasium_rl_agent.agents.double_dqn import DoubleDQNAgent
from gymnasium_rl_agent.config import AgentConfig
from gymnasium_rl_agent.memory.replay_buffer import TransitionBatch


def _agent() -> DoubleDQNAgent:
    config = AgentConfig(
        hidden_layers=(4,), batch_size=2, replay_capacity=8, warmup_steps=2, gamma=0.5
    )
    return DoubleDQNAgent((2,), 2, config, seed=3)


def test_epsilon_schedule_reaches_floor() -> None:
    agent = _agent()
    assert agent.epsilon(0) == 1.0
    assert agent.epsilon(agent.config.epsilon_decay_steps) == pytest.approx(0.05)
    assert agent.epsilon(agent.config.epsilon_decay_steps * 2) == pytest.approx(0.05)


def test_terminated_transition_does_not_bootstrap() -> None:
    agent = _agent()
    online = tf.keras.Sequential([tf.keras.layers.Lambda(lambda x: x)])
    target = tf.keras.Sequential([tf.keras.layers.Lambda(lambda x: x * 2)])
    agent.online_network = online
    agent.target_network = target
    batch = TransitionBatch(
        states=np.zeros((2, 2), dtype=np.float32),
        actions=np.array([0, 0]),
        rewards=np.array([1.0, 1.0], dtype=np.float32),
        next_states=np.array([[1.0, 3.0], [1.0, 3.0]], dtype=np.float32),
        terminated=np.array([1.0, 0.0], dtype=np.float32),
    )
    targets = agent.compute_targets(batch).numpy()
    assert targets[0] == pytest.approx(1.0)
    assert targets[1] == pytest.approx(4.0)


def test_model_round_trip(tmp_path) -> None:
    agent = _agent()
    state = np.array([0.2, -0.1], dtype=np.float32)
    before = agent.online_network(state[None], training=False).numpy()
    destination = tmp_path / "model.keras"
    agent.save(destination)
    restored = _agent()
    restored.load(destination)
    after = restored.online_network(state[None], training=False).numpy()
    np.testing.assert_allclose(before, after)


def test_learn_executes_optimizer_step() -> None:
    agent = _agent()
    for value in (0.0, 1.0):
        state = np.array([value, value + 1], dtype=np.float32)
        agent.remember(state, 0, 1.0, state + 0.5, terminated=False)
    loss = agent.learn()
    assert loss is not None
    assert np.isfinite(loss)
    assert agent.training_steps == 1
