"""Double Deep Q-Network agent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from ..config import AgentConfig
from ..memory.replay_buffer import ReplayBuffer, TransitionBatch
from ..models.q_network import build_q_network


class DoubleDQNAgent:
    """Double DQN with target network and uniform experience replay."""

    def __init__(
        self, state_shape: tuple[int, ...], action_count: int, config: AgentConfig, seed: int
    ) -> None:
        self.config = config
        self.action_count = action_count
        self.rng = np.random.default_rng(seed)
        self.online_network = build_q_network(state_shape, action_count, config.hidden_layers)
        self.target_network = build_q_network(state_shape, action_count, config.hidden_layers)
        self.target_network.set_weights(self.online_network.get_weights())
        self.target_network.trainable = False
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.learning_rate, clipnorm=config.gradient_clip_norm
        )
        self.loss_fn = tf.keras.losses.Huber()
        self.replay_buffer = ReplayBuffer(config.replay_capacity, state_shape, seed)
        self.training_steps = 0

    def epsilon(self, environment_step: int) -> float:
        progress = min(environment_step / self.config.epsilon_decay_steps, 1.0)
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def act(self, state: NDArray[np.floating], epsilon: float = 0.0) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.action_count))
        values = self.online_network(np.asarray(state, dtype=np.float32)[None, ...], training=False)
        return int(tf.argmax(values[0]).numpy())

    def remember(
        self,
        state: NDArray[np.floating],
        action: int,
        reward: float,
        next_state: NDArray[np.floating],
        terminated: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, terminated)

    def compute_targets(self, batch: TransitionBatch) -> tf.Tensor:
        """Compute Double DQN targets; truncations continue to bootstrap."""
        next_online = self.online_network(batch.next_states, training=False)
        best_actions = tf.argmax(next_online, axis=1, output_type=tf.int32)
        next_target = self.target_network(batch.next_states, training=False)
        indices = tf.stack([tf.range(tf.shape(best_actions)[0]), best_actions], axis=1)
        selected_values = tf.gather_nd(next_target, indices)
        rewards = tf.convert_to_tensor(batch.rewards, dtype=tf.float32)
        terminated = tf.convert_to_tensor(batch.terminated, dtype=tf.float32)
        return rewards + self.config.gamma * (1.0 - terminated) * selected_values

    def learn(self) -> float | None:
        if len(self.replay_buffer) < max(self.config.batch_size, self.config.warmup_steps):
            return None
        batch = self.replay_buffer.sample(self.config.batch_size)
        targets = self.compute_targets(batch)
        with tf.GradientTape() as tape:
            q_values = self.online_network(batch.states, training=True)
            indices = tf.stack(
                [
                    tf.range(self.config.batch_size),
                    tf.convert_to_tensor(batch.actions, dtype=tf.int32),
                ],
                axis=1,
            )
            chosen_values = tf.gather_nd(q_values, indices)
            loss = self.loss_fn(targets, chosen_values)
        gradients = tape.gradient(loss, self.online_network.trainable_variables)
        gradient_pairs = (
            (gradient, variable)
            for gradient, variable in zip(
                gradients, self.online_network.trainable_variables, strict=True
            )
            if gradient is not None
        )
        self.optimizer.apply_gradients(gradient_pairs)
        self.training_steps += 1
        if self.training_steps % self.config.target_update_steps == 0:
            self.update_target_network()
        return float(loss.numpy())

    def update_target_network(self) -> None:
        self.target_network.set_weights(self.online_network.get_weights())

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.online_network.save(destination)

    def load(self, path: str | Path) -> None:
        self.online_network = tf.keras.models.load_model(path)
        self.target_network.set_weights(self.online_network.get_weights())
