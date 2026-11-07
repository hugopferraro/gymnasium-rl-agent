"""Q-network construction."""

from __future__ import annotations

import tensorflow as tf


def build_q_network(
    state_shape: tuple[int, ...], action_count: int, hidden_layers: tuple[int, ...]
) -> tf.keras.Model:
    """Build a dense Q-network suitable for vector observations."""
    inputs = tf.keras.Input(shape=state_shape, name="state")
    x = inputs
    for index, width in enumerate(hidden_layers):
        x = tf.keras.layers.Dense(width, activation="relu", name=f"dense_{index + 1}")(x)
    outputs = tf.keras.layers.Dense(action_count, name="q_values")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="q_network")
