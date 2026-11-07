from gymnasium_rl_agent.config import EnvironmentConfig
from gymnasium_rl_agent.environments import make_environment


def test_lunar_lander_smoke() -> None:
    env = make_environment(EnvironmentConfig(max_episode_steps=2), seed=1)
    try:
        observation, info = env.reset(seed=1)
        assert observation.shape == (8,)
        assert isinstance(info, dict)
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()
