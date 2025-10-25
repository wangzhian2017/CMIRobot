from gymnasium.envs.registration import register


# Register the environment so we can create it with gym.make()
register(
    id="cmirobot/SO100-v0",
    entry_point="cmirobot.envs.so100:SO100Env",
    max_episode_steps=300,  # Prevent infinite episodes
)

register(
    id="cmirobot/GridWorld-v0",
    entry_point="cmirobot.envs.grid_world:GridWorldEnv",
)


