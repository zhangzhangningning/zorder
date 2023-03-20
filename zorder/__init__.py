from gym.envs.registration import register

register(
    id="zorder/SelCol-v0",
    entry_point="zorder.envs:SelColEnv",
)
