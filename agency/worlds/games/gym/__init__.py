from gym.envs.registration import register
from agency.worlds.games.gym.simple_identity import IdentityEnv, IdentityEnvContinuous

register(
    id='discrete-identity-env-v0',
    entry_point='agency.worlds.games.gym.simple_identity:IdentityEnv',
)

register(
    id='continuous-identity-env-v0',
    entry_point='agency.worlds.games.gym.simple_identity:IdentityEnvContinuous',
)
