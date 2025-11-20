from sac_envs.ant_multi_old import AntMulti

env = AntMulti()
obs = env.reset()

print("observation shape:", obs.shape)
print("first observation:", obs)
