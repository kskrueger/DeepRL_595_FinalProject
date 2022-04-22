import numpy as np

from panda_env import PandaEnv
p = PandaEnv((180, 240), (180, 240))
b = p.reset()

for i in range(1):
    b, _, _, _ = p.step(7)
    print(b['motors'])
