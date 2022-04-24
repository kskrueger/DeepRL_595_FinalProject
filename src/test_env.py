import time

import numpy as np

from panda_env import PandaEnv
p = PandaEnv((180, 240), (180, 240))
b = p.reset()


p.move_to([.35, .25, 0])
# time.sleep(1)
p.move_to([.35, .25, .75])
# time.sleep(1)

p.move_to([.35, -.25, 0])
# time.sleep(1)
p.move_to([.35, -.25, .75])
# time.sleep(1)

p.move_to([.6, .5, 0])
# time.sleep(1)
p.move_to([.6, .5, .75])
# time.sleep(1)

p.move_to([.35, 0, 0])
# time.sleep(1)
p.move_to([.35, 0, .75])
# time.sleep(1)
p.move_to([.35+.5, 0, 0])
# time.sleep(1)
p.move_to([.35+.5, 0, .75])

# for i in range(100):
#     b, _, _, _ = p.step(0)
#     print(b['motors'])
