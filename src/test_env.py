from panda_env import PandaEnv
p = PandaEnv()
b = p.reset()

for i in range(100):
    p.step([0, 0, 5, .5, 3.14/2])
