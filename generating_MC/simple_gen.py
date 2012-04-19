import numpy as np
import sys

n = int(sys.argv[1])

myrange = 1.5

x = myrange*np.random.rand(n)
y = myrange*np.random.rand(n)

print n
for i,j in zip(x,y):
    print "%f %f" % (i,j)
