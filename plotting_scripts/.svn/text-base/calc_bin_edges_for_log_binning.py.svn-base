import numpy as np

output = "{0.0000,"
npts = 1
for i in range(-3,2):
    x = np.logspace(i, i+1, 6)

    npts += len(x)-1

    for i,n in enumerate(x):
        if i<len(x)-1:
            output += "%f," % (n)

output += "%f}" % (100.00)
npts += 1

print npts
print output
