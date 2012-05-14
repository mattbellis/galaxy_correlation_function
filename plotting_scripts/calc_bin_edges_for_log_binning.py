import numpy as np

output = "{0.0000,"
npts = 1
lo = -3
hi =  2
max_pts = 6
#lo = 0
#hi =  4
#max_pts = 50
for i in range(lo,hi):
    x = np.logspace(i, i+1, max_pts)

    npts += len(x)-1

    for i,n in enumerate(x):
        if i<len(x)-1:
            #output += "%f," % (n)
            output += "%f\n" % (n)

output += "%f}" % (100.00)
npts += 1

print npts
print output
