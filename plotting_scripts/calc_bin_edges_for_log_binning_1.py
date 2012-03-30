# From http://www.esapubs.org/archive/ecol/E089/052/appendix-A.pdf
import numpy as np

output = ""
npts = 1
#lo = -3
#hi =  2
#max_pts = 6
lo = 1.0
hi =  4
bin_width = 0.05
max_pts = 50
i = lo
while i < hi:
    
    pt1 = np.exp(np.log(i) + bin_width)

    output += "%f %f %f %f\n" % (i,pt1,(i+pt1)/2.0,pt1-i)

    i = pt1

    npts += 1

print npts
print output

# This is how you get the index.
#index = int((np.log(x)-np.log(imin))/bin_width)
# Should I use floor?
