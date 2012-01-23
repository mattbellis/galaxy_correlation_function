import os
import sys
import numpy as np

nsamples = 1000

infilename = sys.argv[1]

outdirname = 'outdir'
if len(sys.argv)>2:
    outdirname = sys.argv[2]

infile = open(infilename)

data = []
count = 0
ngals = 0
header = ""
for line in infile:

    if count == 0:
        header = line
    elif count >= 2:
        data.append(line)

    count += 1

ngals = len(data)
print ngals

if not os.access(outdirname,os.W_OK ):
    os.mkdir(outdirname,0744)

for i in range(0,nsamples):

    if i%100==0:
        print "samples: %s" % (i)

    outfilename = "%s/sample_%04d.csv" % (outdirname,i)
    outfile = open(outfilename,"w+")

    output = header
    output += "%d\n" % (ngals)

    for j in range(0,ngals):

        line_num = np.random.randint(0,ngals)

        output += data[line_num]

    outfile.write(output)
    outfile.close()



