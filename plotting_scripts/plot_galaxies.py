#!/usr/bin/env python
"""
Make a plot of some numbers read in from a .csv file with a header.
"""

import sys
from math import *
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.font_manager as fm

from optparse import OptionParser
from pylab import *


################################################################################
# main
################################################################################
def main():

    # Parse the command line options
    myusage = "\nusage: %prog [options] <file1.csv>"

    parser = OptionParser(usage = myusage)
    parser.add_option("-x", "--x-axis",  dest="x_index", default=0, 
            help="Column in file to use as x-axis values.")
    parser.add_option("-y", "--y-axis",  dest="y_index", default=1, 
            help="Column in file to use as y-axis values.")
    parser.add_option("-n", "--plot_every_nth_point",  dest="plot_every_nth_point",
            default=1, 
            help="If you're plotting a really big file, then only plot every nth point. [default=1]")

    # Parse the options
    (options, args) = parser.parse_args()

    plot_every_nth_point = int(options.plot_every_nth_point)

    ################################################################################
    ################################################################################
    # Make a figure on which to plot stuff.
        
    figs = []
    for i in xrange(2):
        figs.append(plt.figure(figsize=(12, 8), dpi=90, facecolor='w', edgecolor='k'))
#    fig2 = plt.figure(figsize=(12, 8), dpi=90, facecolor='w', edgecolor='k')
    #
    # Usage is XYZ: X=how many rows to divide.
    #               Y=how many columns to divide.
    #               Z=which plot to plot based on the first being '1'.
    # So '111' is just one plot on the main figure.
    ################################################################################
    subplots = []
    for i in range(0,2):
        subplots.append(figs[i].add_subplot(1,1,1))
        
    ################################################################################
    ################################################################################
    
    ############################################################################
    # Open the file, assuming that it is .csv format and checking to see
    # that it exists.
    ############################################################################
    filename = args[0]
    
    infile = open(filename,'r')
      
    ############################################################################

#   formatter = ScalarFormatter()
#    formatter.set_scientific(True)
#    formatter.set_powerlimits((-4,4))
    
    plt.xticks(fontsize=14, weight='bold')
    plt.yticks(fontsize=14, weight='bold')

    x_index = int(options.x_index)
    y_index = int(options.y_index)


    xpts = []
    ypts = []  
    zpts = []  
    xaxis_title = ""
    yaxis_title = ""
   
    
    i=0 
    for row in infile:
        #print row
        if i%10000==0:
           print i

        if i>=20000000:
            break

        vals = row.split()
        if i>0:
            if len(vals)>=3:
                if i%plot_every_nth_point==0:
                    xpts.append(float(vals[0]))
                    ypts.append(float(vals[1]))
                    zpts.append(float(vals[2]))

        i += 1
    
    #print xpts[i]
    #print ypts[i]
    #myplots = subplots[0].plot(xpts, ypts,'o',markersize=1)
    h = subplots[1].hist(zpts,bins=500)

    subplots[0].set_xlabel(xaxis_title, fontsize=14, weight='bold')
    subplots[0].set_ylabel(yaxis_title, fontsize=14, weight='bold')
   
    subplots[1].set_xlabel("z", fontsize=24, weight='bold')
    subplots[1].set_ylabel("# galaxies", fontsize=24, weight='bold')

    #infile_basename = filename[0].split('/')[-1].split('.')[0] 
    #output_file_name = "plot_together_%s_x%d_y%d.png" % (infile_basename,x_index,y_index)
    #plt.savefig(output_file_name)
    plt.show()

################################################################################
# Top-level script evironment
################################################################################
if __name__ == "__main__":
    main()
