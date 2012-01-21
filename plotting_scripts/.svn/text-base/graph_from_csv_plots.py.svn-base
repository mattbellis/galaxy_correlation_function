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
    fig1 = plt.figure(figsize=(12, 8), dpi=90, facecolor='w', edgecolor='k')
    #
    # Usage is XYZ: X=how many rows to divide.
    #               Y=how many columns to divide.
    #               Z=which plot to plot based on the first being '1'.
    # So '111' is just one plot on the main figure.
    ################################################################################
    subplots = []
    for i in range(1,2):
        division = 110 + i
        subplots.append(fig1.add_subplot(division))
    
    ################################################################################
    ################################################################################
    
    ############################################################################
    # Open the file, assuming that it is .csv format and checking to see
    # that it exists.
    ############################################################################
    filename = None
    if len(args)==0:
        print "Need to pass in csv file on command line!"
        exit(-1)
    else:
        filename = args[0]


    infile = csv.reader(open(filename, 'rb'), delimiter=',', quotechar='#')
    ############################################################################

    formatter = ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits((-4,4))
    
    plt.xticks(fontsize=14, weight='bold')
    plt.yticks(fontsize=14, weight='bold')

    x_index = int(options.x_index)
    y_index = int(options.y_index)

    pts = [[],[]]
    xaxis_title = ""
    yaxis_title = ""
   
    i = 0
    for row in infile:
        #print row
        if i%10000==0:
            print i

        # Assume that there is header information that we will use for the
        # axis labels
        if len(row)>y_index and i==0:
            xaxis_title = row[x_index]
            yaxis_title = row[y_index]
        # Otherwise, it is data
        elif len(row)>y_index and i>0 and i%plot_every_nth_point==0:
            pts[0].append(float(row[x_index]))
           
            pts[1].append(float(row[y_index]))
            # pts[1].append(log10(float(row[y_index])))
   
        i += 1
    
    
    # The plot of the data
    # Line 
    # my_plot = plot(pts[0], pts[1])
    # Points

    my_plot = scatter(pts[0], pts[1], s = 1)
    
    subplots[0].xaxis.set_major_formatter(formatter)

    subplots[0].set_xlabel(xaxis_title, fontsize=14, weight='bold')
    subplots[0].set_ylabel(yaxis_title, fontsize=14, weight='bold')
   
   # subplots[0].set_xlim(1.5e9)
   # subplots[0].set_ylim(1.5e9)

 
    infile_basename = filename.split('/')[-1].split('.')[0] 
    output_file_name = "plot_%s_x%d_y%d.png" % (infile_basename,x_index,y_index)
    plt.savefig(output_file_name)
    plt.show()

################################################################################
# Top-level script evironment
################################################################################
if __name__ == "__main__":
    main()
