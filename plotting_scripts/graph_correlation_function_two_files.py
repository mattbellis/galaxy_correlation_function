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
    filename0 = None # DD
    filename1 = None # RR
    filename2 = None # DR
    if len(args)<3:
        print "Need to pass in csv file on command line!"
        exit(-1)
    else:
        filename0 = args[0]
        filename1 = args[1]
        filename2 = args[2]


    infiles = [None,None,None]
    infiles[0] = csv.reader(open(filename0, 'rb'), delimiter=',', quotechar='#')
    infiles[1] = csv.reader(open(filename1, 'rb'), delimiter=',', quotechar='#')
    infiles[2] = csv.reader(open(filename2, 'rb'), delimiter=',', quotechar='#')
    ############################################################################

    #formatter = ScalarFormatter()
    #formatter.set_scientific(True)
    #formatter.set_powerlimits((-4,4))
    
    plt.xticks(fontsize=14, weight='bold')
    plt.yticks(fontsize=14, weight='bold')

    x_index = int(options.x_index)
    y_index = int(options.y_index)

    pts = [[],[],[]]
    xaxis_title = ""
    yaxis_title = ""
   
    for j in xrange(0,3):
        pts[j].append([])
        pts[j].append([])
        i = 0
        for row in infiles[j]:
            #print row
            if i%10000==0:
                print i

            if len(row)>1 and i>0 and i%plot_every_nth_point==0:
                print "%d %f %f" % (i,float(row[0]),float(row[1]))
                pts[j][0].append(float(row[0]))
                pts[j][1].append(float(row[1]))
                # pts[1].append(log10(float(row[y_index])))
       
            i += 1
        
        
    # The plot of the data
    # Line 
    # my_plot = plot(pts[0], pts[1])
    # Points
    
    xpts = []
    ypts = []

    npts = len(pts[0][0])
    for i in range(0,npts):

        print pts[0][0][i]

        dd = pts[0][1][i]/2.0
        rr = pts[1][1][i]/2.0
        dr = pts[2][1][i]

        w = 0.0 
        if rr>0:

            #w = (2*dd-(2*dr)+2*rr)/(2*rr)
            w = (dd-(1*dr)+rr)/(rr)

            xpts.append(pts[0][0][i])
            ypts.append(w)
            #print "%f %f" % (degrees(pts[0][0][i]),w)

    print len(xpts)
    print len(ypts)
    print xpts
    print ypts
    my_plot = scatter(xpts, ypts, s = 10)
    
    #subplots[0].xaxis.set_major_formatter(formatter)

    subplots[0].set_xlabel(r"$\theta$ (degrees)", fontsize=24, weight='bold')
    subplots[0].set_ylabel(r"w($\theta$)", fontsize=24, weight='bold')
    subplots[0].set_xscale('log')
    subplots[0].set_yscale('log')
   
    #subplots[0].set_xlim(0.01,60)
    subplots[0].set_ylim(0.01,100)

 
    #infile_basename = filename.split('/')[-1].split('.')[0] 
    infile_basename = "test_cf"
    output_file_name = "plot_%s_x%d_y%d.png" % (infile_basename,x_index,y_index)
    plt.savefig(output_file_name)
    plt.show()

################################################################################
# Top-level script evironment
################################################################################
if __name__ == "__main__":
    main()
