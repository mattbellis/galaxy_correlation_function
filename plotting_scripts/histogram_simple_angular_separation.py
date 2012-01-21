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
from optparse import OptionParser
from pylab import *


################################################################################
# main
################################################################################
def angular_separation(ra0,ra1,dec0,dec1):

    ang_sep = 1.0
    
    # Calculate stuff here. 
    a_diff = ra1-ra0

    sin_a_diff = sin(a_diff)
    cos_a_diff = cos(a_diff)

    sin_d1 = sin(dec0)
    cos_d1 = cos(dec0)

    sin_d2 = sin(dec1)
    cos_d2 = cos(dec1)

    mult1 = cos_d2 * cos_d2 * sin_a_diff * sin_a_diff;
    mult2 = cos_d1 * sin_d2 - sin_d1 * cos_d2 * cos_a_diff;
    mult2 = mult2 * mult2;

    numer = sqrt(mult1 + mult2);

    denom = sin_d1 *sin_d2 + cos_d1 * cos_d2 * cos_a_diff;

    ang_sep = atan2(numer, denom);

    return ang_sep

################################################################################
# main
################################################################################
def main():

    # Parse the command line options
    myusage = "\nusage: %prog [options] <file1.csv>"
    parser = OptionParser(usage = myusage)
    parser.add_option("-x", "--x-axis",  dest="x_index", default=0, 
            help="Column in file to use as x-axis values.")
    parser.add_option("-m", "--max",  dest="max", default=10000, 
            help="Calculate only max number of angular separations")
    parser.add_option("-n", "--plot_every_nth_point",  dest="plot_every_nth_point",
            default=1, 
            help="If you're plotting a really big file, then only plot every nth point. [default=1]")

    # Parse the options
    (options, args) = parser.parse_args()

    plot_every_nth_point = int(options.plot_every_nth_point)

    ################################################################################
    ################################################################################
    # Make a figure on which to plot stuff.
    fig1 = plt.figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')
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
    plt.xticks(fontsize=14, weight='bold')
    plt.yticks(fontsize=14, weight='bold')
   
    x_index = int(options.x_index)
    
    pts = [[],[]]
    xaxis_title = ""
    
    i = 0
    for row in infile:
        #print row
        if i%10000==0:
            print i

        # Assume that there is header information that we will use for the
        # axis labels
        if i==0:
            #xaxis_title = row[x_index]
            xaxis_title = "Something"
        # Otherwise, it is data
        elif i>0 and i%plot_every_nth_point==0:
            pts[0].append(float(row[0]))
            pts[1].append(float(row[1]))

        i += 1
    
    max = int(options.max)

    ang_sep_pts = []
    npoints = len(pts[0])

    if max>npoints:
        max=npoints

    for i in range(0,max):

        ra0 = pts[0][i]
        dec0 = pts[1][i]

        for j in range(i+1,max):

            ra1 = pts[0][j]
            dec1 = pts[1][j]


            ang_sep_pts.append(angular_separation(ra0,ra1,dec0,dec1))

            #print "%f %f %f %f" % (ra0,ra1,dec0,dec1)


    
    #Histogram
    #my_plot = hist(pts[0], bins = 100, facecolor='blue', alpha=0.75, range=(0,6.0)) 
    my_plot = hist(ang_sep_pts, bins = 100, facecolor='blue', alpha=0.75, range=(0,6.0)) 

    subplots[0].set_xlabel(xaxis_title, fontsize=14, weight='bold')

    infile_basename = filename.split('/')[-1].split('.')[0] 
    output_file_name = "plot_%s_x%d.png" % (infile_basename,x_index)
    plt.savefig(output_file_name)
    plt.show()

################################################################################
# Top-level script evironment
################################################################################
if __name__ == "__main__":
    main()
