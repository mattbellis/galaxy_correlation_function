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
#    fig2 = plt.figure(figsize=(12, 8), dpi=90, facecolor='w', edgecolor='k')
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
     #   subplots.append(fig2.add_subplot(division))
        
    ################################################################################
    ################################################################################
    
    ############################################################################
    # Open the file, assuming that it is .csv format and checking to see
    # that it exists.
    ############################################################################
    filename = []
    
    if len(args)<2:
        print "Need to pass in 2 csv files on command line!"
        exit(-1)
    else:
        for i in range(0,2):
           filename.append(args[i])
      
    infile = [] 
    for i in range(0,2):
       infile.append(csv.reader(open(filename[i], 'rb'), delimiter=',', quotechar='#'))
      
    ############################################################################

#   formatter = ScalarFormatter()
#    formatter.set_scientific(True)
#    formatter.set_powerlimits((-4,4))
    
    plt.xticks(fontsize=14, weight='bold')
    plt.yticks(fontsize=14, weight='bold')

    x_index = int(options.x_index)
    y_index = int(options.y_index)


    x_pts = ([],[])
    y_pts = ([],[])  
    xaxis_title = ""
    yaxis_title = ""
   
    
    for j in range(0,2):
        i=0 
        for row in infile[j]:
            print row
            if i%10000==0:
               print i

        # Assume that there is header information that we will use for the
        # axis labels
            if len(row)>y_index and i==0:
                xaxis_title = row[x_index]
                yaxis_title = row[y_index]
        # Otherwise, it is data
            elif len(row)>y_index and i>0 and i%plot_every_nth_point==0:
        #        nvalues = len(row)
        #       for k in range(0,nvalues):
	#          print row[0]
                x_pts[j].append(float(row[0]))
                y_pts[j].append(float(row[1]))

            i += 1
    
    myplots = []
    formatting = ['ro','bo']
    leg_text = ["real RA_Declination data", "random RA_Declination data"]
    for i in range(0,2):
        print x_pts[i]
        print y_pts[i]
        myplots.append(subplots[0].plot(x_pts[i], y_pts[i], formatting[i], markersize=1))
#    for i in range(3,5):
#        myplots.append(subplots[1].plot(x_pts[i], y_pts[i], formatting[i], markersize=10))

#    subplots[0].xaxis.set_major_formatter(formatter)

    subplots[0].set_xlabel(xaxis_title, fontsize=14, weight='bold')
    subplots[0].set_ylabel(yaxis_title, fontsize=14, weight='bold')
   
#    subplots[1].set_xlabel(xaxis_title, fontsize=14, weight='bold')
#    subplots[1].set_ylabel(yaxis_title, fontsize=14, weight='bold')

 # subplots[0].set_xlim(1.5e9)
   # subplots[0].set_ylim(1.5e9)
    fig1.legend((myplots[0],myplots[1]), (leg_text[0], leg_text[1]), 'upper right',1)
  
#    fig2.legend((myplots[3], myplots[4]), (leg_text[3], leg_text[4]), 'upper right', 1)
   
    infile_basename = filename[0].split('/')[-1].split('.')[0] 
    output_file_name = "plot_together_%s_x%d_y%d.png" % (infile_basename,x_index,y_index)
    plt.savefig(output_file_name)
    plt.show()

################################################################################
# Top-level script evironment
################################################################################
if __name__ == "__main__":
    main()
