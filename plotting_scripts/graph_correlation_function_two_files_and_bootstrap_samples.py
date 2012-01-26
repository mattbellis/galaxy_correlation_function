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

# Correlation function
def calc_cf(dd,rr,dr,norms=[100,100,100]):

        dd_norm = norms[0]
        rr_norm = norms[1]
        dr_norm = norms[2]

        dd /= dd_norm
        rr /= rr_norm
        dr /= dr_norm

        w = 0.0 
        if rr>0:
            w = (dd-(2*dr)+rr)/(rr)

        return w



################################################################################
# main
################################################################################
def main():

    # Parse the command line options
    myusage = "\nusage: %prog [options] <file1.csv>"
    parser = OptionParser(usage = myusage)
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
    filename0 = "../data/2MASS_study/out_DD_40k_bins6.dat"
    filename1 = "../data/2MASS_study/out_RR_40k_bins6.dat"
    filename2 = "../data/2MASS_study/out_DR_40k_bins6.dat"


    infiles = [None,None,None]
    infiles[0] = csv.reader(open(filename0, 'rb'), delimiter=',', quotechar='#')
    infiles[1] = csv.reader(open(filename1, 'rb'), delimiter=',', quotechar='#')
    infiles[2] = csv.reader(open(filename2, 'rb'), delimiter=',', quotechar='#')
    ############################################################################

    pts = [[],[],[]]
    for j in xrange(0,3):
        pts[j].append([])
        pts[j].append([])
        i = 0
        for row in infiles[j]:
            #print row
            if i%10000==0:
                print i

            if len(row)>1 and i>0 and i%plot_every_nth_point==0:
                #print "%d %f %f" % (i,float(row[0]),float(row[1]))
                pts[j][0].append(float(row[0]))
                pts[j][1].append(float(row[1]))
                # pts[1].append(log10(float(row[y_index])))
       
            i += 1
        
        #infiles[j].close()
        
    # The plot of the data
    # Line 
    # my_plot = plot(pts[0], pts[1])
    # Points
    
    xpts = []
    ypts = []

    ngalaxies = 40000.0
    #ngalaxies = 160000.0

    dd_norm = ((ngalaxies*ngalaxies)-ngalaxies)/2.0
    rr_norm = ((ngalaxies*ngalaxies)-ngalaxies)/2.0
    dr_norm = (ngalaxies*ngalaxies)

    norms = [dd_norm,rr_norm,dr_norm]

    print dd_norm 
    print rr_norm 
    print dr_norm 

    npts = len(pts[0][0])
    for i in range(0,npts):

        #print pts[0][0][i]

        dd = pts[0][1][i]
        rr = pts[1][1][i]
        dr = pts[2][1][i]

        x = pts[0][0][i]

        w = 0.0
        if rr>0:

            w = calc_cf(dd,rr,dr,norms)

            xpts.append(x)
            ypts.append(w)

    print len(xpts)
    print len(ypts)
    #print xpts
    #print ypts


    ############################################################################
    # Run over the bootstrap files.
    ############################################################################
    max_samples = 1000

    bootstrap_pts = []

    infiles = [None,None,None]

    for k in range(0,max_samples):
        filename0 = "../data/2MASS_study/bootstrapping_output/output_DD_40k_bins6_%04d.dat" % (k)
        filename1 = "../data/2MASS_study/bootstrapping_output/output_RR_40k_bins6_%04d.dat" % (k)
        filename2 = "../data/2MASS_study/bootstrapping_output/output_DR_40k_bins6_%04d.dat" % (k)

        print filename0

        infiles[0] = csv.reader(open(filename0, 'rb'), delimiter=',', quotechar='#')
        infiles[1] = csv.reader(open(filename1, 'rb'), delimiter=',', quotechar='#')
        infiles[2] = csv.reader(open(filename2, 'rb'), delimiter=',', quotechar='#')
        ############################################################################

        bootstrap_pts.append([[],[],[]])
        for j in xrange(0,3):
            bootstrap_pts[k][j].append([])
            bootstrap_pts[k][j].append([])
            i = 0
            for row in infiles[j]:
                #print row
                if i%10000==0:
                    print i

                if len(row)>1 and i>0 and i%plot_every_nth_point==0:
                    #print "%d %f %f" % (i,float(row[0]),float(row[1]))
                    bootstrap_pts[k][j][0].append(float(row[0]))
                    bootstrap_pts[k][j][1].append(float(row[1]))
                    # bootstrap_pts[1].append(log10(float(row[y_index])))
           
                i += 1
                
            #infiles[j].close()
            

    # Calculate limits for first 20 points.
    confidence_interval = 0.68
    lo_ci_index = int(((1.0-confidence_interval)*max_samples)/2.0)
    hi_ci_index = int(confidence_interval*max_samples) + lo_ci_index+1

    print "confidence_intervals: %d %d" % (lo_ci_index,hi_ci_index)

    b_x_pts = []
    b_w_pts = []
    y_err_lo = []
    y_err_hi = []
    for j in range(0,20):
        b_w_pts.append([])
        for i in range(0,max_samples):

            # Grab the x-axis point.
            if i==0:
                b_x_pts.append(bootstrap_pts[i][0][0][j])

            dd = bootstrap_pts[i][0][1][j]
            rr = bootstrap_pts[i][1][1][j]
            dr = bootstrap_pts[i][2][1][j]

            w = calc_cf(dd,rr,dr,norms)

            b_w_pts[j].append(w)

        b_w_pts[j].sort()

        y_err_lo.append(b_w_pts[j][lo_ci_index])
        y_err_hi.append(b_w_pts[j][hi_ci_index])




    ############################################################################
    # Build the plot
    ############################################################################
    #my_plot = scatter(xpts, ypts, yerr=(y_err_lo,y_err_hi), s=30)
    my_plot = errorbar(xpts, ypts, yerr=(y_err_lo,y_err_hi))
    #my_plot = errorbar(xpts, ypts)
    
    #formatter = ScalarFormatter()
    #formatter.set_scientific(True)
    #formatter.set_powerlimits((-4,4))
    
    plt.xticks(fontsize=14, weight='bold')
    plt.yticks(fontsize=14, weight='bold')

    xaxis_title = ""
    yaxis_title = ""
   
    #subplots[0].xaxis.set_major_formatter(formatter)

    subplots[0].set_xlabel(r"$\theta$ (degrees)", fontsize=24, weight='bold')
    subplots[0].set_ylabel(r"w($\theta$)", fontsize=24, weight='bold')
    subplots[0].set_xscale('log')
    subplots[0].set_yscale('log')
   
    subplots[0].set_xlim(0.01,100)
    subplots[0].set_ylim(0.01,100)

 
    #infile_basename = filename.split('/')[-1].split('.')[0] 
    infile_basename = "test_cf"
    output_file_name = "plot_%s_.png" % ("temp")
    plt.savefig(output_file_name)
    plt.show()

################################################################################
# Top-level script evironment
################################################################################
if __name__ == "__main__":
    main()
