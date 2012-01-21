#!/usr/bin/env python

import sys
from numpy import*
from math import pi
infilename = sys.argv[1]
num_lines=40000
Radian = pi / 180
infile = open(infilename,'r')

###outfilename = "parsed_file_%s_line%d.csv" % (infilename.split('.')[0], line_to_parse)
outfilename = sys.argv[2] 
outfile = open(outfilename,'w+')

output = "Right_Ascension , Declination\n"
outfile.write(output)

output = "%i\n" % (num_lines)
outfile.write(output)
output = ""

for i, line in enumerate(infile):
    if i > 11 and i < num_lines+12:
       line_num = line.split()      
       
       for j in range(1,3):     
           RA_Decl = float(line_num[j]) * Radian
           output += "%.4f , " % (RA_Decl)
      
#       output += " %s " % (line_num[24])      
       output += "\n"   
       outfile.write(output)
  
       output = ""

outfile.close()
    

