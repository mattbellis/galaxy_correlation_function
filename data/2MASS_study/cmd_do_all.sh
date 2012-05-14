#!/bin/tcsh

set tag = $1

time ../../CUDA_code/Calculate_arc_length_two_datasets 2MASS_40k_RA_Dec.csv      2MASS_40k_RA_Dec.csv      out_DD_$tag.dat
time ../../CUDA_code/Calculate_arc_length_two_datasets MonteCarlo_40k_RA_Dec.csv MonteCarlo_40k_RA_Dec.csv out_RR_$tag.dat
time ../../CUDA_code/Calculate_arc_length_two_datasets 2MASS_40k_RA_Dec.csv      MonteCarlo_40k_RA_Dec.csv out_DR_$tag.dat
