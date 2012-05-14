#!/bin/tcsh

set tag = "subsamples_"$1

time ../../CUDA_code/Calculate_arc_length_two_datasets 2MASS_"$1"_RA_Dec.csv      2MASS_"$1"_RA_Dec.csv      out_DD_$tag.dat
time ../../CUDA_code/Calculate_arc_length_two_datasets MonteCarlo_"$1"_RA_Dec.csv MonteCarlo_"$1"_RA_Dec.csv out_RR_$tag.dat
time ../../CUDA_code/Calculate_arc_length_two_datasets 2MASS_"$1"_RA_Dec.csv      MonteCarlo_"$1"_RA_Dec.csv out_DR_$tag.dat

#time ../../CUDA_code/Calculate_arc_length_two_datasets_C_version 2MASS_"$1"_RA_Dec.csv      2MASS_"$1"_RA_Dec.csv      out_DD_Cversion_$tag.dat
#time ../../CUDA_code/Calculate_arc_length_two_datasets_C_version MonteCarlo_"$1"_RA_Dec.csv MonteCarlo_"$1"_RA_Dec.csv out_RR_Cversion_$tag.dat
#time ../../CUDA_code/Calculate_arc_length_two_datasets_C_version 2MASS_"$1"_RA_Dec.csv      MonteCarlo_"$1"_RA_Dec.csv out_DR_Cversion_$tag.dat
