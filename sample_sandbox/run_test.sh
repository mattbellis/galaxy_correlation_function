#!/bin/tcsh

#set executable = '../CUDA_code/Calculate_arc_length_two_datasets'
set executable = '../CUDA_code/Calculate_arc_length_two_datasets_shared'

set ngals = $1

set mc   = '../sample_input_data/mc'$ngals'k_ra_dec_arcmin.dat'
set data = '../sample_input_data/wg'$ngals'k_ra_dec_arcmin.dat'

#set global_params = '-w 0.05 -L 1.00 -N187 -l 1 -m'
set global_params = '-w 0.05 -L 1.00 -l 1 -m'

time $executable $data $data $global_params -o logbinning_"$ngals"k_data_data_arcmin.dat 
echo "\n#####################\n"
time $executable $mc $mc     $global_params -o logbinning_"$ngals"k_mc_mc_arcmin.dat 
echo "\n#####################\n"
time $executable $data $mc   $global_params -o logbinning_"$ngals"k_data_mc_arcmin.dat 
echo "\n#####################\n"
