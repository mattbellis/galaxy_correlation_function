#!/bin/tcsh

set executable = '../CUDA_code/Calculate_arc_length_two_datasets'

set mc   = '../sample_input_data/mc100k_ra_dec_arcmin.dat'
set data = '../sample_input_data/wg100k_ra_dec_arcmin.dat'

set global_params = '-w 0.05 -L 1.00 -N187 -l 1 -m'

time $executable $data $data $global_params -o logbinning_100k_data_data_arcmin.dat 
time $executable $mc $mc     $global_params -o logbinning_100k_mc_mc_arcmin.dat 
time $executable $data $mc   $global_params -o logbinning_100k_data_mc_arcmin.dat 



#time ../../CUDA_code/Calculate_arc_length_two_datasets /home/bellis/Work/Astronomy/catalogs/Wechsler/wg_mc_100k_ra_dec_arcmin.dat /home/bellis/Work/Astronomy/catalogs/Wechsler/wg_mc_100k_ra_dec_arcmin.dat -o logbinning_100k_mc_mc_minutes_as_input.dat -w 0.05 -L 1.00 -N187 -l 1 -m

#time ../../CUDA_code/Calculate_arc_length_two_datasets /home/bellis/Work/Astronomy/catalogs/Wechsler/wg100k_ra_dec_arcmin.dat /home/bellis/Work/Astronomy/catalogs/Wechsler/wg100k_ra_dec_arcmin.dat -o logbinning_100k_data_data_minutes_as_input.dat -w 0.05 -L 1.00 -N187 -l 1

#time ../../CUDA_code/Calculate_arc_length_two_datasets /home/bellis/Work/Astronomy/catalogs/Wechsler/wg100k_ra_dec_arcsec.dat /home/bellis/Work/Astronomy/catalogs/Wechsler/wg100k_ra_dec_arcsec.dat -o logbinning_100k_data_data_seconds_as_input.dat -w 0.05 -L 1.00 -N187 -l 1
#time ../../CUDA_code/Calculate_arc_length_two_datasets /home/bellis/Work/Astronomy/catalogs/Wechsler/wg100k_ra_dec_arcmin.dat /home/bellis/Work/Astronomy/catalogs/Wechsler/wg_mc_100k_ra_dec_arcmin.dat -o logbinning_100k_data_mc_minutes_as_input.dat -w 0.05 -L 1.00 -N187 -l 1 -m
#time ../../CUDA_code/Calculate_arc_length_two_datasets /home/bellis/Work/Astronomy/catalogs/Wechsler/wg_mc_100k_ra_dec_arcmin.dat /home/bellis/Work/Astronomy/catalogs/Wechsler/wg_mc_100k_ra_dec_arcmin.dat -o logbinning_100k_mc_mc_minutes_as_input.dat -w 0.05 -L 1.00 -N187 -l 1 -m
