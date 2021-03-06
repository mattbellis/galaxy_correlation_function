time ../../CUDA_code/Calculate_arc_length_two_datasets 2MASS_40k_RA_Dec.dat 2MASS_40k_RA_Dec.dat
time ../../CUDA_code/Calculate_arc_length_two_datasets 2MASS_40k_RA_Dec.dat 2MASS_40k_RA_Dec.dat -o logbinning_test.dat -w 0.2 -L 0.001 -N15 -l 2
time ../../CUDA_code/Calculate_arc_length_two_datasets 2MASS_40k_RA_Dec.dat 2MASS_40k_RA_Dec.dat -o logbinning_test.dat -w 0.5 -L 0.001 -N15 -l 1

time ../../CUDA_code/Calculate_arc_length_two_datasets ~/Work/Astronomy/catalogs/Wechsler/wg100k_ra_dec.dat ~/Work/Astronomy/catalogs/Wechsler/wg100k_ra_dec.dat -o logbinning_test.dat -w 0.5 -L 0.001 -N25 -l 1
time ../../CUDA_code/Calculate_arc_length_two_datasets ../2MASS_study/mc100k_ra_dec.dat ../2MASS_study/mc100k_ra_dec.dat -o logbinning_test_mcmc.dat -w 0.5 -L 0.001 -N25 -l 1
time ../../CUDA_code/Calculate_arc_length_two_datasets ~/Work/Astronomy/catalogs/Wechsler/wg100k_ra_dec.dat ../2MASS_study/mc100k_ra_dec.dat -o logbinning_test_mc_data.dat -w 0.5 -L 0.001 -N25 -l 1


# Testing arcsecond calculations
time ../../CUDA_code/Calculate_arc_length_two_datasets ~/Work/Astronomy/catalogs/Wechsler/wg100k_ra_dec_arcseconds.dat ~/Work/Astronomy/catalogs/Wechsler/wg100k_ra_dec_arcseconds.dat -o logbinning_test_data_data.dat -w 0.5 -L 0.0001 -N35 -l 1

# This gives the right binning for the minutes
time ../../CUDA_code/Calculate_arc_length_two_datasets /home/bellis/Work/Astronomy/catalogs/Wechsler/wg50k_ra_dec_arcminutes.dat /home/bellis/Work/Astronomy/catalogs/Wechsler/wg50k_ra_dec_arcminutes.dat -o logbinning_test_data_data.dat -w 0.05 -L 1.00 -N15 -l 1
