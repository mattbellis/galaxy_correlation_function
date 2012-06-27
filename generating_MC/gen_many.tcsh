#!/bin/tcsh

set div = "min"
python simple_gen.py 1000000 5400 > wg_mc_1000k_ra_dec_arc"$div".dat
#python simple_gen.py 100000 5400 > wg_mc_100k_ra_dec_arc"$div".dat
#python simple_gen.py 200000 5400 > wg_mc_200k_ra_dec_arc"$div".dat
#python simple_gen.py 50000 5400 > wg_mc_50k_ra_dec_arc"$div".dat

#set div = "sec"
#python simple_gen.py 100000 324000 > wg_mc_100k_ra_dec_arc"$div".dat
#python simple_gen.py 200000 324000 > wg_mc_200k_ra_dec_arc"$div".dat
#python simple_gen.py 50000 324000 > wg_mc_50k_ra_dec_arc"$div".dat
