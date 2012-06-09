#!/bin/tcsh

set div = "min"
python simple_gen.py 1000000 5400 > mc1000k_ra_dec_arc"$div".dat
python simple_gen.py 100000 5400 > mc100k_ra_dec_arc"$div".dat
python simple_gen.py 200000 5400 > mc200k_ra_dec_arc"$div".dat
python simple_gen.py 50000 5400 > mc50k_ra_dec_arc"$div".dat
python simple_gen.py 10000 5400 > mc10k_ra_dec_arc"$div".dat

set div = "sec"
python simple_gen.py 1000000 324000 > mc1000k_ra_dec_arc"$div".dat
python simple_gen.py 100000 324000 > mc100k_ra_dec_arc"$div".dat
python simple_gen.py 200000 324000 > mc200k_ra_dec_arc"$div".dat
python simple_gen.py 50000 324000 > mc50k_ra_dec_arc"$div".dat
python simple_gen.py 10000 324000 > mc10k_ra_dec_arc"$div".dat
