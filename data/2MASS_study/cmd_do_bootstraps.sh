#!/bin/tcsh

set data_dir = "bootstrapping_samples_2MASS_40k"
set mc_dir = "bootstrapping_samples_MonteCarlo_40k"
set outdir = "bootstrapping_output"

set tag = "40k_bins6"

#@ i = `echo $1 | awk '{print $1}'`
#@ max = `echo $2 | awk '{print $1}'`
@ i = $1
@ max = $2

echo "Running boostrapping samples from " $i " to " $max

cd ~/CUDA/galaxy_correlation_function/data/2MASS_study

while ( $i < $max )

    set datafile = `printf "$data_dir""/sample_%04d.csv" $i`
    set mcfile = `printf "$mc_dir""/sample_%04d.csv" $i`
    
    #echo $datafile
    #echo $mcfile

    set outfile = `printf "$outdir""/output_DD_"$tag"_%04d.dat" $i`
    echo time ../../CUDA_code/Calculate_arc_length_two_datasets $datafile  $datafile $outfile
    time ../../CUDA_code/Calculate_arc_length_two_datasets $datafile  $datafile $outfile
    set outfile = `printf "$outdir""/output_RR_"$tag"_%04d.dat" $i`
    echo time ../../CUDA_code/Calculate_arc_length_two_datasets $mcfile  $mcfile $outfile
    time ../../CUDA_code/Calculate_arc_length_two_datasets $mcfile  $mcfile $outfile
    set outfile = `printf "$outdir""/output_DR_"$tag"_%04d.dat" $i`
    echo time ../../CUDA_code/Calculate_arc_length_two_datasets $datafile  $mcfile $outfile
    time ../../CUDA_code/Calculate_arc_length_two_datasets $datafile  $mcfile $outfile

    @ i++

end
