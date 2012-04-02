#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include <unistd.h>

#include <cuda_runtime.h>
//#include <cutil_inline.h>

using namespace std;

#define SUBMATRIX_SIZE 10000
#define DEFAULT_NBINS 27 // for log binning

#define CONV_FACTOR 57.2957795 // 180/pi

////////////////////////////////////////////////////////////////////////
// Kernel to calculate angular distances between galaxies and histogram
// the distances.
////////////////////////////////////////////////////////////////////////
__global__ void distance(float *a0, float *d0, float *a1, float *d1, int xind, int yind, int *dev_hist, float hist_min, float hist_max, int nbins, float bin_width, bool log_binning=0, bool two_different_files=1)
{

    ////////////////////////////////////////////////////////////////////////////
    // Idx will keep track of which thread is being calculated within a given 
    // warp.
    ////////////////////////////////////////////////////////////////////////////
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // This should range to SUBMATRIX_SIZE

    int thread_idx = idx;
    idx += xind;

    int i=0;
    //int j=0;

    ////////////////////////////////////////////////////////////////////////////
    // Copy over a local version of the bin edges. This will prevent threads
    // from accessing the same memory that holds this information.
    ////////////////////////////////////////////////////////////////////////////
    /*
    float bin_edges[1024]; // This is hard-coded and will result in bugs if a histogram
    // with greater than 1024 bins is used.
    int bin_offset = thread_idx*nbins;
    for (i=0;i<nbins;i++)
    {
        bin_edges[i] = dev_bin_edges[i+bin_offset];
    }
    */

    //float hist_min = bin_edges[0];
    //float hist_max = bin_edges[nbins-1];

    float alpha = a0[idx], delta0 = d0[idx];
    float cos_d0 = cos(delta0), sin_d0 = sin(delta0), dist;

    int ymax = yind + SUBMATRIX_SIZE;
    int bin_index = 0; 
    int offset = 0;

    float a_diff, sin_a_diff, cos_a_diff;
    float cos_d1, sin_d1, numer, denom, mult1, mult2;    

    bool do_calc = 1;
    for(i=yind; i<ymax; i++)
    {
        if (two_different_files)
        {
            do_calc = 1;
        }
        else // Doing the same file
        {
            if(idx > i)
                do_calc=1;
            else
                do_calc=0;
        }
        //if(idx > i) ///////// CHECK THIS
        if (do_calc)
        {
            a_diff = a1[i] - alpha;

            sin_a_diff = sin(a_diff);
            cos_a_diff = cos(a_diff);

            sin_d1 = sin(d1[i]);
            cos_d1 = cos(d1[i]);

            mult1 = cos_d1 * cos_d1 * sin_a_diff * sin_a_diff;
            mult2 = cos_d0 * sin_d1 - sin_d0 * cos_d1 * cos_a_diff;
            mult2 = mult2 * mult2;

            numer = sqrt(mult1 + mult2); 

            denom = sin_d0 *sin_d1 + cos_d0 * cos_d1 * cos_a_diff;

            //dist = atan(num);  
            dist = atan2(numer,denom);  
            dist *= CONV_FACTOR;  // Convert to degrees

            if(dist < hist_min)
                bin_index = 0; 
            else if(dist >= hist_max)
                bin_index = nbins + 1;
            else
           if (!log_binning)
           {
               bin_index = int((dist-hist_min)/bin_width) + 1;
           }
           else // log binning
           {
               bin_index = int((log(dist)-log(hist_min))/bin_width) + 1;
           }

            /*
            {
                //bin_index = int(((dist - hist_min) * nbins / hist_max) +1);    
                bin_index = 0;
                for (j=0;j<nbins-1;j++)
                {
                    // This works
                    if (dist>=bin_edges[j] && dist<bin_edges[j+1])
                    {
                        bin_index = j+1;
                        break;
                    }
                }
            }
            */

            offset = ((nbins+2)*thread_idx);
            bin_index += offset;

            dev_hist[bin_index]++;

        }
    }
}

////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // Needed for parsing command-line arguments.
    extern char *optarg;
    extern int optind, optopt, opterr;
    int c;
    char *filename;
    char *outfilename = NULL;
    char defaultoutfilename[256];
    sprintf(defaultoutfilename,"default_out.dat");
    char *binning_filename = NULL;
    FILE *binning_file = NULL;

    float hist_lower_range = 0.0000001;
    float hist_upper_range = 0;
    int nbins = 100;
    float hist_bin_width = 0.05;
    bool log_binning_flag = 0; // False

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////


    while ((c = getopt(argc, argv, "ab:o:L:N:lw:")) != -1) {
        switch(c) {
            case 'N':
                printf("N is set\n");
                nbins = atoi(optarg);
                break;
            case 'L':
                printf("L is set\n");
                hist_lower_range = atof(optarg);
                break;
            case 'w':
                hist_bin_width = atof(optarg);
                printf("Histogram bin width: %f\n",hist_bin_width);
                break;
            case 'l':
                log_binning_flag = 1;
                printf("Will use log binning.\n");
                break;
            case 'b':
                binning_filename = optarg;
                printf("Using binning information from file: %s\n",binning_filename);
                break;
            case 'o':
                outfilename = optarg;
                printf("Output filename is %s\n", outfilename);
                break;
            case '?':
                printf("unknown arg %c\n", optopt);
                break;
        }
    }

    if (argc < 2)
    {

        printf("\nMust pass in at least two input files on command line!\n");
        printf("\nUsage: ", argv[0] );
        //printf(" <cluster_data file> <distances file> \n\n");
        exit(1);
    }

    // Set a default output file name, if none was passed in on the 
    // command line.
    if (outfilename == NULL) 
    {
        outfilename = defaultoutfilename;
        printf("Output filename is %s\n", outfilename);
    }

    float temp_lo = hist_lower_range;
    if (hist_upper_range == 0)
    {
        if (log_binning_flag==0)
        {
            for (int i=0;i<nbins;i++)
            {
                hist_upper_range = temp_lo + hist_bin_width;
                temp_lo = hist_upper_range;
            }
        }
        else
        {
            for (int i=0;i<nbins;i++)
            {
                hist_upper_range = exp(log(temp_lo) + hist_bin_width);
                temp_lo = hist_upper_range;
            }
        }
    }
    printf("hist_upper_range: %f\n",hist_upper_range);

    //printf ("%d\n", optind);
    //printf ("%d\n", argc);
    //printf ("%d\n", optind);
    //printf ("%s\n", argv[optind]);
    //printf ("%s\n", argv[optind+1]);

    FILE *infile0, *infile1, *outfile ;
    infile0 = fopen(argv[optind],"r");
    infile1 = fopen(argv[optind+1],"r");
    //outfile = fopen(argv[3], "w");

    printf("Opening input file 0: %s\n",argv[optind]);
    printf("Opening input file 1: %s\n",argv[optind+1]);
    //printf("Outfilename: %s\n",outfilename);
    outfile = fopen(outfilename, "w");

    ////////////////////////////////////////////////////////////////////////////
    // Check to see if the two files are actually the same file.
    // This is the case for the DD and RR calculations and change slightly
    // the exact calculations being performed.
    ////////////////////////////////////////////////////////////////////////////
    bool two_different_files = 1;
    if (strcmp(argv[optind],argv[optind+1])==0)
    {
        two_different_files = 0;
        printf("Using the same file!\n");
    }
    printf("\n");
    ////////////////////////////////////////////////////////////////////////////
    // Now get the info from the device.
    ////////////////////////////////////////////////////////////////////////////
    printf("\n------ CUDA device diagnostics ------\n\n");

    int tot_gals = 40000;
    int nx = SUBMATRIX_SIZE;
    int ncalc = nx * nx;
    int gpu_mem_needed = int(tot_gals * sizeof(float)) * 5; // need to allocate gamma1, gamma2, ra, dec and output.
    printf("Requirements: %d calculations and %d bytes memory on the GPU \n\n", ncalc, gpu_mem_needed);

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        printf( "cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id) );
    }
    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
        printf("There is no device supporting CUDA\n");
    else
        printf("Found %d CUDA Capable device(s)\n", deviceCount);


    int dev, driverVersion = 0, runtimeVersion = 0;
    for (dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
                (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);


        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
                deviceProp.maxThreadsDim[0],
                deviceProp.maxThreadsDim[1],
                deviceProp.maxThreadsDim[2]);
        printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
                deviceProp.maxGridSize[0],
                deviceProp.maxGridSize[1],
                deviceProp.maxGridSize[2]);

        // does this device have enough capcacity for the calculation?
        printf("\n*************\n");

        // check memory
        if((unsigned long long) deviceProp.totalGlobalMem < gpu_mem_needed) printf(" FAILURE: Not eneough memeory on device for this calculation! \n");
        else
        {
            printf("Hurrah! This device has enough memory to perform this calculation\n");

            // check # threads

            int threadsPerBlock = deviceProp.maxThreadsPerBlock; // maximal efficiency exists if we use max # threads per block.
            int blocksPerGrid = int(ceil(ncalc / threadsPerBlock)); // need nx*nx threads total
            if(deviceProp.maxThreadsDim[0] >blocksPerGrid) printf("FAILURE: Not enough threads on the device to do this calculation!\n");
            else
            {
                printf("Hurrah! This device supports enough threads to do this calculation\n");
                // how many kernels can we run at once on this machine?
                int n_mem = floor(deviceProp.totalGlobalMem / float(gpu_mem_needed));
                int n_threads = floor(threadsPerBlock * deviceProp.maxThreadsDim[0]*deviceProp.maxThreadsDim[1] / float(ncalc) ); // max # threads possible?

                printf("%d %d  \n",  n_threads, deviceProp.maxThreadsDim[0]);

                int max_kernels = 0;
                n_mem<n_threads ? max_kernels = n_mem : max_kernels = n_threads;

                printf(" you can run %d kernels at a time on this device without overloading the resources \n", max_kernels);
            }
        }

    }

    printf("\n------ End CUDA device diagnostics ------\n\n");
    ////////////////////////////////////////////////////////////////////////////

    float *d_alpha0, *d_delta0;
    float *h_alpha0, *h_delta0;

    float *d_alpha1, *d_delta1;
    float *h_alpha1, *h_delta1;

    //float *d_bin_edges;
    float *h_bin_edges;

    int NUM_GALAXIES;

    //////////////////////////////////////////////////////////////////////
    // Read in the file that defines the bin edges.
    ////////////////////////////////////////////////////////////////////////////

    /*
    int default_nbins = 27;
    float default_bin_edges[DEFAULT_NBINS] = {0.0000,0.001000,0.001585,0.002512,0.003981,0.006310,0.010000,0.015849,0.025119,0.039811,0.063096,0.100000,0.158489,0.251189,0.398107,0.630957,1.000000,1.584893,2.511886,3.981072,6.309573,10.000000,15.848932,25.118864,39.810717,63.095734,100.000000};

    int nbins=0;
    int size_bin_edges_array=0;
    float temp_bin_edges[4096];
    //FILE *binning_file = NULL;
    if (binning_filename != NULL)
    {
        printf("Binning filename: %s\n",binning_filename);
        binning_file = fopen(binning_filename,"r");

        while(fscanf(binning_file, "%f", &temp_bin_edges[nbins])!=EOF)
        {
            nbins++;
        }

        // Copy over the temp bin edges into the host array.
        // Note we are going to have one array of the bin edges for
        // *each thread*.
        size_bin_edges_array = nbins * sizeof(float)*SUBMATRIX_SIZE;
        h_bin_edges = (float*)malloc(size_bin_edges_array);
        printf("Size of host bin edges array: %d bytes\n",size_bin_edges_array);
        for (int i=0;i<SUBMATRIX_SIZE;i++)
        {
            int thread_index = i*nbins;
            for (int j=0;j<nbins;j++)
            {
                h_bin_edges[j+thread_index] = temp_bin_edges[j];
                //printf("h_bin_edges: %3d %f\n",j+thread_index,h_bin_edges[j+thread_index]);
            }
        }
    }
    else
    {
        // No file containing bin edges was passed in on the 
        // command line, so use the defaults.
        nbins = default_nbins;
        //size = nbins * sizeof(float);    
        //h_bin_edges = (float*)malloc(size);
        size_bin_edges_array = nbins * sizeof(float)*SUBMATRIX_SIZE;
        h_bin_edges = (float*)malloc(size_bin_edges_array);
        printf("Size of host bin edges array: %d bytes\n",size_bin_edges_array);
        for (int i=0;i<SUBMATRIX_SIZE;i++)
        {
            int thread_index = i*nbins;
            for (int j=0;j<nbins;j++)
            {
                h_bin_edges[j+thread_index] = default_bin_edges[j];
                //printf("h_bin_edges: %3d %f\n",j+thread_index,h_bin_edges[j+thread_index]);
            }
        }
        //for (int i=0;i<nbins;i++)
        //{
        //h_bin_edges[i] = default_bin_edges[i];
        ////printf("h_bin_edges: %3d %f\n",i,h_bin_edges[i]);
        //}
    }
    float hist_min = h_bin_edges[0];
    float hist_max = h_bin_edges[nbins-1];
    */

    ////////////////////////////////////////////////////////////////////////////
    // Finished defining the bin edges.
    ////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////
    // Read in the galaxy files.
    ////////////////////////////////////////////////////////////////////////////
    // Read in the first file
    ////////////////////////////////////////////////////////////////////////////

    //fscanf(infile0, "%s %s %s", &axis_titles, &dummy, &axis_titles);
    fscanf(infile0, "%d", &NUM_GALAXIES);

    int size_of_galaxy_array = NUM_GALAXIES * sizeof(float);    
    printf("SIZE 0 # GALAXIES: %d\n",NUM_GALAXIES);

    h_alpha0 = (float*)malloc(size_of_galaxy_array);
    h_delta0 = (float*)malloc(size_of_galaxy_array);

    for(int i=0; i<NUM_GALAXIES; i++)
    {
        fscanf(infile0, "%f %f", &h_alpha0[i], &h_delta0[i]);
        //printf("%e %e\n", h_alpha0[i], h_delta0[i]);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Read in the second file
    ////////////////////////////////////////////////////////////////////////////

    //fscanf(infile1, "%s %s %s", &axis_titles, &dummy, &axis_titles);
    fscanf(infile1, "%d", &NUM_GALAXIES);
    printf("SIZE 1 # GALAXIES: %d\n",NUM_GALAXIES);

    h_alpha1 = (float*)malloc(size_of_galaxy_array);
    h_delta1 = (float*)malloc(size_of_galaxy_array);

    for(int i=0; i<NUM_GALAXIES; i++)
    {
        fscanf(infile1, "%f %f", &h_alpha1[i], &h_delta1[i]);
        //printf("%e %e\n", h_alpha1[i], h_delta1[i]);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Allocation of histogram
    ///////////////////////////////////////////////////////////////////////////

    int *hist, *dev_hist;
    //float *dev_bin_edges;

    //cudaMalloc((void **) &dev_bin_edges,size_bin_edges_array);
    //printf("Size of dev_bin_edges: %d bytes\n",size_bin_edges_array);
    //cudaMemset(dev_bin_edges, 0, nbins*SUBMATRIX_SIZE);
    //cudaMemcpy(dev_bin_edges, h_bin_edges,size_bin_edges_array, cudaMemcpyHostToDevice );

    int size_hist = SUBMATRIX_SIZE * (nbins+2);
    int size_hist_bytes = size_hist*sizeof(int);

    hist = (int*)malloc(size_hist_bytes);
    memset(hist, 0, size_hist_bytes);

    printf("Size of histogram: %d bytes\n",size_hist_bytes);
    cudaMalloc((void **) &dev_hist, (size_hist_bytes));
    cudaMemset(dev_hist, 0, size_hist_bytes);

    unsigned long  *hist_array;

    int hist_array_size = (nbins+2) * sizeof(unsigned long);
    hist_array =  (unsigned long*)malloc(hist_array_size);
    printf("Size of histogram array: %d bytes\n",hist_array_size);
    memset(hist_array,0,hist_array_size); 

    ////////////////////////////////////////////////////////////////////////////
    // Define the grid and block size
    ////////////////////////////////////////////////////////////////////////////
    dim3 grid, block;
    grid.x =100; // Is this the number of blocks?
    block.x = SUBMATRIX_SIZE/grid.x; // Is this the number of threads per block? NUM_GALAXIES/block.x;
    // SUBMATRIX is the number of threads per warp? Per kernel call?
    ////////////////////////////////////////////////////////////////////////////

    cudaMalloc((void **) &d_alpha0, size_of_galaxy_array );
    cudaMalloc((void **) &d_delta0, size_of_galaxy_array );

    cudaMalloc((void **) &d_alpha1, size_of_galaxy_array );
    cudaMalloc((void **) &d_delta1, size_of_galaxy_array );

    // Check to see if we allocated enough memory.
    if (0==d_alpha0 || 0==d_delta0 || 0==d_alpha1 || 0==d_delta1 || 0==dev_hist)
    {
        printf("couldn't allocate memory\n");
        return 1;
    }

    // Initialize array to all 0's
    cudaMemset(d_alpha0,0,size_of_galaxy_array);
    cudaMemset(d_delta0,0,size_of_galaxy_array);
    cudaMemset(d_alpha1,0,size_of_galaxy_array);
    cudaMemset(d_delta1,0,size_of_galaxy_array);

    cudaMemcpy(d_alpha0, h_alpha0, size_of_galaxy_array, cudaMemcpyHostToDevice );
    cudaMemcpy(d_delta0, h_delta0, size_of_galaxy_array, cudaMemcpyHostToDevice );
    cudaMemcpy(d_alpha1, h_alpha1, size_of_galaxy_array, cudaMemcpyHostToDevice );
    cudaMemcpy(d_delta1, h_delta1, size_of_galaxy_array, cudaMemcpyHostToDevice );

    int x, y;
    int num_submatrices = NUM_GALAXIES / SUBMATRIX_SIZE;

    printf("Breaking down the calculations.\n");
    printf("Number of submatrices: %dx%d\n",num_submatrices,num_submatrices);
    printf("Number of calculations per submatrices: %dx%d\n",SUBMATRIX_SIZE,SUBMATRIX_SIZE);


    int bin_index = 0;
    for(int k = 0; k < num_submatrices; k++)
    {
        y = k*SUBMATRIX_SIZE;
        //printf("%d %d\n",k,y);
        for(int j = 0; j < num_submatrices; j++)
        {
            x = j *SUBMATRIX_SIZE; 

            //printf("----\n");
            //printf("%d %d\t\t%d %d\n",k,y,j,x);
            //printf("----\n");

            // Set the histogram to all zeros each time.
            cudaMemset(dev_hist,0,size_hist_bytes);

//__global__ void distance(float *a0, float *d0, float *a1, float *d1, int xind, int yind, int *dev_hist, float hist_min, float hist_max, int nbins, float bin_width, bool log_binning=0, bool two_different_files=1)
            distance<<<grid,block>>>(d_alpha0, d_delta0,d_alpha1, d_delta1, x, y, dev_hist, hist_lower_range, hist_upper_range, nbins, hist_bin_width, log_binning_flag, two_different_files);
            cudaMemcpy(hist, dev_hist, size_hist_bytes, cudaMemcpyDeviceToHost);

            ////////////////////////////////////////////////////////////////////
            // Sum up the histograms from each thread (hist).
            ////////////////////////////////////////////////////////////////////
            for(int m=0; m<size_hist; m++)
            {
                bin_index = m%(nbins+2);
                hist_array[bin_index] += hist[m];
            }    
        }  
    }

    unsigned long total = 0;
    //float  bin_width = (hist_upper_range - hist_lower_range) / nbins;
    float bins_mid = 0;

    fprintf(outfile, "%s %s\n", "Angular Distance(radians)","Number of Entries");      
    float lo = hist_lower_range;
    float hi = 0;
    //printf("hist_lower_range: %f\n",hist_lower_range);
    //printf("hist_upper_range: %f\n",hist_upper_range);
    //printf("hist_bin_width: %f\n",hist_bin_width);
    for(int k=0; k<nbins+1; k++)
    {
        //bins_mid = bin_width*(k - 0.5);

        //float lo = h_bin_edges[k];
        //float hi = h_bin_edges[k+1];
        if (!log_binning_flag)
        {
            hi = lo + hist_bin_width;
        }
        else
        {
            //printf("lo: %f\t\tlog(lo): %f\n",lo,log(lo));
            hi = exp(log(lo) + hist_bin_width);
        }

        bins_mid = (hi+lo)/2.0;

        fprintf(outfile, "%.3e %s %lu \n", bins_mid, ",",  hist_array[k]);
        total += hist_array[k];

        lo = hi;

    }
    printf("total: %lu \n", total);

    fclose(infile0);
    fclose(infile1);
    fclose(outfile);

    free(h_alpha0);
    free(h_delta0);
    free(h_alpha1);
    free(h_delta1);
    free(hist);

    cudaFree(d_alpha0);
    cudaFree(d_delta0);  
    cudaFree(d_alpha1);
    cudaFree(d_delta1);  
    cudaFree(dev_hist);
    //cudaFree(dev_bin_edges);

    return 0;
}  
//////////////////////////////////////////////////////////////////////
