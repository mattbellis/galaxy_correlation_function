#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include <unistd.h>

#include <cuda_runtime.h>
//#include <cutil_inline.h>

using namespace std;

#define SUBMATRIX_SIZE 10000
//#define NUM_BIN 5000
//#define HIST_MIN 0.0
//#define HIST_MAX 3.5 
#define NUM_BIN 27 // for log binning
//#define NUM_BIN 37 // for log binning
#define HIST_MIN 0.0 // for degrees
#define HIST_MAX 100.0 // for degrees

#define CONV_FACTOR 57.2957795 // 180/pi

//float bin_edges[30] = {0.001000,0.001585,0.002512,0.003981,0.006310,0.010000,0.010000,0.015849,0.025119,0.039811,0.063096,0.100000,0.100000,0.158489,0.251189,0.398107,0.630957,1.000000,1.000000,1.584893,2.511886,3.981072,6.309573,10.000000,10.000000,15.848932,25.118864,39.810717,63.095734,100.000000};

////////////////////////////////////////////////////////////////////////
__global__ void distance(float *a0, float *d0, float *a1, float *d1, int xind, int yind, int *dev_hist, float* dev_bin_edges, bool two_different_files=1)
{

    //float bin_edges[30] = {0.001000,0.001585,0.002512,0.003981,0.006310,0.010000,0.010000,0.015849,0.025119,0.039811,0.063096,0.100000,0.100000,0.158489,0.251189,0.398107,0.630957,1.000000,1.000000,1.584893,2.511886,3.981072,6.309573,10.000000,10.000000,15.848932,25.118864,39.810717,63.095734,100.000000};

    // For 27
    float bin_edges[NUM_BIN] = {0.0000,0.001000,0.001585,0.002512,0.003981,0.006310,0.010000,0.015849,0.025119,0.039811,0.063096,0.100000,0.158489,0.251189,0.398107,0.630957,1.000000,1.584893,2.511886,3.981072,6.309573,10.000000,15.848932,25.118864,39.810717,63.095734,100.000000};

    // For 37
    //float bin_edges[NUM_BIN] = {0.0000,0.001000,0.001389,0.001931,0.002683,0.003728,0.005179,0.007197,0.010000,0.013895,0.019307,0.026827,0.037276,0.051795,0.071969,0.100000,0.138950,0.193070,0.268270,0.372759,0.517947,0.719686,1.000000,1.389495,1.930698,2.682696,3.727594,5.179475,7.196857,10.000000,13.894955,19.306977,26.826958,37.275937,51.794747,71.968567,100.000000};


    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_idx = idx;
    idx += xind;

    float alpha = a0[idx], delta0 = d0[idx];
    float cos_d0 = cos(delta0), sin_d0 = sin(delta0), dist;

    int ymax = yind + SUBMATRIX_SIZE;
    int bin_index = 0; 
    int offset = 0;

    float a_diff, sin_a_diff, cos_a_diff;
    float cos_d1, sin_d1, numer, denom, mult1, mult2;    

    bool do_calc = 1;
    for(int i=yind; i<ymax; i++)
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

            if(dist < HIST_MIN)
                bin_index = 0; 
            else if(dist >= HIST_MAX)
                bin_index = NUM_BIN + 1;
            else
            {
                //bin_index = int(((dist - HIST_MIN) * NUM_BIN / HIST_MAX) +1);    
                bin_index = 0;
                for (int j=0;j<NUM_BIN-1;j++)
                {
                    //bin_index = 5;
                    //if (dist>=0.1*j && dist<0.1*(j+1))
                    //if (dist>=dev_bin_edges[j] && dist<dev_bin_edges[j+1])
                    if (dist>=bin_edges[j] && dist<bin_edges[j+1])
                    {
                        bin_index = j+1;
                        break;
                    }
                }
            }

            offset = ((NUM_BIN+2)*thread_idx);
            bin_index += offset;

            dev_hist[bin_index]++;

        }
    }
}

////////////////////////////////////////////////////////////////////////

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

    int size;

    while ((c = getopt(argc, argv, "ab:f:o:")) != -1) {
        switch(c) {
            case 'a':
                printf("a is set\n");
                break;
            case 'b':
                binning_filename = optarg;
                printf("Using binning information from file: %s\n",binning_filename);
                break;
            case 'f':
                filename = optarg;
                printf("filename is %s\n", filename);
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
    ////////////////////////////////////////////////////////////////////////////
    // Now get the info from the device.
    ////////////////////////////////////////////////////////////////////////////
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

    }


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

    int default_nbins = 27;
    float default_bin_edges[NUM_BIN] = {0.0000,0.001000,0.001585,0.002512,0.003981,0.006310,0.010000,0.015849,0.025119,0.039811,0.063096,0.100000,0.158489,0.251189,0.398107,0.630957,1.000000,1.584893,2.511886,3.981072,6.309573,10.000000,15.848932,25.118864,39.810717,63.095734,100.000000};

    int nbins=0;
    float temp_bin_edges[4096];
    //FILE *binning_file = NULL;
    if (binning_filename != NULL)
    {
        printf("BBBBBBBBBBBB: %s\n",binning_filename);
        binning_file = fopen(binning_filename,"r");

        while(fscanf(binning_file, "%f", &temp_bin_edges[nbins])!=EOF)
        {
            nbins++;
        }

        // Copy over the temp bin edges into the host array.
        size = nbins * sizeof(float);    
        h_bin_edges = (float*)malloc(size);
        for (int i=0;i<nbins;i++)
        {
            h_bin_edges[i] = temp_bin_edges[i];
            printf("h_bin_edges: %3d %f\n",i,h_bin_edges[i]);
        }
    }
    else
    {
        // No file containing bin edges was passed in on the 
        // command line, so use the defaults.
        nbins = default_nbins;
        size = nbins * sizeof(float);    
        h_bin_edges = (float*)malloc(size);
        for (int i=0;i<nbins;i++)
        {
            h_bin_edges[i] = default_bin_edges[i];
            printf("h_bin_edges: %3d %f\n",i,h_bin_edges[i]);
        }
    }

    /*
       int size = nbins * sizeof(float);    
       printf("SIZE0 # GALAXIES: %d\n",nbins+1);

       h_bin_edges = (float*)malloc(size);

       for(int i=0; i<nbins; i++)
       {
       fscanf(infile0, "%f", &h_bin_edges[i]);
       }
     */

    //////////////////////////////////////////////////////////////////////
    // Read in the cluster_data file
    ////////////////////////////////////////////////////////////////////////////

    char axis_titles[256];
    char dummy[256];

    ////////////////////////////////////////////////////////////////////////////
    // Read in the first file
    ////////////////////////////////////////////////////////////////////////////

    fscanf(infile0, "%s %s %s", &axis_titles, &dummy, &axis_titles);
    fscanf(infile0, "%d", &NUM_GALAXIES);

    size = NUM_GALAXIES * sizeof(float);    
    printf("SIZE 0 # GALAXIES: %d\n",NUM_GALAXIES);

    h_alpha0 = (float*)malloc(size);
    h_delta0 = (float*)malloc(size);

    for(int i=0; i<NUM_GALAXIES; i++)
    {
        fscanf(infile0, "%f %s %f %s ", &h_alpha0[i], &dummy, &h_delta0[i], &dummy);
        //fscanf(infile, "%f%s %f ", &h_alpha[i], &dummy, &h_delta[i]);
        //printf("%e %s %e\n", h_alpha0[i], dummy, h_delta0[i]);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Read in the second file
    ////////////////////////////////////////////////////////////////////////////

    fscanf(infile1, "%s %s %s", &axis_titles, &dummy, &axis_titles);
    fscanf(infile1, "%d", &NUM_GALAXIES);
    printf("SIZE 1 # GALAXIES: %d\n",NUM_GALAXIES);

    h_alpha1 = (float*)malloc(size);
    h_delta1 = (float*)malloc(size);

    for(int i=0; i<NUM_GALAXIES; i++)
    {
        fscanf(infile1, "%f %s %f %s ", &h_alpha1[i], &dummy, &h_delta1[i], &dummy);
        //fscanf(infile, "%f%s %f ", &h_alpha[i], &dummy, &h_delta[i]);
        //printf("%e %s %e\n", h_alpha1[i], dummy, h_delta1[i]);
    }

    ////////////////////////////////////////////////////////////////////////////
    //allocation of histogram
    ///////////////////////////////////////////////////////////////////////////

    int *hist, *dev_hist;
    // Original
    //int size_hist = SUBMATRIX_SIZE * (NUM_BIN+2);
    //int size_hist_bytes = size_hist*sizeof(int);

    // Log binning
    //float h_bin_edges[30] = {0.001000,0.001585,0.002512,0.003981,0.006310,0.010000,0.010000,0.015849,0.025119,0.039811,0.063096,0.100000,0.100000,0.158489,0.251189,0.398107,0.630957,1.000000,1.000000,1.584893,2.511886,3.981072,6.309573,10.000000,10.000000,15.848932,25.118864,39.810717,63.095734,100.000000};

    ////////////////////////////////////////////////////////////////////////////
    // For 27 bins
    ////////////////////////////////////////////////////////////////////////////
    //float h_bin_edges[NUM_BIN] = {0.0000,0.001000,0.001585,0.002512,0.003981,0.006310,0.010000,0.015849,0.025119,0.039811,0.063096,0.100000,0.158489,0.251189,0.398107,0.630957,1.000000,1.584893,2.511886,3.981072,6.309573,10.000000,15.848932,25.118864,39.810717,63.095734,100.000000};

    // For 37 bins
    //float h_bin_edges[NUM_BIN] = {0.0000,0.001000,0.001389,0.001931,0.002683,0.003728,0.005179,0.007197,0.010000,0.013895,0.019307,0.026827,0.037276,0.051795,0.071969,0.100000,0.138950,0.193070,0.268270,0.372759,0.517947,0.719686,1.000000,1.389495,1.930698,2.682696,3.727594,5.179475,7.196857,10.000000,13.894955,19.306977,26.826958,37.275937,51.794747,71.968567,100.000000};

    /*
       for (int i=0;i<NUM_BIN;i++)
       {
       printf("%d %f\n",i,h_bin_edges[i]);
       }
       printf("\n");
     */
    float *dev_bin_edges;
    cudaMalloc((void **) &dev_bin_edges, (NUM_BIN*sizeof(float)));
    cudaMemset(dev_bin_edges, 0, NUM_BIN);
    cudaMemcpy(dev_bin_edges, h_bin_edges, NUM_BIN, cudaMemcpyHostToDevice );
    //int nbins = 30;

    int size_hist = SUBMATRIX_SIZE * (NUM_BIN+2);
    int size_hist_bytes = size_hist*sizeof(int);

    hist = (int*)malloc(size_hist_bytes);
    memset(hist, 0, size_hist_bytes);

    printf("size_hist: %d\n",size_hist_bytes);
    cudaMalloc((void **) &dev_hist, (size_hist_bytes));
    cudaMemset(dev_hist, 0, size_hist_bytes);

    unsigned long  *hist_array;

    hist_array =  (unsigned long*)malloc((NUM_BIN+2) * sizeof(unsigned long));
    memset(hist_array, 0, (NUM_BIN+2)*sizeof(unsigned long)); 

    ////////////////////////////////////////////////////////////////////////////
    // Define the grid and block size
    ////////////////////////////////////////////////////////////////////////////
    dim3 grid, block;
    grid.x =100;
    block.x = SUBMATRIX_SIZE/grid.x; //NUM_GALAXIES/block.x;
    ////////////////////////////////////////////////////////////////////////////

    cudaMalloc((void **) &d_alpha0, size );
    cudaMalloc((void **) &d_delta0, size );

    cudaMalloc((void **) &d_alpha1, size );
    cudaMalloc((void **) &d_delta1, size );

    // Check to see if we allocated enough memory.
    if (0==d_alpha0 || 0==d_delta0 || 0==d_alpha1 || 0==d_delta1 || 0==dev_hist)
    {
        printf("couldn't allocate memory\n");
        return 1;
    }


    // Initialize array to all 0's
    cudaMemset(d_alpha0,0,size);
    cudaMemset(d_delta0,0,size);
    cudaMemset(d_alpha1,0,size);
    cudaMemset(d_delta1,0,size);

    cudaMemcpy(d_alpha0, h_alpha0, size, cudaMemcpyHostToDevice );
    cudaMemcpy(d_delta0, h_delta0, size, cudaMemcpyHostToDevice );
    cudaMemcpy(d_alpha1, h_alpha1, size, cudaMemcpyHostToDevice );
    cudaMemcpy(d_delta1, h_delta1, size, cudaMemcpyHostToDevice );

    int x, y;
    int num_submatrices = NUM_GALAXIES / SUBMATRIX_SIZE;


    int bin_index = 0;
    for(int k = 0; k < num_submatrices; k++)
    {
        y = k*SUBMATRIX_SIZE;
        //        printf("%d %d\n",k,y);
        for(int j = 0; j < num_submatrices; j++)
        {
            x = j *SUBMATRIX_SIZE; 

            //printf("----\n");
            //printf("%d %d\t\t%d %d\n",k,y,j,x);
            //printf("----\n");

            cudaMemset(dev_hist,0,size_hist_bytes);

            distance<<<grid,block>>>(d_alpha0, d_delta0,d_alpha1, d_delta1, x, y, dev_hist, dev_bin_edges, two_different_files);
            cudaMemcpy(hist, dev_hist, size_hist_bytes, cudaMemcpyDeviceToHost);


            for(int m=0; m<size_hist; m++)
            {

                bin_index = m%(NUM_BIN+2);
                //if(bin_index == 0)
                //printf("\n");

                //printf("%3i:%3i ", m, hist[m]);
                //printf("%3i ", hist[m]);

                hist_array[bin_index] += hist[m];
            }    
            //printf("\n");
        }  
    }

    unsigned long total = 0;
    float  bin_width = (HIST_MAX - HIST_MIN) / NUM_BIN;
    float bins_mid = 0;

    fprintf(outfile, "%s %s\n", "Angular Distance(radians)","Number of Entries");      
    for(int k=0; k<NUM_BIN+1; k++)
    {
        //bins_mid = bin_width*(k - 0.5);

        float lo = h_bin_edges[k];
        float hi = h_bin_edges[k+1];

        bins_mid = (hi+lo)/2.0;

        fprintf(outfile, "%.3e %s %lu \n", bins_mid, ",",  hist_array[k]);
        total += hist_array[k];

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
    cudaFree(dev_bin_edges);

    return 0;
}  
//////////////////////////////////////////////////////////////////////
