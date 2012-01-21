#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>

#include <cuda_runtime.h>
//#include <cutil_inline.h>

using namespace std;

#define SUBMATRIX_SIZE 10000
#define NUM_BIN 500
#define HIST_MIN 0.0
#define HIST_MAX 3.5 

////////////////////////////////////////////////////////////////////////
__global__ void distance(float *a, float *d, int xind, int yind, int *dev_hist)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_idx = idx;
    idx += xind;

    float alpha = a[idx], delta = d[idx];
    float cos_d1 = cos(delta), sin_d1 = sin(delta), dist;

    int ymax = yind + SUBMATRIX_SIZE;
    int bin_index; 
    int offset = 0;

    float a_diff, sin_a_diff, cos_a_diff;
    float cos_d2, sin_d2, numer, denom, mult1, mult2;    

    for(int i=yind; i<ymax; i++)
    {
        if(idx > i)
        {
            a_diff = a[i] - alpha;
            
            sin_a_diff = sin(a_diff);
            cos_a_diff = cos(a_diff);
  
            sin_d2 = sin(d[i]);
            cos_d2 = cos(d[i]);
 
            mult1 = cos_d2 * cos_d2 * sin_a_diff * sin_a_diff;
            mult2 = cos_d1 * sin_d2 - sin_d1 * cos_d2 * cos_a_diff;
            mult2 = mult2 * mult2;
           
            numer = sqrt(mult1 + mult2); 
       
            denom = sin_d1 *sin_d2 + cos_d1 * cos_d2 * cos_a_diff;
            
            //dist = atan(num);  
            dist = atan2(numer,denom);  
            if(dist < HIST_MIN)
                bin_index = 0; 
            else if(dist >= HIST_MAX)
                bin_index = NUM_BIN + 1;
            else
                bin_index = int(((dist - HIST_MIN) * NUM_BIN / HIST_MAX) +1);    

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

    float *d_alpha, *d_delta;
    float *h_alpha, *h_delta;

    int NUM_PARTICLES;

    if (argc < 3)
    {

        printf("\nMust pass in cluster_data file  on command line!\n");
        printf("\nUsage: ", argv[0] );
        printf(" <cluster_data file> <distances file> \n\n");
        exit(1);
    }

    FILE *infile, *outfile ;
    infile = fopen(argv[1],"r");
    outfile = fopen(argv[2], "w");

    //////////////////////////////////////////////////////////////////////
    // Read in the cluster_data file
    ////////////////////////////////////////////////////////////////////////////

    char axis_titles[256];
    char dummy[256];

    fscanf(infile, "%s %s %s", &axis_titles, &dummy, &axis_titles);
    fscanf(infile, "%d", &NUM_PARTICLES);
   
    int size = NUM_PARTICLES * sizeof(float);    
    printf("# particles: %d\n",NUM_PARTICLES);

    h_alpha = (float*)malloc(size);
    h_delta = (float*)malloc(size);


    for(int i=0; i<NUM_PARTICLES; i++)
    {
        fscanf(infile, "%f %s %f %s ", &h_alpha[i], &dummy, &h_delta[i], &dummy);
       //fscanf(infile, "%f%s %f ", &h_alpha[i], &dummy, &h_delta[i]);
       // printf("%e %s %e\n", h_alpha[i], dummy, h_delta[i]);
    }
    ////////////////////////////////////////////////////////////////////////////
    //allocation of histogram
    ///////////////////////////////////////////////////////////////////////////

    int *hist, *dev_hist;
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
    block.x = SUBMATRIX_SIZE/grid.x; //NUM_PARTICLES/block.x;
    ////////////////////////////////////////////////////////////////////////////

    cudaMalloc((void **) &d_alpha, size );
    cudaMalloc((void **) &d_delta, size );
    

    // Check to see if we allocated enough memory.
    if (0==d_alpha || 0==d_delta|| 0==dev_hist)
    {
        printf("couldn't allocate memory\n");
        return 1;
    }


    // Initialize array to all 0's
    cudaMemset(d_alpha,0,size);
    cudaMemset(d_delta,0,size);

    cudaMemcpy(d_alpha, h_alpha, size, cudaMemcpyHostToDevice );
    cudaMemcpy(d_delta, h_delta, size, cudaMemcpyHostToDevice );

    int x, y;
    int num_submatrices = NUM_PARTICLES / SUBMATRIX_SIZE;


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

                distance<<<grid,block>>>(d_alpha, d_delta, x, y, dev_hist);
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
    for(int k=0; k<NUM_BIN+2; k++)
    {
          bins_mid = bin_width*(k - 0.5);
       fprintf(outfile, "%.3e %s %lu \n", bins_mid, ",",  hist_array[k]);
       total += hist_array[k];

    }
    printf("total: %lu \n", total);
    
    fclose(infile);
    fclose(outfile);

    free(h_alpha);
    free(h_delta);
    free(hist);

    cudaFree(d_alpha);
    cudaFree(d_delta);  
    cudaFree(dev_hist);

    return 0;
}  
//////////////////////////////////////////////////////////////////////
