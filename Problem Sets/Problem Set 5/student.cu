/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.


   Evan Cummings
   Homework 6

   The implementation here is derived from the cuda-programming blogspot (link
   below).  It splits the data into blocks as a function of the number of 
   processors and computes each block's histogram counts into a shared memory
   histogram using atomic adds for each thread within a block, then each block 
   does a single atomic add to the output array.

   This method produces code that is readable, with a good amount of speedup
   obtained by reducing the number of atomic adds. the code is about 7x faster
   than the naive parallel approach.
   
   http://cuda-programming.blogspot.com/2013/03/computing-histogram-on-cuda-cuda-code_8.html
*/

#include "utils.h"
#include "reference.cpp"

__global__
void histogram_optimized(const unsigned int* const vals,
                         unsigned int* const histo,
                         const unsigned int numVals)
{
  extern __shared__ unsigned int temp[];
  temp[threadIdx.x] = 0;
  __syncthreads();
  
  int i      = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;  // num threads in block
  while (i < numVals)
  {
    atomicAdd( &(temp[vals[i]]), 1 );
    i += stride;
  }
  __syncthreads();
  atomicAdd( &(histo[threadIdx.x]), temp[threadIdx.x] );
}

__global__
void histogram(const unsigned int* const vals,
               unsigned int* const histo,
               int numVals)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  atomicAdd(&(histo[vals[idx]]), 1);
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  cudaDeviceProp prop;
  checkCudaErrors( cudaGetDeviceProperties( &prop, 0 ) );
  int blocks = prop.multiProcessorCount;
  int shared = numBins * sizeof(unsigned int);
  
  histogram_optimized <<<blocks*8, numBins, shared>>>
      (d_vals, d_histo, numElems);
}



