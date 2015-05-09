/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <stdio.h>

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

__global__ void shmem_reduce_kernel(float * d_out, const float * d_in,
                                    int n, int op)
{
  // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
  extern __shared__ float sdata[];

  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid  = threadIdx.x;

  if (myId >= n)
    return;

  // load shared mem from global mem
  sdata[tid] = d_in[myId];
  __syncthreads();            // make sure entire block is loaded!

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (tid < s)
    {
      if (op == 0)
        sdata[tid] = min(sdata[tid], sdata[tid + s]);
      else if (op == 1)
        sdata[tid] = max(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();        // make sure all adds at one stage are done!
  }

  // only thread 0 writes result for this block back to global mem
  if (tid == 0)
  {
      d_out[blockIdx.x] = sdata[0];
  }
}

__global__ void histogram(const float * const d_logLuminance, 
                          unsigned int *d_hist,
                          float lumMin, 
                          const float range,
                          const int numBins)
{
  int myId     = threadIdx.x + blockIdx.x * blockDim.x;
  float lum    = d_logLuminance[myId];
  int myBin    = (lum - lumMin) / range * numBins;
  atomicAdd(&(d_hist[myBin]), 1);
}

__global__ void exclusiveScan(unsigned int * d_hist, 
                              unsigned int * const d_cdf, 
                              const int numBins)
{
  extern __shared__ unsigned int tmp[];
  int tid = threadIdx.x;
  
  tmp[tid] = (tid > 0) ? d_hist[tid - 1] : 0;
  __syncthreads();
  
  for(int s = 1; s < numBins; s *= 2)
  { 
    unsigned int t = tmp[tid];
    __syncthreads();
    
    if(tid + s < numBins)
    { 
      tmp[tid + s] += t;
    } 
    __syncthreads();
  }
  d_cdf[tid] = tmp[tid];
}


void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  /*1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum */
  int n = numRows * numCols;

  // declare GPU memory pointers
  float * d_intermediate;
  float * d_temp;

  // allocate GPU memory
  checkCudaErrors(cudaMalloc((void **) &d_intermediate, n*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **) &d_temp, sizeof(float)));

  const int maxThreadsPerBlock = 1024;
  int threads = maxThreadsPerBlock;
  int blocks  = n / maxThreadsPerBlock;
  int shared  = threads * sizeof(float);

  shmem_reduce_kernel<<<blocks, threads, shared>>>
      (d_intermediate, d_logLuminance, n, 0);

  shmem_reduce_kernel<<<1, blocks, shared>>>
      (d_temp, d_intermediate, n, 0);

  checkCudaErrors(cudaMemcpy(&min_logLum, d_temp, sizeof(float),
                  cudaMemcpyDeviceToHost));

  shmem_reduce_kernel<<<blocks, threads, shared>>>
      (d_intermediate, d_logLuminance, n, 1);

  shmem_reduce_kernel<<<1, blocks, shared>>>
      (d_temp, d_intermediate, n, 1);

  checkCudaErrors(cudaMemcpy(&max_logLum, d_temp, sizeof(float),
                  cudaMemcpyDeviceToHost));

  /*2) subtract them to find the range */
  float range = max_logLum - min_logLum;

  /*3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins */
  unsigned int * d_hist;

  checkCudaErrors(cudaMalloc((void**) &d_hist, sizeof(unsigned int) * numBins));
  checkCudaErrors(cudaMemset(d_hist, 0, sizeof(int)*numBins));

  histogram<<<blocks, threads>>>
      (d_logLuminance, d_hist, min_logLum, range, numBins);

  /*4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
  exclusiveScan<<<1, threads, sizeof(unsigned int) * threads>>>
      (d_hist, d_cdf, numBins);
}


