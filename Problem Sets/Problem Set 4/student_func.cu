//Udacity HW 4
//Radix Sorting

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
#include "utils.h"
#include <thrust/host_vector.h>
#include <thrust/copy.h>

__global__ void histogram(unsigned int* const d_inputVals,
                          unsigned int* d_hist,
                          unsigned int  mask,
                          unsigned int  i)
{
  int myId = threadIdx.x + blockIdx.x * blockDim.x;

  //perform histogram of data & mask into bins
  unsigned int bin = (d_inputVals[myId] & mask) >> i;
  atomicAdd(&(d_hist[bin]), 1);
}

__global__ void exclusiveScan(unsigned int * d_hist,
                              unsigned int * b_scan,
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
  b_scan[tid] = tmp[tid];
}

__global__ void scan(unsigned int *binHistogram,
                     unsigned int *binScan,
                     const int numBins)
{
  const unsigned int myId = threadIdx.x;
  int tmp=0;
  if(myId > 0 && myId < numBins)
  {
    tmp = binScan[myId-1] + binHistogram[myId-1];
    __syncthreads();
    binScan[myId] = tmp;
  }
}

__global__ void gather(unsigned int* const vals_src,
                       unsigned int* const pos_src,
                       unsigned int* const vals_dst,
                       unsigned int* const pos_dst,
                       unsigned int* b_scan,
                       unsigned int  mask,
                       unsigned int  i)
{
  int myId = threadIdx.x + blockIdx.x * blockDim.x;

  //perform histogram of data & mask into bins
  unsigned int bin = (vals_src[myId] & mask) >> i;

  vals_dst[b_scan[bin]] = vals_src[myId];
  pos_dst[b_scan[bin]]  = pos_src[myId];
  atomicAdd(&(b_scan[bin]), 1);
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
  const int maxThreadsPerBlock = 1024;
  int threads = maxThreadsPerBlock;
  int blocks  = numElems / maxThreadsPerBlock;

  const int numBits = 1;
  const int numBins = 1 << numBits;

  unsigned int* vals_src = d_inputVals;
  unsigned int* pos_src  = d_inputPos;

  unsigned int* vals_dst = d_outputVals;
  unsigned int* pos_dst  = d_outputPos;

  unsigned int* d_hist, * b_scan;

  // create the structure for the histogram and exclusive prefix scan :
  checkCudaErrors(cudaMalloc((void**) &d_hist, sizeof(unsigned int)*numBins));
  checkCudaErrors(cudaMalloc((void**) &b_scan, sizeof(unsigned int)*numBins));

  // loop through a
  for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += numBits)
  {
    unsigned int mask = (numBins - 1) << i;

    // zero out the bins :
    checkCudaErrors(cudaMemset(d_hist, 0, sizeof(unsigned int) * numBins));
    checkCudaErrors(cudaMemset(b_scan, 0, sizeof(unsigned int) * numBins));

    // create histogram :
    histogram<<<blocks, threads>>>(vals_src, d_hist, mask, i);

    // exclusive prefix sum scan on bins : 
    scan<<<1, threads, sizeof(unsigned int) * threads>>>
        (d_hist, b_scan, numBins);

    gather<<<blocks, threads>>>
        (vals_src, pos_src, vals_dst, pos_dst, b_scan, mask, i);
    
    std::swap(vals_dst, vals_src);
    std::swap(pos_dst, pos_src);  
  }

  // we did an even number of iterations, need to copy from input buffer into 
  // output
  checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, 
                  numElems * sizeof(unsigned int),
                  cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos,
                  numElems * sizeof(unsigned int),
                  cudaMemcpyDeviceToDevice));
}



