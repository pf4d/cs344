//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four 
      neighboring pixels:
      Sum1: If the neighbor is in the interior 
               += ImageGuess_prev[neighbor]
            else if the neighbor in on the border
               += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f // floating point
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>
#include "reference_calc.cpp"

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

__global__
void gen_mask(unsigned int * d_mask,
              unsigned char * d_red,
              unsigned char * d_green,
              unsigned char * d_blue,
              const size_t numRows,
              const size_t numCols)
{
  //calculate a 1D offset
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
  if ( thread_2D_pos.y >= numCols || thread_2D_pos.x >= numRows )
    return;
  
  int idx = thread_2D_pos.x * numCols + thread_2D_pos.y;
  
  int2 idx_u, idx_d, idx_r, idx_l;
  int  u, d, l, r;
  
  if (d_red[idx] != 255 || d_green[idx] != 255 || d_blue[idx] != 255)
  {
    idx_u = make_int2( thread_2D_pos.x,
                       blockIdx.y * blockDim.y + threadIdx.y - 1);
    idx_d = make_int2( thread_2D_pos.x,
                       blockIdx.y * blockDim.y + threadIdx.y + 1);
    idx_l = make_int2( blockIdx.x * blockDim.x + threadIdx.x + 1,
                       thread_2D_pos.y);
    idx_r = make_int2( blockIdx.x * blockDim.x + threadIdx.x - 1,
                       thread_2D_pos.y);
    u = idx_u.x * numCols + idx_u.y;
    d = idx_d.x * numCols + idx_d.y;
    l = idx_l.x * numCols + idx_l.y;
    r = idx_r.x * numCols + idx_r.y;
    if(d_red[u]   == 255 || d_red[d]   == 255 || 
       d_red[l]   == 255 || d_red[r]   == 255 ||
       d_blue[u]  == 255 || d_blue[d]  == 255 || 
       d_blue[l]  == 255 || d_blue[r]  == 255 ||
       d_green[u] == 255 || d_green[d] == 255 || 
       d_green[l] == 255 || d_green[r] == 255)
      d_mask[idx] = 2;
    else
      d_mask[idx] = 1;
  }
}

__global__
void apply_mask(unsigned int * d_mask,
                uchar4 * d_image,
                const size_t numRows,
                const size_t numCols)
{
  //calculate a 1D offset
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  if ( thread_2D_pos.y >= numCols || thread_2D_pos.x >= numRows )
    return;
  
  int idx = thread_2D_pos.x * numCols + thread_2D_pos.y;
 
  if (d_mask[idx] == 2)
  {
    d_image[idx].x = 0;
    d_image[idx].y = 0;
    d_image[idx].z = 0;
  }
 
  else if (d_mask[idx] == 1)
  {
    d_image[idx].x = 255;
    d_image[idx].y = 255;
    d_image[idx].z = 0;
  }
}

__global__
void init_guess(unsigned int * d_mask,
                float * d_guess_color,
                unsigned char * d_image_color,
                const size_t numRows,
                const size_t numCols)
{
  //calculate a 1D offset
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  if ( thread_2D_pos.y >= numCols || thread_2D_pos.x >= numRows )
    return;
  
  int idx = thread_2D_pos.x * numCols + thread_2D_pos.y;
 
  if (d_mask[idx] > 0)
  {
    d_guess_color[idx] = (float) d_image_color[idx];
  }
}

__global__
void separateChannels(uchar4* d_sourceImg,
                      int numRows,
                      int numCols,
                      unsigned char* const d_red,
                      unsigned char* const d_green,
                      unsigned char* const d_blue)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  if ( thread_2D_pos.y >= numCols || thread_2D_pos.x >= numRows )
    return;
  
  const int idx = thread_2D_pos.x * numCols + thread_2D_pos.y;
  uchar4 rgba   = d_sourceImg[idx];
  d_red[idx]    = rgba.x;
  d_green[idx]  = rgba.y;
  d_blue[idx]   = rgba.z;
}

__global__
void replace_dest(float * d_guess_red,
                  float * d_guess_green,
                  float * d_guess_blue,
                  unsigned int * d_mask,
                  uchar4* const d_destImg,
                  int numRows,
                  int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
  
  const int idx = thread_2D_pos.x * numCols + thread_2D_pos.y;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.y >= numCols || thread_2D_pos.x >= numRows)
    return;

  unsigned char red   = (unsigned char) d_guess_red[idx];
  unsigned char green = (unsigned char) d_guess_green[idx];
  unsigned char blue  = (unsigned char) d_guess_blue[idx];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  if (d_mask[idx] == 1)
  {
    d_destImg[idx] = outputPixel;
  }
}

__global__
void solve(float * d_guess,
           float * d_guess_prev,
           unsigned int * d_mask,
           unsigned char* const d_source,
           unsigned char* const d_dest,
           const size_t numRows,
           const size_t numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
  
  const int idx = thread_2D_pos.x * numCols + thread_2D_pos.y;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.y >= numCols || thread_2D_pos.x >= numRows)
    return;

  if (d_mask[idx] == 1)
  {
    float sum1 = 0.0;
    float sum2 = 0.0;
    int2 idx_u, idx_d, idx_r, idx_l;
    int  u, d, l, r;
    
    idx_u = make_int2( thread_2D_pos.x,
                       blockIdx.y * blockDim.y + threadIdx.y - 1);
    idx_d = make_int2( thread_2D_pos.x,
                       blockIdx.y * blockDim.y + threadIdx.y + 1);
    idx_l = make_int2( blockIdx.x * blockDim.x + threadIdx.x + 1,
                       thread_2D_pos.y);
    idx_r = make_int2( blockIdx.x * blockDim.x + threadIdx.x - 1,
                       thread_2D_pos.y);
    u = idx_u.x * numCols + idx_u.y;
    d = idx_d.x * numCols + idx_d.y;
    l = idx_l.x * numCols + idx_l.y;
    r = idx_r.x * numCols + idx_r.y;
   
    if (d_mask[u] == 1)
      sum1 += d_guess_prev[u];
    else if (d_mask[u] == 2)
      sum1 += d_dest[u];
   
    if (d_mask[d] == 1)
      sum1 += d_guess_prev[d];
    else if (d_mask[d] == 2)
      sum1 += d_dest[d];
   
    if (d_mask[l] == 1)
      sum1 += d_guess_prev[l];
    else if (d_mask[l] == 2)
      sum1 += d_dest[l];
   
    if (d_mask[r] == 1)
      sum1 += d_guess_prev[r];
    else if (d_mask[r] == 2)
      sum1 += d_dest[r];
    
    d_guess_prev[idx] = d_guess[idx];

    sum2 += 4.f*d_source[idx] - d_source[u] - d_source[d] 
                              - d_source[l] - d_source[r];
    float newVal = (sum1 + sum2) / 4.f;
    d_guess[idx] = min(255.0, max(0.0, newVal));
  }
}
           


void your_blend(const uchar4* const h_sourceImg, //IN
                const size_t numRows,
                const size_t numCols,
                const uchar4* const h_destImg,   //IN
                uchar4* const h_blendedImg)      //OUT
{
  const unsigned int n = numRows * numCols;
  
  int k = 16;
  
  // Set reasonable block size (i.e., number of threads per block)
  const dim3 threads( k, k, 1);
  
  // Compute correct grid size (i.e., number of blocks per kernel launch)
  //   from the image size and and block size.
  const dim3 blocks( numRows/k+1, numCols/k+1, 1);

  //const int maxThreadsPerBlock = 1024;
  //int threads = maxThreadsPerBlock;
  //int blocks  = n / maxThreadsPerBlock + 1;
  //int shared  = threads * sizeof(float);

  unsigned char * d_source_red,     * d_source_green,    * d_source_blue;
  unsigned char * d_dest_red,       * d_dest_green,      * d_dest_blue;
  float         * d_guess_red,      * d_guess_blue,      * d_guess_green;
  float         * d_guess_prev_red, * d_guess_prev_blue, * d_guess_prev_green;
  unsigned int  * d_mask;
  uchar4        * d_sourceImg,      * d_destImg;
  
  // sizes of arrays :
  size_t p_size = sizeof(unsigned char) * n;
  size_t f_size = sizeof(float) * n;
  size_t i_size = sizeof(uchar4) * n;
  size_t m_size = sizeof(unsigned int) * n;

  //============================================================================
  // allocate the memory :

  // allocate memory for the three different channels (source and dest) :
  checkCudaErrors(cudaMalloc(&d_source_red,   p_size));
  checkCudaErrors(cudaMalloc(&d_source_green, p_size));
  checkCudaErrors(cudaMalloc(&d_source_blue,  p_size));
  checkCudaErrors(cudaMalloc(&d_dest_red,   p_size));
  checkCudaErrors(cudaMalloc(&d_dest_green, p_size));
  checkCudaErrors(cudaMalloc(&d_dest_blue,  p_size));

  // allocate memory for the previous and current guesses :
  checkCudaErrors(cudaMalloc(&d_guess_red,   p_size));
  checkCudaErrors(cudaMalloc(&d_guess_green, p_size));
  checkCudaErrors(cudaMalloc(&d_guess_blue,  p_size));
  checkCudaErrors(cudaMalloc(&d_guess_prev_red,   p_size));
  checkCudaErrors(cudaMalloc(&d_guess_prev_green, p_size));
  checkCudaErrors(cudaMalloc(&d_guess_prev_blue,  p_size));
  
  // allocate memory for the source image and copy from host : 
  checkCudaErrors(cudaMalloc((void**) &d_sourceImg, i_size));
  checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, i_size,
                             cudaMemcpyHostToDevice));
  
  // allocate memory for the destination image and copy from host : 
  checkCudaErrors(cudaMalloc((void**) &d_destImg, i_size));
  checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, i_size,
                             cudaMemcpyHostToDevice));
  
  // allocate memory for the mask and init to 0 :
  checkCudaErrors(cudaMalloc((void**) &d_mask, m_size));
  checkCudaErrors(cudaMemset(d_mask, 0, m_size));
  
  //============================================================================
  
  /* 1) Separate out the incoming images into three separate channels */
  separateChannels<<<blocks, threads>>>(d_sourceImg,
                                        numRows,
                                        numCols,
                                        d_source_red,
                                        d_source_green,
                                        d_source_blue);
  
  separateChannels<<<blocks, threads>>>(d_destImg,
                                        numRows,
                                        numCols,
                                        d_dest_red,
                                        d_dest_green,
                                        d_dest_blue);
  
  //cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  
  /* 2) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied. */
  
  /* 3) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't. */

  gen_mask<<<blocks, threads>>>(d_mask,
                                d_source_red,
                                d_source_green,
                                d_source_blue,
                                numRows,
                                numCols);
  
  //cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  /*
  apply_mask<<<blocks, threads>>>(d_mask,
                                  d_sourceImg,
                                  numRows,
                                  numCols);

  checkCudaErrors(cudaMemcpy(h_blendedImg, d_sourceImg, i_size,
                             cudaMemcpyDeviceToHost));
  */

  /* 4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess. */
  init_guess<<<blocks, threads>>>(d_mask,
                                  d_guess_red,
                                  d_source_red,
                                  numRows,
                                  numCols);
  init_guess<<<blocks, threads>>>(d_mask,
                                  d_guess_green,
                                  d_source_green,
                                  numRows,
                                  numCols);
  init_guess<<<blocks, threads>>>(d_mask,
                                  d_guess_blue,
                                  d_source_blue,
                                  numRows,
                                  numCols);
  
  init_guess<<<blocks, threads>>>(d_mask,
                                  d_guess_prev_red,
                                  d_source_red,
                                  numRows,
                                  numCols);
  init_guess<<<blocks, threads>>>(d_mask,
                                  d_guess_prev_green,
                                  d_source_green,
                                  numRows,
                                  numCols);
  init_guess<<<blocks, threads>>>(d_mask,
                                  d_guess_prev_blue,
                                  d_source_blue,
                                  numRows,
                                  numCols);
  
  /* 5) For each color channel perform the Jacobi iteration described 
        above 800 times. */
  for (int i = 0; i < 800; i++)
  {
    solve<<<blocks, threads>>>(d_guess_red,
                               d_guess_prev_red,
                               d_mask,
                               d_source_red,
                               d_dest_red,
                               numRows,
                               numCols);
    solve<<<blocks, threads>>>(d_guess_green,
                               d_guess_prev_green,
                               d_mask,
                               d_source_green,
                               d_dest_green,
                               numRows,
                               numCols);
    solve<<<blocks, threads>>>(d_guess_blue,
                               d_guess_prev_blue,
                               d_mask,
                               d_source_blue,
                               d_dest_blue,
                               numRows,
                               numCols);
    //cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  }

  /* 6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range. */
    
  replace_dest<<<blocks, threads>>>(d_guess_red,
                                    d_guess_green,
                                    d_guess_blue,
                                    d_mask,
                                    d_destImg,
                                    numRows,
                                    numCols);
  
  checkCudaErrors(cudaMemcpy(h_blendedImg, d_destImg, i_size,
                             cudaMemcpyDeviceToHost));
  /*
  uchar4* h_reference = new uchar4[n];
  reference_calc(h_sourceImg, numRows, numCols,
                 h_destImg, h_reference);
  checkResultsEps((unsigned char *)h_reference,
                  (unsigned char *)h_blendedImg, 4 * n, 2, .01);
  delete[] h_reference; */
}

