#include <wb.h>
#include <math.h>

#define NUM_BINS 4096
#define BLOCK_SIZE 1024

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

/*  
   version 1: global memory only coalesced memory access 
              with interleaved partitioning (striding)
*/ 
__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
	unsigned int num_elements, unsigned int num_bins)
{
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int idx;

    for (idx = i; idx < num_elements; idx += stride)
    {
        unsigned int value = input[idx];
        if (value < num_bins)
        {
            atomicAdd(&bins[value], 1);
        }
    }
}

/*  
   version 2: shared memory with privitization
*/
__global__ void histogram_private_kernel(unsigned int *input, unsigned int *bins,
    unsigned int num_elements, unsigned int num_bins)
{
    extern __shared__ unsigned int sBins[];

    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int idx;

    // Phase 1 - Clear Bins
    for (idx = threadIdx.x; idx < num_bins; idx += blockDim.x)
    {
        sBins[idx] = 0;
    }
    __syncthreads();

    // Phase 2 - Add to Shared Bins
    for (idx = i; idx < num_elements; idx += stride)
    {
        unsigned int value = input[idx];
        if (value < num_bins)
        {
            atomicAdd(&sBins[value], 1);
        }
    }
    __syncthreads();

    // Phase 3 - Add Bins to Global Memory
    for (idx = threadIdx.x; idx < num_bins; idx += blockDim.x)
    {
        atomicAdd(&bins[idx], sBins[idx]);
    }
}

/* Cliping Kernel */
__global__ void histogram_cliping(unsigned int * bins, unsigned int num_bins)
{
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < num_bins)
    {
        unsigned int value = bins[i];
        if (value > 127)
        {
            bins[i] = 127;
        }
    }
}

int main(int argc, char *argv[]) {
  cudaEvent_t kstart1, kstop1, kstart2, kstop2;
  
  wbArg_t args;
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins1;
  unsigned int *hostBins2;
  unsigned int *deviceInput;
  unsigned int *deviceBins1;
  unsigned int *deviceBins2;

  cudaEventCreate(&kstart1);
  cudaEventCreate(&kstop1);
  cudaEventCreate(&kstart2);
  cudaEventCreate(&kstop2);

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0),
                                       &inputLength, "Integer");
  hostBins1 = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  hostBins2 = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  wbLog(TRACE, "The number of bins is ", NUM_BINS);

  wbTime_start(GPU, "Allocating GPU memory.");
  if (cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int)) != cudaSuccess)
  {
      wbLog(TRACE, "Unable to Allocation Memory on GPU");
  }

  if (cudaMalloc(&deviceBins1, NUM_BINS * sizeof(unsigned int)) != cudaSuccess)
  {
      wbLog(TRACE, "Unable to Allocation Memory on GPU");
  }

  if (cudaMalloc(&deviceBins2, NUM_BINS * sizeof(unsigned int)) != cudaSuccess)
  {
      wbLog(TRACE, "Unable to Allocation Memory on GPU");
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemset(deviceBins1, 0, NUM_BINS * sizeof(unsigned int));
  cudaMemset(deviceBins2, 0, NUM_BINS * sizeof(unsigned int));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");


  //unsigned int gridSize = static_cast<unsigned int>(ceil(inputLength / (1.0 * BLOCK_SIZE)));
  unsigned int gridSize = NUM_BINS / BLOCK_SIZE;
  wbLog(TRACE, "Grid Size: ", gridSize);

  // Launch kernel 1 - Global
  // ----------------------------------------------------------
  wbLog(TRACE, "Launching kernel 1");
  wbTime_start(Compute, "Kernel 1");
  cudaEventRecord(kstart1);
  histogram_kernel<<< gridSize, BLOCK_SIZE >>>(deviceInput, deviceBins1, inputLength, NUM_BINS);
  cudaDeviceSynchronize();
  histogram_cliping << < gridSize, BLOCK_SIZE >> > (deviceBins1, NUM_BINS);
  cudaEventRecord(kstop1);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Kernel 1");
  
  cudaEventSynchronize(kstop1);
  float  milliseconds1 = 0;
  cudaEventElapsedTime(&milliseconds1, kstart1, kstop1);
  wbLog(TRACE, "Elapsed kernel time (Version 1): " , milliseconds1);
  // ----------------------------------------------------------

  // Launch kernel 2 - Shared
  // ----------------------------------------------------------
  wbLog(TRACE, "Launching kernel 2");
  wbTime_start(Compute, "Kernel 2");
  cudaEventRecord(kstart2);
  histogram_private_kernel << < gridSize, BLOCK_SIZE, NUM_BINS * sizeof(unsigned int) >> >(deviceInput, deviceBins2, inputLength, NUM_BINS);
  cudaDeviceSynchronize();
  histogram_cliping << < gridSize, BLOCK_SIZE >> > (deviceBins2, NUM_BINS);
  cudaEventRecord(kstop2);
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Compute, "Kernel 2");

  cudaEventSynchronize(kstop2);
  float  milliseconds2 = 0;
  cudaEventElapsedTime(&milliseconds2, kstart2, kstop2);
  wbLog(TRACE, "Elapsed kernel time (Version 2): ", milliseconds2);
  // ----------------------------------------------------------
  
  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostBins1, deviceBins1, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostBins2, deviceBins2, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceBins1);
  cudaFree(deviceBins2);
  wbTime_stop(GPU, "Freeing GPU Memory");

  // Verify correctness
  // -----------------------------------------------------
  wbSolution(args, hostBins1, NUM_BINS);
  wbSolution(args, hostBins2, NUM_BINS);

  free(hostBins1);
  free(hostBins2);
  free(hostInput);
  return 0;
}
