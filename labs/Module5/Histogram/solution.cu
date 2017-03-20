#include <wb.h>

#define NUM_BINS 4096
#define BLOCK_SIZE 32

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

    while (i < num_elements)
    {
        unsigned int value = input[i];
        if (value < num_bins)
        {
            atomicAdd(&bins[value], 1);
        }
        i += stride;
    }
}

/*  
   version 2: shared memory with privitization
*/
__global__ void histogram_private_kernel(unsigned int *input, unsigned int *bins,
    unsigned int num_elements, unsigned int num_bins)
{
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
  wbArg_t args;
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0),
                                       &inputLength, "Integer");
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  wbLog(TRACE, "The number of bins is ", NUM_BINS);

  wbTime_start(GPU, "Allocating GPU memory.");
  if (cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int)) != cudaSuccess)
  {
      wbLog(TRACE, "Unable to Allocation Memory on GPU");
  }

  if (cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int)) != cudaSuccess)
  {
      wbLog(TRACE, "Unable to Allocation Memory on GPU");
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // Launch kernel
  // ----------------------------------------------------------
  wbLog(TRACE, "Launching kernel");
  wbTime_start(Compute, "Performing CUDA computation");
  histogram_kernel<<< 4096 / BLOCK_SIZE, BLOCK_SIZE >>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  cudaDeviceSynchronize();
  histogram_cliping << < 4096 / BLOCK_SIZE, BLOCK_SIZE >> > (deviceBins, NUM_BINS);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceBins);
  wbTime_stop(GPU, "Freeing GPU Memory");

  // Verify correctness
  // -----------------------------------------------------
  wbSolution(args, hostBins, NUM_BINS);

  free(hostBins);
  free(hostInput);
  return 0;
}
