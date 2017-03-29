#include <wb.h>

#define NUM_BINS 4096
#define MAX_CNT  127

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

__global__ void histogram_global_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = blockDim.x*gridDim.x;

  // in case there weren't enough total threads for all elements, must stride across input data each thread doing it's share of the work
  while (i < num_elements) {
    int val = input[i];
    atomicAdd(&bins[val], 1);  // update global full histogram held in global memory, by incrementing the bin corresponding to this value
    i += stride;               // proceed to next input data element if there are more left to do
  }
}

__global__ void histogram_shared_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
  // declare shared memory
  extern __shared__ unsigned int bin_s[];

  for (int i=threadIdx.x; i < num_bins;i += blockDim.x) {
    bin_s[i] = 0;
  }
  __syncthreads(); // guarantee that all threads in this block have done their share of the initialization


  // update local partial histogram held in my block's shared memory
  int stride = blockDim.x*gridDim.x; // will stride through full data set, each thread processing inputs on a stride same as other kernel
  for (unsigned int i= blockIdx.x*blockDim.x + threadIdx.x;i<num_elements;i+=stride) { 
    atomicAdd(&bin_s[input[i]], 1);  // update local histogram by incrementing the bin corresponding to this value
  }
  __syncthreads(); // guarantee all threads in block have completed their histograms

  // update global memory, by using atomicAdd to combine local histogram with global histogram
  for (int i=threadIdx.x;i<num_bins;i+=blockDim.x) {
    atomicAdd(&bins[i], bin_s[i]);
  }
}

// kernel that corrects the bins to apply limit 
__global__ void limit_kernel(
  unsigned int *bins,
  unsigned int num_bins) {

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = blockDim.x*gridDim.x;

  while (i < num_bins) {
    bins[i] = min(bins[i], MAX_CNT);  // enforce a limit of MAX_CNT
    i += stride; // in case the thread count was less than the bin count, will need to stride through the bins to complete
  }
}

void histogram(unsigned int *input, unsigned int *bins,
               unsigned int num_elements, unsigned int num_bins, int kernel_version) {


 if (kernel_version == 0) {
  // zero out bins
  CUDA_CHECK(cudaMemset(bins, 0, num_bins * sizeof(unsigned int)));
  // Launch histogram kernel on the bins
  {
    dim3 blockDim(1024), gridDim((num_elements + blockDim.x - 1) / blockDim.x);
    histogram_global_kernel<<<gridDim, blockDim>>>(
        input, bins, num_elements, num_bins);
    // apply limiting kernel to clip bins to MAX_CNT
    dim3 lBlockSize(512);
    dim3 lGridSize((num_bins + lBlockSize.x - 1)/lBlockSize.x);
    limit_kernel             <<< lGridSize, lBlockSize >>> (bins, num_bins);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

 }
 else if (kernel_version==1) {
  // zero out bins
  CUDA_CHECK(cudaMemset(bins, 0, num_bins * sizeof(unsigned int)));
  // Launch histogram kernel on the bins
  {
    dim3 blockDim(1024), gridDim((num_elements + blockDim.x - 1) / blockDim.x);
    histogram_shared_kernel<<<gridDim, blockDim,
                       num_bins * sizeof(unsigned int)>>>(
        input, bins, num_elements, num_bins);
    // apply limiting kernel to clip bins to MAX_CNT
    dim3 lBlockSize(512);
    dim3 lGridSize((num_bins + lBlockSize.x - 1)/lBlockSize.x);
    limit_kernel             <<< lGridSize, lBlockSize >>> (bins, num_bins);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

 }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int inputLength;
  int version; // kernel version global or shared 
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
  //@@ Allocate GPU memory here
  CUDA_CHECK(cudaMalloc((void **)&deviceInput,
                        inputLength * sizeof(unsigned int)));
  CUDA_CHECK(
      cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(unsigned int)));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  CUDA_CHECK(cudaMemcpy(deviceInput, hostInput,
                        inputLength * sizeof(unsigned int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // Launch kernel
  // ----------------------------------------------------------
  wbTime_start(Compute, "Performing CUDA computation");

  version = 0; 
  histogram(deviceInput, deviceBins, inputLength, NUM_BINS,version);
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  CUDA_CHECK(cudaMemcpy(hostBins, deviceBins,
                        NUM_BINS * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  // Verify correctness
  // -----------------------------------------------------
  wbLog(TRACE, "Checking global memory only kernel");
  wbSolution(args, hostBins, NUM_BINS);


 // Launch kernel
  // ----------------------------------------------------------
  wbLog(TRACE, "Launching shared memory kernel");
  wbTime_start(Compute, "Performing CUDA computation");
  version = 1;

  histogram(deviceInput, deviceBins, inputLength, NUM_BINS, version);
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  CUDA_CHECK(cudaMemcpy(hostBins, deviceBins,
                        NUM_BINS * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  // Verify correctness
  // -----------------------------------------------------
  wbLog(TRACE, "Checking shared memory kernel");
  wbSolution(args, hostBins, NUM_BINS);


  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  CUDA_CHECK(cudaFree(deviceInput));
  CUDA_CHECK(cudaFree(deviceBins));
  wbTime_stop(GPU, "Freeing GPU Memory");


  free(hostBins);
  free(hostInput);
  return 0;
}
