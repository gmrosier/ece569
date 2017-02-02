#include <wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  int i = blockDim.x * blockDim.y + threadIdx.x;
  if (i < len)
  {
  	out[i] = in1[i] + in2[i];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;
  size_t size;
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  size = inputLength * sizeof(float);

  wbTime_start(GPU, "Allocating GPU memory.");
  if (cudaMalloc(&deviceInput1, size) != cudaSuccess)
  {
  	wbLog(TRACE, "Unable to Allocation Memory on GPU");
  }

  if (cudaMalloc(&deviceInput2, size) != cudaSuccess)
  {
  	wbLog(TRACE, "Unable to Allocation Memory on GPU");
  }

  if (cudaMalloc(&deviceOutput, size) != cudaSuccess)
  {
  	wbLog(TRACE, "Unable to Allocation Memory on GPU");
  }

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  int threadsPerBlock = 256;
  int blocksPerGrid = (inputLength + threadsPerBlock - 1) / threadsPerBlock;
  
  wbTime_start(Compute, "Performing CUDA computation");
  vecAdd<<<blocksPerGrid, threadsPerBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  wbLog(TRACE, "All Done!");
  return 0;
}
