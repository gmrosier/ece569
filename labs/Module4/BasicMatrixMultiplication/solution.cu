
#include <wb.h>

#define BLOCK_SIZE  (16)

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x; // Column
	int row = threadIdx.y + blockIdx.y * blockDim.y; // Row
	int width = numBColumns;
	int height = numARows;
	int multiSize = (numAColumns < numBRows) ? numAColumns : numBRows;

	if ((row < height) && (col < width))
	{
		float value = 0;
		for (int i = 0; i < multiSize; ++i)
		{
			value += A[row * numAColumns + i] * B[i * numBColumns + col];
		}
		C[row*width + col] = value;
	}
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA; // A matrix on device
  float *deviceB; // B matrix on device
  float *deviceC; // C matrix on device
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows    = numARows;    // set to correct value
  numCColumns = numBColumns; // set to correct value
  //@@ Allocate the hostC matrix
  hostC = (float*) malloc(numCRows * numCColumns * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);
  
  wbTime_start(GPU, "Allocating GPU memory.");

  if (cudaMalloc(&deviceA, numARows*numAColumns*sizeof(float)) != cudaSuccess)
  {
	  wbLog(TRACE, "Unable to Allocation Memory on GPU");
  }

  if (cudaMalloc(&deviceB, numBRows*numBColumns * sizeof(float)) != cudaSuccess)
  {
	  wbLog(TRACE, "Unable to Allocation Memory on GPU");
  }

  if (cudaMalloc(&deviceC, numCRows*numCColumns * sizeof(float)) != cudaSuccess)
  {
	  wbLog(TRACE, "Unable to Allocation Memory on GPU");
  }

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");

  cudaMemcpy(deviceA, hostA, numARows*numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows*numBColumns * sizeof(float), cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  // set block size to 16,16 and determine the grid dimensions
  // use dim3 structure for setting block and grid dimensions
  dim3 DimGrid((numCColumns-1)/BLOCK_SIZE + 1, (numCRows-1)/BLOCK_SIZE + 1, 1);
  dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  matrixMultiply<<<DimGrid, DimBlock >>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCRows*numCColumns * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
